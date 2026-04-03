import logging
import queue
import subprocess
import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from app.config import settings

logger = logging.getLogger(__name__)

class AudioService:
    """
    Multimodal Speech-to-Text Pipeline
    แยกตัวจัดการ FFmpeg และโมเดล Whisper ออกมาเป็น Service โดยเฉพาะ
    """
    def __init__(self, target_language="th"):
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue(maxsize=10)
        self.transcription_queue = queue.Queue(maxsize=20)
        self.target_language = target_language
        self.whisper_model = None
        self._t_audio = None
        self._t_transcribe = None
        self._cuda_failed = False
        
    def ensure_model(self):
        if self.whisper_model is None:
            logger.info("[AudioService] โหลด Whisper '%s' บน %s ...",
                        settings.whisper_model_size, settings.whisper_device)
            try:
                self.whisper_model = WhisperModel(
                    settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute,
                )
            except Exception as e:
                logger.warning("[AudioService] GPU ไม่พร้อม (%s) → CPU/int8", e)
                self.whisper_model = WhisperModel(
                    settings.whisper_model_size,
                    device="cpu", compute_type="int8",
                )
            logger.info("[AudioService] ✅ Whisper พร้อม")

    def start(self, audio_url: str):
        if self._t_transcribe and self._t_transcribe.is_alive():
            return
            
        self.stop_event.clear()
        
        # Flush raw queues
        while not self.transcription_queue.empty(): self.transcription_queue.get_nowait()
        while not self.audio_queue.empty(): self.audio_queue.get_nowait()
        
        self.ensure_model()
        
        self._t_audio = threading.Thread(
            target=self._audio_producer, args=(audio_url,), daemon=True, name="audio-producer"
        )
        self._t_transcribe = threading.Thread(
            target=self._transcribe_worker, daemon=True, name="transcribe-worker"
        )
        
        self._t_audio.start()
        self._t_transcribe.start()

    def stop(self):
        self.stop_event.set()
        if self._t_audio and self._t_audio.is_alive():
            self._t_audio.join(timeout=5)
        if self._t_transcribe and self._t_transcribe.is_alive():
            self._t_transcribe.join(timeout=5)

    def _audio_producer(self, audio_url: str):
        cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-i", audio_url,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(settings.sample_rate),
            "-ac", "1", "-f", "s16le", "pipe:1",
        ]
        logger.info("[Audio] 🎙️ ffmpeg เริ่มสตรีม")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf = b""
        try:
            while not self.stop_event.is_set():
                chunk = proc.stdout.read(4096)
                if not chunk:
                    logger.warning("[Audio] ffmpeg stream หยุดแล้ว")
                    break
                buf += chunk
                while len(buf) >= settings.chunk_bytes:
                    try:
                        self.audio_queue.put_nowait(buf[:settings.chunk_bytes])
                    except queue.Full:
                        pass
                    buf = buf[settings.chunk_bytes:]
        finally:
            proc.terminate()
            self.stop_event.set()
            self.audio_queue.put(None)

    def _transcribe_worker(self):
        logger.info("[Transcribe] 📝  เริ่มทำงาน (VAD=off, CPU-safe mode)")
        _model = self.whisper_model
        while not self.stop_event.is_set():
            try:
                raw_bytes = self.audio_queue.get(timeout=2)
            except queue.Empty:
                continue

            if raw_bytes is None:
                break

            audio_np = (np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
            now = time.time()
            
            try:
                segments, _ = _model.transcribe(
                    audio_np,
                    language=self.target_language,
                    beam_size=5, vad_filter=False, condition_on_previous_text=True,
                    temperature=0.0,
                    initial_prompt="รายการที่ 1 รหัส 2 ตัวที่ 3 ราคา 50 บาท 100 บาท ร้อยนึง เอฟเสื้อ",
                )
                texts = [seg.text.strip() for seg in segments if seg.text.strip()]
                if texts:
                    full_text = " ".join(texts)
                    logger.info("[%s] 📣 %s", time.strftime("%H:%M:%S", time.localtime(now)), full_text)
                    try:
                        self.transcription_queue.put_nowait((now, full_text))
                    except queue.Full:
                        pass
            except RuntimeError as exc:
                err = str(exc).lower()
                if ("cublas" in err or "cuda" in err or "cudnn" in err) and not self._cuda_failed:
                    self._cuda_failed = True
                    logger.error("[Transcribe] ⚠️ CUDA runtime ล้มเหลว → Reload CPU/int8 ...")
                    try:
                        _model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")
                        self.whisper_model = _model
                        logger.info("[Transcribe] ✅ Reload CPU สำเร็จ")
                    except Exception as reload_err:
                        logger.error("[Transcribe] ❌ Reload ล้มเหลวหยุด: %s", reload_err)
                        self.stop_event.set()
                else:
                    logger.error("[Transcribe] ❌ %s", exc)
            except Exception as exc:
                logger.error("[Transcribe] ❌ %s", exc)
