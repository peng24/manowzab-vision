"""
app/services/stream_manager.py
─────────────────────────────────────────────────────────────────
StreamManager — Singleton ที่จัดการ lifecycle ของ pipeline ทั้งหมด

  start(youtube_url, session_id)
    ├── get_stream_urls()         → audio_url, video_url
    ├── Thread: audio_producer    → queue
    └── Thread: transcribe_worker → NLP → Vision → Webhook

  stop()
    └── set stop_event → threads หยุดเรียบร้อย

  status → dict  (สำหรับ GET /status endpoint)
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel
from ultralytics import YOLO

from app.config import settings
from app.utils.nlp import LiveDataExtractor
from app.services.vision_service import ContinuousVisionBuffer
from app.services.webhook_service import send_product_event

logger = logging.getLogger(__name__)

# ─── yt-dlp helpers ────────────────────────────────────────────────────────

def _ytdlp_get_url(youtube_url: str, fmt: str) -> str:
    """เรียก yt-dlp --get-url แบบ synchronous"""
    cmd = ["yt-dlp", "--no-playlist", "--format", fmt, "--get-url", youtube_url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{r.stderr.strip()}")
    return r.stdout.strip().splitlines()[0]


def get_stream_urls(youtube_url: str) -> tuple[str, str]:
    """
    คืน (audio_url, video_url)
    video_url: 720p mp4 ถ้าได้, fallback = audio_url
    """
    audio_url = _ytdlp_get_url(youtube_url, "bestaudio/best")
    try:
        video_url = _ytdlp_get_url(
            youtube_url,
            "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
        )
    except Exception:
        logger.warning("[Stream] ไม่พบ video stream แยก → ใช้ URL เดิม")
        video_url = audio_url
    return audio_url, video_url


# ─── Audio Producer Thread ────────────────────────────────────────────────

def _audio_producer(
    audio_url: str,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """ffmpeg → raw PCM 16kHz mono int16 → queue"""
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", audio_url,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(settings.sample_rate),
        "-ac", "1", "-f", "s16le", "pipe:1",
    ]
    logger.info("[Audio] 🎙️  ffmpeg เริ่มสตรีม")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf = b""
    try:
        while not stop_event.is_set():
            chunk = proc.stdout.read(4096)
            if not chunk:
                logger.warning("[Audio] ffmpeg stream หยุดแล้ว")
                break
            buf += chunk
            while len(buf) >= settings.chunk_bytes:
                audio_queue.put(buf[:settings.chunk_bytes])
                buf = buf[settings.chunk_bytes:]
    finally:
        proc.terminate()
        stop_event.set()
        audio_queue.put(None)  # sentinel


# ─── Transcribe + NLP + Vision Worker ─────────────────────────────────────

def _transcribe_worker(
    whisper_model: WhisperModel,
    vision_buffer: ContinuousVisionBuffer,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    capture_lock: threading.Lock,
    captured_codes: set[int],
    session_id: str,
    on_transcript: Callable[[str], None] | None,
    extractor: LiveDataExtractor,
    session_date_str: str,
) -> None:
    """
    Loop หลัก:
      audio chunk → Whisper → extract_product_data()
        → vision_buffer.get_best_frame_and_save() [synchronous from Memory]
          → send_product_event() [webhook]
    """
    logger.info("[Transcribe] 📝  เริ่มทำงาน (VAD=off, CPU-safe mode)")

    # อนุญาตให้ reload model เป็น CPU ถ้า CUDA crash กลางทาง
    _model = whisper_model
    _cuda_failed = False

    while not stop_event.is_set():
        try:
            raw_bytes = audio_queue.get(timeout=2)
        except queue.Empty:
            continue

        if raw_bytes is None:
            break

        audio_np = (
            np.frombuffer(raw_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )
        ts = time.strftime("%H:%M:%S")

        try:
            # ── Transcribe (VAD ปิด — กรองเสียงไทยออกหมด) ────────────────
            segments, _ = _model.transcribe(
                audio_np,
                language=settings.target_language,
                beam_size=5,
                vad_filter=False,           # ❌ ปิด VAD — กรองเสียงไทยออกหมด
                condition_on_previous_text=True,
                temperature=0.0,            # Greedy decode — เร็วและ stable
                initial_prompt="รายการที่ 1 รหัส 2 ตัวที่ 3 ราคา 50 บาท 100 บาท ร้อยนึง เอฟเสื้อ",
            )

            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if not texts:
                continue

            full_text = " ".join(texts)
            logger.info("[%s] 📣  %s", ts, full_text)

            if on_transcript:
                on_transcript(full_text)

            # ── NLP ──────────────────────────────────────────────────────
            product = extractor.process_text(full_text)
            if not product:
                continue

            status = product.get("status", "new")
            item_code: int = product["item_code"]
            price: int = product["price"]

            logger.info("[NLP] 🛍️  item=%d price=%d status=%s", item_code, price, status)

            # ── Vision Trigger / Webhook ─────────────────────────────────
            with capture_lock:
                if status == "conflict":
                    logger.warning("[Review] 🚨  Price Conflict item=#%d (ก่อน: %s, ใหม่: %s)", 
                                   item_code, product.get("old_price"), price)
                    send_product_event(
                        product=product, image_path=None, session_id=session_id, date_str=session_date_str
                    )
                    continue

                if item_code in captured_codes:
                    continue
                    
                captured_codes.add(item_code)

            # ให้ Buffer ดึงรูปย้อนหลังที่คะแนนดีที่สุดมาบันทึกลง disk ทันที
            saved_path = vision_buffer.get_best_frame_and_save(item_code, session_date_str)
            
            # ยิง Webhook หลังจากดึงภาพสำเร็จ
            send_product_event(
                product=product,
                image_path=saved_path,
                session_id=session_id,
                date_str=session_date_str,
            )

        except RuntimeError as exc:
            err = str(exc).lower()
            if ("cublas" in err or "cuda" in err or "cudnn" in err) and not _cuda_failed:
                # ── CUDA crash กลางทาง → Reload เป็น CPU ──────────────────
                _cuda_failed = True
                logger.error("[Transcribe] ⚠️  CUDA runtime ล้มเหลว → Reload CPU/int8 ...")
                try:
                    _model = WhisperModel(
                        settings.whisper_model_size,
                        device="cpu", compute_type="int8",
                    )
                    logger.info("[Transcribe] ✅  Reload CPU สำเร็จ — ทำงานต่อ")
                except Exception as reload_err:
                    logger.error("[Transcribe] ❌  Reload ล้มเหลว: %s — หยุด", reload_err)
                    stop_event.set()
            else:
                logger.error("[Transcribe] ❌  %s", exc, exc_info=True)

        except Exception as exc:
            logger.error("[Transcribe] ❌  %s", exc, exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# StreamManager — Application-level Singleton
# ══════════════════════════════════════════════════════════════════════════════

class StreamManager:
    """
    จัดการ lifecycle ของ pipeline
    ใช้ผ่าน module-level singleton: `stream_manager`
    """

    def __init__(self) -> None:
        self._whisper: WhisperModel | None  = None
        self._yolo:    YOLO        | None   = None
        self._stop_event:   threading.Event | None = None
        self._capture_lock: threading.Lock         = threading.Lock()
        self._captured_codes: set[int]             = set()
        self._t_audio:      threading.Thread | None = None
        self._t_transcribe: threading.Thread | None = None
        self._extractor:    LiveDataExtractor | None = None
        self._session_id:   str = ""
        self._youtube_url:  str = ""
        self._started_at:   float | None = None
        self._vision_buffer: ContinuousVisionBuffer | None = None
        self._session_date_str: str = ""

    # ── Model Loaders ────────────────────────────────────────────────────

    def _ensure_models(self) -> None:
        """โหลดโมเดลครั้งแรก (lazy init — ไม่โหลดซ้ำ)"""
        if self._whisper is None:
            logger.info("[Models] โหลด Whisper '%s' บน %s ...",
                        settings.whisper_model_size, settings.whisper_device)
            try:
                self._whisper = WhisperModel(
                    settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute,
                )
            except Exception as e:
                logger.warning("[Models] GPU ไม่พร้อม (%s) → CPU/int8", e)
                self._whisper = WhisperModel(
                    settings.whisper_model_size,
                    device="cpu", compute_type="int8",
                )
            logger.info("[Models] ✅  Whisper พร้อม")

        if self._yolo is None:
            logger.info("[Models] โหลด YOLO '%s' ...", settings.yolo_model_name)
            self._yolo = YOLO(settings.yolo_model_name)
            logger.info("[Models] ✅  YOLO พร้อม")

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return (
            self._stop_event is not None
            and not self._stop_event.is_set()
            and self._t_transcribe is not None
            and self._t_transcribe.is_alive()
        )

    def start(self, youtube_url: str, session_id: str) -> None:
        """
        เริ่ม pipeline:
          1) โหลดโมเดล (lazy)
          2) ดึง stream URLs
          3) spawn threads
        Raises RuntimeError ถ้า pipeline กำลังรันอยู่แล้ว
        """
        if self.is_running:
            raise RuntimeError("Pipeline กำลังทำงานอยู่ — กรุณา /stop-vision ก่อน")

        self._ensure_models()

        logger.info("[Manager] ดึง stream URLs ...")
        audio_url, video_url = get_stream_urls(youtube_url)

        self._stop_event     = threading.Event()
        self._captured_codes = set()
        self._session_id     = session_id
        self._youtube_url    = youtube_url
        self._started_at     = time.time()
        self._extractor      = LiveDataExtractor()
        
        # วันที่และปี พ.ศ. ของ Session นี้เพื่อใช้เป็นชื่อโฟลเดอร์ (ex: 1-4-69)
        from datetime import datetime
        now = datetime.now()
        year_be = str(now.year + 543)[-2:]
        self._session_date_str = f"{now.day}-{now.month}-{year_be}"

        audio_queue = queue.Queue(maxsize=10)

        # เริ่ม Vision Buffer (4 FPS)
        self._vision_buffer = ContinuousVisionBuffer(self._yolo, buffer_seconds=15, fps=4)
        self._vision_buffer.start(video_url, self._youtube_url)

        self._t_audio = threading.Thread(
            target=_audio_producer,
            args=(audio_url, audio_queue, self._stop_event),
            daemon=True,
            name="audio-producer",
        )
        self._t_transcribe = threading.Thread(
            target=_transcribe_worker,
            args=(
                self._whisper,
                self._vision_buffer,   
                audio_queue,
                self._stop_event,
                self._capture_lock,
                self._captured_codes,
                session_id,
                None,          # on_transcript callback (None = log เท่านั้น)
                self._extractor,
                self._session_date_str,
            ),
            daemon=True,
            name="transcribe-worker",
        )

        self._t_audio.start()
        self._t_transcribe.start()
        logger.info("[Manager] 🚀  Pipeline เริ่มแล้ว  session=%s", session_id)

    def stop(self) -> None:
        """หยุด pipeline — thread จะเสร็จสิ้นภายใน 5 วินาที"""
        if self._stop_event:
            self._stop_event.set()
        if self._vision_buffer:
            self._vision_buffer.stop()
        if self._t_audio and self._t_audio.is_alive():
            self._t_audio.join(timeout=5)
        if self._t_transcribe and self._t_transcribe.is_alive():
            self._t_transcribe.join(timeout=5)
        logger.info("[Manager] 🛑  Pipeline หยุดแล้ว")

    def status(self) -> dict:
        return {
            "running":     self.is_running,
            "session_id":  self._session_id or None,
            "youtube_url": self._youtube_url or None,
            "started_at":  self._started_at,
            "captured_count": len(self._captured_codes),
            "captured_codes": sorted(self._captured_codes),
        }


# Module-level singleton
stream_manager = StreamManager()
