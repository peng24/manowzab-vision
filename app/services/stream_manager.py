"""
app/services/stream_manager.py
─────────────────────────────────────────────────────────────────
StreamManager — Singleton ที่จัดการ lifecycle ของ pipeline ทั้งหมด
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
from ultralytics import YOLO

from app.config import settings
from app.utils.nlp import LiveDataExtractor
from app.services.vision_service import ContinuousVisionBuffer
from app.services.webhook_service import send_product_event
from app.services.audio_service import AudioService

logger = logging.getLogger(__name__)

def _ytdlp_get_url(youtube_url: str, fmt: str) -> str:
    cmd = ["yt-dlp", "--no-playlist", "--format", fmt, "--get-url", youtube_url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{r.stderr.strip()}")
    return r.stdout.strip().splitlines()[0]


def _ytdlp_get_metadata(youtube_url: str) -> dict:
    """Fetch stream title and upload_date from yt-dlp (JSON dump)."""
    cmd = ["yt-dlp", "--no-playlist", "--dump-json", "--skip-download", youtube_url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8")
        if r.returncode == 0 and r.stdout.strip():
            import json as _json
            info = _json.loads(r.stdout.strip().splitlines()[0])
            title = info.get("title", "") or info.get("fulltitle", "")
            # upload_date is YYYYMMDD string; for live streams it may be today
            raw_date = info.get("upload_date") or info.get("release_date") or ""
            if raw_date and len(raw_date) == 8:
                date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            else:
                from datetime import date
                date_str = date.today().isoformat()
            return {"title": title, "date_str": date_str}
    except Exception as e:
        logger.warning("[Stream] yt-dlp metadata error: %s", e)
    from datetime import date
    return {"title": "", "date_str": date.today().isoformat()}


def get_stream_urls(youtube_url: str) -> tuple[str, str]:
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


# ══════════════════════════════════════════════════════════════════════════════
# StreamManager — Application-level Singleton
# ══════════════════════════════════════════════════════════════════════════════

class StreamManager:
    def __init__(self) -> None:
        self._yolo: YOLO | None = None
        self._stop_event: threading.Event | None = None
        self._capture_lock: threading.Lock = threading.Lock()
        self._captured_codes: set[int] = set()
        
        self._audio_service = AudioService(target_language=settings.target_language)
        self._t_multimodal: threading.Thread | None = None
        
        self._extractor: LiveDataExtractor | None = None
        self._session_id: str = ""
        self._youtube_url: str = ""
        self._started_at: float | None = None
        self._vision_buffer: ContinuousVisionBuffer | None = None
        self._session_date_str: str = ""
        self._stream_title: str = ""
        self._output_dir: Path = settings.output_dir  # default; overridden on start()

    def _ensure_models(self) -> None:
        if self._yolo is None:
            logger.info("[Models] โหลด YOLO '%s' ...", settings.yolo_model_name)
            self._yolo = YOLO(settings.yolo_model_name)
            logger.info("[Models] ✅ YOLO พร้อม")

    @property
    def is_running(self) -> bool:
        return (
            self._stop_event is not None
            and not self._stop_event.is_set()
            and self._t_multimodal is not None
            and self._t_multimodal.is_alive()
        )

    def _multimodal_sync_worker(self) -> None:
        """
        เชื่อมโยงข้อมูลเสียงพูดจาก AudioService เข้ากับข้อความภาพจาก VisionService
        ในหน้าต่างเวลา (Time Window) ใกล้เคียงกัน แล้วส่งให้ NLP
        """
        logger.info("[Multimodal] 🧠 เริ่มต้น Synchronization Engine")
        
        while not self._stop_event.is_set():
            try:
                # 1. รับเสียงพูดมา (Blocking request)
                spoken_ts, spoken_text = self._audio_service.transcription_queue.get(timeout=2)
            except queue.Empty:
                continue

            # 2. ส่งต่อให้ NLP ประมวลผล (Hybrid: Gemini → Ollama)
            product = self._extractor.process_text(spoken_text)
            if not product:
                continue

            status = product.get("status", "new")
            item_code: int = product["item_code"]
            price: int = product["price"]

            logger.info("[NLP] 🛍️ item=%d price=%d status=%s", item_code, price, status)

            # 4. แคปเจอร์ Vision และส่ง Webhook
            with self._capture_lock:
                if status == "conflict":
                    logger.warning("[Review] 🚨 Price Conflict item=#%d (ก่อน: %s, ใหม่: %s)",
                                   item_code, product.get("old_price"), price)
                    send_product_event(
                        product=product, image_path=None,
                        session_id=self._session_id, date_str=self._session_date_str,
                        output_dir=self._output_dir,
                    )
                    continue

                if item_code in self._captured_codes:
                    continue
                    
                self._captured_codes.add(item_code)

            logger.info("[Vision] ⏳ รอ 3 วินาที เพื่อเก็บเฟรมภาพ (Look-Ahead)...")
            time.sleep(3)
            
            saved_path = self._vision_buffer.get_best_frame_and_save(
                item_code, prefix=self._session_date_str
            )

            send_product_event(
                product=product,
                image_path=saved_path,
                session_id=self._session_id,
                date_str=self._session_date_str,
                output_dir=self._output_dir,
            )


    def start(self, youtube_url: str, session_id: str) -> None:
        if self.is_running:
            raise RuntimeError("Pipeline กำลังทำงานอยู่ — กรุณา /stop-vision ก่อน")

        self._ensure_models()

        logger.info("[Manager] ดึง stream URLs ...")
        audio_url, video_url = get_stream_urls(youtube_url)

        # --- Fetch stream metadata (title + date) ---
        logger.info("[Manager] ดึง stream metadata ...")
        meta = _ytdlp_get_metadata(youtube_url)
        self._stream_title    = meta["title"]
        self._session_date_str = meta["date_str"]
        # Build date-specific output directory
        self._output_dir = Path("data/outputs") / self._session_date_str
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Manager] 📁 Output dir: %s | Title: %s", self._output_dir, self._stream_title or '(ไม่พบ)')

        self._stop_event = threading.Event()
        self._captured_codes = set()
        self._session_id = session_id
        self._youtube_url = youtube_url
        self._started_at = time.time()
        self._extractor = LiveDataExtractor()

        # เริ่ม Vision (ส่ง output_dir แบบ dynamic)
        self._vision_buffer = ContinuousVisionBuffer(
            self._yolo, buffer_seconds=15, fps=4,
            output_dir=self._output_dir,
        )
        self._vision_buffer.start(video_url, self._youtube_url)

        # เริ่ม Audio
        self._audio_service.start(audio_url)

        # เริ่ม Multimodal Fusion
        self._t_multimodal = threading.Thread(
            target=self._multimodal_sync_worker,
            daemon=True,
            name="multimodal-sync",
        )
        self._t_multimodal.start()
        
        logger.info("[Manager] 🚀 Pipeline (Multimodal) เริ่มแล้ว session=%s", session_id)


    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._vision_buffer:
            self._vision_buffer.stop()
        self._audio_service.stop()

        if self._t_multimodal and self._t_multimodal.is_alive():
            self._t_multimodal.join(timeout=5)

        # หยุด Ollama model (unload จาก VRAM/RAM)
        try:
            model = settings.ollama_model
            result = subprocess.run(
                ["ollama", "stop", model],
                capture_output=True, text=True, timeout=8,
            )
            if result.returncode == 0:
                logger.info("[Manager] 🦙 Ollama หยุดโมเดล '%s' เรียบร้อย", model)
            else:
                logger.warning("[Manager] ⚠️ ollama stop: %s", result.stderr.strip() or "no output")
        except FileNotFoundError:
            logger.warning("[Manager] ⚠️ ไม่พบคำสั่ง ollama — ข้ามการหยุด")
        except Exception as e:
            logger.warning("[Manager] ⚠️ ollama stop error: %s", e)

        logger.info("[Manager] 🛑 Pipeline หยุดแล้ว")


    def status(self) -> dict:
        return {
            "running": self.is_running,
            "session_id": self._session_id or None,
            "youtube_url": self._youtube_url or None,
            "started_at": self._started_at,
            "captured_count": len(self._captured_codes),
            "captured_codes": sorted(self._captured_codes),
            "stream_title": self._stream_title or None,
            "output_dir": str(self._output_dir),
        }

stream_manager = StreamManager()
