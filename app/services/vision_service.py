"""
app/services/vision_service.py
─────────────────────────────────────────────────────────────────
Computer Vision Pipeline:
  - variance_of_laplacian()  → วัด sharpness ของเฟรม
  - score_frame()            → ให้คะแนนเฟรม (sharpness × YOLO bonus)
  - capture_best_frame()     → บันทึกเฟรมที่ดีที่สุดใน CAPTURE_DURATION วินาที
  - trigger_capture()        → spawn daemon thread (non-blocking)
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
import yt_dlp

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)


# ─── Sharpness Metric ──────────────────────────────────────────────────────

def variance_of_laplacian(frame_bgr: np.ndarray) -> float:
    """
    คำนวณ sharpness ด้วย Variance of Laplacian
    ค่าสูง = คมชัด / ค่าต่ำ = เบลอ
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─── Frame Scoring ─────────────────────────────────────────────────────────

def score_frame(frame_bgr: np.ndarray, yolo_model: YOLO) -> float:
    """
    ให้คะแนนเฟรม:
      score = sharpness × highest_confidence (ความมั่นใจของ AI ว่าเจอเสื้อ)

    เฟรมที่ sharpness < BLUR_THRESHOLD หรือหาเสื้อไม่เจอ จะได้ score = 0 (ตัดทิ้ง)
    """
    sharpness = variance_of_laplacian(frame_bgr)
    if sharpness < settings.blur_threshold:
        return 0.0

    # ให้ AI หาเสื้อที่เราเทรนมา (ไม่ต้องระบุ classes=[0] แล้ว เพราะโมเดลเรามีแค่คลาสเสื้อคลาสเดียว)
    results = yolo_model(
        frame_bgr,
        verbose=False,
        imgsz=640,
    )
    
    confs = [float(box.conf) for r in results for box in r.boxes]
    
    # ดึงค่าความมั่นใจสูงสุดที่ AI หาเสื้อเจอ
    highest_conf = max(confs) if confs else 0.0
    
    # ถ้า AI ไม่เจอเสื้อเลย หรือเจอแต่มั่นใจน้อยกว่า 60% (เช่น ถูกแม่ค้าบัง) ให้ข้ามเฟรมนี้ไป
    if highest_conf < 0.60:
        return 0.0

    # เอาคะแนนความคมชัดคูณกับความมั่นใจของ AI ยิ่งชัดและ AI มั่นใจมาก คะแนนยิ่งพุ่งสูงปรี๊ด
    return sharpness * highest_conf


# ─── Capture Worker ────────────────────────────────────────────────────────


def get_live_m3u8(youtube_url: str) -> str:
    try:
        import yt_dlp
        ydl_opts = {"format": "best", "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("url", "")
    except Exception as e:
        logger.warning("YT-DLP get_live_m3u8 error: %s", e)
        return ""

import shutil

def capture_best_frame(
    youtube_url: str,
    video_url: str,
    item_code: int,
    yolo_model: YOLO,
    stop_event: threading.Event,
    on_complete: "Callable[[int, Path | None], None] | None" = None,
) -> Path | None:
    output_dir: Path = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{item_code}.jpg"

    tmp_dir = output_dir / f"tmp_{item_code}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[Vision] 📷 เริ่มจับภาพ item=#%d จากสตรีมสด (%.1fs)", item_code, settings.capture_duration)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "quiet",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-live_start_index", "-1",  # 🟢 สำคัญสุด! บังคับดึงภาพจากวินาทีล่าสุดของ Live
        "-i", video_url,
        "-t", "4",                  # เวลาในการแคป
        "-vf", "fps=2",             # แคป 2 ภาพต่อวินาที (ได้ราวๆ 8 ภาพมาเลือก)
        str(tmp_dir / "%03d.jpg")
    ]

    try:
        subprocess.run(cmd, timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("[Vision] ⚠️ Timeout จาก FFmpeg")
    except Exception as e:
        logger.error("[Vision] ❌ FFmpeg error: %s", e)

    # เช็คว่ามีรูปถูกสร้างมาไหม
    img_files = list(tmp_dir.glob("*.jpg"))
    
    # ถ้า URL เก่าใช้ไม่ได้ (หมดอายุ) ดึงใหม่ผ่าน yt-dlp
    if not img_files:
        logger.warning("[Vision] ⚠️ FFmpeg ไม่ได้ภาพ (ลิงก์อาจตาย) ขอดึงลิงก์ YouTube ใหม่...")
        fresh_url = get_live_m3u8(youtube_url)
        if fresh_url:
            cmd[13] = fresh_url # แก้ URL
            try:
                subprocess.run(cmd, timeout=15)
                img_files = list(tmp_dir.glob("*.jpg"))
            except:
                pass

    best_score: float = -1.0
    best_frame_path: Path | None = None

    for f in img_files:
        frame = cv2.imread(str(f))
        if frame is not None:
            s = score_frame(frame, yolo_model)
            if s > best_score:
                best_score = s
                best_frame_path = f

    saved: Path | None = None
    if best_frame_path is not None:
        shutil.copy(best_frame_path, out_path)
        saved = out_path
        logger.info("[Vision] 📸 บันทึกภาพเสร็จสิ้น → %s (score=%.1f)", out_path, best_score)
    else:
        logger.warning("[Vision] ⚠️ ไม่ได้เฟรมภาพสำหรับ item=#%d สตรีมอาจจะหยุดไปแล้ว", item_code)

    # ล้างไฟล์ temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if on_complete:
        on_complete(item_code, saved)
    return saved


# ─── Non-blocking Trigger ──────────────────────────────────────────────────

def trigger_capture(
    youtube_url: str,
    video_url: str,
    item_code: int,
    yolo_model: YOLO,
    stop_event: threading.Event,
    on_complete: "Callable[[int, Path | None], None] | None" = None,
) -> threading.Thread:
    """
    เรียก capture_best_frame ใน daemon thread แยก
    คืน thread object เพื่อให้ caller .join() ได้ถ้าต้องการ
    """
    t = threading.Thread(
        target=capture_best_frame,
        args=(youtube_url, video_url, item_code, yolo_model, stop_event, on_complete),
        daemon=True,
        name=f"vision-item{item_code}",
    )
    t.start()
    return t
