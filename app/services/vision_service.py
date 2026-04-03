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
import urllib.request
import urllib.parse

def _extract_video_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed.path[1:]
    if parsed.hostname in ('youtube.com', 'www.youtube.com'):
        if parsed.path == '/watch':
            qs = urllib.parse.parse_qs(parsed.query)
            return qs.get('v', [None])[0]
        if parsed.path.startswith('/live/'):
            return parsed.path.split('/')[2]
    return None

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

    logger.info("[Vision] 📷 เริ่มจับภาพ item=#%d ตรงลง /%s", item_code, output_dir.name)

    # 1. แคปภาพ 1 เฟรมตรงๆ 
    cmd = [
        "ffmpeg", "-y", "-loglevel", "quiet",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", video_url,
        "-vframes", "1",
        "-q:v", "2",
        str(out_path)
    ]

    try:
        subprocess.run(cmd, timeout=12)
    except subprocess.TimeoutExpired:
        logger.warning("[Vision] ⚠️ Timeout จาก FFmpeg")
    except Exception as e:
        logger.error("[Vision] ❌ FFmpeg error: %s", e)

    # 2. ถ้าไม่ได้ภาพ (ลิงก์อาจตายหรือเป็นแค่ audio-only) ลองให้ yt-dlp ดึง URL ใหม่
    if not out_path.exists():
        logger.warning("[Vision] ⚠️ FFmpeg ไม่ได้ภาพ (ลิงก์หลักพัง/ไม่มีภาพ) ขอดึงลิงก์ YouTube ใหม่...")
        fresh_url = get_live_m3u8(youtube_url)
        if fresh_url:
            cmd[11] = fresh_url # index ของ video_url
            try:
                subprocess.run(cmd, timeout=12)
            except:
                pass

    # 3. ถ้ายังไม่ได้อีก (เช่น yt-dlp แจ้ง JS Runtime missing) ให้โหลด Live Thumbnail
    if not out_path.exists():
        logger.warning("[Vision] ⚠️ FFmpeg พังหมด! ดึง Live Thumbnail ของช่องมาแก้ขัดแทน")
        vid = _extract_video_id(youtube_url)
        if vid:
            urls = [
                f"https://i.ytimg.com/vi/{vid}/maxresdefault_live.jpg",
                f"https://i.ytimg.com/vi/{vid}/hqdefault_live.jpg"
            ]
            for tu in urls:
                try:
                    urllib.request.urlretrieve(tu, str(out_path))
                    break
                except:
                    pass

    # บันทึกภาพ พร้อมกับประเมินคะแนนเพื่อเก็บเป็น Log ตามโมเดล
    saved: Path | None = None
    if out_path.exists():
        frame = cv2.imread(str(out_path))
        if frame is not None:
            s = score_frame(frame, yolo_model)
            logger.info("[Vision] 📸 บันทึกภาพเสร็จสิ้น → %s (score=%.1f)", out_path.name, s)
            saved = out_path
        else:
            logger.warning("[Vision] ⚠️ ไฟล์เสีย ลบออก")
            out_path.unlink(missing_ok=True)
    else:
        logger.warning("[Vision] ⚠️ ไม่ได้เฟรมภาพสำหรับ item=#%d สตรีมอาจจะหยุดไปแล้ว", item_code)

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
