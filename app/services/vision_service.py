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
      score = sharpness × (1 + 0.5 × max_person_confidence)

    เฟรมที่ sharpness < BLUR_THRESHOLD จะได้ score = 0 (ตัดทิ้ง)
    """
    sharpness = variance_of_laplacian(frame_bgr)
    if sharpness < settings.blur_threshold:
        return 0.0

    results = yolo_model(
        frame_bgr,
        classes=[0],    # class 0 = person
        verbose=False,
        imgsz=640,
    )
    confs = [float(box.conf) for r in results for box in r.boxes]
    bonus = max(confs) * 0.5 if confs else 0.0
    return sharpness * (1.0 + bonus)


# ─── Capture Worker ────────────────────────────────────────────────────────

def capture_best_frame(
    video_url: str,
    item_code: int,
    yolo_model: YOLO,
    stop_event: threading.Event,
    on_complete: "Callable[[int, Path | None], None] | None" = None,
) -> Path | None:
    """
    เปิด video_url ด้วย OpenCV, วิ่งวนเป็นเวลา CAPTURE_DURATION วินาที
    ประเมินแต่ละเฟรมด้วย score_frame() → บันทึกเฟรมคะแนนสูงสุด

    Args:
        on_complete: callback(item_code, saved_path | None) ถูกเรียกเมื่อเสร็จ

    Returns:
        Path ของไฟล์ที่บันทึก หรือ None ถ้าไม่มีเฟรมพอ
    """
    output_dir: Path = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{item_code}.jpg"

    logger.info("[Vision] 📷  เริ่มจับภาพ item=#%d  (%.1fs)", item_code, settings.capture_duration)

    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-live_start_index", "-1",  # 🟢 สำคัญมาก! บังคับให้เริ่มที่ Live Edge เสมอ
        "-i", video_url,
        "-t", str(settings.capture_duration),
        "-vf", f"fps={settings.capture_fps_target}",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error("[Vision] ❌  เปิด ffmpeg ไม่ได้: %s", e)
        if on_complete: on_complete(item_code, None)
        return None

    best_score: float = -1.0
    best_frame: np.ndarray | None = None
    buffer = b""

    try:
        while not stop_event.is_set():
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            buffer += chunk

            # ค้นหาจุดเริ่มต้นและจุดสิ้นสุดของไฟล์ JPEG
            while True:
                start = buffer.find(b"\xff\xd8")
                if start == -1:
                    break
                end = buffer.find(b"\xff\xd9", start)
                if end == -1:
                    break

                # สกัดไฟล์ JPEG 1 ภาพ
                jpg_data = buffer[start:end+2]
                buffer = buffer[end+2:]

                # Decode กลับเป็น BGR frame สำหรับ OpenCV
                frame_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    s = score_frame(frame, yolo_model)
                    if s > best_score:
                        best_score, best_frame = s, frame.copy()
    finally:
        proc.stdout.close()
        proc.wait(timeout=2)

    saved: Path | None = None
    if best_frame is not None and best_score > 0:
        cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved = out_path
        logger.info("[Vision] 📸  บันทึก → %s  (score=%.1f)", out_path, best_score)
    else:
        logger.warning("[Vision] ⚠️  ไม่มีเฟรมที่คมชัดพอสำหรับ item=#%d", item_code)

    if on_complete:
        on_complete(item_code, saved)
    return saved


# ─── Non-blocking Trigger ──────────────────────────────────────────────────

def trigger_capture(
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
        args=(video_url, item_code, yolo_model, stop_event, on_complete),
        daemon=True,
        name=f"vision-item{item_code}",
    )
    t.start()
    return t
