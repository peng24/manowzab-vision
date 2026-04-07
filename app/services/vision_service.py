"""
app/services/vision_service.py
─────────────────────────────────────────────────────────────────
Computer Vision Pipeline (Continuous Buffer Architecture):
  - variance_of_laplacian()  → วัด sharpness ของเฟรม
  - score_frame()            → ให้คะแนนเฟรม (sharpness × YOLO highest_confidence)
  - ContinuousVisionBuffer   → Class จัดการ Background Thread เก็บภาพ 10-15s ล่าสุด
"""
from __future__ import annotations

import collections
import logging
import threading
import time
import queue
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Sharpness Metric ──────────────────────────────────────────────────────

def variance_of_laplacian(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ─── Frame Scoring ─────────────────────────────────────────────────────────

last_al_save_time = 0.0

def score_frame(frame_bgr: np.ndarray, yolo_model: YOLO) -> tuple[float, np.ndarray]:
    global last_al_save_time
    sharpness = variance_of_laplacian(frame_bgr)
    if sharpness < settings.blur_threshold:
        return 0.0, frame_bgr

    results = yolo_model(
        frame_bgr,
        verbose=False,
        imgsz=640,
    )
    
    confs = [float(box.conf) for r in results for box in r.boxes]
    highest_conf = max(confs) if confs else 0.0
    
    annotated = results[0].plot()
    now = time.time()
    
    # ── Active Learning Hook (1 Frame / Sec Cooldown) ──
    if 0.0 < highest_conf < 0.60 and (now - last_al_save_time > 1.0):
        al_dir = Path("data/needs_training")
        al_dir.mkdir(parents=True, exist_ok=True)
        filename = al_dir / f"al_{int(now)}.jpg"
        cv2.imwrite(str(filename), frame_bgr)  # บันทึกรูปดิบ (ไม่มีกรอบ YOLO)
        last_al_save_time = now
        logger.info("[ActiveLearning] 📸 เซฟภาพ confidence ต่ำ (%.2f) เข้า needs_training/", highest_conf)

    if highest_conf < 0.60:
        return 0.0, annotated

    return sharpness * highest_conf, annotated


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


# ─── Continuous Buffer Manager ─────────────────────────────────────────────

class ContinuousVisionBuffer:
    """
    ดึงภาพจาก Stream ตลอดเวลาใส่ไว้ใน Deque Buffer อย่างจำกัด
    """
    def __init__(self, yolo_model: YOLO, buffer_seconds: int = 15, fps: int = 4,
                 output_dir: Path | None = None):
        self.yolo_model = yolo_model
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = buffer_seconds * fps
        self.output_dir: Path = output_dir or settings.output_dir
        
        # ใช้ deque พร้อม maxlen ป้องกัน Memory Leak เด็ดขาด (จำกัดรูปตาม Queue Size)
        self.frame_buffer = collections.deque(maxlen=self.max_frames)
        self.lock = threading.Lock()
        
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.video_url: str = ""
        self.youtube_url: str = ""
        self._cap: cv2.VideoCapture | None = None  # track active capture for force-release

    def start(self, video_url: str, youtube_url: str):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.video_url = video_url
        self.youtube_url = youtube_url
        self.stop_event.clear()

        with self.lock:
            self.frame_buffer.clear()

        self.worker_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="vision-buffer-worker"
        )
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        # Force-release VideoCapture so grab() unblocks immediately
        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=6)

    def _capture_loop(self):
        logger.info("[Vision] 📽️ เริ่ม Continuous Frame Buffer (%d FPS, จุสูงสุด %d รูป)",
                    self.fps, self.max_frames)

        cap = cv2.VideoCapture(self.video_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._cap = cap  # expose for stop() to force-release

        last_score_time = 0.0
        frame_interval = 1.0 / self.fps

        while not self.stop_event.is_set():
            # อ่านเฉพาะ header เพื่อให้ stream เดินหน้าแบบ Real-time (ล้างคิวภาพเก่าทิ้ง)
            ret = cap.grab()

            if not ret:
                logger.warning("[Vision] ⚠️ สตรีมภาพสะดุดหรือลิงก์หมดอายุ กำลังเชื่อมต่อใหม่...")
                cap.release()

                # ลิงก์อาจจะหมดอายุ ลองดึงใหม่ผ่าน yt-dlp
                fresh_url = get_live_m3u8(self.youtube_url)
                if fresh_url:
                    self.video_url = fresh_url

                time.sleep(2)
                if self.stop_event.is_set(): break

                cap = cv2.VideoCapture(self.video_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                self._cap = cap  # update reference after reconnect
                continue

            now = time.time()
            # ตัดสินใจดึงภาพมาประมวลผล (Retrieve) ตาม FPS ที่กำหนด
            if now - last_score_time >= frame_interval:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    last_score_time = now
                    s, annotated = score_frame(frame, self.yolo_model)
                    
                    # Pre-encode JPEG for the MJPEG stream to save CPU
                    # ใช้ frame ดิบ (ไม่มีกรอบ) เพื่อให้ preview สะอาด
                    ret_enc, buffer_enc = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer_enc.tobytes() if ret_enc else None
                    
                    # ロックก่อนจัดการคิว
                    with self.lock:
                        # ถึงคะแนนต่ำกว่า 60% ก็เก็บเผื่อไว้ก่อน (เผื่อแม่ค้าบังยาว) 
                        # พอของชิ้นใหม่มา เราจะเอารูปคะแนนสูงสุด เท่าที่มีใน 15 วิมาใช้
                        self.frame_buffer.append((now, s, frame))
                        self.latest_frame_bytes = frame_bytes
                        
        cap.release()
        self._cap = None
        logger.info("[Vision] 🛑 หยุด Continuous Frame Buffer เรียบร้อย")

    def generate_mjpeg_stream(self):
        """Generator สำหรับส่งภาพแบบ MJPEG"""
        while not self.stop_event.is_set():
            frame_bytes = None
            with self.lock:
                if hasattr(self, 'latest_frame_bytes') and self.latest_frame_bytes is not None:
                    frame_bytes = self.latest_frame_bytes
            
            if frame_bytes is None:
                time.sleep(0.1)
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # 10 FPS

    def get_best_frame_and_save(self, item_code: int, prefix: str = "") -> Path | None:
        """
        เมื่อได้ยินรหัสสินค้า ดึงหน้าภาพย้อนหลัง (Retrospective) ทันที (0.1s)
        และเคลียคิวเพื่อไม่ให้ตั๋วใบถัดไปรับภาพนี้ซ้ำ
        """
        logger.info("[Vision] 🔍 ค้นหาภาพย้อนหลังสำหรับ item=#%d จาก Buffer", item_code)

        output_dir = self.output_dir
        if prefix:
            output_dir = output_dir / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{item_code}.jpg" if prefix else f"{item_code}.jpg"
        target_path = output_dir / filename
        
        best_score = -1.0
        best_frame = None
        
        with self.lock:
            if not self.frame_buffer:
                logger.warning("[Vision] ⚠️ ตะกร้าภาพบอกว่าว่างเปล่า! ไม่มีให้ดึง")
                return None
                
            for timestamp, score, frame in self.frame_buffer:
                if score > best_score:
                    best_score = score
                    best_frame = frame
                    
            # เคลียร์ buffer ทิ้ง ป้องกันสินค้าโค้ดถัดไป ดึงภาพเสื้อของสินค้านี้ไปใช้
            self.frame_buffer.clear()
            
        if best_frame is not None:
            cv2.imwrite(str(target_path), best_frame)
            logger.info("[Vision] 📸 ดึงภาพ Retrospective สำเร็จ → %s (score=%.1f)", target_path.name, best_score)
            return target_path
            
        return None
