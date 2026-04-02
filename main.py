"""
╔══════════════════════════════════════════════════════════════╗
║           Manowzab Vision — Live Stream Transcription        ║
║  Phase 3 : Audio + Video → NLP → CV Capture (YOLOv8 + OCV)  ║
╚══════════════════════════════════════════════════════════════╝

วิธีรัน:
    python main.py
    แล้วป้อน URL ของ YouTube Live เมื่อระบบถาม

ข้อกำหนด:
    - ติดตั้ง CUDA Toolkit (สำหรับ GPU)
    - pip install -r requirements.txt
    - ffmpeg ต้องอยู่ใน PATH (https://ffmpeg.org/download.html)
"""

import subprocess
import threading
import queue
import sys
import re
import json
import time
import os
import logging
import numpy as np
import cv2
from pathlib import Path
from colorama import init, Fore, Style
from faster_whisper import WhisperModel
from ultralytics import YOLO

# ─── Silence ultralytics verbose output ──────────────────────────────────────
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ─── Init colorama ───────────────────────────────────────────────────────────
init(autoreset=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

WHISPER_MODEL_SIZE  = "medium"      # tiny / base / small / medium / large-v3
WHISPER_DEVICE      = "cuda"        # "cuda" | "cpu"
WHISPER_COMPUTE     = "float16"     # float16 (GPU) | int8 (CPU)
TARGET_LANGUAGE     = "th"

CHUNK_SECONDS       = 10
SAMPLE_RATE         = 16000
BYTES_PER_SAMPLE    = 2
CHUNK_BYTES         = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SECONDS

# Vision config
YOLO_MODEL_NAME     = "yolov8n.pt"  # auto-download on first run
CAPTURE_DURATION    = 3.0           # วินาทีที่เปิดโหมดจับภาพหลัง trigger
CAPTURE_FPS_TARGET  = 10           # เก็บเฟรมต่อวินาทีระหว่างจับ
BLUR_THRESHOLD      = 80.0          # Variance of Laplacian ต่ำกว่านี้ = เบลอ
OUTPUT_DIR          = Path("captured_images")
VIDEO_FRAME_W       = 1280
VIDEO_FRAME_H       = 720

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Thai Number-Word → Arabic Numeral Converter
# ══════════════════════════════════════════════════════════════════════════════

_ONES = {
    "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1,
    "สอง": 2,  "ยี่": 2,
    "สาม": 3,  "สี่": 4,  "ห้า": 5,
    "หก": 6,   "เจ็ด": 7, "แปด": 8,  "เก้า": 9,
}
_TENS = {
    "สิบ": 10,  "ยี่สิบ": 20, "สามสิบ": 30,
    "สี่สิบ": 40, "ห้าสิบ": 50, "หกสิบ": 60,
    "เจ็ดสิบ": 70, "แปดสิบ": 80, "เก้าสิบ": 90,
}
_HUNDREDS = {
    "ร้อย": 100, "สองร้อย": 200, "สามร้อย": 300,
    "สี่ร้อย": 400, "ห้าร้อย": 500, "หกร้อย": 600,
    "เจ็ดร้อย": 700, "แปดร้อย": 800, "เก้าร้อย": 900,
}

_ALL_TOKENS: list[tuple[str, int]] = sorted(
    list(_HUNDREDS.items()) + list(_TENS.items()) + list(_ONES.items()),
    key=lambda x: -len(x[0]),
)

_NUM_WORD_PATTERN = re.compile(
    r"(?:" + r"|".join(re.escape(k) for k, _ in _ALL_TOKENS) + r")+"
)


def thai_words_to_int(text: str) -> int | None:
    pos, total, current_hundred, current_rest = 0, 0, 0, 0
    while pos < len(text):
        matched = False
        for token, val in _ALL_TOKENS:
            if text.startswith(token, pos):
                if val >= 100:
                    total += current_hundred + current_rest
                    current_hundred, current_rest = val, 0
                elif val >= 10:
                    current_rest = val
                else:
                    current_rest += val
                pos += len(token)
                matched = True
                break
        if not matched:
            pos += 1
    total += current_hundred + current_rest
    return total if total > 0 else None


def replace_thai_numbers(text: str) -> str:
    def _replacer(m: re.Match) -> str:
        val = thai_words_to_int(m.group())
        return str(val) if val is not None else m.group()
    return _NUM_WORD_PATTERN.sub(_replacer, text)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Product Data Extraction via Regex
# ══════════════════════════════════════════════════════════════════════════════

_KW_CODE   = r"(?:รายการ(?:ที่)?|รหัส|ไอเท็ม|item|no\.?|ลำดับ(?:ที่)?)\s*"
_KW_PRICE  = r"(?:ราคา|ขาย|บาท|฿)"
_KW_CHEST  = r"(?:อก|รอบอก|chest)"
_KW_LENGTH = r"(?:ยาว|ความยาว|length)"
_UNIT_PRICE = r"(?:\s*บาท)?"
_UNIT_SIZE  = r"(?:\s*(?:นิ้ว|cm|ซม\.?|นิ|นิ้ว))?"
_NUM        = r"([0-9๐-๙]+(?:\.[0-9๐-๙]+)?)"

_PAT_CODE   = re.compile(_KW_CODE  + _NUM, re.IGNORECASE)
_PAT_PRICE  = re.compile(_KW_PRICE + r"\s*" + _NUM + _UNIT_PRICE, re.IGNORECASE)
_PAT_PRICE2 = re.compile(_NUM + r"\s*บาท", re.IGNORECASE)
_PAT_CHEST  = re.compile(_KW_CHEST  + r"\s*" + _NUM + _UNIT_SIZE, re.IGNORECASE)
_PAT_LENGTH = re.compile(_KW_LENGTH + r"\s*" + _NUM + _UNIT_SIZE, re.IGNORECASE)


def _to_arabic(s: str) -> int | None:
    _THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
    try:
        return int(float(s.translate(_THAI_DIGITS)))
    except (ValueError, TypeError):
        return None


def extract_product_data(text: str) -> dict | None:
    normalized = replace_thai_numbers(text)

    m_code = _PAT_CODE.search(normalized)
    if not m_code:
        return None
    item_code = _to_arabic(m_code.group(1))
    if item_code is None:
        return None

    price = None
    m_price = _PAT_PRICE.search(normalized)
    if m_price:
        price = _to_arabic(m_price.group(1))
    if price is None:
        m_price2 = _PAT_PRICE2.search(normalized)
        if m_price2:
            price = _to_arabic(m_price2.group(1))
    if price is None:
        return None

    chest = length = None
    m_chest = _PAT_CHEST.search(normalized)
    if m_chest:
        chest = _to_arabic(m_chest.group(1))
    m_len = _PAT_LENGTH.search(normalized)
    if m_len:
        length = _to_arabic(m_len.group(1))

    result: dict = {"item_code": item_code, "price": price}
    if chest  is not None: result["chest"]  = chest
    if length is not None: result["length"] = length
    result["_raw"]        = text
    result["_normalized"] = normalized
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Computer Vision Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def variance_of_laplacian(frame_bgr: np.ndarray) -> float:
    """
    คำนวณ sharpness ของเฟรมด้วย Variance of Laplacian
    ค่าสูง = ภาพคมชัด / ค่าต่ำ = ภาพเบลอ
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_frame(frame_bgr: np.ndarray, yolo_model: YOLO) -> float:
    """
    ให้คะแนนเฟรม:
      - ถ้าไม่พบ person → score = sharpness เท่านั้น
      - ถ้าพบ person    → score = sharpness × (1 + 0.5 × confidence_max)
    เฟรมที่ผ่าน BLUR_THRESHOLD จะได้ score > 0
    """
    sharpness = variance_of_laplacian(frame_bgr)
    if sharpness < BLUR_THRESHOLD:
        return 0.0  # เบลอเกินไป → ตัดทิ้ง

    results = yolo_model(
        frame_bgr,
        classes=[0],           # class 0 = person
        verbose=False,
        imgsz=640,
    )
    person_confs = [
        float(box.conf)
        for r in results
        for box in r.boxes
    ]
    bonus = max(person_confs) * 0.5 if person_confs else 0.0
    return sharpness * (1.0 + bonus)


def capture_best_frame(
    video_url: str,
    item_code: int,
    yolo_model: YOLO,
    stop_event: threading.Event,
) -> None:
    """
    ดึง video stream ด้วย OpenCV เป็นเวลา CAPTURE_DURATION วินาที
    ประเมินแต่ละเฟรมด้วย score_frame() แล้วบันทึกเฟรมที่ดีที่สุด
    ทำงานใน Thread แยกต่างหาก (non-blocking)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{item_code}.jpg"

    print(Fore.CYAN + f"\n📷  [Vision] เริ่มจับภาพสินค้า #{item_code} เป็นเวลา {CAPTURE_DURATION}s ...")

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(Fore.RED + f"[Vision] ❌  เปิด video stream ไม่ได้: {video_url[:60]}...")
        return

    # ตั้งความละเอียด (แนะนำ 720p เพื่อสมดุล speed/quality)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIDEO_FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_H)

    best_score: float = -1.0
    best_frame: np.ndarray | None = None
    frame_interval = 1.0 / CAPTURE_FPS_TARGET  # วินาทีต่อเฟรม

    deadline = time.monotonic() + CAPTURE_DURATION
    next_capture = time.monotonic()

    while time.monotonic() < deadline and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        now = time.monotonic()
        if now < next_capture:
            continue                    # ยังไม่ถึงเวลา sample ถัดไป
        next_capture = now + frame_interval

        score = score_frame(frame, yolo_model)
        if score > best_score:
            best_score = score
            best_frame = frame.copy()

    cap.release()

    if best_frame is not None and best_score > 0:
        cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(
            Fore.GREEN + Style.BRIGHT
            + f"📸  [Vision] บันทึกภาพสินค้า #{item_code} → {out_path}  "
            + Fore.WHITE + f"(score={best_score:.1f})\n"
        )
    else:
        print(Fore.YELLOW + f"[Vision] ⚠️  ไม่พบเฟรมที่ชัดพอสำหรับสินค้า #{item_code}\n")


def trigger_capture(
    video_url: str,
    item_code: int,
    yolo_model: YOLO,
    stop_event: threading.Event,
) -> None:
    """
    เรียก capture_best_frame() ใน daemon thread แยก
    เพื่อไม่บล็อก audio/transcription pipeline
    """
    t = threading.Thread(
        target=capture_best_frame,
        args=(video_url, item_code, yolo_model, stop_event),
        daemon=True,
        name=f"vision-item{item_code}",
    )
    t.start()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Stream URL Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_stream_urls(youtube_url: str) -> tuple[str, str]:
    """
    ดึง Audio URL และ Video URL แยกกัน
    Returns (audio_url, video_url)
    """
    print(Fore.YELLOW + f"[*] กำลังดึง Stream URLs จาก: {youtube_url}")

    # Audio stream
    audio_cmd = [
        "yt-dlp", "--no-playlist",
        "--format", "bestaudio/best",
        "--get-url", youtube_url,
    ]
    ar = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=30)
    if ar.returncode != 0:
        raise RuntimeError(f"yt-dlp (audio) error:\n{ar.stderr.strip()}")
    audio_url = ar.stdout.strip().splitlines()[0]

    # Video stream (720p หรือเล็กกว่า เพื่อประหยัด bandwidth)
    video_cmd = [
        "yt-dlp", "--no-playlist",
        "--format", "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]",
        "--get-url", youtube_url,
    ]
    vr = subprocess.run(video_cmd, capture_output=True, text=True, timeout=30)
    if vr.returncode != 0:
        # Fallback: ใช้ URL เดียวกับ audio (บาง stream รวม av)
        print(Fore.YELLOW + "[!] ไม่พบ video stream แยก → ใช้ URL เดิม (av combined)")
        video_url = audio_url
    else:
        video_url = vr.stdout.strip().splitlines()[0]

    print(Fore.GREEN + "[✓] ได้ Stream URLs แล้ว\n")
    return audio_url, video_url


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Audio Pipeline (Phase 1 – unchanged logic, new signature)
# ══════════════════════════════════════════════════════════════════════════════

def audio_producer(
    audio_url: str,
    audio_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """ffmpeg → raw PCM 16kHz mono → queue"""
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-i", audio_url,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "s16le", "pipe:1",
    ]
    print(Fore.CYAN + "[*] เริ่มต้น ffmpeg สตรีมเสียง...\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buffer = b""
    try:
        while not stop_event.is_set():
            chunk = proc.stdout.read(4096)
            if not chunk:
                print(Fore.RED + "[!] ffmpeg audio: stream หยุดแล้ว")
                break
            buffer += chunk
            while len(buffer) >= CHUNK_BYTES:
                audio_queue.put(buffer[:CHUNK_BYTES])
                buffer = buffer[CHUNK_BYTES:]
    finally:
        proc.terminate()
        stop_event.set()
        audio_queue.put(None)  # sentinel


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Transcription + NLP + Vision Trigger Worker
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_worker(
    whisper_model: WhisperModel,
    yolo_model: YOLO,
    audio_queue: queue.Queue,
    video_url: str,
    stop_event: threading.Event,
    capture_lock: threading.Lock,
) -> None:
    """
    ดึง audio chunk จาก queue:
      1) Whisper transcribe → Thai text
      2) extract_product_data() → JSON
      3) ถ้าพบ item_code ใหม่ → trigger_capture() (non-blocking)

    capture_lock ป้องกันการ trigger ซ้ำพร้อมกันหลาย thread
    """
    # ชุด item_code ที่จับภาพไปแล้วในการไลฟ์นี้ (ป้องกัน duplicate)
    captured_codes: set[int] = set()

    print(Fore.MAGENTA + Style.BRIGHT + "─" * 62)
    print(Fore.MAGENTA + Style.BRIGHT + "  📝  เริ่มถอดเสียง+จับภาพ — กด Ctrl+C เพื่อหยุด")
    print(Fore.MAGENTA + Style.BRIGHT + "─" * 62 + "\n")

    while not stop_event.is_set():
        try:
            raw_bytes = audio_queue.get(timeout=2)
        except queue.Empty:
            continue

        if raw_bytes is None:   # sentinel
            break

        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        ts = time.strftime("%H:%M:%S")

        try:
            segments, _ = whisper_model.transcribe(
                audio_np,
                language=TARGET_LANGUAGE,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if not texts:
                continue

            full_text = " ".join(texts)

            # ── 1) แสดง transcript ───────────────────────────────────────────
            print(Fore.WHITE + f"[{ts}] " + Fore.GREEN + Style.BRIGHT + full_text)

            # ── 2) สกัดข้อมูลสินค้า ─────────────────────────────────────────
            product = extract_product_data(full_text)
            if not product:
                continue

            json_str = json.dumps(product, ensure_ascii=False, indent=2)
            print(Fore.YELLOW + Style.BRIGHT + "\n🛍️  พบข้อมูลสินค้า:")
            print(Fore.YELLOW + json_str + "\n")

            # ── 3) Trigger Vision Capture (non-blocking) ─────────────────────
            item_code: int = product["item_code"]
            with capture_lock:
                if item_code not in captured_codes:
                    captured_codes.add(item_code)
                    trigger_capture(video_url, item_code, yolo_model, stop_event)
                else:
                    print(
                        Fore.CYAN + f"[Vision] ℹ️  สินค้า #{item_code} "
                        "เคยจับภาพไปแล้ว — ข้าม\n"
                    )

        except Exception as e:
            print(Fore.RED + f"[{ts}] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Model Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_whisper_model() -> WhisperModel:
    print(Fore.YELLOW + f"[*] โหลด Whisper '{WHISPER_MODEL_SIZE}' บน {WHISPER_DEVICE.upper()}...")
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        print(Fore.GREEN + f"[✓] Whisper โหลดสำเร็จ ({WHISPER_MODEL_SIZE}/{WHISPER_DEVICE}/{WHISPER_COMPUTE})\n")
        return model
    except Exception as e:
        print(Fore.RED + f"[!] GPU ไม่พร้อม ({e}) → Fallback CPU/int8")
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print(Fore.GREEN + "[✓] Whisper โหลดบน CPU สำเร็จ\n")
        return model


def load_yolo_model() -> YOLO:
    print(Fore.YELLOW + f"[*] โหลด YOLO '{YOLO_MODEL_NAME}' ...")
    model = YOLO(YOLO_MODEL_NAME)   # auto-download จาก ultralytics hub
    print(Fore.GREEN + f"[✓] YOLO โหลดสำเร็จ\n")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Banner & Main
# ══════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    print(Fore.CYAN + Style.BRIGHT + """
╔══════════════════════════════════════════════════════════════╗
║           Manowzab Vision — Live Transcription               ║
║   Phase 3 : Audio+Video → NLP → CV Capture (YOLOv8+OCV)     ║
╚══════════════════════════════════════════════════════════════╝
""")


def main() -> None:
    print_banner()

    youtube_url = input(Fore.CYAN + "🎬  กรอก YouTube Live URL: " + Style.RESET_ALL).strip()
    if not youtube_url:
        print(Fore.RED + "[!] ไม่ได้ป้อน URL")
        sys.exit(1)

    # ── โหลดโมเดลทั้งสอง ─────────────────────────────────────────────────────
    whisper_model = load_whisper_model()
    yolo_model    = load_yolo_model()

    # ── ดึง Stream URLs ───────────────────────────────────────────────────────
    try:
        audio_url, video_url = get_stream_urls(youtube_url)
    except Exception as e:
        print(Fore.RED + f"[!] ไม่สามารถดึง stream: {e}")
        sys.exit(1)

    # ── เตรียม synchronization primitives ────────────────────────────────────
    audio_queue  = queue.Queue(maxsize=10)
    stop_event   = threading.Event()
    capture_lock = threading.Lock()     # ป้องกัน race condition บน captured_codes

    # ── Thread 1: Audio Producer ──────────────────────────────────────────────
    t_audio = threading.Thread(
        target=audio_producer,
        args=(audio_url, audio_queue, stop_event),
        daemon=True,
        name="audio-producer",
    )

    # ── Thread 2: Transcribe + NLP + Vision Trigger ───────────────────────────
    t_transcribe = threading.Thread(
        target=transcribe_worker,
        args=(whisper_model, yolo_model, audio_queue, video_url, stop_event, capture_lock),
        daemon=True,
        name="transcribe-worker",
    )

    t_audio.start()
    t_transcribe.start()

    try:
        t_transcribe.join()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n[*] กำลังหยุดโปรแกรม...")
        stop_event.set()
        t_audio.join(timeout=5)
        t_transcribe.join(timeout=5)
        print(Fore.GREEN + "[✓] หยุดเรียบร้อยแล้ว")


if __name__ == "__main__":
    main()
