"""
app/main.py
─────────────────────────────────────────────────────────────────
FastAPI Application Entry Point
─────────────────────────────────────────────────────────────────
"""
import logging
import signal
import sys

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import vision as vision_router
from app.services.stream_manager import stream_manager

# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ─── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Manowzab Vision API",
    description=(
        "Real-time YouTube Live stream transcription + computer vision pipeline.\n\n"
        "**Flow:** YouTube Live → yt-dlp → ffmpeg → Whisper (STT) → "
        "Regex NLP → YOLOv8 Capture → Webhook"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ปรับให้ restrictive ก่อน deploy production
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from app.routers import dataset_review

# ─── Static Mounts ─────────────────────────────────────────────────────────
train_images_dir = Path("data/train/images")
train_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data/train/images", StaticFiles(directory=str(train_images_dir)), name="train_images")

# ─── Routers ───────────────────────────────────────────────────────────────
app.include_router(vision_router.router)
app.include_router(dataset_review.router)

# ─── Web UI Endpoints ─────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """เปิด Web Dashboard สำหรับทดสอบ API"""
    html_path = Path(__file__).parent.parent / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/review", response_class=HTMLResponse, include_in_schema=False)
async def review_tool() -> HTMLResponse:
    """เปิด Web UI สำหรับรีวิว Dataset"""
    html_path = Path(__file__).parent.parent / "review_tool.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/", tags=["Health"])
async def root() -> dict:
    return {
        "service": "Manowzab Vision API",
        "version": "1.0.0",
        "status":  "ok",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "pipeline": stream_manager.status()})


# ─── Events (Startup / Shutdown) ─────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    print("\n" + "="*50)
    print("🚀 ระบบ Vision Server เริ่มทำงานแล้ว!")
    print(f"👉 แดชบอร์ดพร้อมใช้งาน: http://localhost:{settings.api_port}/dashboard")
    print("="*50 + "\n")


@app.on_event("shutdown")
async def _shutdown() -> None:
    logger.info("[App] Shutdown → หยุด pipeline ...")
    if stream_manager.is_running:
        stream_manager.stop()
    logger.info("[App] Bye 👋")
