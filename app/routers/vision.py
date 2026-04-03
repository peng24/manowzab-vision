"""
app/routers/vision.py
─────────────────────────────────────────────────────────────────
FastAPI Router สำหรับ Vision Pipeline

  POST /start-vision  → เริ่ม pipeline
  POST /stop-vision   → หยุด pipeline
  GET  /status        → ดูสถานะปัจจุบัน
  GET  /captured/{item_code} → ดาวน์โหลดภาพที่จับไว้
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, field_validator

from app.config import settings
from app.services.stream_manager import stream_manager
import json

router = APIRouter(prefix="/vision", tags=["Vision Pipeline"])


# ─── Request / Response Schemas ────────────────────────────────────────────

class StartVisionRequest(BaseModel):
    youtube_url: str
    webhook_url: str | None = None  # override WEBHOOK_URL ใน .env ได้ (optional)

    @field_validator("youtube_url")
    @classmethod
    def must_be_youtube(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("youtube_url ต้องไม่ว่าง")
        if "youtube.com" not in v and "youtu.be" not in v:
            raise ValueError("youtube_url ต้องเป็น URL ของ YouTube")
        return v


class StartVisionResponse(BaseModel):
    message:    str
    session_id: str
    youtube_url: str


class StopVisionResponse(BaseModel):
    message: str
    session_id: str | None


class StatusResponse(BaseModel):
    running:          bool
    session_id:       str | None
    youtube_url:      str | None
    started_at:       float | None
    captured_count:   int
    captured_codes:   list[int]


# ─── Endpoints ─────────────────────────────────────────────────────────────

@router.post(
    "/start-vision",
    response_model=StartVisionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="เริ่ม pipeline (Audio → Transcription → NLP → Vision → Webhook)",
)
async def start_vision(body: StartVisionRequest) -> StartVisionResponse:
    """
    รับ YouTube Live URL แล้วเริ่ม pipeline แบบ background thread

    - โหลด Whisper + YOLOv8 (lazy — โหลดแค่ครั้งแรก)
    - ดึง audio/video stream URL ผ่าน yt-dlp
    - spawn threads สำหรับ audio production และ transcription
    - เมื่อพบสินค้าใหม่ → จับภาพ → ส่ง Webhook
    """
    if stream_manager.is_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pipeline กำลังทำงานอยู่แล้ว กรุณา POST /vision/stop-vision ก่อน",
        )

    # Override webhook URL ต่อ session ถ้าส่งมา
    if body.webhook_url:
        settings.webhook_url = body.webhook_url

    session_id = str(uuid.uuid4())
    try:
        # get_stream_urls + spawn threads (blocking นิดนึงเพราะ yt-dlp)
        # ใช้ run_in_executor เพื่อไม่บล็อก event loop
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, stream_manager.start, body.youtube_url, session_id
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ไม่สามารถเชื่อมต่อ stream: {e}",
        )

    return StartVisionResponse(
        message=f"Pipeline เริ่มทำงานแล้ว (session={session_id})",
        session_id=session_id,
        youtube_url=body.youtube_url,
    )


@router.post(
    "/stop-vision",
    response_model=StopVisionResponse,
    summary="หยุด pipeline",
)
async def stop_vision() -> StopVisionResponse:
    """หยุด pipeline ที่กำลังรัน — thread จะ gracefully terminate ภายใน 5 วินาที"""
    if not stream_manager.is_running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ไม่มี pipeline ที่กำลังทำงาน",
        )

    session_id = stream_manager._session_id

    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, stream_manager.stop)

    return StopVisionResponse(
        message="Pipeline หยุดเรียบร้อยแล้ว",
        session_id=session_id,
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="ดูสถานะ pipeline ปัจจุบัน",
)
async def get_status() -> StatusResponse:
    """คืน JSON สถานะ pipeline รวมถึงรายการ item_code ที่จับภาพไปแล้ว"""
    return StatusResponse(**stream_manager.status())


@router.get(
    "/captured/{item_code}",
    summary="ดาวน์โหลดภาพสินค้าที่จับไว้",
    response_class=FileResponse,
)
async def get_captured_image(item_code: int) -> FileResponse:
    """
    คืนไฟล์ภาพ JPEG ของสินค้าตาม item_code
    ค้นหาแบบ recursive ทั่วทั้งโฟลเดอร์ภาพ เช่น {prefix}_{item_code}.jpg
    """
    found_path = None
    for p in settings.output_dir.rglob("*.jpg"):
        if p.name == f"{item_code}.jpg" or p.name.endswith(f"_{item_code}.jpg"):
            found_path = p
            break
            
    if not found_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ยังไม่มีภาพสินค้า #{item_code}",
        )
    return FileResponse(
        path=str(found_path),
        media_type="image/jpeg",
        filename=found_path.name,
    )


class UpdatePriceRequest(BaseModel):
    price: int

@router.get("/results", summary="ดึงข้อมูลราคาสินค้าจาก local")
async def get_results() -> list[dict]:
    results_file = settings.output_dir / "results.json"
    if not results_file.exists():
        return []
    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
        return data
    except:
        return []

@router.post("/results/{item_code}/price", summary="อัปเดตราคาสินค้า")
async def update_item_price(item_code: int, req: UpdatePriceRequest):
    results_file = settings.output_dir / "results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="No results.json found")
    
    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
        updated = False
        for item in data:
            if item.get("product", {}).get("item_code") == item_code:
                item["product"]["price"] = req.price
                updated = True
                break
        
        if not updated:
            raise HTTPException(status_code=404, detail="Item not found")
            
        results_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        
        # update in memory extractor so it won't conflict later
        if stream_manager._extractor and item_code in stream_manager._extractor.history:
            stream_manager._extractor.history[item_code]["price"] = req.price
            
        return {"success": True, "new_price": req.price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
