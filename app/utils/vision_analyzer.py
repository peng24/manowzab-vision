"""
app/utils/vision_analyzer.py
─────────────────────────────────────────────────────────────────
วิเคราะห์สีและประเภทเสื้อผ้าจากภาพโดยใช้ Ollama Vision LLM
ไม่ใช้ OCR — ใช้ multimodal inference เพียงอย่างเดียว
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_PROMPT = (
    "Analyze the main clothing item shown in this image. "
    "What is its primary color, and what type of clothing is it "
    "(e.g., T-shirt, dress, jacket, pants, skirt)? "
    "Return ONLY a valid JSON object with the keys 'color' (string) and 'type' (string). "
    "Do not output any markdown code blocks or extra explanations."
)

_FALLBACK: dict = {"color": None, "type": None}


def analyze_clothing(image_path: str) -> dict:
    """
    ส่งภาพไปยัง Ollama Vision model แล้วคืนค่า dict ที่มีกุญแจ:
        - color (str | None): สีหลักของเสื้อผ้า
        - type  (str | None): ประเภทเสื้อผ้า

    คืน fallback dict หาก Ollama ไม่พร้อมใช้งานหรือเกิดข้อผิดพลาด
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("[Vision] ไม่พบไฟล์ภาพ: %s", image_path)
        return _FALLBACK

    try:
        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    except OSError as exc:
        logger.error("[Vision] อ่านภาพไม่ได้: %s", exc)
        return _FALLBACK

    payload = {
        "model":  settings.ollama_vision_model,
        "prompt": _PROMPT,
        "images": [image_b64],
        "format": "json",
        "stream": False,
    }

    url = f"{settings.ollama_endpoint}/api/generate"

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("[Vision] Ollama request ล้มเหลว: %s", exc)
        return _FALLBACK

    try:
        raw_text: str = response.json().get("response", "")
        result: dict = json.loads(raw_text)
        color = result.get("color") or None
        clothing_type = result.get("type") or None
        logger.info("[Vision] ✅ วิเคราะห์เสื้อผ้า → color=%s, type=%s", color, clothing_type)
        return {"color": color, "type": clothing_type}
    except (json.JSONDecodeError, AttributeError, KeyError) as exc:
        logger.error("[Vision] แปลง JSON จาก Ollama ไม่สำเร็จ: %s", exc)
        return _FALLBACK
