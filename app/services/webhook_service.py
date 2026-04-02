"""
app/services/webhook_service.py
─────────────────────────────────────────────────────────────────
ส่ง HTTP POST ไปยัง WEBHOOK_URL พร้อม payload ข้อมูลสินค้า
ใช้ httpx.Client (synchronous) เนื่องจากถูกเรียกจาก thread ปกติ
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _build_payload(
    product: dict,
    image_path: Path | None,
    session_id: str,
) -> dict:
    """สร้าง payload มาตรฐานสำหรับ webhook"""
    # แยก internal keys (_raw, _normalized) ออกจาก public data
    public_product = {k: v for k, v in product.items() if not k.startswith("_")}

    return {
        "event":      "product_detected",
        "session_id": session_id,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "product":    public_product,
        "image": {
            "path":     str(image_path) if image_path else None,
            "captured": image_path is not None and image_path.exists(),
        },
    }


def send_product_event(
    product: dict,
    image_path: Path | None,
    session_id: str,
) -> None:
    """
    ส่ง webhook event ไปยัง WEBHOOK_URL
    ถ้าไม่ได้ตั้งค่า WEBHOOK_URL → ข้ามโดยไม่ error

    Args:
        product:    dict จาก extract_product_data()
        image_path: Path ของไฟล์ภาพที่บันทึก (None ถ้าไม่ได้จับ)
        session_id: UUID ของ session ปัจจุบัน
    """
    url = settings.webhook_url
    if not url:
        logger.debug("[Webhook] WEBHOOK_URL ไม่ได้ตั้งค่า — ข้าม")
        return

    payload = _build_payload(product, image_path, session_id)

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            logger.info(
                "[Webhook] ✅  ส่งสำเร็จ item=%s → %s  HTTP %d",
                product.get("item_code"),
                url,
                resp.status_code,
            )
    except httpx.TimeoutException:
        logger.warning("[Webhook] ⚠️  Timeout ขณะส่ง item=%s", product.get("item_code"))
    except httpx.HTTPStatusError as e:
        logger.error("[Webhook] ❌  HTTP %d: %s", e.response.status_code, e.response.text[:200])
    except Exception as e:
        logger.error("[Webhook] ❌  ส่งไม่ได้: %s", e)
