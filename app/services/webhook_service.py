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
from app.utils.vision_analyzer import analyze_clothing

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
            "image_filename": image_path.name if image_path else None,
            "captured": image_path is not None and image_path.exists(),
        },
    }


import json
from filelock import FileLock

def _save_local_result(payload: dict, output_dir: Path | None = None) -> None:
    """บันทึกข้อมูลลงไฟล์ results.json ในโฟเดอร์รากเพื่อใช้ร่วมกับ Dashboard
    ใช้ FileLock ป้องกัน Race Condition เมื่อมีหลาย Thread เขียนพร้อมกัน
    """
    target_dir = output_dir or settings.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    results_file = target_dir / "results.json"
    lock_file = FileLock(str(target_dir / "results.json.lock"))

    item_code = payload["product"].get("item_code")

    with lock_file:
        data = []
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text(encoding="utf-8"))
            except Exception:
                data = []  # ไฟล์เสียหาย → เริ่มใหม่

        # ตรวจสอบว่าเคยมีสินค้ารหัสนี้แล้วหรือยัง (Update ถ้าเคยมีแล้ว)
        updated = False
        for i, item in enumerate(data):
            if item.get("product", {}).get("item_code") == item_code:
                data[i] = payload
                updated = True
                break

        if not updated:
            data.append(payload)

        results_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("[Local] 💾 บันทึกข้อมูลสินค้า #%s ลง results.json เรียบร้อย", item_code)


def send_product_event(
    product: dict,
    image_path: Path | None,
    session_id: str,
    date_str: str = "",
    output_dir: Path | None = None,
) -> None:
    """
    บันทึกผลลงไฟล์ในเครื่อง (results.json) ในโฟลเดอร์ตามวันที่โดย dynamic
    วิเคราะห์สีและประเภทเสื้อผ้าด้วย Ollama Vision ก่อนส่ง payload
    """
    # ── Vision Analysis ──────────────────────────────────────────
    if image_path is not None and image_path.exists():
        logger.info("[Webhook] 🔍 ส่งภาพไปวิเคราะห์ด้วย Ollama Vision: %s", image_path)
        clothing_info = analyze_clothing(str(image_path))
        product = {**product, **clothing_info}  # merge color + type into product
        logger.info(
            "[Webhook] 👗 ผลวิเคราะห์ → color=%s, type=%s",
            clothing_info.get("color"),
            clothing_info.get("type"),
        )
    else:
        logger.warning("[Webhook] ⚠️ ไม่มีภาพให้วิเคราะห์ — ข้ามขั้นตอน Vision Analysis")

    payload = _build_payload(product, image_path, session_id)

    # 1. เก็บรูปลงเครื่องอยู่แล้ว + เพิ่มการเก็บ JSON ลงเครื่อง (Root folder)
    _save_local_result(payload, output_dir=output_dir)
