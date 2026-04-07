"""
app/utils/nlp.py
─────────────────────────────────────────────────────────────────
สกัดข้อมูลสินค้าจากข้อความภาษาไทย
Hybrid NLP: Gemini Primary → Ollama Local Fallback
รองรับ State Management: โหมดทวนสินค้า (Review Mode) และ Price Conflict
รองรับการดึงข้อมูลไซส์เสื้อผ้า: อก, ยาว, ไซส์
"""
import json
import logging
import requests
from typing import Dict, Any, Optional

from pydantic import BaseModel

from app.utils.thai_numbers import replace_thai_numbers

logger = logging.getLogger(__name__)

# ─── Pydantic schema สำหรับ Structured Output ────────────────────────────────
class ProductResponse(BaseModel):
    item:       Optional[int] = None
    price:      Optional[int] = None
    chest:      Optional[int] = None
    length:     Optional[int] = None
    size_label: Optional[str] = None


# ─── Shared system instruction ────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = (
    "Extract product information from the Spoken Audio Transcript of a Thai secondhand clothing livestream. "
    "Return ONLY a valid JSON object with these exact keys: "
    "'item' (integer or null), 'price' (integer or null), "
    "'chest' (integer or null), 'length' (integer or null), 'size_label' (string or null). "

    "IMPORTANT RULES: "

    "1. NUMBER MERGING: The seller often merges the item code and price into a single spoken number. "
    "You MUST intelligently split these merged numbers based on context and realistic secondhand clothing prices "
    "(common prices are: 50, 60, 80, 100, 120, 150, 200 baht). "
    "Examples: '3180' → item=31, price=80. '600' → item=6, price=100 (ร้อยนึง). "
    "'280' → item=2, price=80. '1380' → item=13, price=80. "
    "Always choose the split that gives a realistic clothing price for the right-hand portion. "

    "2. ITEM CODE: You MUST NOT extract an item code unless it is explicitly preceded by context words like "
    "'รายการที่', 'รหัส', or 'ตัวที่', OR unless a Number Merging pattern is clearly present. "

    "3. SIZES: Extract chest measurement from keywords like 'อก' (e.g. 'อก 48' → chest=48). "
    "Extract length from keywords like 'ยาว' (e.g. 'ยาว 29' → length=29). "
    "Extract size label from keywords like 'ไซส์' or 'size' (e.g. 'ไซส์ XL' → size_label='XL'). "

    "4. EMPTY RESPONSE: If the Spoken Data is just greetings, casual chat, or general rules with no product info, "
    "return an empty JSON {}."
)

_EXTRACT_RESULT = tuple[int | None, int | None, int | None, int | None, str | None]


class LiveDataExtractor:
    """
    คลาสสกัดข้อมูลแบบมี State รองรับ:
    - Hybrid NLP: Gemini Primary → Ollama Local Fallback
    - ตรวจจับ Price Conflict
    - Review mode (ราคาทวนซ้ำเลขน้อยลงกะทันหัน)
    - ดึงข้อมูลไซส์เสื้อผ้า (chest, length, size_label)
    """
    def __init__(self):
        self.history: Dict[int, Dict[str, Any]] = {}
        self.highest_item_number = 0
        self.current_code = None

        from app.config import settings
        self._settings = settings

        # ── Gemini client (optional) ──
        self.client = None
        if settings.gemini_api_key:
            try:
                from google import genai
                from google.genai import types as genai_types
                self._genai_types = genai_types
                self.client = genai.Client(api_key=settings.gemini_api_key)
                logger.info("[NLP] ✅ Gemini client พร้อม")
            except Exception as e:
                logger.warning("[NLP] ⚠️ ไม่สามารถเริ่ม Gemini client: %s", e)
        else:
            logger.warning("[NLP] ⚠️ ไม่พบ GEMINI_API_KEY → ใช้ Ollama เท่านั้น")

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: parse raw dict → safe typed tuple
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_dict(data: dict) -> _EXTRACT_RESULT:
        def _int(v):
            try:
                return int(v) if v is not None else None
            except (ValueError, TypeError):
                return None

        size_label = data.get("size_label")
        if size_label is not None:
            size_label = str(size_label).strip() or None

        return _int(data.get("item")), _int(data.get("price")), \
               _int(data.get("chest")), _int(data.get("length")), size_label

    # ─────────────────────────────────────────────────────────────────────────
    # Ollama fallback
    # ─────────────────────────────────────────────────────────────────────────
    def extract_with_ollama(self, text: str) -> _EXTRACT_RESULT:
        """ส่ง prompt ไปยัง Ollama local LLM แล้วแยกผลลัพธ์ JSON"""
        s = self._settings
        endpoint = f"{s.ollama_endpoint}/api/generate"
        prompt = f"{_SYSTEM_INSTRUCTION}\n\nSpoken Data: {text}"
        payload = {
            "model": s.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=15)
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
            data = json.loads(raw)
            result = self._parse_dict(data)
            logger.info("[NLP] 🦙 Ollama สกัดได้: item=%s price=%s", result[0], result[1])
            return result
        except requests.exceptions.ConnectionError:
            logger.error("[NLP] ❌ Ollama ไม่ได้รัน หรือ endpoint (%s) ผิด", endpoint)
        except Exception as e:
            logger.error("[NLP] ❌ Ollama error: %s", e)
        return None, None, None, None, None

    # ─────────────────────────────────────────────────────────────────────────
    # Gemini primary (with Ollama fallback on 429 / any failure)
    # ─────────────────────────────────────────────────────────────────────────
    def extract_with_gemini(self, text: str) -> _EXTRACT_RESULT:
        """Gemini API พร้อม fallback ไปยัง Ollama เมื่อ quota หมดหรือ error"""
        # ถ้าไม่มี Gemini key → ใช้ Ollama ทันที
        if not self.client:
            return self.extract_with_ollama(text)

        try:
            from google.genai import types as genai_types
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=ProductResponse,
                ),
            )

            parsed: ProductResponse = response.parsed
            if parsed is None:
                parsed = ProductResponse.model_validate_json(response.text)

            size_label = parsed.size_label
            if size_label is not None:
                size_label = size_label.strip() or None

            return parsed.item, parsed.price, parsed.chest, parsed.length, size_label

        except Exception as e:
            err_str = str(e)
            # ตรวจจับ 429 quota / spending cap
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                logger.warning(
                    "[NLP] ⚠️ Gemini quota exceeded or failed → falling back to Ollama... (%s)", e
                )
            else:
                logger.error("[NLP] ❌ Gemini API Error: %s", e)
            return self.extract_with_ollama(text)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────
    def process_text(self, text: str) -> Dict[str, Any] | None:
        """
        วิเคราะห์ข้อความด้วย Hybrid NLP (Gemini → Ollama)
        พร้อมระบุ state ว่าเป็น 'new', 'review', or 'conflict'
        """
        normalized = replace_thai_numbers(text)
        prompt = f"Spoken Data: {normalized}"

        # 1. เรียก NLP (Gemini หรือ Ollama)
        code, price, chest, length, size_label = self.extract_with_gemini(prompt)

        # หากมีการพูดถึงรหัสใหม่ เปลี่ยน State
        if code is not None:
            self.current_code = code

        if self.current_code is None:
            return None  # ยังไม่มีการตั้งต้นรายการใดๆ

        active_code = self.current_code

        # Determine Mode
        is_review = False
        conflict  = False
        old_price = None

        # Outlier Rejection: STT hallucinate เลขใหญ่มาก (+100) → ไม่อัปเดต highest
        jump = active_code - self.highest_item_number
        if jump > 100:
            logger.warning(
                "[NLP] ⚠️ Outlier item code: %d (jump=+%d) → ข้ามการอัปเดต highest",
                active_code, jump,
            )
        elif active_code > self.highest_item_number:
            self.highest_item_number = active_code
        elif active_code < self.highest_item_number and (self.highest_item_number - active_code) > 5:
            is_review = True

        # Save or Update State
        if active_code not in self.history:
            self.history[active_code] = {
                "price": price, "chest": chest,
                "length": length, "size_label": size_label,
            }
            is_review = False  # บังคับเป็นของใหม่เพราะไม่เคยมีประวัติ
        else:
            hist = self.history[active_code]
            if price is not None:
                if hist["price"] is not None and hist["price"] != price and hist["price"] != 0:
                    conflict  = True
                    old_price = hist["price"]
                hist["price"] = price
            if chest      is not None: hist["chest"]      = chest
            if length     is not None: hist["length"]     = length
            if size_label is not None: hist["size_label"] = size_label

        hist      = self.history[active_code]
        out_price = hist["price"]

        if out_price is None:
            return None  # มีแต่รหัส ยังไม่มีราคา

        result = {
            "item_code":  active_code,
            "price":      out_price,
            "chest":      hist.get("chest"),
            "length":     hist.get("length"),
            "size_label": hist.get("size_label"),
            "_raw":       text,
            "status": "conflict" if conflict else ("review" if is_review else "new"),
        }
        if old_price is not None:
            result["old_price"] = old_price

        return result
