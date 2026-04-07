"""
app/utils/nlp.py
─────────────────────────────────────────────────────────────────
สกัดข้อมูลสินค้าจากข้อความภาษาไทยด้วย Google Gemini API
รองรับ State Management: โหมดทวนสินค้า (Review Mode) และ Price Conflict
รองรับการดึงข้อมูลไซส์เสื้อผ้า: อก, ยาว, ไซส์
"""
import os
import json
import logging
from typing import Dict, Any

from google import genai
from google.genai import types

from app.utils.thai_numbers import replace_thai_numbers, thai_digit_to_arabic

logger = logging.getLogger(__name__)

class LiveDataExtractor:
    """
    คลาสสกัดข้อมูลแบบมี State รองรับ:
    - ตรวจจับ Price Conflict
    - Review mode (ราคาทวนซ้ำเลขน้อยลงกะทันหัน)
    - ดึงข้อมูลไซส์เสื้อผ้า (chest, length, size_label)
    """
    def __init__(self):
        self.history: Dict[int, Dict[str, Any]] = {}
        self.highest_item_number = 0
        self.current_code = None

        # ตั้งค่า Gemini Client ตรวจสอบว่ามี API Key หรือไม่
        from app.config import settings
        self.api_key = settings.gemini_api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("[NLP] ⚠️ ไม่พบ GEMINI_API_KEY ใน Environment ระบบสกัดข้อมูลจะไม่ทำงาน!")

        self.system_instruction = (
            "Extract product information from the following multimodal data (Spoken Audio Transcript from a Thai secondhand clothing livestream). "
            "You may also be provided with 'Visual Data' (OCR Tags). "
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
            "'รายการที่', 'รหัส', or 'ตัวที่' in the Spoken Data, OR unless a Number Merging pattern is clearly present. "

            "3. SIZES: Extract chest measurement from keywords like 'อก' (e.g. 'อก 48' → chest=48). "
            "Extract length from keywords like 'ยาว' (e.g. 'ยาว 29' → length=29). "
            "Extract size label from keywords like 'ไซส์' or 'size' (e.g. 'ไซส์ XL' → size_label='XL'). "

            "4. EMPTY RESPONSE: If the Spoken Data is just greetings, casual chat, or general rules with no product info, "
            "return an empty JSON {}. "

            "5. VISUAL DATA: Use Visual Data strictly to correct or validate misheard numbers/strings from the Spoken Data."
        )

    def extract_with_gemini(self, text: str) -> tuple[int | None, int | None, int | None, int | None, str | None]:
        """เรียกใช้ Gemini API สกัด item, price, chest, length, size_label"""
        if not self.client:
            return None, None, None, None, None

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            
            output_text = response.text.strip()
            # อาจมี Markdown โอบล้อม ให้เอาออกถ้าจำเป็น (แต่ response_mime_type มักจะป้องกันได้)
            if output_text.startswith("```json"):
                output_text = output_text[7:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]

            data = json.loads(output_text.strip())
            
            code = data.get("item")
            price = data.get("price")
            chest = data.get("chest")
            length = data.get("length")
            size_label = data.get("size_label")
            
            # แปลงเป็น int อย่างปลอดภัย
            try:
                code = int(code) if code is not None else None
            except (ValueError, TypeError):
                code = None
                
            try:
                price = int(price) if price is not None else None
            except (ValueError, TypeError):
                price = None

            try:
                chest = int(chest) if chest is not None else None
            except (ValueError, TypeError):
                chest = None

            try:
                length = int(length) if length is not None else None
            except (ValueError, TypeError):
                length = None

            # size_label คือ string เช่น "XL", "L", "M"
            if size_label is not None:
                size_label = str(size_label).strip() or None

            return code, price, chest, length, size_label

        except Exception as e:
            logger.error(f"[NLP] ❌ Gemini API Error / JSON Parsing: {e}")
            return None, None, None, None, None

    def process_text(self, text: str, ocr_data: str = "") -> Dict[str, Any] | None:
        """
        วิเคราะห์ข้อความด้วย Gemini ถ้าได้ข้อมูลมา ให้คืน Dict
        พร้อมระบุ state ว่าเป็น 'new', 'review', or 'conflict'
        รองรับข้อมูลไซส์: chest, length, size_label
        """
        normalized = replace_thai_numbers(text)
        
        prompt = f"Spoken Data: {normalized}"
        if ocr_data:
            prompt += f"\nVisual Data (OCR Extracted tags): {ocr_data}"
        
        # 1. Call Gemini API
        code, price, chest, length, size_label = self.extract_with_gemini(prompt)
        
        # หากมีการพูดถึงรหัสใหม่ เปลี่ยน State มาเล็งที่ตัวนี้
        if code is not None:
            self.current_code = code
            
        if self.current_code is None:
            return None  # ยังไม่มีการตั้งต้นรายการใดๆ
            
        active_code = self.current_code
        
        # Determine Mode
        is_review = False
        conflict = False
        old_price = None
        
        if active_code > self.highest_item_number:
            self.highest_item_number = active_code
        elif active_code < self.highest_item_number and (self.highest_item_number - active_code) > 5:
            # เลขลดลงเยอะผิดปกติ -> เข้าโหมด Review!
            is_review = True
            
        # Save or Update State
        if active_code not in self.history:
            self.history[active_code] = {
                "price": price,
                "chest": chest,
                "length": length,
                "size_label": size_label,
            }
            is_review = False  # บังคับเป็นของใหม่เพราะไม่เคยมีประวัติ
        else:
            hist = self.history[active_code]
            if price is not None:
                # ตรวจจับ Price Conflict ในโหมดทวน
                if hist["price"] is not None and hist["price"] != price and hist["price"] != 0:
                    conflict = True
                    old_price = hist["price"]
                hist["price"] = price
            # อัปเดตข้อมูลไซส์ถ้ามี (ประโยคถัดๆ ไปอาจพูดเพิ่มเติม)
            if chest is not None:
                hist["chest"] = chest
            if length is not None:
                hist["length"] = length
            if size_label is not None:
                hist["size_label"] = size_label
        
        # เตรียมส่งคืนข้อมูล
        hist = self.history[active_code]
        out_price = hist["price"]
        
        if out_price is None:
            return None  # มีแต่รหัส ยังไม่มีราคา (รอประโยคถัดไป)
            
        result = {
            "item_code": active_code,
            "price": out_price,
            "chest": hist.get("chest"),
            "length": hist.get("length"),
            "size_label": hist.get("size_label"),
            "_raw": text,
            "status": "conflict" if conflict else ("review" if is_review else "new"),
        }
        
        if old_price is not None:
            result["old_price"] = old_price
            
        return result
