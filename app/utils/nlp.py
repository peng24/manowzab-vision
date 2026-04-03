"""
app/utils/nlp.py
─────────────────────────────────────────────────────────────────
สกัดข้อมูลสินค้าจากข้อความภาษาไทยด้วย Google Gemini API
รองรับ State Management: โหมดทวนสินค้า (Review Mode) และ Price Conflict
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
            "Extract the item code (รหัสสินค้า/รายการที่) and price (ราคา) from the following Thai live-selling transcript. "
            "Return ONLY a valid JSON object with keys 'item' (string or integer) and 'price' (integer). "
            "IMPORTANT RULES: You MUST NOT extract an item code unless it is explicitly preceded by context words like "
            "'รายการที่', 'รหัส', or 'ตัวที่'. If the transcript is just greetings, casual chat, or general rules "
            "(e.g., 'ราคาเริ่มต้น 20 ถึง 100 บาท', 'ตัวนี้สวยมาก'), you MUST return an empty JSON {}."
        )

    def extract_with_gemini(self, text: str) -> tuple[int | None, int | None]:
        """เรียกใช้ Gemini API สกัด item และ price เป็นคู่ tuple"""
        if not self.client:
            return None, None

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
            
            # แปลงเป็น int อย่างปลอดภัย
            try:
                code = int(code) if code is not None else None
            except ValueError:
                code = None
                
            try:
                price = int(price) if price is not None else None
            except ValueError:
                price = None

            return code, price

        except Exception as e:
            logger.error(f"[NLP] ❌ Gemini API Error / JSON Parsing: {e}")
            return None, None

    def process_text(self, text: str) -> Dict[str, Any] | None:
        """
        วิเคราะห์ข้อความด้วย Gemini ถ้าได้ข้อมูลมา ให้คืน Dict
        พร้อมระบุ state ว่าเป็น 'new', 'review', or 'conflict'
        """
        normalized = replace_thai_numbers(text)
        
        # 1. Сall Gemini API
        code, price = self.extract_with_gemini(normalized)
        
        # หากมีการพูดถึงรหัสใหม่ เปลี่ยน State มาเล็งที่ตัวนี้
        if code is not None:
            self.current_code = code
            
        if self.current_code is None:
            return None # ยังไม่มีการตั้งต้นรายการใดๆ
            
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
            self.history[active_code] = {"price": price}
            is_review = False # บังคับเป็นของใหม่เพราะไม่เคยมีประวัติ
        else:
            hist = self.history[active_code]
            if price is not None:
                # ตรวจจับ Price Conflict ในโหมดทวน
                if hist["price"] is not None and hist["price"] != price and hist["price"] != 0:
                    conflict = True
                    old_price = hist["price"]
                hist["price"] = price
        
        # เตรียมส่งคืนข้อมูล
        hist = self.history[active_code]
        out_price = hist["price"]
        
        if out_price is None:
            return None # มีแต่รหัส ยังไม่มีราคา (รอประโยคถัดไป)
            
        result = {
            "item_code": active_code,
            "price": out_price,
            "_raw": text,
            "status": "conflict" if conflict else ("review" if is_review else "new")
        }
        
        if old_price is not None:
            result["old_price"] = old_price
            
        return result
