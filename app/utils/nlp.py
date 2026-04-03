"""
app/utils/nlp.py
─────────────────────────────────────────────────────────────────
สกัดข้อมูลสินค้าจากข้อความภาษาไทยด้วย Regular Expression แบบมี Context
รองรับดักจับราคาหลอมติดกันของ ASR, จัดการโหมดทวนสินค้า (Review Mode)
"""
import re
from typing import Dict, Any

from app.utils.thai_numbers import replace_thai_numbers, thai_digit_to_arabic

# ─── Keyword aliases ───────────────────────────────────────────────────────
_KW_CODE   = r"(?:รายการ(?:ที่)?|รหัส|ไอเท็ม|item|no\.?|ลำดับ(?:ที่)?|ตัวที่|เบอร์|หมายเลข)\s*"
_KW_PRICE  = r"(?:ราคา|ขาย|฿|รา(?:คา)?)\s*"
_KW_CHEST  = r"(?:อก|รอบอก|ขออก|chest)\s*"
_KW_LENGTH = r"(?:ยาว|ความยาว|length)\s*"

# ─── Unit suffixes (optional) ──────────────────────────────────────────────
_UNIT_PRICE = r"(?:\s*บาท)?"
_UNIT_SIZE  = r"(?:\s*(?:นิ้ว|cm|ซม\.?|นิ้ว))?"

# ─── Numeric token (Arabic + Thai digits) ─────────────────────────────────
_NUM = r"([0-9๐-๙]+(?:\.[0-9๐-๙]+)?)"

# ราคามาตรฐานที่เจอบ่อย (ลงท้ายด้วยศูนย์)
STANDARD_PRICES = {20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 190, 200, 250, 300, 350}

# ─── Compiled patterns ─────────────────────────────────────────────────────
# การบอกรหัสสินค้า: ต้องไม่ตามด้วยคำว่า บาท/฿ เพื่อแยกรหัสกับราคา
_PAT_CODE   = re.compile(_KW_CODE  + _NUM + r"(?!\s*(?:บาท|฿)|[0-9๐-๙])", re.IGNORECASE)

# การรับราคาแบบเจาะจง
_PAT_PRICE  = re.compile(_KW_PRICE + r"\s*" + _NUM + _UNIT_PRICE, re.IGNORECASE)
_PAT_PRICE2 = re.compile(_NUM + r"\s*บาท", re.IGNORECASE)

# อกและยาว: ตัดคำว่า "ตึงหน้าผ้า" หรืออะไรก็ตามออก ดึงเฉพาะตัวเลขที่ติดกับคีย์เวิร์ด
_PAT_CHEST  = re.compile(_KW_CHEST  + _NUM + _UNIT_SIZE, re.IGNORECASE)
_PAT_LENGTH = re.compile(_KW_LENGTH + _NUM + _UNIT_SIZE, re.IGNORECASE)


class LiveDataExtractor:
    """
    คลาสสกัดข้อมูลแบบมี State รองรับ:
    - Fused numbers (เช่น 'รหัส 1180' -> ถอดเป็น รหัส 11, ราคา 80)
    - Review mode (ราคาทวนซ้ำเลขน้อยลงกะทันหัน)
    - ตรวจจับ Price Conflict
    """
    def __init__(self):
        self.history: Dict[int, Dict[str, Any]] = {}
        self.highest_item_number = 0
        self.current_code = None

    def process_text(self, text: str) -> Dict[str, Any] | None:
        """
        วิเคราะห์ประโยค ถ้าได้ข้อมูลชิ้นหลักครบถ้วน (code+price) ให้คืน Dict
        พร้อมระบุ state ว่าเป็น 'new', 'review', or 'conflict'
        """
        normalized = replace_thai_numbers(text)
        
        chest = length = price = code = None
        
        # 1. Size
        mc = _PAT_CHEST.search(normalized)
        if mc: 
            chest = thai_digit_to_arabic(mc.group(1))
            
        ml = _PAT_LENGTH.search(normalized)
        if ml: 
            length = thai_digit_to_arabic(ml.group(1))
            
        # 2. Price
        mp = _PAT_PRICE.search(normalized)
        if mp:
            price = thai_digit_to_arabic(mp.group(1))
        else:
            mp2 = _PAT_PRICE2.search(normalized)
            if mp2:
                price = thai_digit_to_arabic(mp2.group(1))
                
        # 3. Item Code & Fused Numbers (ASR error e.g. 380 -> 3: 80)
        mc_code = _PAT_CODE.search(normalized)
        if mc_code:
            code_raw = int(thai_digit_to_arabic(mc_code.group(1)))
            
            # ถ้าไม่ได้เจอราคาด้วยคีย์เวิร์ด และตัวเลขรหัสที่จับได้มี 3 หลักขึ้นไป -> สงสัยว่า Fused!
            if price is None and code_raw > 100:
                s_code = str(code_raw)
                # ลองตัด 3 หลัก หรือ 2 หลักสุดท้ายไปเทียบเป็นราคามาตรฐาน
                found_fused = False
                for suffix_len in (3, 2):
                    if len(s_code) > suffix_len:
                        p_val = int(s_code[-suffix_len:])
                        if p_val in STANDARD_PRICES:
                            price = p_val
                            code = int(s_code[:-suffix_len])
                            found_fused = True
                            break
                if not found_fused:
                    code = code_raw # ถ้าไม่เข้าข่ายราคามาตรฐาน ก็ให้เป็นรหัสอ้วนๆไป
            else:
                code = code_raw
                
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
            self.history[active_code] = {"price": price, "chest": chest, "length": length}
            is_review = False # บังคับเป็นของใหม่เพราะไม่เคยมีประวัติ
        else:
            hist = self.history[active_code]
            if chest: hist["chest"] = chest
            if length: hist["length"] = length
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
        if hist["chest"] is not None:
            result["chest"] = hist["chest"]
        if hist["length"] is not None:
            result["length"] = hist["length"]
            
        return result
