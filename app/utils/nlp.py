"""
app/utils/nlp.py
─────────────────────────────────────────────────────────────────
สกัดข้อมูลสินค้าจากข้อความภาษาไทยด้วย Regular Expression
รองรับบริบทการพูดของแม่ค้าในไลฟ์สด
"""
import re
from app.utils.thai_numbers import replace_thai_numbers, thai_digit_to_arabic

# ─── Keyword aliases ───────────────────────────────────────────────────────
_KW_CODE   = r"(?:รายการ(?:ที่)?|รหัส|ไอเท็ม|item|no\.?|ลำดับ(?:ที่)?|ตัวที่|เบอร์|หมายเลข)\s*"
_KW_PRICE  = r"(?:ราคา|ขาย|บาท|฿|รา(?:คา)?)\s*"
_KW_CHEST  = r"(?:อก|รอบอก|chest)\s*"
_KW_LENGTH = r"(?:ยาว|ความยาว|length)\s*"

# ─── Unit suffixes (optional) ──────────────────────────────────────────────
_UNIT_PRICE = r"(?:\s*บาท)?"
_UNIT_SIZE  = r"(?:\s*(?:นิ้ว|cm|ซม\.?|นิ้ว))?"

# ─── Numeric token (Arabic + Thai digits) ─────────────────────────────────
_NUM = r"([0-9๐-๙]+(?:\.[0-9๐-๙]+)?)"

# ─── Compiled patterns ─────────────────────────────────────────────────────
_PAT_CODE   = re.compile(_KW_CODE  + _NUM + r"(?!\s*(?:บาท|฿)|[0-9๐-๙])", re.IGNORECASE)
_PAT_PRICE  = re.compile(_KW_PRICE + r"\s*" + _NUM + _UNIT_PRICE, re.IGNORECASE)
_PAT_PRICE2 = re.compile(_NUM + r"\s*บาท", re.IGNORECASE)   # '80 บาท' / '80บาท'
_PAT_CHEST  = re.compile(_KW_CHEST  + r"\s*" + _NUM + _UNIT_SIZE, re.IGNORECASE)
_PAT_LENGTH = re.compile(_KW_LENGTH + r"\s*" + _NUM + _UNIT_SIZE, re.IGNORECASE)


def extract_product_data(text: str) -> dict | None:
    """
    สกัดข้อมูลสินค้าจากข้อความภาษาไทย

    ขั้นตอน:
      1) แปลงคำอ่านตัวเลขไทยทั้งหมด → Arabic
      2) Regex ดักจับ item_code, price, chest, length
      3) คืน dict เมื่อพบ item_code + price ครบ / คืน None เมื่อข้อมูลไม่ครบ

    รูปแบบที่รองรับ:
      'รายการที่ยี่สิบสี่ อกสี่สิบแปด ยาวสามสิบ ราคาแปดสิบบาท'
      'รหัสสิบห้า หกสิบบาท'
      'ไอเท็ม 3 อก 36 นิ้ว ยาว 26 ราคา 120 บาท'
    """
    # Step 1: แปลงเลขคำอ่านไทย
    normalized = replace_thai_numbers(text)

    # Step 2: item_code (บังคับ)
    m = _PAT_CODE.search(normalized)
    if not m:
        return None
    item_code = thai_digit_to_arabic(m.group(1))
    if item_code is None:
        return None

    # Step 3: price (บังคับ)
    price: int | None = None
    m2 = _PAT_PRICE.search(normalized)
    if m2:
        price = thai_digit_to_arabic(m2.group(1))
    if price is None:
        m3 = _PAT_PRICE2.search(normalized)
        if m3:
            price = thai_digit_to_arabic(m3.group(1))
    if price is None:
        return None

    # Step 4: chest / length (optional)
    chest = length = None
    mc = _PAT_CHEST.search(normalized)
    if mc:
        chest = thai_digit_to_arabic(mc.group(1))
    ml = _PAT_LENGTH.search(normalized)
    if ml:
        length = thai_digit_to_arabic(ml.group(1))

    result: dict = {"item_code": item_code, "price": price}
    if chest  is not None: result["chest"]  = chest
    if length is not None: result["length"] = length
    result["_raw"]        = text
    result["_normalized"] = normalized
    return result
