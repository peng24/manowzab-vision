"""
app/utils/thai_numbers.py
─────────────────────────────────────────────────────────────────
แปลงคำอ่านตัวเลขภาษาไทย → ตัวเลขฮินดูอารบิก
รองรับ 0–999  (ร้อย, สิบ, หน่วย)
"""
import re

# ─── Token tables ──────────────────────────────────────────────────────────
_ONES: dict[str, int] = {
    "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1,
    "สอง": 2,  "ยี่": 2,
    "สาม": 3,  "สี่": 4,  "ห้า": 5,
    "หก": 6,   "เจ็ด": 7, "แปด": 8, "เก้า": 9,
}
_TENS: dict[str, int] = {
    "สิบ": 10,  "ยี่สิบ": 20, "สามสิบ": 30,
    "สี่สิบ": 40, "ห้าสิบ": 50, "หกสิบ": 60,
    "เจ็ดสิบ": 70, "แปดสิบ": 80, "เก้าสิบ": 90,
}
_HUNDREDS: dict[str, int] = {
    "ร้อย": 100,   "สองร้อย": 200, "สามร้อย": 300,
    "สี่ร้อย": 400, "ห้าร้อย": 500, "หกร้อย": 600,
    "เจ็ดร้อย": 700, "แปดร้อย": 800, "เก้าร้อย": 900,
}

# เรียงจากยาวไปสั้น → greedy match
_ALL_TOKENS: list[tuple[str, int]] = sorted(
    list(_HUNDREDS.items()) + list(_TENS.items()) + list(_ONES.items()),
    key=lambda x: -len(x[0]),
)

# Compiled regex จับกลุ่มคำตัวเลขในประโยค
NUM_WORD_PATTERN: re.Pattern = re.compile(
    r"(?:" + r"|".join(re.escape(k) for k, _ in _ALL_TOKENS) + r")+"
)


def thai_words_to_int(text: str) -> int | None:
    """
    แปลง string คำอ่านตัวเลขภาษาไทย → int
    คืน None หากแปลงไม่ได้

    ตัวอย่าง:
        'ยี่สิบสี่'     → 24
        'แปดสิบ'        → 80
        'หนึ่งร้อยยี่สิบ' → 120
    """
    pos = total = current_hundred = current_rest = 0
    while pos < len(text):
        matched = False
        for token, val in _ALL_TOKENS:
            if text.startswith(token, pos):
                if val >= 100:
                    total += current_hundred + current_rest
                    current_hundred, current_rest = val, 0
                elif val >= 10:
                    current_rest = val
                else:
                    current_rest += val
                pos += len(token)
                matched = True
                break
        if not matched:
            pos += 1
    total += current_hundred + current_rest
    return total if total > 0 else None


def replace_thai_numbers(text: str) -> str:
    """
    แทนที่คำอ่านตัวเลขไทยทุกตำแหน่งในประโยคด้วยตัวเลข Arabic
    'รายการที่ยี่สิบสี่ อกสี่สิบแปด' → 'รายการที่24 อก48'
    """
    def _replacer(m: re.Match) -> str:
        val = thai_words_to_int(m.group())
        return str(val) if val is not None else m.group()

    return NUM_WORD_PATTERN.sub(_replacer, text)


def thai_digit_to_arabic(s: str) -> int | None:
    """แปลงเลขไทย (๑๒๓) หรือ Arabic digit string → int"""
    _MAP = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
    try:
        return int(float(s.translate(_MAP)))
    except (ValueError, TypeError):
        return None
