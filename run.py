"""
run.py
─────────────────────────────────────────────────────────────────
Entry point — รัน FastAPI ด้วย Uvicorn

วิธีใช้:
    python run.py

หรือรันตรงด้วย Uvicorn:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,           # ปิด reload เพราะ model ใหญ่ (เปิดได้ระหว่าง dev)
        log_level="info",
        access_log=True,
    )
