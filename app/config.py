"""
app/config.py
─────────────────────────────────────────────────────────────────
อ่านค่า Environment Variables จากไฟล์ .env ด้วย pydantic-settings
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Webhook
    webhook_url: str = ""

    # Whisper
    whisper_model_size: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute: str = "float16"
    target_language: str = "th"

    # Audio
    chunk_seconds: int = 10
    sample_rate: int = 16000

    # Computer Vision
    yolo_model_name: str = "runs/detect/train2/weights/best.pt"
    capture_duration: float = 3.0
    capture_fps_target: int = 10
    blur_threshold: float = 80.0
    video_frame_w: int = 1280
    video_frame_h: int = 720
    output_dir: Path = Path("captured_images")

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def bytes_per_sample(self) -> int:
        return 2  # int16

    @property
    def chunk_bytes(self) -> int:
        return self.sample_rate * self.bytes_per_sample * self.chunk_seconds


# Singleton — import ตรงๆ ในทุกโมดูล
settings = Settings()
