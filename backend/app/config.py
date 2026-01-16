from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Face Blur Service"
    debug: bool = False

    # Paths
    upload_dir: Path = Path("uploads")
    model_dir: Path = Path("models")
    output_dir: Path = Path("outputs")

    # File limits
    max_video_size_mb: int = 500
    max_image_size_mb: int = 10
    max_reference_images: int = 5
    allowed_video_extensions: set = {".mp4", ".mov"}
    allowed_image_extensions: set = {".jpg", ".jpeg", ".png"}

    # Processing
    frame_sample_rate: int = 5  # Process every N frames
    face_detection_confidence: float = 0.1  # Very low = aggressive detection
    face_similarity_threshold: float = 0.6

    # Data retention
    retention_hours: int = 24

    # CORS
    cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
