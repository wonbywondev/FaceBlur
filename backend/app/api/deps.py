from typing import Dict, Any
from pathlib import Path
import uuid

from ..config import get_settings
from ..services.face_detector import FaceDetector
from ..services.face_embedder import FaceEmbedder
from ..services.face_matcher import FaceMatcher
from ..services.video_processor import VideoProcessor
from ..core.security import DataRetentionManager

settings = get_settings()

# In-memory storage for session data
# In production, use Redis or similar
sessions: Dict[str, Any] = {}
analyses: Dict[str, Any] = {}
processes: Dict[str, Any] = {}


def get_face_detector() -> FaceDetector:
    """Get or create face detector instance."""
    model_path = settings.model_dir / "yolov8n-face.pt"
    return FaceDetector(
        model_path=str(model_path) if model_path.exists() else None,
        conf_threshold=settings.face_detection_confidence
    )


def get_face_embedder() -> FaceEmbedder:
    """Get or create face embedder instance."""
    return FaceEmbedder()


def get_face_matcher(expected_persons: str = "10") -> FaceMatcher:
    """Get or create face matcher instance."""
    return FaceMatcher(
        similarity_threshold=settings.face_similarity_threshold,
        expected_persons=expected_persons
    )


def get_video_processor() -> VideoProcessor:
    """Get or create video processor instance."""
    return VideoProcessor()


def get_retention_manager() -> DataRetentionManager:
    """Get data retention manager."""
    return DataRetentionManager(
        base_path=settings.upload_dir,
        retention_hours=settings.retention_hours
    )


def create_session() -> str:
    """Create a new session and return session ID."""
    session_id = str(uuid.uuid4())
    session_dir = settings.upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    sessions[session_id] = {
        "id": session_id,
        "created_at": None,  # Will be set when video is uploaded
        "video_id": None,
        "reference_id": None,
        "analysis_id": None
    }

    return session_id


def get_session_dir(session_id: str) -> Path:
    """Get the directory for a session."""
    return settings.upload_dir / session_id


def cleanup_session(session_id: str):
    """Clean up session data."""
    retention_manager = get_retention_manager()
    retention_manager.delete_session_data(session_id)

    if session_id in sessions:
        del sessions[session_id]
