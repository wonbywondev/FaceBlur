from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from uuid import UUID
from datetime import datetime


class BlurType(str, Enum):
    GAUSSIAN = "gaussian"
    MOSAIC = "mosaic"
    BLACKOUT = "blackout"


class ExpectedPersonCount(str, Enum):
    """Expected number of people in video for clustering optimization."""
    FIVE = "5"        # ~5 people (stricter matching)
    TEN = "10"        # ~10 people
    TWENTY = "20"     # ~20 people
    MANY = "many"     # Many people / crowd (more lenient matching)


class ProcessStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class Resolution(BaseModel):
    width: int
    height: int


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    duration: float
    resolution: Resolution
    status: ProcessStatus


class ReferenceUploadResponse(BaseModel):
    reference_id: str
    face_count: int
    embeddings_generated: bool


class AnalyzeRequest(BaseModel):
    video_id: str
    reference_id: Optional[str] = None  # Optional - for video-first flow
    expected_persons: Optional[ExpectedPersonCount] = ExpectedPersonCount.TEN  # Default ~10 people


class AnalyzeResponse(BaseModel):
    analysis_id: str
    status: ProcessStatus
    estimated_time: int


class FaceAppearance(BaseModel):
    start: float
    end: float
    bbox: List[int]


class DetectedFace(BaseModel):
    face_id: str
    thumbnail: str  # base64 encoded
    first_appearance: float
    appearances: List[FaceAppearance]
    appearance_count: int  # Total number of appearances in video
    similarity_to_reference: float = Field(ge=0, le=100)
    is_reference: bool
    blur_enabled: bool


class AnalysisResult(BaseModel):
    analysis_id: str
    faces: List[DetectedFace]
    total_faces: int
    reference_matches: int
    status: ProcessStatus


class BlurSettings(BaseModel):
    type: BlurType = BlurType.GAUSSIAN
    intensity: int = Field(default=25, ge=1, le=50)
    face_ids: List[str]


class ProcessRequest(BaseModel):
    analysis_id: str
    blur_settings: BlurSettings


class ProcessResponse(BaseModel):
    process_id: str
    status: ProcessStatus


class ProgressUpdate(BaseModel):
    progress: float
    faces_detected: int = 0
    frames_processed: int = 0
    status: ProcessStatus
    message: str = ""


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
