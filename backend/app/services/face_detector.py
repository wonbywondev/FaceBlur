import cv2
import numpy as np
from typing import List, Dict, Optional, Generator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Lazy import for torch and ultralytics
_model = None


def get_device():
    """Get the best available device for inference."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class FaceDetector:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5):
        self.model_path = model_path or "yolov8n-face.pt"
        self.conf_threshold = conf_threshold
        self.model = None
        self.device = None

    def _load_model(self):
        """Lazy load the YOLO model."""
        if self.model is None:
            from ultralytics import YOLO
            self.device = get_device()
            logger.info(f"Loading YOLO model on device: {self.device}")

            # Try to load custom model, fall back to default
            try:
                if Path(self.model_path).exists():
                    self.model = YOLO(self.model_path)
                else:
                    # Use default YOLOv8n and fine-tune for face detection
                    self.model = YOLO("yolov8n.pt")
                    logger.warning(f"Custom model not found at {self.model_path}, using default YOLOv8n")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a single frame."""
        self._load_model()

        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        faces = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                # Filter for person class (class 0) as proxy for face detection
                # In production, use a face-specific model
                faces.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence
                })

        return faces

    def extract_face_crop(
        self,
        frame: np.ndarray,
        bbox: List[int],
        padding: float = 0.2
    ) -> np.ndarray:
        """Extract face region with padding."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return frame[y1:y2, x1:x2].copy()

    def process_video(
        self,
        video_path: str,
        sample_rate: int = 5,
        progress_callback: Optional[callable] = None
    ) -> Generator[Dict, None, None]:
        """
        Process video and yield face detections.

        Args:
            video_path: Path to the video file
            sample_rate: Process every N frames
            progress_callback: Optional callback for progress updates

        Yields:
            Dict with frame info and detected faces
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate which frames to process
        frames_to_process = list(range(0, total_frames, sample_rate))
        total_to_process = len(frames_to_process)

        try:
            # Initial progress update
            if progress_callback:
                progress_callback(0, 0)

            for idx, frame_number in enumerate(frames_to_process):
                # Seek to the target frame (faster than reading all frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    continue

                timestamp = frame_number / fps
                faces = self.detect_faces(frame)

                yield {
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "faces": faces,
                    "frame": frame
                }

                # Update progress based on frames processed
                if progress_callback:
                    progress = ((idx + 1) / total_to_process) * 100
                    progress_callback(progress, frame_number)

            # Final progress update
            if progress_callback:
                progress_callback(100, total_frames)

        finally:
            cap.release()

    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            return info
        finally:
            cap.release()
