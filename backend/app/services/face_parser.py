"""
Face Parsing Service using facer library

Provides pixel-level face segmentation for precise blur masks.
Uses facer library which includes pretrained face parsing models.
"""
import logging
import numpy as np
import cv2
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Face classes to include in blur mask
# facer uses: 0=background, 1=face skin, 2=left eyebrow, 3=right eyebrow,
# 4=left eye, 5=right eye, 6=nose, 7=upper lip, 8=inner mouth, 9=lower lip, 10=hair
FACE_SKIN_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Exclude background and hair


class FaceParser:
    """
    Face parsing using facer library for precise face segmentation.
    Falls back to ellipse mask if facer is not available.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize face parser.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
        """
        self.device = self._get_device(device)
        self.face_detector = None
        self.face_parser = None
        self._initialized = False
        self._init_error = None
        self._facer_available = False

    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device != "auto":
            return device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model(self) -> bool:
        """Load facer face parser model."""
        if self._initialized:
            return self._facer_available

        self._initialized = True

        try:
            import facer

            # Initialize face detector and parser (cached)
            self.face_detector = facer.face_detector("retinaface/mobilenet", device=self.device)
            self.face_parser = facer.face_parser("farl/lapa/448", device=self.device)
            self._facer_available = True
            logger.info(f"[FACE_PARSER] facer face parser loaded on {self.device}")
            return True

        except ImportError as e:
            self._init_error = f"facer not installed: {e}"
            logger.warning(f"[FACE_PARSER] {self._init_error}")
            logger.info("[FACE_PARSER] Install with: pip install pyfacer")
            return False
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"[FACE_PARSER] Failed to load facer: {e}")
            return False

    def parse_face(
        self,
        image: np.ndarray,
        bbox: Optional[List[int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Parse face and return segmentation mask.

        Args:
            image: Input image (BGR)
            bbox: Optional bounding box [x1, y1, x2, y2] to focus on face region

        Returns:
            Binary mask of face region (0-255), or None if parsing failed
        """
        if not self._ensure_model():
            return None

        try:
            import torch

            h, w = image.shape[:2]

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to tensor and add batch dimension
            # facer expects (B, C, H, W) with values in [0, 255]
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()
            image_tensor = image_tensor.to(self.device)

            # Detect faces
            with torch.no_grad():
                faces = self.face_detector({"image": image_tensor})

            if faces is None or "image_ids" not in faces or len(faces["image_ids"]) == 0:
                logger.debug("[FACE_PARSER] No faces detected by facer")
                return None

            # Clone tensors to avoid the 'dict has no clone' error
            faces_input = {"image": image_tensor.clone()}
            for key, value in faces.items():
                if isinstance(value, torch.Tensor):
                    faces_input[key] = value.clone()
                else:
                    faces_input[key] = value

            # Run face parsing
            with torch.no_grad():
                result = self.face_parser(faces_input)

            if "seg" not in result or "logits" not in result["seg"]:
                logger.debug("[FACE_PARSER] No segmentation result")
                return None

            # Get segmentation logits and convert to class predictions
            seg_logits = result["seg"]["logits"]  # (B, num_faces, C, H, W)
            seg_probs = seg_logits.softmax(dim=2)  # Softmax over classes

            # Find the face closest to the bbox (if provided)
            best_face_idx = 0
            if bbox is not None and "rects" in result:
                x1, y1, x2, y2 = bbox
                bbox_cx, bbox_cy = (x1 + x2) / 2, (y1 + y2) / 2

                rects = result["rects"].cpu().numpy()  # (B, num_faces, 4)
                min_dist = float('inf')

                for i, rect in enumerate(rects[0]):  # First batch
                    face_cx = (rect[0] + rect[2]) / 2
                    face_cy = (rect[1] + rect[3]) / 2
                    dist = (face_cx - bbox_cx)**2 + (face_cy - bbox_cy)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_face_idx = i

            # Get the segmentation for the selected face
            if seg_probs.shape[1] <= best_face_idx:
                best_face_idx = 0

            face_seg = seg_probs[0, best_face_idx]  # (C, H, W)
            face_classes = face_seg.argmax(dim=0).cpu().numpy()  # (H, W)

            # Create binary mask from face skin classes
            mask = np.zeros(face_classes.shape, dtype=np.uint8)
            for class_id in FACE_SKIN_CLASSES:
                mask[face_classes == class_id] = 255

            # Resize mask to original image size if needed
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Smooth edges for natural blending
            mask = cv2.GaussianBlur(mask, (15, 15), 0)

            # Fill holes in the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            return mask

        except Exception as e:
            logger.error(f"[FACE_PARSER] Face parsing failed: {e}")
            return None

    def is_available(self) -> bool:
        """Check if face parser is available and working."""
        return self._ensure_model()


# Singleton instance
_face_parser: Optional[FaceParser] = None


def get_face_parser() -> FaceParser:
    """Get or create face parser singleton."""
    global _face_parser
    if _face_parser is None:
        _face_parser = FaceParser()
    return _face_parser
