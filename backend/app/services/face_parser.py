"""
Face Parsing Service using facer library

Provides pixel-level face segmentation for precise blur masks.
Uses facer library which includes pretrained face parsing models.
"""
import logging
import numpy as np
import cv2
from typing import Optional, List

logger = logging.getLogger(__name__)

# Face classes to include in blur mask
# facer/LaPa uses: 0=background, 1=face skin, 2=left eyebrow, 3=right eyebrow,
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
            # Note: MPS has issues with adaptive_avg_pool2d in facer model
            # Fall back to CPU for face parsing on Apple Silicon
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

            # Initialize only the face parser (we use existing bbox from detector)
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
            bbox: Bounding box [x1, y1, x2, y2] for the face

        Returns:
            Binary mask of face region (0-255), or None if parsing failed
        """
        if not self._ensure_model():
            return None

        if bbox is None:
            return None

        try:
            import torch

            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to tensor (B, C, H, W) with uint8 values in [0, 255]
            # facer expects: b x 3 x h x w, torch.uint8, [0, 255]
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Create data dict with bbox info for facer
            # facer expects: rects (N, 4), points (N, 5, 2), scores (N,), image_ids (N,)
            rects = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=self.device)
            scores = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            image_ids = torch.tensor([0], dtype=torch.long, device=self.device)

            # Create 5-point landmarks (approximate from bbox)
            # Points order: left_eye, right_eye, nose, left_mouth, right_mouth
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            face_w, face_h = x2 - x1, y2 - y1
            points = torch.tensor([[
                [x1 + face_w * 0.3, y1 + face_h * 0.35],  # left eye
                [x1 + face_w * 0.7, y1 + face_h * 0.35],  # right eye
                [cx, y1 + face_h * 0.55],                  # nose
                [x1 + face_w * 0.35, y1 + face_h * 0.75], # left mouth
                [x1 + face_w * 0.65, y1 + face_h * 0.75], # right mouth
            ]], dtype=torch.float32, device=self.device)

            # Data dict (without image - image is passed separately)
            data = {
                "rects": rects,
                "points": points,
                "scores": scores,
                "image_ids": image_ids,
            }

            # Run face parsing - facer.forward(images, data)
            result = self.face_parser(image_tensor, data)

            if "seg" not in result or "logits" not in result["seg"]:
                logger.debug("[FACE_PARSER] No segmentation result")
                return None

            # Get segmentation logits and convert to class predictions
            seg_logits = result["seg"]["logits"]  # (B, num_faces, C, H, W)

            # Get the first face's segmentation
            face_seg = seg_logits[0, 0]  # (C, H, W)
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
