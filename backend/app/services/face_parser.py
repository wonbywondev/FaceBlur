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
                logger.info("[FACE_PARSER] CUDA available, using GPU")
                return "cuda"
            elif torch.backends.mps.is_available():
                # Try MPS (Apple Silicon GPU)
                # Note: Older PyTorch versions had issues with adaptive_avg_pool2d
                # but newer versions may work. We'll try and fallback to CPU if needed.
                logger.info("[FACE_PARSER] MPS available, attempting to use Apple Silicon GPU")
                return "mps"
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[FACE_PARSER] Error checking GPU availability: {e}")
        
        logger.info("[FACE_PARSER] Using CPU (no GPU available)")
        return "cpu"

    def _ensure_model(self) -> bool:
        """Load facer face parser model."""
        if self._initialized:
            return self._facer_available

        self._initialized = True

        try:
            import facer

            # Initialize only the face parser (we use existing bbox from detector)
            try:
                self.face_parser = facer.face_parser("farl/lapa/448", device=self.device)
                self._facer_available = True
                logger.info(f"[FACE_PARSER] facer face parser loaded on {self.device}")
                return True
            except RuntimeError as e:
                # MPS might fail on some operations, fallback to CPU
                if self.device == "mps" and "mps" in str(e).lower():
                    logger.warning(f"[FACE_PARSER] MPS failed ({e}), falling back to CPU")
                    self.device = "cpu"
                    self.face_parser = facer.face_parser("farl/lapa/448", device=self.device)
                    self._facer_available = True
                    logger.info(f"[FACE_PARSER] facer face parser loaded on {self.device}")
                    return True
                else:
                    raise

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
        landmarks: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Parse face and return segmentation mask.

        Args:
            image: Input image (BGR)
            bbox: Bounding box [x1, y1, x2, y2] for the face
            landmarks: Optional InsightFace 5-point landmarks (5, 2) array

        Returns:
            Binary mask of face region (0-255), or None if parsing failed
        """
        if not self._ensure_model():
            logger.debug("[FACE_PARSER] Model not available")
            return None

        if bbox is None:
            logger.debug("[FACE_PARSER] No bbox provided")
            return None

        try:
            logger.debug(f"[FACE_PARSER] Parsing face at bbox {bbox}")
            import torch

            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox

            # ===== OPTIMIZATION: Crop bbox region with padding =====
            # This reduces input size by ~10-20x, dramatically improving speed
            padding_ratio = 0.3  # 30% padding around bbox
            face_w, face_h = x2 - x1, y2 - y1
            pad_w = int(face_w * padding_ratio)
            pad_h = int(face_h * padding_ratio)
            
            # Crop coordinates with padding (clamped to image bounds)
            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(w, x2 + pad_w)
            crop_y2 = min(h, y2 + pad_h)
            
            # Crop image
            cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_h, crop_w = cropped_image.shape[:2]
            logger.debug(f"[FACE_PARSER] Cropped: {w}x{h} → {crop_w}x{crop_h} (reduction: {(w*h)/(crop_w*crop_h):.1f}x)")
            
            # Convert cropped image BGR to RGB
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            # Convert to tensor (B, C, H, W) with uint8 values in [0, 255]
            # facer expects: b x 3 x h x w, torch.uint8, [0, 255]
            image_tensor = torch.from_numpy(cropped_rgb).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Adjust bbox coordinates to cropped image space
            bbox_in_crop = [x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1]
            rects = torch.tensor([bbox_in_crop], dtype=torch.float32, device=self.device)
            scores = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            image_ids = torch.tensor([0], dtype=torch.long, device=self.device)

            # Create 5-point landmarks
            # Priority: Use InsightFace landmarks if available, else approximate from bbox
            if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape == (5, 2):
                # Adjust landmarks to cropped image space
                landmarks_in_crop = landmarks.copy()
                landmarks_in_crop[:, 0] -= crop_x1  # Adjust x
                landmarks_in_crop[:, 1] -= crop_y1  # Adjust y
                points = torch.from_numpy(landmarks_in_crop).unsqueeze(0).float().to(self.device)
                logger.debug("[FACE_PARSER] Using InsightFace landmarks")
            else:
                # Fallback: approximate from bbox (in cropped space)
                logger.warning("[FACE_PARSER] No landmarks provided, using bbox approximation")
                bbox_cx = (bbox_in_crop[0] + bbox_in_crop[2]) / 2
                bbox_cy = (bbox_in_crop[1] + bbox_in_crop[3]) / 2
                bbox_w = bbox_in_crop[2] - bbox_in_crop[0]
                bbox_h = bbox_in_crop[3] - bbox_in_crop[1]
                points = torch.tensor([[
                    [bbox_in_crop[0] + bbox_w * 0.3, bbox_in_crop[1] + bbox_h * 0.35],  # left eye
                    [bbox_in_crop[0] + bbox_w * 0.7, bbox_in_crop[1] + bbox_h * 0.35],  # right eye
                    [bbox_cx, bbox_in_crop[1] + bbox_h * 0.55],                          # nose
                    [bbox_in_crop[0] + bbox_w * 0.35, bbox_in_crop[1] + bbox_h * 0.75], # left mouth
                    [bbox_in_crop[0] + bbox_w * 0.65, bbox_in_crop[1] + bbox_h * 0.75], # right mouth
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
            seg_logits = result["seg"]["logits"]  # Actual: (B, C, H, W)
            
            # Shape validation and logging
            logger.debug(f"[FACE_PARSER] seg_logits shape: {seg_logits.shape}")
            
            if seg_logits.dim() != 4:
                logger.error(f"[FACE_PARSER] Unexpected dim: {seg_logits.dim()}, expected 4")
                return None
            
            batch_size, num_classes, height, width = seg_logits.shape
            logger.debug(f"[FACE_PARSER] Parsed: B={batch_size}, C={num_classes}, H={height}, W={width}")

            # Get the first batch's segmentation (correct indexing for (B, C, H, W))
            face_seg = seg_logits[0]  # (C, H, W)
            face_classes = face_seg.argmax(dim=0).cpu().numpy()  # (H, W)

            # Log class distribution
            unique_classes, counts = np.unique(face_classes, return_counts=True)
            class_dist = dict(zip(unique_classes.tolist(), counts.tolist()))
            logger.debug(f"[FACE_PARSER] Class distribution: {class_dist}")
            
            # Create binary mask from face skin classes
            mask_crop = np.zeros(face_classes.shape, dtype=np.uint8)
            for class_id in FACE_SKIN_CLASSES:
                mask_crop[face_classes == class_id] = 255
            
            # Log mask coverage before resize
            mask_pixels = (mask_crop > 0).sum()
            mask_coverage = (mask_pixels / mask_crop.size) * 100
            logger.debug(f"[FACE_PARSER] Mask before resize: {mask_pixels} pixels ({mask_coverage:.2f}%)")

            # Resize mask to cropped image size if needed
            if mask_crop.shape[0] != crop_h or mask_crop.shape[1] != crop_w:
                mask_crop = cv2.resize(mask_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

            # ===== OPTIMIZATION: Restore mask to full image size =====
            # Create full-size mask and place cropped mask in correct position
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask_crop

            # Smooth edges for natural blending
            mask = cv2.GaussianBlur(mask, (15, 15), 0)

            # Fill holes in the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            logger.debug(f"[FACE_PARSER] Successfully created mask, non-zero pixels: {mask.sum() / 255:.0f}")
            return mask

        except Exception as e:
            logger.error(f"[FACE_PARSER] Face parsing failed: {e}", exc_info=True)
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
