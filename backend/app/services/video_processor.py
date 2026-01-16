import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# MediaPipe face mesh contour indices for face outline
# These indices form the outer boundary of the face
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


class BlurType(str, Enum):
    GAUSSIAN = "gaussian"
    MOSAIC = "mosaic"
    BLACKOUT = "blackout"


class VideoProcessor:
    def __init__(self, use_segmentation: bool = True):
        self.use_segmentation = use_segmentation
        self._face_mesh = None

    def _get_face_mesh(self):
        """Lazy load MediaPipe Face Mesh."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,  # Better for individual frames
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=0.1,  # Very low for aggressive detection
                    min_tracking_confidence=0.1
                )
                logger.info("[SEGMENTATION] MediaPipe Face Mesh loaded with confidence=0.1")
            except ImportError as e:
                logger.warning(f"[SEGMENTATION] MediaPipe not available: {e}, using bbox fallback")
                self.use_segmentation = False
            except Exception as e:
                logger.error(f"[SEGMENTATION] Error loading MediaPipe: {e}", exc_info=True)
                self.use_segmentation = False
        return self._face_mesh

    def _get_face_contour_mask(
        self,
        frame: np.ndarray,
        bbox: List[int]
    ) -> Optional[np.ndarray]:
        """
        Get face contour mask using MediaPipe Face Mesh.

        Args:
            frame: Input frame (BGR)
            bbox: Face bounding box [x1, y1, x2, y2]

        Returns:
            Binary mask of face contour, or None if detection failed
        """
        if not self.use_segmentation:
            logger.debug("[SEGMENTATION] Segmentation disabled")
            return None

        face_mesh = self._get_face_mesh()
        if face_mesh is None:
            logger.debug("[SEGMENTATION] Face mesh not available")
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Expand bbox for better face mesh detection
        pad_w = int((x2 - x1) * 0.5)  # More padding for better detection
        pad_h = int((y2 - y1) * 0.5)
        ex1 = max(0, x1 - pad_w)
        ey1 = max(0, y1 - pad_h)
        ex2 = min(w, x2 + pad_w)
        ey2 = min(h, y2 + pad_h)

        # Extract ROI and convert to RGB
        roi = frame[ey1:ey2, ex1:ex2]
        if roi.size == 0:
            logger.debug("[SEGMENTATION] ROI is empty")
            return None

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(roi_rgb)

        if not results.multi_face_landmarks:
            logger.debug(f"[SEGMENTATION] No face landmarks found in ROI {roi.shape}")
            return None

        logger.debug(f"[SEGMENTATION] Found {len(results.multi_face_landmarks)} face(s) in ROI")

        # Find the face closest to center of bbox
        roi_h, roi_w = roi.shape[:2]
        bbox_center = ((x1 + x2) / 2 - ex1, (y1 + y2) / 2 - ey1)

        best_landmarks = None
        min_dist = float('inf')

        for face_landmarks in results.multi_face_landmarks:
            # Calculate face center from landmarks
            nose_tip = face_landmarks.landmark[1]
            face_center = (nose_tip.x * roi_w, nose_tip.y * roi_h)
            dist = ((face_center[0] - bbox_center[0])**2 +
                    (face_center[1] - bbox_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_landmarks = face_landmarks

        if best_landmarks is None:
            return None

        # Extract face oval points
        points = []
        for idx in FACE_OVAL_INDICES:
            lm = best_landmarks.landmark[idx]
            px = int(lm.x * roi_w) + ex1
            py = int(lm.y * roi_h) + ey1
            points.append([px, py])

        # Create mask from convex hull
        mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        cv2.fillConvexPoly(mask, hull, 255)

        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask

    def apply_blur(
        self,
        frame: np.ndarray,
        bbox: List[int],
        blur_type: BlurType,
        intensity: int = 25,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply blur effect to a region of the frame.

        Args:
            frame: Input frame (BGR)
            bbox: Bounding box [x1, y1, x2, y2]
            blur_type: Type of blur to apply
            intensity: Blur intensity (1-50)
            mask: Optional face contour mask for precise blur

        Returns:
            Frame with blur applied
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Ensure bbox is within frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # Try to get face contour mask if segmentation is enabled
        if mask is None and self.use_segmentation:
            mask = self._get_face_contour_mask(frame, bbox)
            if mask is not None:
                logger.info(f"[BLUR] Using face contour mask for bbox {bbox}")
            else:
                logger.debug(f"[BLUR] Segmentation failed, using rectangle for bbox {bbox}")

        # If we have a mask, use it for precise blur
        if mask is not None:
            return self._apply_blur_with_mask(frame, mask, blur_type, intensity)

        # Fallback to rectangular blur
        logger.debug(f"[BLUR] Applying rectangular blur for bbox {bbox}")
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return frame

        if blur_type == BlurType.GAUSSIAN:
            kernel_size = intensity * 2 + 1
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

        elif blur_type == BlurType.MOSAIC:
            roi_h, roi_w = roi.shape[:2]
            scale = max(1, intensity // 3)
            small_w = max(1, roi_w // scale)
            small_h = max(1, roi_h // scale)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            blurred_roi = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        elif blur_type == BlurType.BLACKOUT:
            blurred_roi = np.zeros_like(roi)

        else:
            return frame

        frame[y1:y2, x1:x2] = blurred_roi
        return frame

    def _apply_blur_with_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        blur_type: BlurType,
        intensity: int
    ) -> np.ndarray:
        """
        Apply blur using a face contour mask for precise edges.

        Args:
            frame: Input frame (BGR)
            mask: Binary mask (255 = blur area)
            blur_type: Type of blur
            intensity: Blur intensity

        Returns:
            Frame with masked blur applied
        """
        # Create blurred version of entire frame
        if blur_type == BlurType.GAUSSIAN:
            kernel_size = intensity * 2 + 1
            blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        elif blur_type == BlurType.MOSAIC:
            h, w = frame.shape[:2]
            scale = max(1, intensity // 3)
            small_w = max(1, w // scale)
            small_h = max(1, h // scale)
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif blur_type == BlurType.BLACKOUT:
            blurred = np.zeros_like(frame)

        else:
            return frame

        # Normalize mask to 0-1 range for blending
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)

        # Blend original and blurred using mask
        result = (frame * (1 - mask_3ch) + blurred * mask_3ch).astype(np.uint8)
        return result

    def interpolate_bbox(
        self,
        bbox1: List[int],
        bbox2: List[int],
        t: float
    ) -> List[int]:
        """
        Interpolate between two bounding boxes.

        Args:
            bbox1: First bbox
            bbox2: Second bbox
            t: Interpolation factor (0-1)

        Returns:
            Interpolated bbox
        """
        return [
            int(bbox1[i] + (bbox2[i] - bbox1[i]) * t)
            for i in range(4)
        ]

    def get_bbox_at_time(
        self,
        appearances: List[Dict],
        timestamp: float,
        fps: float
    ) -> Optional[List[int]]:
        """
        Get interpolated bounding box at a specific timestamp.

        Args:
            appearances: List of appearance ranges with bbox
            timestamp: Target timestamp
            fps: Video FPS for frame-level precision

        Returns:
            Interpolated bbox or None if not visible
        """
        for i, app in enumerate(appearances):
            if app["start"] <= timestamp <= app["end"]:
                # Face is visible at this time
                # For simplicity, return the stored bbox
                # In production, you'd interpolate between keyframes
                return app["bbox"]

        return None

    def process_video(
        self,
        input_path: str,
        output_path: str,
        blur_targets: List[Dict],
        blur_type: BlurType,
        intensity: int,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict:
        """
        Process entire video and apply blur to specified faces.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            blur_targets: List of faces to blur with their appearances
            blur_type: Type of blur effect
            intensity: Blur intensity
            progress_callback: Optional progress callback

        Returns:
            Processing result dict
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Write to temp file first (OpenCV doesn't handle audio)
        temp_video_path = str(output_path) + ".temp_video.mp4"

        # Setup output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        blurs_applied = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_idx / fps

                # Apply blur for each target face
                for target in blur_targets:
                    bbox = self.get_bbox_at_time(
                        target["appearances"],
                        current_time,
                        fps
                    )

                    if bbox is not None:
                        frame = self.apply_blur(frame, bbox, blur_type, intensity)
                        blurs_applied += 1

                out.write(frame)
                frame_idx += 1

                if progress_callback and frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    progress_callback(progress)

        finally:
            cap.release()
            out.release()

        # Finalize: merge processed video with original audio + compress
        logger.info(f"[VIDEO] Frame processing complete. Finalizing with audio...")
        self._finalize_with_audio(input_path, temp_video_path, output_path)

        return {
            "frames_processed": frame_idx,
            "blurs_applied": blurs_applied,
            "output_path": output_path
        }

    def _finalize_with_audio(
        self,
        original_path: str,
        processed_video_path: str,
        output_path: str
    ):
        """
        Merge processed video with original audio and compress with H.264.

        Args:
            original_path: Original video with audio
            processed_video_path: Processed video (no audio)
            output_path: Final output path
        """
        import subprocess
        import shutil

        logger.info(f"[FFMPEG] Merging audio and compressing...")

        try:
            # Check if ffmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("[FFMPEG] ffmpeg not available, using raw output")
                shutil.move(processed_video_path, output_path)
                return

            # Merge processed video + original audio with H.264 compression
            ffmpeg_result = subprocess.run([
                "ffmpeg", "-y",
                "-i", processed_video_path,  # Processed video (no audio)
                "-i", original_path,          # Original video (for audio)
                "-c:v", "libx264",            # H.264 video codec
                "-preset", "medium",          # Balance speed/quality
                "-crf", "23",                 # Quality (18-28, lower=better)
                "-c:a", "aac",                # AAC audio codec
                "-b:a", "192k",               # Audio bitrate
                "-map", "0:v:0",              # Use video from first input
                "-map", "1:a:0?",             # Use audio from second input (optional)
                "-movflags", "+faststart",    # Web optimization
                "-shortest",                  # End when shortest stream ends
                output_path
            ], capture_output=True, text=True, timeout=1800)  # 30 min timeout

            if ffmpeg_result.returncode != 0:
                logger.error(f"[FFMPEG] Failed: {ffmpeg_result.stderr}")
                # Fallback: just move the processed video
                shutil.move(processed_video_path, output_path)
            else:
                logger.info("[FFMPEG] Video finalized with audio successfully")
                # Clean up temp file
                if Path(processed_video_path).exists():
                    Path(processed_video_path).unlink()

        except subprocess.TimeoutExpired:
            logger.error("[FFMPEG] Timeout during finalization")
            shutil.move(processed_video_path, output_path)
        except FileNotFoundError:
            logger.warning("[FFMPEG] ffmpeg not found, using raw output")
            shutil.move(processed_video_path, output_path)
        except Exception as e:
            logger.error(f"[FFMPEG] Error: {e}", exc_info=True)
            if Path(processed_video_path).exists():
                shutil.move(processed_video_path, output_path)

    def _optimize_output(self, video_path: str):
        """
        Optimize output video for better compression.
        Requires ffmpeg to be installed.
        (Deprecated - use _finalize_with_audio instead)
        """
        import subprocess
        import shutil

        video_path_str = str(video_path)
        temp_path = video_path_str + ".temp.mp4"

        logger.info(f"[FFMPEG] Starting optimization for: {video_path_str}")

        try:
            # Check if ffmpeg is available
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("[FFMPEG] ffmpeg not available, skipping optimization")
                return

            logger.info("[FFMPEG] Running ffmpeg re-encode...")
            # Re-encode with H.264 for better compatibility
            ffmpeg_result = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path_str,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                temp_path
            ], capture_output=True, text=True, timeout=600)

            if ffmpeg_result.returncode != 0:
                logger.error(f"[FFMPEG] ffmpeg failed: {ffmpeg_result.stderr}")
                return

            # Replace original with optimized
            if Path(temp_path).exists():
                shutil.move(temp_path, video_path_str)
                logger.info("[FFMPEG] Video optimized successfully")
            else:
                logger.error("[FFMPEG] Temp file not created")

        except subprocess.TimeoutExpired:
            logger.error("[FFMPEG] ffmpeg timed out")
            if Path(temp_path).exists():
                Path(temp_path).unlink()
        except FileNotFoundError:
            logger.warning("[FFMPEG] ffmpeg not found, skipping optimization")
        except Exception as e:
            logger.error(f"[FFMPEG] Error optimizing video: {e}", exc_info=True)
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def generate_thumbnail(
        self,
        frame: np.ndarray,
        bbox: List[int],
        size: int = 128
    ) -> np.ndarray:
        """
        Generate a square thumbnail of a face.

        Args:
            frame: Source frame
            bbox: Face bounding box
            size: Output thumbnail size

        Returns:
            Square thumbnail image
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Add padding for context
        pad = int((x2 - x1) * 0.2)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = frame[y1:y2, x1:x2]

        # Make square
        crop_h, crop_w = face_crop.shape[:2]
        if crop_h != crop_w:
            max_dim = max(crop_h, crop_w)
            square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            y_offset = (max_dim - crop_h) // 2
            x_offset = (max_dim - crop_w) // 2
            square[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = face_crop
            face_crop = square

        # Resize to target size
        thumbnail = cv2.resize(face_crop, (size, size))
        return thumbnail
