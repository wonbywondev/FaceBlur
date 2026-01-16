import cv2
import numpy as np
from typing import List, Dict, Optional, Callable
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BlurType(str, Enum):
    GAUSSIAN = "gaussian"
    MOSAIC = "mosaic"
    BLACKOUT = "blackout"


class VideoProcessor:
    def __init__(self):
        pass

    def apply_blur(
        self,
        frame: np.ndarray,
        bbox: List[int],
        blur_type: BlurType,
        intensity: int = 25
    ) -> np.ndarray:
        """
        Apply blur effect to a region of the frame.

        Args:
            frame: Input frame (BGR)
            bbox: Bounding box [x1, y1, x2, y2]
            blur_type: Type of blur to apply
            intensity: Blur intensity (1-50)

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

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return frame

        if blur_type == BlurType.GAUSSIAN:
            # Kernel size must be odd
            kernel_size = intensity * 2 + 1
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

        elif blur_type == BlurType.MOSAIC:
            # Mosaic: downscale then upscale
            roi_h, roi_w = roi.shape[:2]
            scale = max(1, intensity // 3)

            # Downscale
            small_w = max(1, roi_w // scale)
            small_h = max(1, roi_h // scale)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

            # Upscale with nearest neighbor for pixelated effect
            blurred_roi = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        elif blur_type == BlurType.BLACKOUT:
            blurred_roi = np.zeros_like(roi)

        else:
            return frame

        frame[y1:y2, x1:x2] = blurred_roi
        return frame

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
