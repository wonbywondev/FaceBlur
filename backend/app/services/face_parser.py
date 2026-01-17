"""
BiSeNet Face Parsing Service

Provides pixel-level face segmentation for precise blur masks.
Uses a pretrained BiSeNet model for 19-class face parsing.

Classes:
    0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
    6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: ear_r, 10: nose,
    11: mouth, 12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth,
    17: hair, 18: hat
"""
import os
import logging
import numpy as np
import cv2
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Face classes to include in blur mask (skin + facial features, excluding hair/neck/cloth)
FACE_CLASSES = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]  # skin, brows, eyes, glasses, nose, mouth, lips

# Model download URL (CelebAMask-HQ pretrained BiSeNet)
MODEL_URL = "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "bisenet_face_parsing.pth"


class FaceParser:
    """
    Face parsing using BiSeNet for precise face segmentation.
    Falls back to ellipse mask if model loading fails.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize face parser.

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
        """
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self._initialized = False
        self._init_error = None

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
        """Download model if not exists and load it."""
        if self._initialized:
            return self.model is not None

        self._initialized = True

        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms

            # Create model directory
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # Download model if needed
            if not MODEL_PATH.exists():
                logger.info(f"Downloading BiSeNet face parsing model to {MODEL_PATH}...")
                self._download_model()

            if not MODEL_PATH.exists():
                logger.warning("BiSeNet model not available, using ellipse fallback")
                return False

            # Build BiSeNet model
            self.model = self._build_bisenet()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            # Setup transform
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            logger.info(f"BiSeNet face parser loaded successfully on {self.device}")
            return True

        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Failed to load BiSeNet model: {e}")
            return False

    def _download_model(self):
        """Download the pretrained model."""
        try:
            import urllib.request
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("BiSeNet model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download BiSeNet model: {e}")

    def _build_bisenet(self):
        """Build BiSeNet architecture."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class ConvBNReLU(nn.Module):
            def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)

            def forward(self, x):
                return F.relu(self.bn(self.conv(x)))

        class AttentionRefinementModule(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
                self.conv_atten = nn.Conv2d(out_ch, out_ch, 1, bias=False)
                self.bn_atten = nn.BatchNorm2d(out_ch)

            def forward(self, x):
                feat = self.conv(x)
                atten = F.adaptive_avg_pool2d(feat, 1)
                atten = self.conv_atten(atten)
                atten = self.bn_atten(atten)
                atten = torch.sigmoid(atten)
                return feat * atten

        class ContextPath(nn.Module):
            def __init__(self):
                super().__init__()
                from torchvision.models import resnet18
                resnet = resnet18(weights=None)
                self.conv1 = resnet.conv1
                self.bn1 = resnet.bn1
                self.relu = resnet.relu
                self.maxpool = resnet.maxpool
                self.layer1 = resnet.layer1
                self.layer2 = resnet.layer2
                self.layer3 = resnet.layer3
                self.layer4 = resnet.layer4
                self.arm16 = AttentionRefinementModule(256, 128)
                self.arm32 = AttentionRefinementModule(512, 128)
                self.conv_head32 = ConvBNReLU(128, 128, 3, 1, 1)
                self.conv_head16 = ConvBNReLU(128, 128, 3, 1, 1)
                self.conv_avg = ConvBNReLU(512, 128, 1, 1, 0)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                feat4 = self.layer1(x)
                feat8 = self.layer2(feat4)
                feat16 = self.layer3(feat8)
                feat32 = self.layer4(feat16)

                avg = F.adaptive_avg_pool2d(feat32, 1)
                avg = self.conv_avg(avg)
                avg_up = F.interpolate(avg, size=feat32.shape[2:], mode='nearest')

                feat32_arm = self.arm32(feat32)
                feat32_sum = feat32_arm + avg_up
                feat32_up = F.interpolate(feat32_sum, size=feat16.shape[2:], mode='nearest')
                feat32_up = self.conv_head32(feat32_up)

                feat16_arm = self.arm16(feat16)
                feat16_sum = feat16_arm + feat32_up
                feat16_up = F.interpolate(feat16_sum, size=feat8.shape[2:], mode='nearest')
                feat16_up = self.conv_head16(feat16_up)

                return feat8, feat16_up, feat32_up

        class SpatialPath(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = ConvBNReLU(3, 64, 7, 2, 3)
                self.conv2 = ConvBNReLU(64, 64, 3, 2, 1)
                self.conv3 = ConvBNReLU(64, 64, 3, 2, 1)
                self.conv_out = ConvBNReLU(64, 128, 1, 1, 0)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                return self.conv_out(x)

        class FeatureFusionModule(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.convblk = ConvBNReLU(in_ch, out_ch, 1, 1, 0)
                self.conv1 = nn.Conv2d(out_ch, out_ch // 4, 1, bias=False)
                self.conv2 = nn.Conv2d(out_ch // 4, out_ch, 1, bias=False)

            def forward(self, fsp, fcp):
                feat = torch.cat([fsp, fcp], dim=1)
                feat = self.convblk(feat)
                atten = F.adaptive_avg_pool2d(feat, 1)
                atten = F.relu(self.conv1(atten))
                atten = torch.sigmoid(self.conv2(atten))
                return feat + feat * atten

        class BiSeNet(nn.Module):
            def __init__(self, n_classes=19):
                super().__init__()
                self.cp = ContextPath()
                self.sp = SpatialPath()
                self.ffm = FeatureFusionModule(256, 256)
                self.conv_out = nn.Sequential(
                    ConvBNReLU(256, 256, 3, 1, 1),
                    nn.Conv2d(256, n_classes, 1, bias=False)
                )
                self.conv_out16 = nn.Sequential(
                    ConvBNReLU(128, 64, 3, 1, 1),
                    nn.Conv2d(64, n_classes, 1, bias=False)
                )
                self.conv_out32 = nn.Sequential(
                    ConvBNReLU(128, 64, 3, 1, 1),
                    nn.Conv2d(64, n_classes, 1, bias=False)
                )

            def forward(self, x):
                feat_sp = self.sp(x)
                feat8, feat16, feat32 = self.cp(x)
                feat_fuse = self.ffm(feat_sp, feat8)
                feat_out = self.conv_out(feat_fuse)
                feat_out = F.interpolate(feat_out, size=x.shape[2:], mode='bilinear', align_corners=True)
                return feat_out

        return BiSeNet(n_classes=19)

    def parse_face(
        self,
        image: np.ndarray,
        bbox: Optional[List[int]] = None,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Optional[np.ndarray]:
        """
        Parse face and return segmentation mask.

        Args:
            image: Input image (BGR)
            bbox: Optional bounding box [x1, y1, x2, y2] to crop face region
            target_size: Size to resize image for model input

        Returns:
            Binary mask of face region (0-255), or None if parsing failed
        """
        if not self._ensure_model():
            return None

        try:
            import torch

            h, w = image.shape[:2]

            # Crop to bbox if provided
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Add padding around face
                pad = int(max(x2 - x1, y2 - y1) * 0.3)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                face_crop = image[y1:y2, x1:x2]
                crop_bbox = (x1, y1, x2, y2)
            else:
                face_crop = image
                crop_bbox = (0, 0, w, h)

            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            # Resize for model
            crop_h, crop_w = face_crop.shape[:2]
            face_resized = cv2.resize(face_rgb, target_size)

            # Transform and run model
            input_tensor = self.transform(face_resized).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                parsing = output.squeeze(0).argmax(0).cpu().numpy()

            # Create binary mask from face classes
            mask = np.zeros(parsing.shape, dtype=np.uint8)
            for class_id in FACE_CLASSES:
                mask[parsing == class_id] = 255

            # Resize mask back to crop size
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

            # Create full-frame mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = crop_bbox
            full_mask[y1:y2, x1:x2] = mask

            # Smooth edges for natural blending
            full_mask = cv2.GaussianBlur(full_mask, (15, 15), 0)

            # Fill holes in the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)

            return full_mask

        except Exception as e:
            logger.error(f"Face parsing failed: {e}")
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
