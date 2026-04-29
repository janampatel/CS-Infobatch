"""
5-channel DeepLabV3+ wrapper (ResNet50 backbone).
Follows the treefinder baseline implementation of create_deeplabv3plus().
Replaces backbone conv1 to accept in_channels, replaces classifier head,
disables auxiliary classifier, and returns (B, num_classes, H, W) logits.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

logger = logging.getLogger(__name__)


class DeepLabV3Wrapper(nn.Module):
    def __init__(self, num_classes: int, image_size: int, in_channels: int,
                 pretrained: bool = True):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        logger.info(f"Loading DeepLabV3+ ResNet50 (pretrained={pretrained})")
        self.deeplabv3 = deeplabv3_resnet50(weights=weights, progress=False)

        # ------------------------------------------------------------------
        # Fix 1: Replace backbone conv1 to accept in_channels instead of 3
        # ------------------------------------------------------------------
        old_conv = self.deeplabv3.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    # Treefinder approach: replicate mean of RGB weights for new channels
                    mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:, 3:] = mean_rgb.expand(-1, in_channels - 3, -1, -1)
        self.deeplabv3.backbone.conv1 = new_conv
        logger.info(f"Replaced backbone conv1: 3 → {in_channels} input channels")

        # ------------------------------------------------------------------
        # Fix 2: Replace classifier head for num_classes output channels
        # ------------------------------------------------------------------
        old_cls = self.deeplabv3.classifier[-1]
        self.deeplabv3.classifier[-1] = nn.Conv2d(
            old_cls.in_channels,
            num_classes,
            kernel_size=old_cls.kernel_size,
            stride=1,
        )
        logger.info(f"Replaced classifier head: → {num_classes} output classes")

        # ------------------------------------------------------------------
        # Fix 3: Disable auxiliary classifier (not needed for inference/our loss)
        # ------------------------------------------------------------------
        self.deeplabv3.aux_classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        Returns: (B, num_classes, H, W) logits
        """
        out = self.deeplabv3(x)["out"]  # torchvision returns dict; 'out' is main head
        # Upsample to image_size in case input was not exactly image_size
        # (torchvision already upsamples to input resolution, but kept for safety)
        return F.interpolate(
            out, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )


def build_deeplabv3(cfg: dict) -> DeepLabV3Wrapper:
    num_classes = int(cfg.get("num_classes", 1))
    image_size  = int(cfg.get("image_size", 224))
    in_channels = int(cfg.get("in_channels", 5))
    pretrained  = bool(
        cfg.get("deeplabv3_pretrained", {}).get("pretrained", True)
    )
    return DeepLabV3Wrapper(
        num_classes=num_classes,
        image_size=image_size,
        in_channels=in_channels,
        pretrained=pretrained,
    )
