"""
5-channel Mask2Former wrapper.
Fixes the bug in treefinder-main where only 3 channels were passed (x[:, :3]).
Replaces the Swin patch embedding Conv2d to accept in_channels input channels.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, AutoConfig

logger = logging.getLogger(__name__)


def _find_patch_embed_conv(model) -> tuple:
    """
    Locate the patch embedding Conv2d in the Swin backbone.
    Returns (parent_module, attribute_name, conv_module).
    Tries the standard path first; falls back to scanning the pixel_level_module.
    """
    # Standard path for facebook/mask2former-swin-tiny-ade-semantic
    try:
        ppm = model.model.pixel_level_module
        conv = ppm.encoder.embeddings.patch_embeddings.projection
        if isinstance(conv, nn.Conv2d):
            return ppm.encoder.embeddings.patch_embeddings, "projection", conv
    except AttributeError:
        pass

    # Fallback: scan for first Conv2d in pixel_level_module
    ppm = model.model.pixel_level_module
    for name, module in ppm.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            parts = name.split(".")
            parent = ppm
            for p in parts[:-1]:
                parent = getattr(parent, p)
            logger.warning(f"Patch embed found at non-standard path: pixel_level_module.{name}")
            return parent, parts[-1], module

    raise RuntimeError(
        "Could not find a 3-channel Conv2d in pixel_level_module. "
        "Print the model architecture and update _find_patch_embed_conv()."
    )


class Mask2FormerWrapper(nn.Module):
    def __init__(self, num_classes: int, image_size: int, in_channels: int,
                 pretrained_name: str):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        logger.info(f"Loading Mask2Former: {pretrained_name}")
        config = AutoConfig.from_pretrained(pretrained_name)
        config.num_labels = num_classes

        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # ------------------------------------------------------------------
        # Fix 1: Replace patch embedding to accept in_channels instead of 3
        # ------------------------------------------------------------------
        if in_channels != 3:
            parent, attr, old_conv = _find_patch_embed_conv(self.mask2former)
            old_weight = old_conv.weight.data  # (out, 3, kH, kW)
            new_conv = nn.Conv2d(
                in_channels, old_weight.shape[0],
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Copy RGB pretrained weights; init new channels (NIR, NDVI) with small noise
            new_conv.weight.data[:, :3] = old_weight
            nn.init.normal_(new_conv.weight.data[:, 3:], mean=0.0, std=0.02)
            if old_conv.bias is not None:
                new_conv.bias = nn.Parameter(old_conv.bias.data.clone())
            setattr(parent, attr, new_conv)
            logger.info(f"Replaced patch embedding Conv2d: 3 → {in_channels} input channels")

        # ------------------------------------------------------------------
        # Fix 2: Replace classifier head for num_classes
        # ------------------------------------------------------------------
        in_ch = self.mask2former.class_predictor.in_features
        self.mask2former.class_predictor = nn.Linear(in_ch, num_classes + 1)
        logger.info(f"Replaced class_predictor head: → {num_classes + 1} outputs")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)
        Returns: (B, num_classes, H, W) logits
        """
        outputs = self.mask2former(pixel_values=x)  # all channels

        class_logits = outputs.class_queries_logits[:, :, :-1]  # (B, Q, C)  drop "no-object"
        mask_logits  = outputs.masks_queries_logits              # (B, Q, H', W')

        class_probs = torch.softmax(class_logits, dim=-1)
        seg = torch.einsum("bqc,bqhw->bchw", class_probs, mask_logits)  # (B, C, H', W')

        return F.interpolate(
            seg, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )


def build_mask2former(cfg: dict) -> Mask2FormerWrapper:
    num_classes  = int(cfg.get("num_classes", 1))
    image_size   = int(cfg.get("image_size", 224))
    in_channels  = int(cfg.get("in_channels", 5))
    pretrained   = cfg["mask2former_pretrained"].get(
        "mask2former_weights", "facebook/mask2former-swin-tiny-ade-semantic"
    )
    return Mask2FormerWrapper(
        num_classes=num_classes,
        image_size=image_size,
        in_channels=in_channels,
        pretrained_name=pretrained,
    )
