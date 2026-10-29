import torch
import torch.nn as nn


def per_sample_bce(raw_loss: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean valid-pixel BCE loss per sample. raw_loss/valid_mask: (B,1,H,W) -> (B,)."""
    B = raw_loss.shape[0]
    vpx = valid_mask.view(B, -1).sum(1).clamp(min=1)
    return (raw_loss * valid_mask).view(B, -1).sum(1) / vpx


def per_sample_cs_score(raw_loss: torch.Tensor, labels: torch.Tensor,
                         valid_mask: torch.Tensor,
                         w_fg: float, w_bg: float) -> torch.Tensor:
    """Class-balanced score H(z_i) = w_fg*L_fg(z_i)*has_fg(z_i) + w_bg*L_bg(z_i). -> (B,)."""
    B = raw_loss.shape[0]
    fg_mask = (labels > 0.5).float() * valid_mask
    bg_mask = (labels <= 0.5).float() * valid_mask

    n_fg = fg_mask.view(B, -1).sum(1).clamp(min=1)
    n_bg = bg_mask.view(B, -1).sum(1).clamp(min=1)

    L_fg = (raw_loss * fg_mask).view(B, -1).sum(1) / n_fg
    L_bg = (raw_loss * bg_mask).view(B, -1).sum(1) / n_bg

    has_fg = (fg_mask.view(B, -1).sum(1) > 0).float()
    return w_fg * L_fg * has_fg + w_bg * L_bg


class DiceLoss(nn.Module):
    """Batch-level Dice loss. Used only for total_loss gradient, never for IB scoring."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        if valid_mask is not None and valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=False)

        probs = torch.sigmoid(logits)
        if valid_mask is not None:
            probs = probs * valid_mask

        p = probs.view(-1)
        t = targets.float().view(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        return 1.0 - (inter + 1e-6) / (union + 1e-6)
