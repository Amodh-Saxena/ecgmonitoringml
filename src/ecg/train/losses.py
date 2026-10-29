"""Losses supporting unknown (masked) labels."""
import numpy as np
import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                           pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def masked_focal_with_logits(logits, targets, mask, gamma: float = 2.0, pos_weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    p_t = torch.exp(-F.binary_cross_entropy_with_logits(logits, targets, reduction="none"))
    return (((1 - p_t) ** gamma * bce) * mask).sum() / mask.sum().clamp_min(1.0)


def class_pos_weights(targets: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """neg/pos ratio per class over known labels (clipped to keep gradients bounded)."""
    pos = (targets * mask).sum(0)
    neg = ((1 - targets) * mask).sum(0)
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 20.0), dtype=torch.float32)
