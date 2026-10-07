"""Losses for the single semantic segmentation model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1.0):
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1 - ((2 * intersection + smooth) / (denominator + smooth))).mean()


class GlobalLoss(nn.Module):
    def __init__(self, mode: str = "bce_dice", max_positive_weight: float = 20.0):
        super().__init__()
        if mode not in {"bce_dice", "balanced_bce_dice"}:
            raise ValueError("loss mode must be bce_dice or balanced_bce_dice")
        if max_positive_weight < 1:
            raise ValueError("max_positive_weight must be >= 1")
        self.mode = mode
        self.max_positive_weight = float(max_positive_weight)

    def forward(self, logits, target):
        positive_weight = None
        if self.mode == "balanced_bce_dice":
            positives = target.sum().clamp_min(1.0)
            negatives = (target.numel() - target.sum()).clamp_min(1.0)
            positive_weight = torch.sqrt(negatives / positives).clamp(
                1.0, self.max_positive_weight
            )
        return (
            F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight
            )
            + dice_loss(logits, target)
        )
