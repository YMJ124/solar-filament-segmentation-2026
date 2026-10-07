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
    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)
