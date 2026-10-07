"""Losses shared by the three segmentation models."""

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


def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.exp(-bce)
    return (alpha * (1 - p).pow(gamma) * bce).mean()


class GlobalLoss(nn.Module):
    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


class DetailLoss(nn.Module):
    def forward(self, logits, target):
        return dice_loss(logits, target) + focal_loss(logits, target)


class MultiTaskLoss(nn.Module):
    def forward(self, output, semantic, boundary, distance):
        semantic_loss = F.binary_cross_entropy_with_logits(output["semantic"], semantic) + dice_loss(output["semantic"], semantic)
        boundary_loss = focal_loss(output["boundary"], boundary)
        distance_loss = F.smooth_l1_loss(torch.sigmoid(output["distance"]), distance)
        return semantic_loss + 0.5 * boundary_loss + 0.5 * distance_loss
