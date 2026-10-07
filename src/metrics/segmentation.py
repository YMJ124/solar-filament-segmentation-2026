"""Pixel-level binary segmentation metrics."""

from __future__ import annotations

import numpy as np


def segmentation_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    return {
        "dice": 2 * tp / dice_den if dice_den else 1.0,
        "iou": tp / iou_den if iou_den else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float(truth.sum() == 0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn,
    }
