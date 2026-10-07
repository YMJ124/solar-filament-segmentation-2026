"""Submission-legal post-processing for overlapping instance masks."""

from __future__ import annotations

import numpy as np


def exclusive_label_map(
    masks: np.ndarray,
    confidences: np.ndarray,
    *,
    confidence_threshold: float,
    max_instances: int,
    min_area: int = 1,
    force_one: bool = False,
) -> np.ndarray:
    """Assign overlapping pixels to the highest-confidence retained instance."""
    masks = np.asarray(masks, dtype=bool)
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [instances, height, width]")
    if len(masks) != len(confidences):
        raise ValueError("mask and confidence counts differ")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be within [0, 1]")
    if max_instances < 1 or min_area < 1:
        raise ValueError("max_instances and min_area must be positive")
    labels = np.zeros(masks.shape[1:], dtype=np.int32)
    if not len(masks):
        return labels
    order = np.argsort(-confidences, kind="stable")
    retained = order[confidences[order] >= confidence_threshold]
    if not len(retained) and force_one:
        retained = order[:1]
    retained = retained[:max_instances]
    claimed = np.zeros(labels.shape, dtype=bool)
    next_label = 1
    for index in retained:
        available = masks[index] & ~claimed
        if int(available.sum()) < min_area:
            continue
        labels[available] = next_label
        claimed |= available
        next_label += 1
    return labels
