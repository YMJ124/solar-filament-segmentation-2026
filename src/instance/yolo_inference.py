"""Small adapters from Ultralytics results to repository-native arrays."""

from __future__ import annotations

import cv2
import numpy as np


def prediction_arrays(result, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return thresholded instance masks and confidences at ``shape``."""
    if result.masks is None or result.boxes is None or not len(result.boxes):
        return np.zeros((0, *shape), dtype=bool), np.zeros(0, dtype=np.float32)
    masks = result.masks.data.detach().cpu().numpy().astype(np.float32, copy=False)
    if masks.shape[1:] != shape:
        masks = np.stack(
            [
                cv2.resize(
                    mask,
                    (shape[1], shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for mask in masks
            ]
        )
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    return masks >= 0.5, confidences
