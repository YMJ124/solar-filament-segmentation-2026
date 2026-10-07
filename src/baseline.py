"""Fast grayscale image-processing baseline for a valid first submission."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .postprocess.instances import filter_instance_labels, labels_to_instances


def predict_instance_labels(
    image_path: str | Path,
    threshold_percentile: float = 8.0,
    min_component_area: int = 80,
    max_component_area_fraction: float = 0.35,
    morphology_kernel: int = 3,
    close_iterations: int = 1,
    open_iterations: int = 1,
    max_instances: int = 128,
) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    threshold = np.percentile(enhanced, threshold_percentile)
    binary = (enhanced <= threshold).astype(np.uint8)
    kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
    if close_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    if open_iterations:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    max_area = image.shape[0] * image.shape[1] * max_component_area_fraction
    return filter_instance_labels(
        labels,
        min_component_area,
        int(max_area),
        max_instances=max_instances,
    )


def predict_instances(
    image_path: str | Path,
    threshold_percentile: float = 8.0,
    min_component_area: int = 80,
    max_component_area_fraction: float = 0.35,
    morphology_kernel: int = 3,
    close_iterations: int = 1,
    open_iterations: int = 1,
    max_instances: int = 128,
) -> list[np.ndarray]:
    """Compatibility wrapper returning dense masks for each baseline instance."""
    labels = predict_instance_labels(
        image_path,
        threshold_percentile,
        min_component_area,
        max_component_area_fraction,
        morphology_kernel,
        close_iterations,
        open_iterations,
        max_instances,
    )
    return labels_to_instances(labels)
