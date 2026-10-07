"""Instance extraction from a single semantic probability map."""

from __future__ import annotations

import numpy as np


def filter_instance_labels(
    labels: np.ndarray,
    min_area: int,
    max_area: int | None,
    max_instances: int = 128,
) -> np.ndarray:
    """Filter and compact a label map without materializing full-size masks."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or np.any(labels < 0):
        raise ValueError("labels must be a non-negative 2D array")
    if min_area < 1 or max_instances < 1:
        raise ValueError("min_area and max_instances must be positive")
    areas = np.bincount(labels.ravel())
    valid = np.flatnonzero(areas >= min_area)
    valid = valid[valid != 0]
    if max_area is not None:
        valid = valid[areas[valid] <= max_area]
    if len(valid) > max_instances:
        order = np.argsort(areas[valid], kind="stable")
        valid = np.sort(valid[order[-max_instances:]])
    lookup = np.zeros(len(areas), dtype=np.int32)
    lookup[valid] = np.arange(1, len(valid) + 1, dtype=np.int32)
    return lookup[labels]


def labels_to_instances(labels: np.ndarray) -> list[np.ndarray]:
    """Compatibility adapter for callers that explicitly need dense masks."""
    labels = np.asarray(labels)
    return [labels == index for index in range(1, int(labels.max()) + 1)]


def probability_to_component_labels(
    probability: np.ndarray,
    threshold: float = 0.5,
    closing_iterations: int = 1,
) -> np.ndarray:
    """Threshold, optionally close, and label 8-connected foreground components."""
    from scipy import ndimage

    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("probability must be a 2D array")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")
    if isinstance(closing_iterations, bool) or int(closing_iterations) != closing_iterations:
        raise ValueError("closing_iterations must be a non-negative integer")
    closing_iterations = int(closing_iterations)
    if closing_iterations < 0:
        raise ValueError("closing_iterations must be a non-negative integer")
    foreground = probability >= threshold
    if closing_iterations:
        foreground = ndimage.binary_closing(
            foreground,
            iterations=closing_iterations,
        )
    labels, _ = ndimage.label(
        foreground, structure=np.ones((3, 3), dtype=np.uint8)
    )
    return np.asarray(labels, dtype=np.int32)


def probability_to_instance_labels(
    probability: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 80,
    *,
    closing_iterations: int = 1,
    max_area_fraction: float = 0.35,
    max_instances: int = 128,
) -> np.ndarray:
    """Convert probabilities into a bounded, compact component label map."""
    probability = np.asarray(probability, dtype=np.float32)
    component_labels = probability_to_component_labels(
        probability,
        threshold=threshold,
        closing_iterations=closing_iterations,
    )
    foreground = component_labels > 0
    max_area = int(foreground.size * max_area_fraction)
    return filter_instance_labels(
        component_labels, min_area, max_area, max_instances
    )


def probability_to_instances(
    probability: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 80,
    *,
    closing_iterations: int = 1,
    max_area_fraction: float = 0.35,
    max_instances: int = 128,
) -> list[np.ndarray]:
    """Compatibility wrapper returning independent dense instance masks."""
    labels = probability_to_instance_labels(
        probability,
        threshold=threshold,
        min_area=min_area,
        closing_iterations=closing_iterations,
        max_area_fraction=max_area_fraction,
        max_instances=max_instances,
    )
    return labels_to_instances(labels)
