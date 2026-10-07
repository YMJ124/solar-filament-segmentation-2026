"""Instance extraction and ensemble probability fusion."""

from __future__ import annotations

import numpy as np


def fuse_probabilities(probabilities: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one probability map is required")
    arrays = [np.asarray(x, dtype=np.float32) for x in probabilities]
    if any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("all probability maps must have the same shape")
    weights = weights or [1.0 / len(arrays)] * len(arrays)
    if len(weights) != len(arrays) or sum(weights) <= 0:
        raise ValueError("weights must match probabilities and have a positive sum")
    return np.clip(sum(w * x for w, x in zip(weights, arrays)) / sum(weights), 0, 1)


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
    boundary: np.ndarray | None = None,
    boundary_threshold: float = 0.5,
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
    if boundary is not None:
        boundary = np.asarray(boundary)
        if boundary.shape != probability.shape:
            raise ValueError("boundary must have the same shape as probability")
        foreground &= boundary < boundary_threshold
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
    boundary: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    boundary_threshold: float = 0.5,
    max_area_fraction: float = 0.35,
    use_watershed: bool = True,
    marker_threshold: float = 0.15,
    marker_min_distance: int = 8,
    max_markers: int = 256,
    max_instances: int = 128,
) -> np.ndarray:
    """Convert fused probabilities into a bounded, compact instance label map."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    probability = np.asarray(probability, dtype=np.float32)
    component_labels = probability_to_component_labels(
        probability,
        threshold=threshold,
        closing_iterations=closing_iterations,
        boundary=boundary,
        boundary_threshold=boundary_threshold,
    )
    foreground = component_labels > 0
    max_area = int(foreground.size * max_area_fraction)
    if use_watershed and distance is not None and foreground.any():
        distance = np.asarray(distance, dtype=np.float32) * foreground
        coordinates = peak_local_max(distance, min_distance=marker_min_distance,
                                     threshold_abs=marker_threshold, labels=foreground,
                                     exclude_border=False, num_peaks=max_markers)
        markers = np.zeros(foreground.shape, dtype=np.int32)
        for idx, (y, x) in enumerate(coordinates, start=1):
            markers[y, x] = idx
        if markers.max() > 0:
            labels = watershed(-distance, markers, mask=foreground)
            return filter_instance_labels(labels, min_area, max_area, max_instances)
    return filter_instance_labels(
        component_labels, min_area, max_area, max_instances
    )


def probability_to_instances(
    probability: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 80,
    *,
    closing_iterations: int = 1,
    boundary: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    boundary_threshold: float = 0.5,
    max_area_fraction: float = 0.35,
    use_watershed: bool = True,
    marker_threshold: float = 0.15,
    marker_min_distance: int = 8,
    max_markers: int = 256,
    max_instances: int = 128,
) -> list[np.ndarray]:
    """Compatibility wrapper returning independent dense instance masks."""
    labels = probability_to_instance_labels(
        probability,
        threshold=threshold,
        min_area=min_area,
        closing_iterations=closing_iterations,
        boundary=boundary,
        distance=distance,
        boundary_threshold=boundary_threshold,
        max_area_fraction=max_area_fraction,
        use_watershed=use_watershed,
        marker_threshold=marker_threshold,
        marker_min_distance=marker_min_distance,
        max_markers=max_markers,
        max_instances=max_instances,
    )
    return labels_to_instances(labels)
