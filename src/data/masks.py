"""COCO polygon conversion utilities."""

from __future__ import annotations

import numpy as np


def annotations_to_rles(
    annotations: list[dict], height: int, width: int
) -> list[dict]:
    """Convert polygon annotations to merged COCO RLE objects without decoding."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for polygon encoding") from exc
    rles = []
    for annotation in annotations:
        polygons = annotation["segmentation"]
        polygon_rles = mask_utils.frPyObjects(polygons, height, width)
        rles.append(mask_utils.merge(polygon_rles))
    return rles


def annotations_to_masks(annotations: list[dict], height: int, width: int) -> list[np.ndarray]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for polygon decoding") from exc
    return [
        np.asarray(mask_utils.decode(rle), dtype=bool)
        for rle in annotations_to_rles(annotations, height, width)
    ]
