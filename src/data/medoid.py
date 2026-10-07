"""Deterministic representative-annotator selection for training targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics.pq import pairwise_rle_iou

from .coco import CocoAnnotations
from .masks import annotations_to_rles


@dataclass(frozen=True)
class MedoidSelection:
    file_name: str
    selected_image_id: str
    selected_annotations: int
    medoid_mean_iou: float


def _semantic_rle(annotations: list[dict], height: int, width: int) -> dict:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pycocotools is required for medoid selection") from exc
    instances = annotations_to_rles(annotations, height, width)
    if instances:
        return mask_utils.merge(instances)
    return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))


def select_medoid_batch(coco: CocoAnnotations, file_name: str) -> MedoidSelection:
    """Choose the annotator batch with maximum mean semantic IoU to its peers."""
    batches = sorted(coco.batches_for_file(file_name), key=lambda row: str(row["id"]))
    if not batches:
        raise KeyError(f"unknown physical image: {file_name}")
    semantic_rles = []
    annotation_counts = []
    for batch in batches:
        annotations = coco.annotations_for_batch(str(batch["id"]))
        annotation_counts.append(len(annotations))
        semantic_rles.append(
            _semantic_rle(annotations, int(batch["height"]), int(batch["width"]))
        )
    if len(batches) == 1:
        means = np.array([1.0], dtype=np.float64)
    else:
        ious = pairwise_rle_iou(semantic_rles, semantic_rles)
        means = (ious.sum(axis=1) - np.diag(ious)) / (len(batches) - 1)
    selected_index = min(
        range(len(batches)), key=lambda idx: (-float(means[idx]), str(batches[idx]["id"]))
    )
    return MedoidSelection(
        file_name=file_name,
        selected_image_id=str(batches[selected_index]["id"]),
        selected_annotations=annotation_counts[selected_index],
        medoid_mean_iou=float(means[selected_index]),
    )
