"""Reversible transform/model consensus primitives for P8 submissions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.metrics.pq import pairwise_rle_iou
from src.submission.rle import decode_rle, label_map_to_rles


TTA_NAMES = ("identity", "hflip", "vflip", "rot180")


def apply_tta(image: np.ndarray, name: str) -> np.ndarray:
    """Apply one shape-preserving flip transform to an image."""
    image = np.asarray(image)
    if name == "identity":
        transformed = image
    elif name == "hflip":
        transformed = np.flip(image, axis=1)
    elif name == "vflip":
        transformed = np.flip(image, axis=0)
    elif name == "rot180":
        transformed = np.flip(image, axis=(0, 1))
    else:
        raise ValueError(f"unsupported TTA transform: {name}")
    return np.ascontiguousarray(transformed)


def invert_tta_masks(masks: np.ndarray, name: str) -> np.ndarray:
    """Map `[N,H,W]` masks from transform coordinates to the source image."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("TTA masks must have shape [N,H,W]")
    if name == "identity":
        restored = masks
    elif name == "hflip":
        restored = np.flip(masks, axis=2)
    elif name == "vflip":
        restored = np.flip(masks, axis=1)
    elif name == "rot180":
        restored = np.flip(masks, axis=(1, 2))
    else:
        raise ValueError(f"unsupported TTA transform: {name}")
    return np.ascontiguousarray(restored)


def cluster_instance_records(
    records: Sequence[Mapping[str, Any]],
    *,
    height: int,
    width: int,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Greedily cluster RLE masks around fixed high-confidence anchors.

    Anchors never change, which avoids single-link chains joining two nearby
    filaments. Fold support is counted later and is independent of TTA count.
    """
    if not 0.0 < iou_threshold < 1.0:
        raise ValueError("iou_threshold must lie strictly between zero and one")
    normalized = [dict(record) for record in records]
    if not normalized:
        return []
    for record in normalized:
        for key in ("segmentation_rle", "score", "area", "fold", "transform"):
            if key not in record:
                raise ValueError(f"instance record is missing {key}")
    order = sorted(
        range(len(normalized)),
        key=lambda index: (
            -float(normalized[index]["score"]),
            -int(normalized[index]["area"]),
            int(normalized[index]["fold"]),
            str(normalized[index]["transform"]),
        ),
    )
    from src.submission.rle import rle_from_counts

    rles = [
        rle_from_counts(str(record["segmentation_rle"]), height, width)
        for record in normalized
    ]
    ious = pairwise_rle_iou(rles, rles)
    clusters: list[dict[str, Any]] = []
    for index in order:
        best_cluster = None
        best_iou = iou_threshold
        for cluster_index, cluster in enumerate(clusters):
            overlap = float(ious[int(cluster["anchor_index"]), index])
            if overlap > best_iou:
                best_iou = overlap
                best_cluster = cluster_index
        if best_cluster is None:
            clusters.append({"anchor_index": index, "member_indices": [index]})
        else:
            clusters[best_cluster]["member_indices"].append(index)
    return [
        {
            "anchor_index": int(cluster["anchor_index"]),
            "member_indices": [int(index) for index in cluster["member_indices"]],
            "anchor_score": float(normalized[int(cluster["anchor_index"])]["score"]),
        }
        for cluster in clusters
    ]


def select_consensus_records(
    records: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    *,
    minimum_fold_support: int,
    representative: str = "max_area",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one auditable representative for clusters supported by folds."""
    if minimum_fold_support < 1:
        raise ValueError("minimum_fold_support must be positive")
    if representative not in {"max_area", "anchor"}:
        raise ValueError("representative must be max_area or anchor")
    normalized = [dict(record) for record in records]
    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for cluster_id, cluster in enumerate(clusters):
        indices = [int(index) for index in cluster["member_indices"]]
        members = [normalized[index] for index in indices]
        best_by_fold: dict[int, dict[str, Any]] = {}
        for member in members:
            fold = int(member["fold"])
            current = best_by_fold.get(fold)
            if current is None or float(member["score"]) > float(current["score"]):
                best_by_fold[fold] = member
        folds = sorted(best_by_fold)
        fold_support = len(folds)
        if representative == "anchor":
            representative_index = int(cluster["anchor_index"])
        else:
            representative_index = max(
                indices,
                key=lambda index: (
                    int(normalized[index]["area"]),
                    float(normalized[index]["score"]),
                ),
            )
        representative_record = dict(normalized[representative_index])
        cluster_score = float(
            np.mean([float(best_by_fold[fold]["score"]) for fold in folds])
        )
        accepted = fold_support >= minimum_fold_support
        row = {
            "cluster_id": int(cluster_id),
            "accepted": bool(accepted),
            "fold_support": int(fold_support),
            "supporting_folds": folds,
            "member_count": len(members),
            "transforms": sorted({str(member["transform"]) for member in members}),
            "cluster_score": cluster_score,
            "representative_index": int(representative_index),
            "representative_area": int(representative_record["area"]),
        }
        audit.append(row)
        if accepted:
            representative_record.update(row)
            selected.append(representative_record)
    selected.sort(
        key=lambda row: (
            -int(row["fold_support"]),
            -float(row["cluster_score"]),
            -int(row["area"]),
        )
    )
    return selected, audit


def compose_exclusive_consensus(
    records: Sequence[Mapping[str, Any]],
    *,
    height: int,
    width: int,
    min_area: int,
    max_instances: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Paint selected masks in consensus rank order without overlap."""
    if min_area < 1 or max_instances < 1:
        raise ValueError("min_area and max_instances must be positive")
    labels = np.zeros((height, width), dtype=np.int32)
    kept_meta: list[dict[str, Any]] = []
    for record in list(records)[:max_instances]:
        mask = decode_rle(str(record["segmentation_rle"]), height, width)
        available = mask & (labels == 0)
        area = int(available.sum())
        if area < min_area:
            continue
        labels[available] = len(kept_meta) + 1
        metadata = dict(record)
        metadata["area_after_deoverlap"] = area
        metadata.pop("segmentation_rle", None)
        kept_meta.append(metadata)
    return labels, label_map_to_rles(labels), kept_meta
