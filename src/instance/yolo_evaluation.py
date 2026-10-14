"""CPU-only post-processing and official scoring of cached YOLO candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.metrics.pq import PQCounts, aggregate_counts, pairwise_rle_iou, score_iou_matrix
from src.submission.rle import label_map_to_rles

from .postprocess import exclusive_label_map
from .yolo_cache import load_candidate_record


@dataclass(frozen=True, order=True)
class YoloPostprocessConfig:
    confidence: float
    min_area: int
    max_instances: int
    force_one: bool = True

    def validate(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if self.min_area < 1 or self.max_instances < 1:
            raise ValueError("min_area and max_instances must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predictions_for_config(
    masks,
    scores,
    config: YoloPostprocessConfig,
) -> tuple[Any, list[dict]]:
    """Apply legal exclusive-mask post-processing to cached candidates."""
    config.validate()
    labels = exclusive_label_map(
        masks,
        scores,
        confidence_threshold=config.confidence,
        max_instances=config.max_instances,
        min_area=config.min_area,
        force_one=config.force_one,
    )
    return labels, label_map_to_rles(labels)


def truth_batches_for_image(
    coco: CocoAnnotations, file_name: str, height: int, width: int
) -> list[tuple[str, list[dict]]]:
    """Return every annotator batch for one physical image."""
    rows = []
    for batch in coco.batches_for_file(file_name):
        batch_shape = (int(batch["height"]), int(batch["width"]))
        if batch_shape != (height, width):
            raise ValueError(
                f"annotation shape {batch_shape} does not match cached image "
                f"shape {(height, width)} for {file_name}"
            )
        rows.append(
            (
                str(batch["id"]),
                annotations_to_rles(
                    coco.annotations_for_batch(str(batch["id"])), height, width
                ),
            )
        )
    return rows


def evaluate_cached_grid(
    cache_dir: str | Path,
    manifest: Mapping[str, Any],
    coco: CocoAnnotations,
    configs: Iterable[YoloPostprocessConfig],
    *,
    selected_names: set[str] | None = None,
    verify_cache_hashes: bool = True,
) -> list[dict[str, Any]]:
    """Decode each image once and score a post-process grid with official PQ."""
    configs = list(dict.fromkeys(configs))
    if not configs:
        raise ValueError("post-process grid is empty")
    for config in configs:
        config.validate()
    images = [
        row
        for row in manifest["images"]
        if selected_names is None or str(row["file_name"]) in selected_names
    ]
    if selected_names is not None:
        found = {str(row["file_name"]) for row in images}
        missing = sorted(selected_names - found)
        if missing:
            raise ValueError(f"selected images are absent from cache: {missing[:5]}")
    if not images:
        raise ValueError("evaluation selection is empty")

    states: dict[YoloPostprocessConfig, list[PQCounts]] = {
        config: [] for config in configs
    }
    prediction_totals = {config: 0 for config in configs}
    for index, image_entry in enumerate(images, start=1):
        masks, scores, _ = load_candidate_record(
            cache_dir, image_entry, verify_hash=verify_cache_hashes
        )
        height, width = int(image_entry["height"]), int(image_entry["width"])
        truth_batches = truth_batches_for_image(
            coco, str(image_entry["file_name"]), height, width
        )
        for config in configs:
            _, predicted = predictions_for_config(masks, scores, config)
            prediction_totals[config] += len(predicted)
            for _, truth in truth_batches:
                matrix = pairwise_rle_iou(truth, predicted)
                states[config].append(
                    score_iou_matrix(matrix, len(truth), len(predicted))
                )
        print(
            f"offline YOLO PQ {index}/{len(images)}: {image_entry['file_name']}",
            flush=True,
        )

    rows = []
    for config in configs:
        row = aggregate_counts(
            states[config], physical_images=len(images)
        ).to_dict()
        row.update(config.to_dict())
        row["prediction_instances"] = prediction_totals[config]
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["official_pq"],
            row["hit_pairs"],
            -row["false_positives"],
            -row["false_negatives"],
        ),
        reverse=True,
    )
    return rows
