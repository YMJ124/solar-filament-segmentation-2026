"""OOF-only post-processing search and detailed evaluation utilities."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.metrics.pq import aggregate_counts, pairwise_rle_iou, score_iou_matrix
from src.metrics.segmentation import segmentation_metrics
from src.postprocess.instances import (
    filter_instance_labels,
    probability_to_component_labels,
    probability_to_instance_labels,
)
from src.submission.rle import label_map_to_rles


@dataclass(frozen=True, order=True)
class PostprocessConfig:
    threshold: float = 0.5
    min_area: int = 80
    closing_iterations: int = 1
    max_instances: int = 128
    max_area_fraction: float = 0.35

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be finite and within [0, 1]")
        if self.min_area < 1:
            raise ValueError("min_area must be positive")
        if self.closing_iterations < 0:
            raise ValueError("closing_iterations must be non-negative")
        if self.max_instances < 1:
            raise ValueError("max_instances must be positive")
        if not 0 < self.max_area_fraction <= 1:
            raise ValueError("max_area_fraction must be within (0, 1]")

    @property
    def candidate_id(self) -> str:
        threshold = f"{self.threshold:.6f}".rstrip("0").rstrip(".")
        return (
            f"threshold={threshold};min_area={self.min_area};"
            f"closing={self.closing_iterations};max_instances={self.max_instances}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expected_oof_paths(
    oof_dir: Path, records: Iterable[Mapping[str, Any]]
) -> dict[str, Path]:
    return {
        str(row["file_name"]): oof_dir / f"{Path(str(row['file_name'])).stem}.npz"
        for row in records
    }


def validate_oof_directory(
    oof_dir: str | Path,
    records: Iterable[Mapping[str, Any]],
    fold: int,
) -> dict[str, Any]:
    """Reject incomplete, extra, misnamed, or stale OOF probability sets."""
    oof_dir = Path(oof_dir)
    selected = [dict(row) for row in records]
    if not selected:
        raise ValueError(f"fold {fold} contains no physical images")
    wrong_fold = [row["file_name"] for row in selected if int(row["fold"]) != fold]
    if wrong_fold:
        raise ValueError(f"records contain files outside fold {fold}: {wrong_fold[:5]}")
    if not oof_dir.is_dir():
        raise FileNotFoundError(f"OOF directory not found: {oof_dir}")

    expected = _expected_oof_paths(oof_dir, selected)
    expected_names = {path.name for path in expected.values()}
    actual_names = {path.name for path in oof_dir.glob("*.npz")}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "OOF directory does not exactly match the selected fold; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    manifest_path = oof_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"OOF manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("physical_images", -1)) != len(selected):
        raise ValueError("OOF manifest physical image count does not match the fold")
    manifest_files = set(manifest.get("oof_probability_files", []))
    if manifest_files and manifest_files != expected_names:
        raise ValueError("OOF manifest filenames do not exactly match the selected fold")
    evaluated = set(manifest.get("evaluated_file_names", []))
    selected_names = {str(row["file_name"]) for row in selected}
    if evaluated and evaluated != selected_names:
        raise ValueError("OOF manifest evaluated filenames do not match the selected fold")
    return manifest


def load_oof_probability(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {"probability", "file_name"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"OOF file {path} is missing arrays: {sorted(missing)}")
        stored_name = str(np.asarray(payload["file_name"]).item())
        expected_name = str(record["file_name"])
        if stored_name != expected_name:
            raise ValueError(
                f"OOF file {path} stores {stored_name!r}, expected {expected_name!r}"
            )
        probability = np.asarray(payload["probability"], dtype=np.float32)
    expected_shape = (int(record["height"]), int(record["width"]))
    if probability.shape != expected_shape:
        raise ValueError(
            f"OOF probability {path} has shape {probability.shape}, expected {expected_shape}"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"OOF probability {path} contains non-finite values")
    if probability.min(initial=0.0) < 0 or probability.max(initial=1.0) > 1:
        raise ValueError(f"OOF probability {path} contains values outside [0, 1]")
    return probability


def build_truth_rles(
    coco: CocoAnnotations,
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[list[dict]]]:
    """Cache every annotator batch as RLE without unioning annotators."""
    truth: dict[str, list[list[dict]]] = {}
    for record in records:
        file_name = str(record["file_name"])
        batches = []
        for batch in coco.batches_for_file(file_name):
            batches.append(
                annotations_to_rles(
                    coco.annotations_for_batch(str(batch["id"])),
                    int(batch["height"]),
                    int(batch["width"]),
                )
            )
        if not batches:
            raise ValueError(f"no annotator batches found for {file_name}")
        truth[file_name] = batches
    return truth


def search_oof_postprocess(
    oof_dir: str | Path,
    records: Iterable[Mapping[str, Any]],
    coco: CocoAnnotations,
    configs: Iterable[PostprocessConfig],
    fold: int,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Score a configuration grid while loading one full probability map at a time."""
    oof_dir = Path(oof_dir)
    selected = [dict(row) for row in records]
    validate_oof_directory(oof_dir, selected, fold)
    candidates = sorted(set(configs))
    if not candidates:
        raise ValueError("at least one post-processing configuration is required")
    truth_by_file = build_truth_rles(coco, selected)
    grouped: dict[tuple[float, int], list[PostprocessConfig]] = defaultdict(list)
    for config in candidates:
        grouped[(config.threshold, config.closing_iterations)].append(config)

    states = {
        config: {
            "hit_pairs": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "hit_iou_sum": 0.0,
            "prediction_instances": 0,
            "max_prediction_instances": 0,
            "images_at_safety_cap": 0,
        }
        for config in candidates
    }
    annotator_records = sum(len(value) for value in truth_by_file.values())
    started = perf_counter()
    for image_index, record in enumerate(selected, start=1):
        file_name = str(record["file_name"])
        probability = load_oof_probability(
            oof_dir / f"{Path(file_name).stem}.npz", record
        )
        for (threshold, closing_iterations), group_configs in grouped.items():
            component_labels = probability_to_component_labels(
                probability,
                threshold=threshold,
                closing_iterations=closing_iterations,
            )
            max_area = int(component_labels.size * group_configs[0].max_area_fraction)
            for config in group_configs:
                labels = filter_instance_labels(
                    component_labels,
                    min_area=config.min_area,
                    max_area=max_area,
                    max_instances=config.max_instances,
                )
                prediction_rles = label_map_to_rles(labels)
                state = states[config]
                prediction_count = len(prediction_rles)
                state["prediction_instances"] += prediction_count
                state["max_prediction_instances"] = max(
                    state["max_prediction_instances"], prediction_count
                )
                state["images_at_safety_cap"] += int(
                    prediction_count == config.max_instances
                )
                for truth_rles in truth_by_file[file_name]:
                    matrix = pairwise_rle_iou(truth_rles, prediction_rles)
                    counts = score_iou_matrix(
                        matrix,
                        n_truth=len(truth_rles),
                        n_prediction=len(prediction_rles),
                    )
                    state["hit_pairs"] += counts.hit_pairs
                    state["false_positives"] += counts.false_positives
                    state["false_negatives"] += counts.false_negatives
                    state["hit_iou_sum"] += counts.hit_iou_sum
        if progress:
            print(f"OOF search {image_index}/{len(selected)}: {file_name}", flush=True)

    elapsed = perf_counter() - started
    rows = []
    for config in candidates:
        state = states[config]
        denominator = (
            state["hit_pairs"]
            + 0.5 * state["false_positives"]
            + 0.5 * state["false_negatives"]
        )
        rows.append(
            {
                "candidate_id": config.candidate_id,
                **config.to_dict(),
                "physical_images": len(selected),
                "annotator_records": annotator_records,
                **state,
                "mean_prediction_instances": (
                    state["prediction_instances"] / len(selected)
                ),
                "official_pq": (
                    state["hit_iou_sum"] / denominator if denominator else 0.0
                ),
                "search_elapsed_seconds": elapsed,
            }
        )
    return rows


def _degree_distribution(degrees: list[int]) -> dict[str, Any]:
    counts = Counter(int(value) for value in degrees)
    return {
        "degree_counts": {str(key): counts[key] for key in sorted(counts)},
        "total_instances": len(degrees),
        "misses": counts.get(0, 0),
        "single_partner": counts.get(1, 0),
        "multiple_partners": sum(value for key, value in counts.items() if key > 1),
        "overlap_rule": "pairwise_iou_strictly_greater_than_zero",
    }


def evaluate_oof_configuration(
    oof_dir: str | Path,
    records: Iterable[Mapping[str, Any]],
    coco: CocoAnnotations,
    config: PostprocessConfig,
    fold: int,
    progress: bool = True,
) -> dict[str, Any]:
    """Produce a full official-PQ and pixel-metric report for one OOF config."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("pycocotools is required for OOF evaluation") from error

    oof_dir = Path(oof_dir)
    selected = [dict(row) for row in records]
    validate_oof_directory(oof_dir, selected, fold)
    truth_by_file = build_truth_rles(coco, selected)
    all_counts = []
    pixel_rows = []
    gt_degrees: list[int] = []
    prediction_degrees: list[int] = []
    prediction_counts: list[int] = []
    started = perf_counter()
    for image_index, record in enumerate(selected, start=1):
        file_name = str(record["file_name"])
        probability = load_oof_probability(
            oof_dir / f"{Path(file_name).stem}.npz", record
        )
        labels = probability_to_instance_labels(
            probability,
            threshold=config.threshold,
            min_area=config.min_area,
            closing_iterations=config.closing_iterations,
            use_watershed=False,
            max_area_fraction=config.max_area_fraction,
            max_instances=config.max_instances,
        )
        prediction_rles = label_map_to_rles(labels)
        prediction_counts.append(len(prediction_rles))
        predicted_semantic = labels > 0
        for truth_rles in truth_by_file[file_name]:
            matrix = pairwise_rle_iou(truth_rles, prediction_rles)
            all_counts.append(
                score_iou_matrix(
                    matrix,
                    n_truth=len(truth_rles),
                    n_prediction=len(prediction_rles),
                )
            )
            overlap = matrix > 0
            gt_degrees.extend(overlap.sum(axis=1).astype(int).tolist())
            prediction_degrees.extend(overlap.sum(axis=0).astype(int).tolist())
            truth_masks = [
                np.asarray(mask_utils.decode(rle), dtype=bool) for rle in truth_rles
            ]
            truth_semantic = (
                np.logical_or.reduce(truth_masks)
                if truth_masks
                else np.zeros(predicted_semantic.shape, dtype=bool)
            )
            pixel_rows.append(segmentation_metrics(truth_semantic, predicted_semantic))
        if progress:
            print(f"OOF evaluate {image_index}/{len(selected)}: {file_name}", flush=True)

    report = aggregate_counts(all_counts, physical_images=len(selected)).to_dict()
    report.update(
        {
            "fold": fold,
            "config": config.to_dict(),
            "dice_annotator_macro": float(np.mean([row["dice"] for row in pixel_rows])),
            "iou_annotator_macro": float(np.mean([row["iou"] for row in pixel_rows])),
            "one_to_many_gt_distribution": _degree_distribution(gt_degrees),
            "many_to_one_prediction_distribution": _degree_distribution(
                prediction_degrees
            ),
            "prediction_count_summary": {
                "total": int(sum(prediction_counts)),
                "mean_per_physical_image": float(np.mean(prediction_counts)),
                "max_per_physical_image": int(max(prediction_counts, default=0)),
                "images_at_safety_cap": int(
                    sum(value == config.max_instances for value in prediction_counts)
                ),
                "safety_cap": config.max_instances,
            },
            "evaluated_file_names": [str(row["file_name"]) for row in selected],
            "evaluation_seconds": perf_counter() - started,
            "oof_probability_dtype": "float16",
        }
    )
    return report
