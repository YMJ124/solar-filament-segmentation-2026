"""Full-image validation with the competition's annotator-aware PQ."""

from __future__ import annotations

import os
import resource
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.metrics.pq import aggregate_counts, pairwise_rle_iou, score_iou_matrix
from src.metrics.segmentation import segmentation_metrics
from src.inference import predict_sliding_window
from src.postprocess.instances import probability_to_instance_labels
from src.submission.rle import label_map_to_rles
from src.io_utils import atomic_write_json


def _atomic_save_probability(path: Path, probability: np.ndarray, file_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            payload = {
                "probability": np.asarray(probability, dtype=np.float16),
                "file_name": np.asarray(file_name),
            }
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _positive_score_distribution(values: list[np.ndarray]) -> dict[str, Any]:
    """Summarize positive pairwise scores like the official evaluation notebook."""
    positive = (
        np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])
        if values else np.zeros(0, dtype=np.float64)
    )
    positive = positive[positive > 0]
    edges = np.linspace(0.0, 1.0, 51)
    counts, _ = np.histogram(positive, bins=edges)
    return {
        "selection": "pairwise_score_strictly_greater_than_zero",
        "count": int(positive.size),
        "mean": float(positive.mean()) if positive.size else None,
        "median": float(np.median(positive)) if positive.size else None,
        "p10": float(np.percentile(positive, 10)) if positive.size else None,
        "p90": float(np.percentile(positive, 90)) if positive.size else None,
        "histogram_bin_edges": edges.tolist(),
        "histogram_counts": counts.tolist(),
    }


def _degree_distribution(degrees: list[int]) -> dict[str, Any]:
    counts = Counter(int(degree) for degree in degrees)
    return {
        "overlap_rule": "pairwise_iou_strictly_greater_than_zero",
        "degree_counts": {str(key): counts[key] for key in sorted(counts)},
        "total_instances": len(degrees),
        "misses": counts.get(0, 0),
        "single_partner": counts.get(1, 0),
        "multiple_partners": sum(value for key, value in counts.items() if key > 1),
    }


@torch.inference_mode()
def evaluate_full_images(
    model,
    records: Iterable[Mapping[str, Any]],
    coco: CocoAnnotations,
    image_dir: str | Path,
    device,
    validation_config: Mapping[str, Any],
    oof_dir: str | Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate one model once per image and score every annotator batch."""
    selected = list(records)
    max_images = validation_config.get("max_images")
    if max_images is not None:
        selected = selected[:int(max_images)]
    if not selected:
        raise ValueError("full-image validation received no records")

    image_dir = Path(image_dir)
    output_dir = Path(oof_dir) if oof_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    training_state = model.training
    model.eval()
    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pq_counts = []
    pixel_metrics = []
    positive_iou_matrices = []
    positive_dice_matrices = []
    gt_overlap_degrees = []
    prediction_overlap_degrees = []
    prediction_counts = []
    saved_files = []
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise RuntimeError("pycocotools is required for full-image validation") from exc
    try:
        for index, record in enumerate(selected, start=1):
            file_name = str(record["file_name"])
            with Image.open(image_dir / file_name) as image:
                image_array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            prediction = predict_sliding_window(
                model, image_array, device,
                tile_size=int(validation_config["tile_size"]),
                overlap=float(validation_config.get("overlap", 0.25)),
                tile_batch_size=int(validation_config.get("tile_batch_size", 8)),
            )
            if output_dir is not None:
                probability_path = output_dir / f"{Path(file_name).stem}.npz"
                _atomic_save_probability(probability_path, prediction, file_name)
                saved_files.append(probability_path.name)
            predicted_labels = probability_to_instance_labels(
                prediction,
                threshold=float(validation_config.get("threshold", 0.5)),
                min_area=int(validation_config.get("min_area", 80)),
                closing_iterations=int(
                    validation_config.get("closing_iterations", 1)
                ),
                max_instances=int(validation_config.get("max_instances", 128)),
            )
            prediction_rles = label_map_to_rles(predicted_labels)
            prediction_counts.append(len(prediction_rles))
            predicted_semantic = predicted_labels > 0
            for batch in coco.batches_for_file(file_name):
                truth_rles = annotations_to_rles(
                    coco.annotations_for_batch(str(batch["id"])), *image_array.shape
                )
                iou_matrix = pairwise_rle_iou(truth_rles, prediction_rles)
                pq_counts.append(
                    score_iou_matrix(
                        iou_matrix,
                        n_truth=len(truth_rles),
                        n_prediction=len(prediction_rles),
                    )
                )
                dice_matrix = np.divide(
                    2.0 * iou_matrix,
                    1.0 + iou_matrix,
                    out=np.zeros_like(iou_matrix),
                    where=(1.0 + iou_matrix) != 0,
                )
                positive_iou_matrices.append(iou_matrix)
                positive_dice_matrices.append(dice_matrix)
                overlap_matrix = iou_matrix > 0
                gt_overlap_degrees.extend(overlap_matrix.sum(axis=1).astype(int).tolist())
                prediction_overlap_degrees.extend(
                    overlap_matrix.sum(axis=0).astype(int).tolist()
                )
                truth_instances = [
                    np.asarray(mask_utils.decode(rle), dtype=bool) for rle in truth_rles
                ]
                truth_semantic = (
                    np.logical_or.reduce(truth_instances)
                    if truth_instances else np.zeros(image_array.shape, dtype=bool)
                )
                pixel_metrics.append(segmentation_metrics(truth_semantic, predicted_semantic))
            if progress:
                print(f"full-image validation {index}/{len(selected)}: {file_name}")
    finally:
        model.train(training_state)

    report = aggregate_counts(pq_counts, physical_images=len(selected)).to_dict()
    elapsed = perf_counter() - started
    max_instances = int(validation_config.get("max_instances", 128))
    report.update({
        "dice_annotator_macro": float(np.mean([row["dice"] for row in pixel_metrics])),
        "iou_annotator_macro": float(np.mean([row["iou"] for row in pixel_metrics])),
        "pairwise_iou_distribution": _positive_score_distribution(positive_iou_matrices),
        "pairwise_dice_distribution": _positive_score_distribution(positive_dice_matrices),
        "one_to_many_gt_distribution": _degree_distribution(gt_overlap_degrees),
        "many_to_one_prediction_distribution": _degree_distribution(
            prediction_overlap_degrees
        ),
        "prediction_count_summary": {
            "total": int(sum(prediction_counts)),
            "mean_per_physical_image": float(np.mean(prediction_counts)),
            "max_per_physical_image": int(max(prediction_counts, default=0)),
            "images_at_safety_cap": int(
                sum(count == max_instances for count in prediction_counts)
            ),
            "safety_cap": max_instances,
        },
        "inference_config": {
            key: validation_config.get(key, default)
            for key, default in {
                "tile_size": 512,
                "tile_batch_size": 8,
                "overlap": 0.25,
                "threshold": 0.5,
                "min_area": 80,
                "closing_iterations": 1,
                "max_instances": 128,
            }.items()
        },
        "model_mode": "single",
        "evaluated_file_names": [str(row["file_name"]) for row in selected],
        "oof_probability_dtype": "float16" if output_dir is not None else None,
        "oof_probability_files": saved_files,
        "inference_gpu_peak_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        "evaluation_seconds": elapsed,
        "seconds_per_physical_image": elapsed / len(selected),
        "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    if output_dir is not None:
        atomic_write_json(output_dir / "manifest.json", report)
    return report


def evaluate_global_full_images(
    model,
    records: Iterable[Mapping[str, Any]],
    coco: CocoAnnotations,
    image_dir: str | Path,
    device,
    validation_config: Mapping[str, Any],
    oof_dir: str | Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Backward-compatible alias for single-model full-image validation."""
    return evaluate_full_images(
        model, records, coco, image_dir, device, validation_config,
        oof_dir=oof_dir, progress=progress,
    )
