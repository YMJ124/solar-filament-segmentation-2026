#!/usr/bin/env python3
"""Calibrate one YOLO segmentation checkpoint with the official Fold PQ."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.data.masks import annotations_to_rles
from src.instance import exclusive_label_map
from src.io_utils import atomic_write_json
from src.metrics.pq import aggregate_counts, pairwise_rle_iou, score_iou_matrix
from src.submission.rle import label_map_to_rles


def _values(raw: list[str], cast) -> list:
    tokens = [token for value in raw for token in value.split(",") if token]
    return sorted(set(cast(token) for token in tokens))


def _prediction_arrays(result, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if result.masks is None or result.boxes is None or not len(result.boxes):
        return np.zeros((0, *shape), dtype=bool), np.zeros(0, dtype=np.float32)
    masks = result.masks.data.detach().cpu().numpy()
    if masks.shape[1:] != shape:
        masks = np.stack([
            cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            for mask in masks
        ])
    confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    return masks >= 0.5, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--fold-manifest", default="data/manifests/train_folds.csv")
    parser.add_argument("--validation-file-list")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidences", nargs="+", default=["0.10", "0.20", "0.30"])
    parser.add_argument("--max-instances", nargs="+", default=["5", "10", "20"])
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--force-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="artifacts/instance/yolo_fold0_pq.json")
    args = parser.parse_args()
    confidences = _values(args.confidences, float)
    instance_limits = _values(args.max_instances, int)
    if not confidences or min(confidences) <= 0 or max(confidences) >= 1:
        raise ValueError("confidences must be inside (0, 1)")
    if not instance_limits or min(instance_limits) < 1:
        raise ValueError("max-instances must be positive")

    root = Path(args.data_root)
    annotation = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    records = [
        row for row in load_fold_manifest(args.fold_manifest, annotation)
        if int(row["fold"]) == args.fold
    ]
    if args.validation_file_list:
        names = set(json.loads(Path(args.validation_file_list).read_text()))
        records = [row for row in records if str(row["file_name"]) in names]
    if not records:
        raise ValueError("validation selection is empty")
    config_dir = Path("artifacts/.ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-yolo.txt before evaluation") from exc

    candidates = list(itertools.product(confidences, instance_limits))
    states = {candidate: [] for candidate in candidates}
    prediction_totals = {candidate: 0 for candidate in candidates}
    image_paths = [root / "train/train_images" / row["file_name"] for row in records]
    model = YOLO(args.checkpoint)
    started = perf_counter()
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=min(0.01, min(confidences)),
        max_det=max(instance_limits),
        retina_masks=True,
        device=args.device,
        stream=True,
        verbose=False,
    )
    for index, (record, result) in enumerate(zip(records, results, strict=True), start=1):
        shape = (int(record["height"]), int(record["width"]))
        masks, scores = _prediction_arrays(result, shape)
        truth_batches = [
            annotations_to_rles(
                coco.annotations_for_batch(str(batch["id"])), *shape
            )
            for batch in coco.batches_for_file(str(record["file_name"]))
        ]
        for candidate in candidates:
            confidence, max_instances = candidate
            labels = exclusive_label_map(
                masks,
                scores,
                confidence_threshold=confidence,
                max_instances=max_instances,
                min_area=args.min_area,
                force_one=args.force_one,
            )
            predicted = label_map_to_rles(labels)
            prediction_totals[candidate] += len(predicted)
            for truth in truth_batches:
                matrix = pairwise_rle_iou(truth, predicted)
                states[candidate].append(
                    score_iou_matrix(matrix, len(truth), len(predicted))
                )
        print(f"YOLO PQ {index}/{len(records)}: {record['file_name']}", flush=True)

    rows = []
    for confidence, max_instances in candidates:
        report = aggregate_counts(
            states[(confidence, max_instances)], physical_images=len(records)
        ).to_dict()
        report.update({
            "confidence": confidence,
            "max_instances": max_instances,
            "min_area": args.min_area,
            "force_one": args.force_one,
            "prediction_instances": prediction_totals[(confidence, max_instances)],
        })
        rows.append(report)
    rows.sort(
        key=lambda row: (
            row["official_pq"], row["hit_pairs"], -row["false_positives"]
        ),
        reverse=True,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "fold": args.fold,
        "physical_images": len(records),
        "imgsz": args.imgsz,
        "evaluation_seconds": perf_counter() - started,
        "best": rows[0],
        "candidates": rows,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["best"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
