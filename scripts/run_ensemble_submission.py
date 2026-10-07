#!/usr/bin/env python3
"""Run available ensemble checkpoints and write an RLE submission."""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.io_utils import atomic_write_json
from src.models.ensemble import predict_sliding_window
from src.models import load_model_from_checkpoint
from src.postprocess.instances import probability_to_instance_labels
from src.submission.rle import label_map_to_rles, rle_counts, write_submission
from src.submission.validate import audit_image_labels, audit_submission_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--global-checkpoint", required=True)
    parser.add_argument("--detail-checkpoint")
    parser.add_argument("--instance-checkpoint")
    parser.add_argument("--output", default="artifacts/submissions/ensemble.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--global-weight", type=float, default=0.45)
    parser.add_argument("--detail-weight", type=float, default=0.30)
    parser.add_argument("--instance-weight", type=float, default=0.25)
    parser.add_argument("--audit-report", default="artifacts/submissions/ensemble_audit.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    models = {"global": load_model_from_checkpoint(
        args.global_checkpoint, device, expected_kind="global"
    )}
    if args.detail_checkpoint:
        models["detail"] = load_model_from_checkpoint(
            args.detail_checkpoint, device, expected_kind="detail"
        )
    if args.instance_checkpoint:
        models["instance"] = load_model_from_checkpoint(
            args.instance_checkpoint, device, expected_kind="instance"
        )
    rows = []
    image_reports = []
    image_dir = Path(args.data_root) / "test/test_images"
    image_paths = sorted(image_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"no test JPEG images found in {image_dir}")
    weights = (args.global_weight, args.detail_weight, args.instance_weight)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            original = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        fused, boundary, distance = predict_sliding_window(
            models,
            original,
            device,
            args.tile_size,
            overlap=args.overlap,
            weights=weights,
            tile_batch_size=args.tile_batch_size,
        )
        labels = probability_to_instance_labels(
            fused,
            threshold=args.threshold,
            min_area=args.min_area,
            closing_iterations=args.closing_iterations,
            boundary=boundary,
            distance=distance,
            use_watershed=distance is not None,
            max_instances=args.max_instances,
        )
        image_reports.append(audit_image_labels(image_path.stem, labels, original.shape))
        for index, rle in enumerate(label_map_to_rles(labels), start=1):
            rows.append({
                "filament_id": f"{image_path.stem}_{index}",
                "segmentation_rle": rle_counts(rle),
            })
    expected = {path.stem for path in image_paths}
    # Masks were already checked before encoding; avoid a duplicate full decode.
    # Use scripts.audit_submission as the final independent submission gate.
    summary = audit_submission_rows(rows, expected, validate_rles=False)
    elapsed = perf_counter() - started
    summary.update({
        "inference_and_audit_seconds": elapsed,
        "seconds_per_test_image": elapsed / max(1, len(expected)),
        "gpu_peak_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        "device": str(device),
        "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    report_path = Path(args.audit_report)
    output = write_submission(rows, args.output)
    atomic_write_json(
        report_path,
        {
            "summary": summary,
            "images": image_reports,
            "output": str(output),
            "checkpoints": {
                "global": args.global_checkpoint,
                "detail": args.detail_checkpoint,
                "instance": args.instance_checkpoint,
            },
            "inference": {
                "tile_size": args.tile_size,
                "tile_batch_size": args.tile_batch_size,
                "overlap": args.overlap,
                "weights": weights,
                "threshold": args.threshold,
                "min_area": args.min_area,
                "closing_iterations": args.closing_iterations,
                "max_instances": args.max_instances,
            },
        },
    )
    print(f"wrote {len(rows)} predicted instances to {output}")
    print(f"wrote submission audit to {report_path}")


if __name__ == "__main__":
    main()
