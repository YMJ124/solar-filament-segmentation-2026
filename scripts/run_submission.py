#!/usr/bin/env python3
"""Run one checkpoint and write an audited competition submission."""

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

from src.inference import predict_sliding_window
from src.io_utils import atomic_write_json
from src.legacy_semantic import add_legacy_semantic_flag, require_legacy_semantic_opt_in
from src.models import load_model_from_checkpoint
from src.postprocess.instances import probability_to_instance_labels
from src.submission.rle import label_map_to_rles, rle_counts, write_submission
from src.submission.validate import audit_image_labels, audit_submission_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    add_legacy_semantic_flag(parser)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/submissions/single_model.csv")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument(
        "--audit-report", default="artifacts/submissions/single_model_audit.json"
    )
    args = parser.parse_args()
    require_legacy_semantic_opt_in(args.allow_legacy_semantic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    model = load_model_from_checkpoint(args.checkpoint, device, expected_kind="global")
    rows, image_reports = [], []
    image_dir = Path(args.data_root) / "test/test_images"
    image_paths = sorted(image_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"no test JPEG images found in {image_dir}")
    for image_path in image_paths:
        with Image.open(image_path) as image:
            original = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        probability = predict_sliding_window(
            model, original, device, args.tile_size,
            overlap=args.overlap, tile_batch_size=args.tile_batch_size,
        )
        labels = probability_to_instance_labels(
            probability,
            threshold=args.threshold,
            min_area=args.min_area,
            closing_iterations=args.closing_iterations,
            max_instances=args.max_instances,
        )
        image_reports.append(audit_image_labels(image_path.stem, labels, original.shape))
        for index, rle in enumerate(label_map_to_rles(labels), start=1):
            rows.append({
                "filament_id": f"{image_path.stem}_{index}",
                "segmentation_rle": rle_counts(rle),
            })

    expected = {path.stem for path in image_paths}
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
    output = write_submission(rows, args.output)
    report_path = Path(args.audit_report)
    atomic_write_json(report_path, {
        "summary": summary,
        "images": image_reports,
        "output": str(output),
        "checkpoint": args.checkpoint,
        "inference": {
            "tile_size": args.tile_size,
            "tile_batch_size": args.tile_batch_size,
            "overlap": args.overlap,
            "threshold": args.threshold,
            "min_area": args.min_area,
            "closing_iterations": args.closing_iterations,
            "max_instances": args.max_instances,
        },
    })
    print(f"wrote {len(rows)} predicted instances to {output}")
    print(f"wrote submission audit to {report_path}")


if __name__ == "__main__":
    main()
