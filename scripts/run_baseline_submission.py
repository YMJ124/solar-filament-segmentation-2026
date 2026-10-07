#!/usr/bin/env python3
"""Run the no-training classical baseline and write a Kaggle CSV."""

from __future__ import annotations

import argparse
import resource
from pathlib import Path
from time import perf_counter

from src.baseline import predict_instance_labels
from src.submission.rle import label_map_to_rles, rle_counts, write_submission
from src.submission.validate import audit_image_labels, audit_submission_rows
from src.io_utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output", default="artifacts/submissions/classical_baseline.csv")
    parser.add_argument("--threshold-percentile", type=float, default=8.0)
    parser.add_argument("--min-component-area", type=int, default=80)
    parser.add_argument("--max-instances", type=int, default=128)
    parser.add_argument(
        "--audit-report", default="artifacts/submissions/classical_baseline_audit.json"
    )
    args = parser.parse_args()
    started = perf_counter()
    image_dir = Path(args.data_root) / "test/test_images"
    image_paths = sorted(image_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"no test JPEG images found in {image_dir}")
    rows = []
    image_reports = []
    for image_path in image_paths:
        labels = predict_instance_labels(
            image_path,
            threshold_percentile=args.threshold_percentile,
            min_component_area=args.min_component_area,
            max_instances=args.max_instances,
        )
        image_reports.append(audit_image_labels(image_path.stem, labels))
        for index, rle in enumerate(label_map_to_rles(labels), start=1):
            rows.append({
                "filament_id": f"{image_path.stem}_{index}",
                "segmentation_rle": rle_counts(rle),
            })
    expected = {path.stem for path in image_paths}
    # Masks were already checked before encoding; avoid decoding every RLE again
    # during generation. Use scripts.audit_submission for the independent audit.
    summary = audit_submission_rows(rows, expected, validate_rles=False)
    elapsed = perf_counter() - started
    summary.update({
        "generation_and_audit_seconds": elapsed,
        "seconds_per_test_image": elapsed / len(expected),
        "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    output = write_submission(rows, args.output)
    atomic_write_json(
        args.audit_report,
        {
            "summary": summary,
            "images": image_reports,
            "output": str(output),
            "parameters": {
                "threshold_percentile": args.threshold_percentile,
                "min_component_area": args.min_component_area,
                "max_instances": args.max_instances,
            },
        },
    )
    print(f"wrote {len(rows)} predicted instances to {output}")
    print(f"wrote submission audit to {args.audit_report}")


if __name__ == "__main__":
    main()
