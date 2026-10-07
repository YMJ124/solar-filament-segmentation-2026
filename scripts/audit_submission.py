#!/usr/bin/env python3
"""Validate every compressed COCO RLE in a competition submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import atomic_write_json
from src.submission.validate import audit_submission_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output")
    args = parser.parse_args()
    submission = Path(args.submission)
    with submission.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ["filament_id", "segmentation_rle"]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"CSV columns must be exactly {expected_columns}, got {reader.fieldnames}"
            )
        rows = list(reader)
    image_dir = Path(args.data_root) / "test/test_images"
    expected_images = {path.stem for path in image_dir.glob("*.jpeg")}
    if not expected_images:
        raise FileNotFoundError(f"no test JPEG images found in {image_dir}")
    report = audit_submission_rows(rows, expected_images)
    report.update({
        "submission": str(submission),
        "submission_bytes": submission.stat().st_size,
        "columns": expected_columns,
    })
    print(json.dumps(report, indent=2))
    if args.output:
        atomic_write_json(args.output, report)


if __name__ == "__main__":
    main()
