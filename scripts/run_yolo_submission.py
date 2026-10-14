#!/usr/bin/env python3
"""Generate and fully audit a competition CSV from one YOLO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import resource
from pathlib import Path
from time import perf_counter

from src.instance import exclusive_label_map, prediction_arrays
from src.io_utils import atomic_write_json
from src.submission.rle import label_map_to_rles, rle_counts, write_submission
from src.submission.validate import audit_image_labels, audit_submission_rows
from src.ultralytics_runtime import prepare_ultralytics_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument(
        "--output", default="artifacts/submissions/yolo26n_fold0_proxy.csv"
    )
    parser.add_argument("--audit-report")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--quantize", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--retina-masks", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument(
        "--force-one", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not 0 < args.confidence < 1:
        raise ValueError("confidence must be inside (0, 1)")
    if args.max_instances < 1 or args.min_area < 1:
        raise ValueError("max-instances and min-area must be positive")

    image_dir = Path(args.data_root) / "test/test_images"
    image_paths = sorted(image_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"no test JPEG images found in {image_dir}")

    prepare_ultralytics_runtime()
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-yolo.txt before inference") from exc

    cuda = args.device != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(checkpoint))
    started = perf_counter()

    rows: list[dict] = []
    image_reports: list[dict] = []
    for index, image_path in enumerate(image_paths, start=1):
        # A list of paths is interpreted by Ultralytics as one in-memory batch.
        # Pass one path at a time so GPU use stays bounded on 8 GB cards.
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=min(0.01, args.confidence),
            max_det=args.max_instances,
            retina_masks=args.retina_masks,
            device=args.device,
            quantize=args.quantize,
            batch=1,
            verbose=False,
        )[0]
        shape = tuple(int(value) for value in result.orig_shape)
        masks, scores = prediction_arrays(result, shape)
        labels = exclusive_label_map(
            masks,
            scores,
            confidence_threshold=args.confidence,
            max_instances=args.max_instances,
            min_area=args.min_area,
            force_one=args.force_one,
        )
        image_reports.append(audit_image_labels(image_path.stem, labels, shape))
        for instance_id, rle in enumerate(label_map_to_rles(labels), start=1):
            rows.append(
                {
                    "filament_id": f"{image_path.stem}_{instance_id}",
                    "segmentation_rle": rle_counts(rle),
                }
            )
        print(
            f"YOLO submission {index}/{len(image_paths)}: "
            f"{image_path.name} instances={int(labels.max())}",
            flush=True,
        )
        del result, masks, scores, labels

    expected = {path.stem for path in image_paths}
    summary = audit_submission_rows(rows, expected, validate_rles=True)
    elapsed = perf_counter() - started
    summary.update(
        {
            "inference_and_audit_seconds": elapsed,
            "seconds_per_test_image": elapsed / len(image_paths),
            "device": args.device,
            "gpu_peak_mb": (
                torch.cuda.max_memory_allocated() / 1024**2 if cuda else 0.0
            ),
            "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        }
    )

    output = write_submission(rows, args.output)
    report_path = (
        Path(args.audit_report)
        if args.audit_report
        else output.with_suffix(".audit.json")
    )
    atomic_write_json(
        report_path,
        {
            "summary": summary,
            "images": image_reports,
            "output": str(output),
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            },
            "inference": {
                "imgsz": args.imgsz,
                "device": args.device,
                "quantize": args.quantize,
                "retina_masks": args.retina_masks,
                "confidence": args.confidence,
                "max_instances": args.max_instances,
                "min_area": args.min_area,
                "force_one": args.force_one,
            },
        },
    )
    print(f"wrote {len(rows)} predicted instances to {output}")
    print(f"wrote full submission audit to {report_path}")


if __name__ == "__main__":
    main()
