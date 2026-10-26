#!/usr/bin/env python3
"""Build OOF T1 crop pairs for the P5-C1 local mask refiner."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.instance.p5_refiner import (
    crop_bounds,
    postprocessed_instances_from_rles,
    resize_refiner_crop,
)
from src.instance.yolo_cache import (
    load_cache_manifest,
    load_candidate_rles,
    sha256_file,
    validate_annotation_provenance,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_multifold import validate_multifold_manifests
from src.io_utils import atomic_write_json
from src.metrics.pq import pairwise_rle_iou
from src.submission.rle import mask_to_rle


def _config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("P5 refiner config must use schema_version=1")
    return payload


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/instance/stage_p5/p5_c1_crop_refiner_fold0.yaml",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override config data.training_crop_dir (useful for smoke tests).",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--no-verify-cache-hashes", action="store_true")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive")

    config = _config(args.config)
    data_cfg = config["data"]
    experiment = config["experiment"]
    crop_cfg = config["crop"]
    annotation = Path(data_cfg["annotation_file"])
    image_dir = Path(data_cfg["image_dir"])
    output_dir = args.output_dir or Path(data_cfg["training_crop_dir"])
    if (output_dir / "index.json").exists():
        raise FileExistsError(f"refiner dataset already completed: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coco = CocoAnnotations(annotation)
    manifests: dict[int, dict] = {}
    cache_dirs: dict[int, Path] = {}
    for raw in data_cfg["cache_dirs"]:
        cache_dir = Path(raw)
        manifest = load_cache_manifest(cache_dir)
        fold = int(manifest["dataset"]["fold"])
        validate_annotation_provenance(manifest, annotation)
        manifests[fold] = manifest
        cache_dirs[fold] = cache_dir
    identity = validate_multifold_manifests(manifests, {0, 1, 2})
    final_fit = bool(experiment.get("final_fit", False))
    raw_heldout = experiment.get("heldout_fold")
    heldout = None if raw_heldout is None else int(raw_heldout)
    training_folds = [int(value) for value in experiment["training_folds"]]
    if final_fit:
        if heldout is not None or set(training_folds) != set(manifests):
            raise ValueError(
                "final-fit refiner must use all cache folds and no heldout fold"
            )
    elif (
        heldout is None
        or heldout in training_folds
        or set(training_folds) != set(manifests) - {heldout}
    ):
        raise ValueError("training_folds must be exactly the two non-heldout folds")
    postprocess = YoloPostprocessConfig(**config["postprocess"])

    records = [
        (fold, entry)
        for fold in sorted(training_folds)
        for entry in manifests[fold]["images"]
    ]
    records.sort(key=lambda item: (item[0], str(item[1]["file_name"])))
    complete = args.max_images is None
    if args.max_images is not None:
        records = records[: args.max_images]

    started = perf_counter()
    samples = []
    image_rows = []
    iou_bands = Counter()
    for image_number, (fold, entry) in enumerate(records, start=1):
        file_name = str(entry["file_name"])
        height, width = int(entry["height"]), int(entry["width"])
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (height, width):
            raise ValueError(f"cannot read expected grayscale image: {image_dir / file_name}")
        rles, scores, _, _ = load_candidate_rles(
            cache_dirs[fold],
            entry,
            verify_hash=not args.no_verify_cache_hashes,
        )
        coarse_masks, coarse_scores, source_indices = postprocessed_instances_from_rles(
            rles,
            scores,
            height=height,
            width=width,
            config=postprocess,
        )
        prediction_rles = [mask_to_rle(mask) for mask in coarse_masks]
        before = len(samples)
        for batch in coco.batches_for_file(file_name):
            batch_id = str(batch["id"])
            annotations = coco.annotations_for_batch(batch_id)
            truth_rles = annotations_to_rles(annotations, height, width)
            matrix = pairwise_rle_iou(truth_rles, prediction_rles)
            for pred_index, coarse in enumerate(coarse_masks):
                if len(truth_rles):
                    gt_index = int(np.argmax(matrix[:, pred_index]))
                    maximum_iou = float(matrix[gt_index, pred_index])
                else:
                    gt_index = -1
                    maximum_iou = 0.0
                if maximum_iou > 0:
                    from pycocotools import mask as mask_utils

                    target = np.asarray(mask_utils.decode(truth_rles[gt_index]), dtype=bool)
                    target_annotation_id = str(annotations[gt_index]["id"])
                else:
                    target = np.zeros((height, width), dtype=bool)
                    target_annotation_id = None
                if maximum_iou > 0.5:
                    iou_band = "matched_gt_0_5"
                elif maximum_iou > 0.25:
                    iou_band = "near_threshold_gt_0_25_to_0_5"
                elif maximum_iou > 0:
                    iou_band = "low_overlap_le_0_25"
                else:
                    iou_band = "no_overlap_background_target"
                iou_bands[iou_band] += 1
                bounds = crop_bounds(
                    coarse,
                    context_scale=float(crop_cfg["context_scale"]),
                    minimum_side=int(crop_cfg["minimum_side"]),
                )
                image_crop, coarse_crop, target_crop = resize_refiner_crop(
                    image,
                    coarse,
                    target,
                    bounds,
                    output_size=int(crop_cfg["output_size"]),
                )
                sample_id = len(samples)
                relative = Path("samples") / f"{sample_id:06d}.npz"
                _atomic_npz(
                    output_dir / relative,
                    image=image_crop.astype(np.uint8),
                    coarse=coarse_crop.astype(np.uint8),
                    target=target_crop.astype(np.uint8),
                )
                samples.append(
                    {
                        "sample_id": sample_id,
                        "sample_file": str(relative),
                        "file_name": file_name,
                        "fold": fold,
                        "annotation_batch_id": batch_id,
                        "prediction_index": pred_index,
                        "source_candidate_index": int(source_indices[pred_index]),
                        "source_confidence": float(coarse_scores[pred_index]),
                        "target_annotation_id": target_annotation_id,
                        "target_max_iou": maximum_iou,
                        "target_iou_band": iou_band,
                        "crop_bounds": list(bounds),
                    }
                )
        image_rows.append(
            {
                "file_name": file_name,
                "fold": fold,
                "annotation_batches": len(coco.batches_for_file(file_name)),
                "postprocessed_instances": len(coarse_masks),
                "samples": len(samples) - before,
            }
        )
        print(
            f"P5-C1 crop data {image_number}/{len(records)}: {file_name} "
            f"samples={len(samples) - before}",
            flush=True,
        )

    if not samples:
        raise RuntimeError("P5 refiner builder produced no samples")
    payload = {
        "report_type": "p5_refiner_crop_dataset_v1",
        "complete": complete,
        "config": config,
        "config_sha256": sha256_file(args.config),
        "heldout_fold": heldout,
        "final_fit": final_fit,
        "training_folds": sorted(training_folds),
        "experiment_family": identity["experiment_family"],
        "training_signature_sha256": identity["training_signature_sha256"],
        "inference_signature": identity["inference_signature"],
        "annotation_sha256": sha256_file(annotation),
        "cache_manifest_sha256_by_fold": {
            str(fold): sha256_file(cache_dirs[fold] / "manifest.json")
            for fold in sorted(training_folds)
        },
        "postprocess": postprocess.to_dict(),
        "physical_images": len(records),
        "samples_count": len(samples),
        "target_iou_bands": dict(sorted(iou_bands.items())),
        "elapsed_seconds": perf_counter() - started,
        "images": image_rows,
        "samples": samples,
    }
    atomic_write_json(output_dir / "index.json", payload)
    print(f"wrote P5-C1 crop dataset to {output_dir}")


if __name__ == "__main__":
    main()
