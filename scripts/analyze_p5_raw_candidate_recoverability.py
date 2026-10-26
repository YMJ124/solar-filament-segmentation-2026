#!/usr/bin/env python3
"""Audit whether final T1 misses are recoverable from existing raw candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.instance.p5_diagnostics import (
    audit_candidate_batch,
    summarize_recoverability_rows,
)
from src.instance.yolo_cache import (
    load_cache_manifest,
    load_candidate_rles,
    sha256_file,
    validate_annotation_provenance,
    validate_search_space,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_multifold import validate_multifold_manifests
from src.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


def _load_residual_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"residual CSV contains no rows: {path}")
    required = {
        "file_name",
        "image_id",
        "annotation_id",
        "fold",
        "t1_outcome",
        "t1_max_prediction_iou",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"residual CSV is missing columns: {sorted(missing)}")
    keys = [
        (row["file_name"], row["image_id"], row["annotation_id"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("residual CSV contains duplicate GT identities")
    return rows


def _validate_residual_source(index: dict, rows: list[dict[str, str]]) -> None:
    if index.get("report_type") != "p4_paired_residual_audit_v1":
        raise ValueError("P5-D1 requires a P4 paired residual audit")
    expected = index["paired_residuals"]["global"]
    if len(rows) != int(expected["instances"]):
        raise ValueError("residual CSV GT count does not match P4 audit index")
    misses = [row for row in rows if row["t1_outcome"] != "matched_gt_0_5"]
    if len(misses) != int(expected["t1_misses"]):
        raise ValueError("residual CSV T1 miss count does not match P4 audit index")
    observed = Counter(row["t1_outcome"] for row in misses)
    expected_outcomes = {
        str(key): int(value)
        for key, value in expected["remaining_t1_miss_outcomes"].items()
    }
    if dict(sorted(observed.items())) != dict(sorted(expected_outcomes.items())):
        raise ValueError("residual CSV T1 outcomes do not match P4 audit index")


def _markdown(payload: dict) -> str:
    if not payload["complete"]:
        return (
            "# P5-D1 raw-candidate recoverability smoke run\n\n"
            f"This partial run processed {payload['processed_physical_images']} physical "
            "images. Routing decisions are intentionally disabled.\n"
        )
    summary = payload["recoverability"]
    global_row = summary["global"]
    checks = summary["routing_checks"]
    lines = [
        "# P5-D1 原始候选可恢复性审计",
        "",
        "本报告只读取 P4-T1 的三折 OOF 原始候选缓存，不重新训练或推理。",
        "官方匹配阈值保持严格 `IoU > 0.5`。",
        "",
        "## 全局结果",
        "",
        f"- 最终 T1 misses：{global_row['miss_instances']}。",
        f"- 最终无重叠：{global_row['final_no_overlap_instances']}。",
        f"- 无重叠 GT 中，现有原始候选可达到 `IoU>0.5`："
        f"{global_row['final_no_overlap_raw_recoverable_gt_0_5']} "
        f"({global_row['final_no_overlap_raw_recoverable_rate']:.2%})。",
        f"- 无重叠 GT 中，原始候选仍 `IoU<=0.25`："
        f"{global_row['final_no_overlap_raw_le_0_25']} "
        f"({global_row['final_no_overlap_raw_le_0_25_rate']:.2%})。",
        f"- 缓存达到 max_candidates 的 miss 记录："
        f"{global_row['cache_limit_hit_instances']} "
        f"({global_row['cache_limit_hit_rate']:.2%})。",
        "",
        "## 无重叠 GT 的可恢复路径",
        "",
        "| 路径 | GT |",
        "|---|---:|",
    ]
    for name, count in global_row[
        "final_no_overlap_recoverability_routes"
    ].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## 预注册路由",
            "",
            f"- Mask-quality 重排证据：`{checks['mask_quality_reranking_supported']}`。",
            f"- 新候选生成证据：`{checks['candidate_generation_supported']}`。",
            f"- Crop refiner 证据：`{checks['crop_refiner_supported']}`。",
            f"- 推荐召回路线：`{summary['recommended_recall_route']}`。",
            f"- 推荐掩膜路线：`{summary['recommended_mask_route']}`。",
            "",
            "注意：候选数达到缓存上限的图像上，‘原始候选缺失’只能解释为当前",
            "`conf>=0.01/max_candidates=50` 缓存范围内的下界，不能外推为模型绝对无候选。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument(
        "--residual-dir",
        default="artifacts/analysis/stage_p4/t1_vs_i1_residuals",
        type=Path,
    )
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis/stage_p5/raw_candidate_recoverability",
        type=Path,
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Smoke-test only; partial runs never emit routing decisions.",
    )
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive")

    annotation = args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    residual_index_path = args.residual_dir / "index.json"
    residual_csv_path = args.residual_dir / "instance_transitions.csv"
    residual_index = json.loads(residual_index_path.read_text())
    residual_rows = _load_residual_rows(residual_csv_path)
    _validate_residual_source(residual_index, residual_rows)
    config = YoloPostprocessConfig(**residual_index["postprocess"])

    manifests: dict[int, dict] = {}
    cache_by_fold: dict[int, Path] = {}
    for cache_dir in args.cache_dir:
        manifest = load_cache_manifest(cache_dir)
        fold = int(manifest["dataset"]["fold"])
        if fold in manifests:
            raise ValueError(f"duplicate cache fold: {fold}")
        validate_annotation_provenance(manifest, annotation)
        validate_search_space(
            manifest,
            confidences=[config.confidence],
            max_instances=[config.max_instances],
        )
        expected_hash = residual_index["cache_manifest_sha256"]["t1"].get(str(fold))
        actual_hash = sha256_file(cache_dir / "manifest.json")
        if expected_hash != actual_hash:
            raise ValueError(f"Fold{fold} cache manifest differs from P4 residual audit")
        manifests[fold] = manifest
        cache_by_fold[fold] = cache_dir
    identity = validate_multifold_manifests(manifests, {0, 1, 2})
    if identity != residual_index["t1_identity"]:
        raise ValueError("T1 cache identity differs from P4 residual audit")

    residual_by_batch: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    missed_files: set[str] = set()
    for row in residual_rows:
        key = (row["file_name"], row["image_id"])
        residual_by_batch[key][row["annotation_id"]] = row
        if row["t1_outcome"] != "matched_gt_0_5":
            missed_files.add(row["file_name"])

    records: list[tuple[int, dict]] = []
    for fold in sorted(manifests):
        records.extend(
            (fold, record)
            for record in manifests[fold]["images"]
            if str(record["file_name"]) in missed_files
        )
    records.sort(key=lambda item: (item[0], str(item[1]["file_name"])))
    complete = args.max_images is None
    if args.max_images is not None:
        records = records[: args.max_images]

    instance_rows: list[dict] = []
    image_rows: list[dict] = []
    for completed, (fold, record) in enumerate(records, start=1):
        candidate_rles, scores, areas, _ = load_candidate_rles(
            cache_by_fold[fold], record
        )
        file_name = str(record["file_name"])
        before = len(instance_rows)
        for batch in coco.batches_for_file(file_name):
            batch_id = str(batch["id"])
            annotations = coco.annotations_for_batch(batch_id)
            instance_rows.extend(
                audit_candidate_batch(
                    annotations,
                    annotations_to_rles(
                        annotations, int(record["height"]), int(record["width"])
                    ),
                    residual_by_batch[(file_name, batch_id)],
                    candidate_rles,
                    scores,
                    areas,
                    record,
                    config,
                    cache_max_candidates=int(
                        manifests[fold]["inference"]["max_candidates"]
                    ),
                    inference_size=int(manifests[fold]["inference"]["imgsz"]),
                )
            )
        image_rows.append(
            {
                "file_name": file_name,
                "fold": fold,
                "miss_instances": len(instance_rows) - before,
                "cached_candidate_count": len(candidate_rles),
                "cache_max_candidates": int(
                    manifests[fold]["inference"]["max_candidates"]
                ),
                "cache_limit_hit": len(candidate_rles)
                >= int(manifests[fold]["inference"]["max_candidates"]),
            }
        )
        print(
            f"P5-D1 raw candidate audit {completed}/{len(records)}: {file_name}",
            flush=True,
        )

    if not instance_rows:
        raise RuntimeError("selected images contain no T1 misses")
    recoverability = summarize_recoverability_rows(instance_rows)
    if complete:
        expected = residual_index["paired_residuals"]["global"]
        if len(instance_rows) != int(expected["t1_misses"]):
            raise RuntimeError("P5-D1 did not reproduce the expected T1 miss count")
        no_overlap = recoverability["global"]["final_no_overlap_instances"]
        expected_no_overlap = int(
            expected["remaining_t1_miss_outcomes"]["no_overlap"]
        )
        if no_overlap != expected_no_overlap:
            raise RuntimeError("P5-D1 did not reproduce the expected no-overlap count")
    else:
        recoverability["routing_checks"] = None
        recoverability["recommended_recall_route"] = "not_evaluated_partial_run"
        recoverability["recommended_mask_route"] = "not_evaluated_partial_run"

    payload = {
        "report_type": "p5_raw_candidate_recoverability_v1",
        "complete": complete,
        "processed_physical_images": len(records),
        "processed_miss_instances": len(instance_rows),
        "experiment_family": identity["experiment_family"],
        "training_signature_sha256": identity["training_signature_sha256"],
        "inference_signature": identity["inference_signature"],
        "postprocess": config.to_dict(),
        "annotation_sha256": sha256_file(annotation),
        "residual_index_sha256": sha256_file(residual_index_path),
        "residual_csv_sha256": sha256_file(residual_csv_path),
        "cache_manifest_sha256_by_fold": {
            str(fold): sha256_file(cache_by_fold[fold] / "manifest.json")
            for fold in sorted(cache_by_fold)
        },
        "raw_candidate_scope": {
            "minimum_confidence": float(identity["inference_signature"][3]),
            "maximum_candidates": int(identity["inference_signature"][4]),
            "cache_limit_interpretation": "absence_is_lower_bound_when_limit_hit",
        },
        "recoverability": recoverability,
        "image_cache_censoring": {
            "physical_images": len(image_rows),
            "cache_limit_hit_images": sum(row["cache_limit_hit"] for row in image_rows),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "index.json", payload)
    atomic_write_csv(
        args.output_dir / "instance_recoverability.csv",
        list(instance_rows[0]),
        instance_rows,
    )
    atomic_write_csv(
        args.output_dir / "image_cache_censoring.csv",
        list(image_rows[0]),
        image_rows,
    )
    atomic_write_text(args.output_dir / "README.md", _markdown(payload))
    print(f"wrote P5-D1 raw-candidate audit to {args.output_dir}")


if __name__ == "__main__":
    main()
