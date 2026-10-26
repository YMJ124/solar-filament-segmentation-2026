#!/usr/bin/env python3
"""Refine B0 FN/FP evidence from existing fold caches without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.coco import CocoAnnotations
from src.instance.yolo_cache import (
    load_cache_manifest,
    load_candidate_record,
    validate_annotation_provenance,
    validate_search_space,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_failure_modes import (
    PREDICTION_AREA_EDGES,
    TRUTH_AREA_EDGES,
    analyze_yolo_failure_modes,
    summarize_failure_mode_rows,
)
from src.instance.yolo_multifold import validate_multifold_manifests
from src.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


def _markdown(report: dict) -> str:
    total = report["global"]
    fold2 = report["fold_summaries"]["2"]
    lines = [
        "# B0 细分失败模式",
        "",
        "本报告只读取已有三 Fold 候选缓存；没有重新推理。面积单位为原图像素，",
        "IoU 分层只统计官方阈值 0.5 以下的未匹配实例。",
        "",
        "## FN 面积分层",
        "",
        "| GT 面积 | GT 数 | FN 数 | 该面积漏检率 | FN 占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, count in total["truth_instances_by_area"].items():
        misses = total["false_negatives_by_area"].get(key, 0)
        lines.append(
            f"| `{key}` | {count} | {misses} | "
            f"{total['false_negative_area_miss_rates'][key]:.2%} | "
            f"{total['false_negative_area_shares'].get(key, 0):.2%} |"
        )
    lines.extend(["", "## FN 最大 IoU", ""])
    for key, count in total["false_negative_iou_bands"].items():
        lines.append(
            f"- `{key}`：{count}（{total['false_negative_iou_band_shares'][key]:.2%}）"
        )
    lines.extend(["", "## FP 最大 IoU", ""])
    for key, count in total["official_false_positive_iou_bands"].items():
        lines.append(
            f"- `{key}`：{count}（{total['official_false_positive_iou_band_shares'][key]:.2%}）"
        )
    lines.extend(
        [
            "",
            "## 标注者分歧分解",
            "",
            f"- 共识 FP 预测：{total['consensus_false_positive_instances']} 个，对官方 FP "
            f"贡献 {total['consensus_false_positive_records']} 条记录。",
            f"- 只被部分标注者接受的预测："
            f"{total['partial_annotator_false_positive_instances']} 个，对官方 FP 贡献 "
            f"{total['partial_annotator_false_positive_records']} 条记录。",
            "",
            "## Fold2",
            "",
            f"- 每标注记录 FP：{fold2['false_positives_per_annotator_record']:.4f}；"
            f"每物理图预测：{fold2['predictions_per_physical_image']:.4f}。",
            f"- 共识 FP 预测：{fold2['consensus_false_positive_instances']}；"
            f"部分标注者 FP 记录：{fold2['partial_annotator_false_positive_records']}。",
            "",
            "### Fold2 分站点",
            "",
            "| 站点 | 图像 | PQ | 每标注记录 FP | 每图预测 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for station, row in report["station_summaries"]["2"].items():
        lines.append(
            f"| {station} | {row['physical_images']} | {row['official_pq']:.4f} | "
            f"{row['false_positives_per_annotator_record']:.3f} | "
            f"{row['predictions_per_physical_image']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=600)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--force-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir", default="artifacts/analysis/stage_p3/b0_failure_modes", type=Path
    )
    args = parser.parse_args()
    annotation = args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    config = YoloPostprocessConfig(
        args.confidence, args.min_area, args.max_instances, args.force_one
    )
    manifests = {}
    cache_by_fold = {}
    for cache_dir in args.cache_dir:
        manifest = load_cache_manifest(cache_dir)
        fold = int(manifest["dataset"]["fold"])
        if fold in manifests:
            raise ValueError(f"duplicate cache fold: {fold}")
        validate_annotation_provenance(manifest, annotation)
        validate_search_space(
            manifest, confidences=[config.confidence], max_instances=[config.max_instances]
        )
        manifests[fold] = manifest
        cache_by_fold[fold] = cache_dir
    identity = validate_multifold_manifests(manifests, {0, 1, 2})

    rows = []
    total_images = sum(len(manifest["images"]) for manifest in manifests.values())
    done = 0
    for fold in sorted(manifests):
        manifest = manifests[fold]
        for record in manifest["images"]:
            masks, scores, _ = load_candidate_record(cache_by_fold[fold], record)
            rows.append(analyze_yolo_failure_modes(masks, scores, record, coco, config))
            done += 1
            print(f"failure modes {done}/{total_images}: {record['file_name']}", flush=True)

    payload = {
        "report_type": "yolo_failure_modes_v1",
        "postprocess": config.to_dict(),
        "truth_area_edges": list(TRUTH_AREA_EDGES),
        "prediction_area_edges": list(PREDICTION_AREA_EDGES),
        "experiment_family": identity["experiment_family"],
        "training_signature_sha256": identity["training_signature_sha256"],
        "cache_run_config_sha256_by_fold": {
            str(fold): manifests[fold]["run_config_sha256"] for fold in sorted(manifests)
        },
        **summarize_failure_mode_rows(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "index.json", payload)
    atomic_write_text(args.output_dir / "README.md", _markdown(payload))
    fields = [
        "file_name", "fold", "year", "station", "annotator_batches",
        "prediction_instances", "truth_instances", "hit_pairs", "false_positives",
        "false_negatives", "consensus_false_positive_instances",
        "partial_annotator_false_positive_instances",
        "consensus_false_positive_records",
        "partial_annotator_false_positive_records",
    ]
    atomic_write_csv(
        args.output_dir / "image_metrics.csv",
        fields,
        [{key: row[key] for key in fields} for row in rows],
    )
    print(f"wrote failure-mode report to {args.output_dir}")


if __name__ == "__main__":
    main()
