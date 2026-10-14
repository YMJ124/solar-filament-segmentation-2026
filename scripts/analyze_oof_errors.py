#!/usr/bin/env python3
"""Build the Fold-0 OOF error atlas using the selected post-process config."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.io_utils import atomic_write_json
from src.legacy_semantic import add_legacy_semantic_flag, require_legacy_semantic_opt_in
from src.oof_error_analysis import (
    CATEGORY_LABELS,
    analyze_oof_image,
    category_counts,
    render_error_canvas,
    select_error_groups,
)
from src.oof_postprocess import (
    PostprocessConfig,
    load_oof_probability,
    validate_oof_directory,
)


GROUP_LABELS = {
    "lowest_pq": "PQ 最低 10 张",
    "most_false_positives": "FP 最多 10 张",
    "most_false_negatives": "FN 最多 10 张",
    "most_fragmented": "断裂最明显 10 张",
    "best_pq": "表现最好 5 张",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_legacy_semantic_flag(parser)
    parser.add_argument(
        "--oof-dir",
        default="artifacts/runs/global_resnet34_fold0/oof_best",
        type=Path,
    )
    parser.add_argument(
        "--best-config",
        default="artifacts/postprocess/fold0/best_config.yaml",
        type=Path,
    )
    parser.add_argument(
        "--stage1-report",
        default="artifacts/postprocess/fold0/best_report.json",
        type=Path,
    )
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument(
        "--fold-manifest", default="data/manifests/train_folds.csv", type=Path
    )
    parser.add_argument(
        "--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis/errors_fold0",
        type=Path,
    )
    parser.add_argument("--display-size", default=512, type=int)
    parser.add_argument("--tile-size", default=256, type=int)
    parser.add_argument("--overlap", default=0.25, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def load_config(path: Path) -> PostprocessConfig:
    payload = yaml.safe_load(path.read_text())
    values = payload["selected_config"]
    return PostprocessConfig(
        threshold=float(values["threshold"]),
        min_area=int(values["min_area"]),
        closing_iterations=int(values["closing_iterations"]),
        max_instances=int(values["max_instances"]),
        max_area_fraction=float(values.get("max_area_fraction", 0.35)),
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hit_pairs = sum(row["hit_pairs"] for row in rows)
    false_positives = sum(row["false_positives"] for row in rows)
    false_negatives = sum(row["false_negatives"] for row in rows)
    hit_iou_sum = float(sum(row["hit_iou_sum"] for row in rows))
    denominator = hit_pairs + 0.5 * false_positives + 0.5 * false_negatives
    return {
        "physical_images": len(rows),
        "annotator_records": sum(row["annotator_batches"] for row in rows),
        "hit_pairs": hit_pairs,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hit_iou_sum": hit_iou_sum,
        "official_pq": hit_iou_sum / denominator if denominator else 0.0,
        "prediction_instances": sum(row["prediction_instances"] for row in rows),
        "fragmented_gt_instances": sum(
            row["fragmented_gt_instances"] for row in rows
        ),
        "merged_prediction_instances": sum(
            row["merged_prediction_instances"] for row in rows
        ),
    }


def stage_decision(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    stage1_report: dict[str, Any],
) -> dict[str, Any]:
    tile_threshold = max(10, math.ceil(0.10 * len(rows)))
    tile_significant = counts["tile_seam"] >= tile_threshold
    images_with_merge = sum(row["merged_prediction_instances"] > 0 for row in rows)
    many_to_one = stage1_report["best_report"][
        "many_to_one_prediction_distribution"
    ]
    merged_total = int(many_to_one["multiple_partners"])
    prediction_relations = int(many_to_one["total_instances"])
    merged_ratio = merged_total / prediction_relations if prediction_relations else 0.0
    model_c_triggered = bool(
        merged_ratio >= 0.01 or merged_total >= 20 or images_with_merge >= 10
    )
    if tile_significant:
        next_action = "compare_tile_context_before_model_b"
        rationale = (
            "tile 接缝启发式命中达到显著阈值；先比较 tile 256/512 与 overlap 0.25/0.5。"
        )
    else:
        next_action = "proceed_to_model_b"
        rationale = (
            "tile 接缝未达到显著阈值；漏检、断裂和背景误检仍是主要诊断方向，进入 Model B。"
        )
    return {
        "next_action": next_action,
        "rationale": rationale,
        "tile_seam_significant": tile_significant,
        "tile_seam_images": counts["tile_seam"],
        "tile_seam_significance_threshold": tile_threshold,
        "images_with_false_negatives": sum(row["false_negatives"] > 0 for row in rows),
        "images_with_false_positives": sum(row["false_positives"] > 0 for row in rows),
        "images_with_fragmentation": counts["structure_middle_break"],
        "model_c_triggered": model_c_triggered,
        "model_c_evidence": {
            "merged_prediction_relations": merged_total,
            "prediction_relations": prediction_relations,
            "merged_ratio": merged_ratio,
            "physical_images_with_merge": images_with_merge,
            "thresholds": {
                "ratio": 0.01,
                "absolute": 20,
                "physical_images": 10,
            },
        },
    }


def build_summary(index: dict[str, Any]) -> str:
    aggregate = index["aggregate"]
    baseline = index["baseline_comparison"]
    decision = index["decision"]
    lines = [
        "# Fold 0 OOF 错误分析",
        "",
        "## 结论",
        "",
        f"- 阶段决策：`{decision['next_action']}`。{decision['rationale']}",
        f"- 调优后 PQ：`{aggregate['official_pq']:.6f}`；TP/FP/FN："
        f"`{aggregate['hit_pairs']} / {aggregate['false_positives']} / {aggregate['false_negatives']}`。",
        f"- 相对冻结基线：PQ `{baseline['pq_delta']:+.6f}`，TP "
        f"`{baseline['tp_delta']:+d}`，FP `{baseline['fp_delta']:+d}`，FN "
        f"`{baseline['fn_delta']:+d}`。",
        f"- tile 接缝启发式命中 `{decision['tile_seam_images']}` 张；显著阈值为 "
        f"`{decision['tile_seam_significance_threshold']}` 张。",
        f"- Model C 条件：`{'触发' if decision['model_c_triggered'] else '未触发'}`。",
        "",
        "官方 PQ 排名和汇总始终对每个 annotator batch 分别评分；图中的 GT/TP/FP/FN "
        "定位使用训练 fold manifest 指定的 medoid annotator，仅用于可视化，不替代官方评分。",
        "",
        "## 错误分类",
        "",
        "| 分类 | 全部 138 张 | 入选可视化 |",
        "|---|---:|---:|",
    ]
    for key, label in CATEGORY_LABELS.items():
        lines.append(
            f"| {label} | {index['category_counts_all'][key]} | "
            f"{index['category_counts_selected'][key]} |"
        )
    lines.extend(["", "分类是确定性的诊断启发式，不是新增人工标注。", ""])
    for group, names in index["selection_groups"].items():
        lines.extend([f"## {GROUP_LABELS[group]}", ""])
        for name in names:
            stem = Path(name).stem
            lines.append(f"- [{name}](images/{stem}.png)")
        lines.append("")
    lines.extend(
        [
            "## 可视化图例",
            "",
            "每张五联图依次为：原图、medoid GT 实例边界、概率热力图、预测实例、"
            "TP/FP/FN 叠加图。最后一栏中绿色为 TP、红色为 FP、蓝色边界为 FN。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    require_legacy_semantic_opt_in(args.allow_legacy_semantic)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; pass --overwrite to replace files"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.best_config)
    stage1_report = json.loads(args.stage1_report.read_text())
    annotation_file = (
        args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    image_dir = args.data_root / "train/train_images"
    coco = CocoAnnotations(annotation_file)
    all_records = load_fold_manifest(args.fold_manifest, annotation_file)
    records = [row for row in all_records if int(row["fold"]) == args.fold]
    validate_oof_directory(args.oof_dir, records, args.fold)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        file_name = str(record["file_name"])
        probability = load_oof_probability(
            args.oof_dir / f"{Path(file_name).stem}.npz", record
        )
        with Image.open(image_dir / file_name) as source:
            image = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
        row, _, _, _ = analyze_oof_image(
            probability,
            image,
            record,
            coco,
            config,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        rows.append(row)
        print(f"OOF error analysis {index}/{len(records)}: {file_name}", flush=True)

    aggregate = aggregate_rows(rows)
    expected = stage1_report["best_report"]
    verification = {
        "hit_pairs_match": aggregate["hit_pairs"] == expected["hit_pairs"],
        "false_positives_match": (
            aggregate["false_positives"] == expected["false_positives"]
        ),
        "false_negatives_match": aggregate["false_negatives"] == expected["false_negatives"],
        "official_pq_match": abs(aggregate["official_pq"] - expected["official_pq"])
        <= 1e-12,
        "tolerance": 1e-12,
    }
    verification["passed"] = all(
        value is True for key, value in verification.items() if key != "tolerance"
    )
    if not verification["passed"]:
        raise RuntimeError(f"per-image aggregate does not reproduce stage 1: {verification}")

    groups = select_error_groups(rows)
    selected_names = list(
        dict.fromkeys(name for group in groups.values() for name in group)
    )
    row_by_name = {row["file_name"]: row for row in rows}
    record_by_name = {str(record["file_name"]): record for record in records}
    for index, file_name in enumerate(selected_names, start=1):
        record = record_by_name[file_name]
        probability = load_oof_probability(
            args.oof_dir / f"{Path(file_name).stem}.npz", record
        )
        with Image.open(image_dir / file_name) as source:
            image = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
        repeated_row, labels, truth_masks, medoid_matrix = analyze_oof_image(
            probability,
            image,
            record,
            coco,
            config,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        if repeated_row != row_by_name[file_name]:
            raise RuntimeError(f"non-deterministic per-image analysis for {file_name}")
        output_path = images_dir / f"{Path(file_name).stem}.png"
        render_error_canvas(
            image,
            probability,
            labels,
            truth_masks,
            medoid_matrix,
            repeated_row,
            output_path,
            display_size=args.display_size,
        )
        row_by_name[file_name]["visualization"] = str(
            output_path.relative_to(args.output_dir)
        )
        print(f"render error atlas {index}/{len(selected_names)}: {file_name}", flush=True)

    counts_all = category_counts(rows)
    selected_rows = [row_by_name[name] for name in selected_names]
    counts_selected = category_counts(selected_rows)
    baseline_manifest = json.loads((args.oof_dir / "manifest.json").read_text())
    decision = stage_decision(rows, counts_all, stage1_report)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "config": config.to_dict(),
        "metric": {
            "name": "official_pq_v6",
            "iou_rule": "strictly_greater_than_0.5",
            "scoring_annotator_policy": "all annotator batches separately",
            "visualization_gt_policy": "fold-manifest medoid annotator",
        },
        "inputs": {
            "oof_dir": str(args.oof_dir),
            "oof_manifest_sha256": sha256_file(args.oof_dir / "manifest.json"),
            "best_config": str(args.best_config),
            "best_config_sha256": sha256_file(args.best_config),
            "stage1_report": str(args.stage1_report),
            "stage1_report_sha256": sha256_file(args.stage1_report),
            "fold_manifest": str(args.fold_manifest),
            "fold_manifest_sha256": sha256_file(args.fold_manifest),
            "annotation_file": str(annotation_file),
            "annotation_sha256": sha256_file(annotation_file),
            "test_images_read": 0,
        },
        "aggregate": aggregate,
        "stage1_reproduction": verification,
        "baseline_comparison": {
            "baseline_official_pq": baseline_manifest["official_pq"],
            "pq_delta": aggregate["official_pq"] - baseline_manifest["official_pq"],
            "tp_delta": aggregate["hit_pairs"] - baseline_manifest["hit_pairs"],
            "fp_delta": aggregate["false_positives"]
            - baseline_manifest["false_positives"],
            "fn_delta": aggregate["false_negatives"]
            - baseline_manifest["false_negatives"],
        },
        "category_definitions": {
            key: {"label": label} for key, label in CATEGORY_LABELS.items()
        },
        "category_counts_all": counts_all,
        "category_counts_selected": counts_selected,
        "selection_groups": groups,
        "selected_unique_images": selected_names,
        "selected_unique_count": len(selected_names),
        "decision": decision,
        "images": sorted(rows, key=lambda row: row["file_name"]),
    }
    atomic_write_json(args.output_dir / "index.json", payload)
    atomic_write_text(args.output_dir / "summary.md", build_summary(payload))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "selected_unique_images": len(selected_names),
                "category_counts": counts_all,
                "decision": decision,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
