#!/usr/bin/env python3
"""Audit thin-structure representation and archived P3 sampling shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.instance.p4_diagnostics import (
    analyze_representation_image,
    physical_feature_rows,
    summarize_p3_sampling_shift,
    summarize_representation_rows,
)
from src.instance.yolo_cache import (
    load_cache_manifest,
    load_candidate_record,
    sha256_file,
    validate_annotation_provenance,
    validate_search_space,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_multifold import validate_multifold_manifests
from src.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


def _markdown(payload: dict) -> str:
    representation = payload["representation"]
    lines = [
        "# P4 readiness audit",
        "",
        "本报告使用 B0 已有三 Fold 缓存，不重新执行 GPU 推理。等效宽度优先按",
        "`polygon area / spine length` 计算；缺失 spine 时使用 bbox 短边。",
        "",
        "## 1024 输入下的 stride-4 表示分层",
        "",
        "| 原型宽度 | 实例 | 漏检 | 漏检率 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in representation["by_representation_band"].items():
        lines.append(
            f"| `{name}` | {row['instances']} | {row['misses']} | {row['miss_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## P3 静态重复的预算与分布偏移",
            "",
            "| 验证 Fold | 物理训练图 | P3 样本 | 步数倍率 | 重复物理图 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, row in payload["p3_sampling_shift"].items():
        lines.append(
            f"| {fold} | {row['physical_train_images']} | {row['p3_train_samples']} | "
            f"{row['optimizer_step_multiplier']:.4f} | {row['repeated_physical_images']} |"
        )
    lines.extend(
        [
            "",
            "P4-T1 将保持每 epoch 抽样数等于物理训练图数，并在每个站点内部",
            "提高中等面积实例丰富图的采样概率。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument(
        "--fold-manifest", default="data/manifests/train_folds.csv", type=Path
    )
    parser.add_argument(
        "--p3-sampling-manifest", action="append", required=True, type=Path
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=600)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument(
        "--force-one", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--output-dir", default="artifacts/analysis/stage_p4/readiness", type=Path
    )
    args = parser.parse_args()

    annotation = args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    config = YoloPostprocessConfig(
        args.confidence, args.min_area, args.max_instances, args.force_one
    )
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
        manifests[fold] = manifest
        cache_by_fold[fold] = cache_dir
    identity = validate_multifold_manifests(manifests, {0, 1, 2})

    instance_rows = []
    total_images = sum(len(manifest["images"]) for manifest in manifests.values())
    completed = 0
    for fold in sorted(manifests):
        for record in manifests[fold]["images"]:
            masks, scores, _ = load_candidate_record(cache_by_fold[fold], record)
            instance_rows.extend(
                analyze_representation_image(masks, scores, record, coco, config)
            )
            completed += 1
            print(
                f"P4 representation audit {completed}/{total_images}: {record['file_name']}",
                flush=True,
            )

    fold_records = load_fold_manifest(args.fold_manifest, annotation)
    feature_rows = physical_feature_rows(fold_records, coco)
    p3_manifests: dict[int, dict] = {}
    for path in args.p3_sampling_manifest:
        payload = json.loads(path.read_text())
        validation_fold = next(
            (
                fold
                for fold in (0, 1, 2)
                if f"fold{fold}" in str(path)
            ),
            None,
        )
        if validation_fold is None:
            raise ValueError(f"cannot infer validation Fold from path: {path}")
        p3_manifests[validation_fold] = payload
    if set(p3_manifests) != {0, 1, 2}:
        raise ValueError("P3 sampling manifests must cover Fold0/1/2 exactly")

    payload = {
        "report_type": "p4_readiness_v1",
        "experiment_family": identity["experiment_family"],
        "training_signature_sha256": identity["training_signature_sha256"],
        "postprocess": config.to_dict(),
        "annotation_sha256": sha256_file(annotation),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
        "cache_run_config_sha256_by_fold": {
            str(fold): manifests[fold]["run_config_sha256"] for fold in sorted(manifests)
        },
        "representation": summarize_representation_rows(instance_rows),
        "p3_sampling_shift": summarize_p3_sampling_shift(
            feature_rows, p3_manifests
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "index.json", payload)
    atomic_write_csv(
        args.output_dir / "instance_metrics.csv",
        list(instance_rows[0]),
        instance_rows,
    )
    atomic_write_csv(
        args.output_dir / "physical_image_features.csv",
        list(feature_rows[0]),
        feature_rows,
    )
    atomic_write_text(args.output_dir / "README.md", _markdown(payload))
    print(f"wrote P4 readiness audit to {args.output_dir}")


if __name__ == "__main__":
    main()
