#!/usr/bin/env python3
"""Compare P4-T1 and P4-I1 residual errors from existing three-fold caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.coco import CocoAnnotations
from src.instance.p4_diagnostics import (
    analyze_representation_image,
    pair_residual_instance_rows,
    summarize_cap_pressure,
    summarize_paired_residual_rows,
)
from src.instance.yolo_cache import (
    load_cache_manifest,
    load_candidate_record,
    sha256_file,
    validate_annotation_provenance,
    validate_search_space,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig, predictions_for_config
from src.instance.yolo_multifold import validate_multifold_manifests
from src.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


def _load_family(
    paths: list[Path], annotation: Path, config: YoloPostprocessConfig
) -> tuple[dict[int, dict], dict[int, Path], dict]:
    manifests: dict[int, dict] = {}
    cache_dirs: dict[int, Path] = {}
    for path in paths:
        manifest = load_cache_manifest(path)
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
        cache_dirs[fold] = path
    identity = validate_multifold_manifests(manifests, {0, 1, 2})
    return manifests, cache_dirs, identity


def _cap_row(
    masks: np.ndarray,
    scores: np.ndarray,
    config: YoloPostprocessConfig,
    *,
    cache_max_candidates: int,
) -> dict[str, int]:
    order = np.argsort(-scores, kind="stable")
    qualified = order[scores[order] >= config.confidence]
    beyond = qualified[config.max_instances :]
    raw_areas = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.zeros(0)
    _, predictions = predictions_for_config(masks, scores, config)
    return {
        "candidate_count": len(masks),
        "cache_max_candidates": int(cache_max_candidates),
        "candidates_above_confidence": len(qualified),
        "raw_area_eligible_beyond_cap": int(
            sum(raw_areas[index] >= config.min_area for index in beyond)
        ),
        "prediction_instances": len(predictions),
    }


def _markdown(payload: dict) -> str:
    paired = payload["paired_residuals"]["global"]
    cap = payload["cap_pressure"]
    lines = [
        "# P4 T1 vs I1 paired residual audit",
        "",
        "本报告只读取已有 1536 三折候选缓存，不重新推理。GT 以物理图、标注批次和",
        "annotation ID 配对；`matched` 使用官方严格 `IoU > 0.5` 判定。",
        "",
        "## GT 转移",
        "",
        f"- GT 实例记录：{paired['instances']}。",
        f"- I1/T1 漏检：{paired['i1_misses']} / {paired['t1_misses']} "
        f"（变化 {paired['miss_delta']:+d}）。",
        f"- T1 救回/退化/净救回："
        f"{paired['transitions'].get('recovered_by_t1', 0)} / "
        f"{paired['transitions'].get('regressed_in_t1', 0)} / "
        f"{paired['net_recovered']}。",
        "",
        "## max_instances=20 压力",
        "",
        "| 模型 | confidence 池被截断图 | 截断率 | cap 外仍有原始面积合格候选图 | 输出达到 20 图 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("i1", "t1"):
        row = cap[name]
        lines.append(
            f"| {name.upper()} | {row['confidence_pool_truncated_images']} | "
            f"{row['confidence_pool_truncated_rate']:.2%} | "
            f"{row['raw_area_eligible_beyond_cap_images']} | "
            f"{row['output_at_cap_images']} |"
        )
    lines.extend(["", "## 1536 等效宽度分层", "", "| 分层 | 实例 | I1 FN | T1 FN | 净救回 |", "|---|---:|---:|---:|---:|"])
    for name, row in payload["paired_residuals"]["by_representation_band_1536"].items():
        lines.append(
            f"| `{name}` | {row['instances']} | {row['i1_misses']} | "
            f"{row['t1_misses']} | {row['net_recovered']:+d} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--t1-cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=600)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--force-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir", default="artifacts/analysis/stage_p4/t1_vs_i1_residuals", type=Path
    )
    args = parser.parse_args()

    annotation = args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    config = YoloPostprocessConfig(
        args.confidence, args.min_area, args.max_instances, args.force_one
    )
    i1_manifests, i1_dirs, i1_identity = _load_family(
        args.i1_cache_dir, annotation, config
    )
    t1_manifests, t1_dirs, t1_identity = _load_family(
        args.t1_cache_dir, annotation, config
    )
    if i1_identity["inference_signature"] != t1_identity["inference_signature"]:
        raise ValueError("I1 and T1 caches use different inference configurations")
    if int(i1_identity["inference_signature"][0]) != 1536:
        raise ValueError("paired P4 residual audit requires 1536 inference caches")

    i1_instances: list[dict] = []
    t1_instances: list[dict] = []
    image_rows: list[dict] = []
    total = sum(len(manifest["images"]) for manifest in i1_manifests.values())
    completed = 0
    for fold in sorted(i1_manifests):
        i1_entries = {str(row["file_name"]): row for row in i1_manifests[fold]["images"]}
        t1_entries = {str(row["file_name"]): row for row in t1_manifests[fold]["images"]}
        if set(i1_entries) != set(t1_entries):
            raise ValueError(f"I1/T1 physical-image mismatch in Fold{fold}")
        for file_name in sorted(i1_entries):
            i1_entry, t1_entry = i1_entries[file_name], t1_entries[file_name]
            for key in ("fold", "height", "width"):
                if i1_entry[key] != t1_entry[key]:
                    raise ValueError(f"I1/T1 {key} mismatch for {file_name}")
            i1_masks, i1_scores, _ = load_candidate_record(i1_dirs[fold], i1_entry)
            t1_masks, t1_scores, _ = load_candidate_record(t1_dirs[fold], t1_entry)
            i1_instances.extend(
                analyze_representation_image(
                    i1_masks, i1_scores, i1_entry, coco, config, inference_sizes=(1536,)
                )
            )
            t1_instances.extend(
                analyze_representation_image(
                    t1_masks, t1_scores, t1_entry, coco, config, inference_sizes=(1536,)
                )
            )
            row = {"file_name": file_name, "fold": fold}
            for prefix, masks, scores, manifest in (
                ("i1", i1_masks, i1_scores, i1_manifests[fold]),
                ("t1", t1_masks, t1_scores, t1_manifests[fold]),
            ):
                for key, value in _cap_row(
                    masks,
                    scores,
                    config,
                    cache_max_candidates=int(manifest["inference"]["max_candidates"]),
                ).items():
                    row[f"{prefix}_{key}"] = value
            image_rows.append(row)
            completed += 1
            print(f"paired P4 residual audit {completed}/{total}: {file_name}", flush=True)

    paired_rows = pair_residual_instance_rows(i1_instances, t1_instances)
    payload = {
        "report_type": "p4_paired_residual_audit_v1",
        "postprocess": config.to_dict(),
        "annotation_sha256": sha256_file(annotation),
        "i1_identity": i1_identity,
        "t1_identity": t1_identity,
        "cache_manifest_sha256": {
            "i1": {
                str(fold): sha256_file(i1_dirs[fold] / "manifest.json")
                for fold in sorted(i1_dirs)
            },
            "t1": {
                str(fold): sha256_file(t1_dirs[fold] / "manifest.json")
                for fold in sorted(t1_dirs)
            },
        },
        "paired_residuals": summarize_paired_residual_rows(paired_rows),
        "cap_pressure": summarize_cap_pressure(
            image_rows, max_instances=config.max_instances
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "index.json", payload)
    atomic_write_csv(args.output_dir / "instance_transitions.csv", list(paired_rows[0]), paired_rows)
    atomic_write_csv(args.output_dir / "image_cap_pressure.csv", list(image_rows[0]), image_rows)
    atomic_write_text(args.output_dir / "README.md", _markdown(payload))
    print(f"wrote paired P4 residual audit to {args.output_dir}")


if __name__ == "__main__":
    main()
