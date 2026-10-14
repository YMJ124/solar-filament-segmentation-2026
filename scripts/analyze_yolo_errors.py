#!/usr/bin/env python3
"""Build a per-image YOLO error report and visual atlas from a candidate cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.coco import CocoAnnotations
from src.instance.yolo_cache import (
    load_cache_manifest,
    load_candidate_record,
    validate_annotation_provenance,
    validate_search_space,
)
from src.instance.yolo_error_analysis import (
    analyze_yolo_image,
    render_yolo_error_canvas,
    select_yolo_error_groups,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


GROUP_LABELS = {
    "worst_pq": "PQ 最低",
    "best_pq": "PQ 最高",
    "most_large_false_positives": "大面积假阳性最多",
    "most_small_instance_misses": "小实例漏检最多",
    "most_fragmented": "同一条日珥切分最明显",
    "most_merged": "多条日珥合并最明显",
    "largest_annotator_variation": "标注者间 PQ 波动最大",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--max-instances", type=int, required=True)
    parser.add_argument(
        "--force-one", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--small-instance-area", type=int, default=400)
    parser.add_argument("--large-fp-area", type=int, default=10_000)
    parser.add_argument("--diagnostic-iou", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--display-size", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        default="artifacts/analysis/yolo_errors_fold0",
        type=Path,
    )
    parser.add_argument(
        "--no-verify-cache-hashes",
        action="store_true",
        help="skip per-image SHA256 verification",
    )
    return parser.parse_args()


def aggregate_rows(rows: list[dict]) -> dict:
    hit_pairs = sum(row["hit_pairs"] for row in rows)
    false_positives = sum(row["false_positives"] for row in rows)
    false_negatives = sum(row["false_negatives"] for row in rows)
    hit_iou_sum = float(sum(row["hit_iou_sum"] for row in rows))
    denominator = hit_pairs + 0.5 * false_positives + 0.5 * false_negatives
    return {
        "physical_images": len(rows),
        "annotator_records": sum(row["annotator_batches"] for row in rows),
        "official_pq": hit_iou_sum / denominator if denominator else 0.0,
        "hit_pairs": hit_pairs,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hit_iou_sum": hit_iou_sum,
        "prediction_instances": sum(row["prediction_instances"] for row in rows),
        "truth_instances_total": sum(row["truth_instances_total"] for row in rows),
        "consensus_false_positive_instances": sum(
            row["consensus_false_positive_instances"] for row in rows
        ),
        "large_false_positive_instances": sum(
            row["large_false_positive_instances"] for row in rows
        ),
        "small_instance_misses": sum(row["small_instance_misses"] for row in rows),
        "fragmented_gt_instances": sum(
            row["fragmented_gt_instances"] for row in rows
        ),
        "merged_prediction_instances": sum(
            row["merged_prediction_instances"] for row in rows
        ),
        "mean_annotator_pq_std": float(
            np.mean([row["annotator_pq_std"] for row in rows])
        ),
        "max_annotator_pq_range": max(row["annotator_pq_range"] for row in rows),
    }


def summary_markdown(index: dict) -> str:
    aggregate = index["aggregate"]
    config = index["postprocess"]
    lines = [
        "# YOLO 错误分析",
        "",
        "## 汇总",
        "",
        f"- 官方 PQ：`{aggregate['official_pq']:.6f}`；TP/FP/FN："
        f"`{aggregate['hit_pairs']} / {aggregate['false_positives']} / "
        f"{aggregate['false_negatives']}`。",
        f"- 预测实例 / 标注实例记录：`{aggregate['prediction_instances']} / "
        f"{aggregate['truth_instances_total']}`。",
        f"- 大面积 FP：`{aggregate['large_false_positive_instances']}`；"
        f"小实例漏检：`{aggregate['small_instance_misses']}`。",
        f"- 断裂关系：`{aggregate['fragmented_gt_instances']}`；"
        f"粘连关系：`{aggregate['merged_prediction_instances']}`。",
        f"- 每图标注者 PQ 标准差的均值："
        f"`{aggregate['mean_annotator_pq_std']:.6f}`；最大极差："
        f"`{aggregate['max_annotator_pq_range']:.6f}`。",
        f"- 后处理：`confidence={config['confidence']}, min_area="
        f"{config['min_area']}, max_instances={config['max_instances']}, "
        f"force_one={config['force_one']}`。",
        "",
        "官方 TP/FP/FN 使用 IoU > 0.5，并对每个标注者批次分别计分。断裂/粘连是诊断关系，",
        f"使用 IoU > `{index['diagnostics']['diagnostic_iou']}`，不参与官方 PQ 计算。",
        "",
    ]
    for group, names in index["selection_groups"].items():
        lines.extend([f"## {GROUP_LABELS[group]}", ""])
        for name in names:
            lines.append(f"- [{name}](images/{Path(name).stem}.png)")
        lines.append("")
    lines.extend(
        [
            "## 可视化图例",
            "",
            "四联图依次为原图、fold manifest 指定标注者的 GT 边界、预测实例、",
            "TP/FP/FN。最后一栏中绿色为 TP 预测，红色为 FP 预测，蓝色边界为 FN GT。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = load_cache_manifest(args.cache_dir)
    validate_search_space(
        manifest,
        confidences=[args.confidence],
        max_instances=[args.max_instances],
    )
    config = YoloPostprocessConfig(
        args.confidence, args.min_area, args.max_instances, args.force_one
    )
    config.validate()
    annotation = (
        args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    validate_annotation_provenance(manifest, annotation)
    coco = CocoAnnotations(annotation)
    image_dir = args.data_root / "train/train_images"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, record in enumerate(manifest["images"], start=1):
        masks, scores, _ = load_candidate_record(
            args.cache_dir,
            record,
            verify_hash=not args.no_verify_cache_hashes,
        )
        row, _, _, _ = analyze_yolo_image(
            masks,
            scores,
            record,
            coco,
            config,
            small_instance_area=args.small_instance_area,
            large_false_positive_area=args.large_fp_area,
            diagnostic_iou=args.diagnostic_iou,
        )
        rows.append(row)
        print(
            f"YOLO error analysis {index}/{len(manifest['images'])}: "
            f"{record['file_name']}",
            flush=True,
        )

    aggregate = aggregate_rows(rows)
    groups = select_yolo_error_groups(rows, top_k=args.top_k)
    selected_names = list(
        dict.fromkeys(name for names in groups.values() for name in names)
    )
    entry_by_name = {str(row["file_name"]): row for row in manifest["images"]}
    metric_by_name = {str(row["file_name"]): row for row in rows}
    for index, file_name in enumerate(selected_names, start=1):
        record = entry_by_name[file_name]
        masks, scores, _ = load_candidate_record(
            args.cache_dir,
            record,
            verify_hash=not args.no_verify_cache_hashes,
        )
        repeated, labels, truth_masks, matrix = analyze_yolo_image(
            masks,
            scores,
            record,
            coco,
            config,
            small_instance_area=args.small_instance_area,
            large_false_positive_area=args.large_fp_area,
            diagnostic_iou=args.diagnostic_iou,
        )
        if repeated != metric_by_name[file_name]:
            raise RuntimeError(f"non-deterministic YOLO analysis for {file_name}")
        with Image.open(image_dir / file_name) as source:
            image = np.asarray(source.convert("L"))
        render_yolo_error_canvas(
            image,
            labels,
            truth_masks,
            matrix,
            repeated,
            images_dir / f"{Path(file_name).stem}.png",
            display_size=args.display_size,
        )
        print(f"YOLO visualization {index}/{len(selected_names)}: {file_name}")

    category_counts = Counter(category for row in rows for category in row["categories"])
    index_payload = {
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_run_config_sha256": manifest["run_config_sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "fold": manifest["dataset"].get("fold"),
        "postprocess": config.to_dict(),
        "diagnostics": {
            "small_instance_area": args.small_instance_area,
            "large_false_positive_area": args.large_fp_area,
            "diagnostic_iou": args.diagnostic_iou,
            "split_merge_are_diagnostic_not_official": True,
        },
        "aggregate": aggregate,
        "category_counts": dict(sorted(category_counts.items())),
        "selection_groups": groups,
        "visualized_images": selected_names,
        "per_image": rows,
    }
    atomic_write_json(args.output_dir / "index.json", index_payload)

    image_fields = [
        "file_name",
        "fold",
        "annotator_batches",
        "official_pq",
        "hit_pairs",
        "false_positives",
        "false_negatives",
        "prediction_instances",
        "truth_instances_total",
        "truth_instances_mean",
        "consensus_false_positive_instances",
        "large_false_positive_instances",
        "largest_consensus_false_positive_area",
        "small_instance_misses",
        "fragmented_gt_instances",
        "merged_prediction_instances",
        "annotator_pq_mean",
        "annotator_pq_std",
        "annotator_pq_min",
        "annotator_pq_max",
        "annotator_pq_range",
        "categories",
    ]
    csv_rows = [
        {
            **{key: row.get(key) for key in image_fields},
            "categories": ";".join(row["categories"]),
        }
        for row in rows
    ]
    atomic_write_csv(args.output_dir / "image_metrics.csv", image_fields, csv_rows)
    annotator_fields = [
        "file_name",
        "image_id",
        "truth_instances",
        "official_pq",
        "hit_pairs",
        "false_positives",
        "false_negatives",
        "small_instance_misses",
        "fragmented_gt_instances",
        "merged_prediction_instances",
    ]
    annotator_rows = [
        {"file_name": row["file_name"], **annotator}
        for row in rows
        for annotator in row["annotators"]
    ]
    atomic_write_csv(
        args.output_dir / "annotator_metrics.csv", annotator_fields, annotator_rows
    )
    atomic_write_text(args.output_dir / "README.md", summary_markdown(index_payload))
    print(f"wrote YOLO error atlas to {args.output_dir}")


if __name__ == "__main__":
    main()
