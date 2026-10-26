#!/usr/bin/env python3
"""Summarize compatible per-fold YOLO error reports without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.instance.yolo_multifold_error_analysis import summarize_multifold_error_reports
from src.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text


def _markdown(report: dict) -> str:
    total = report["global"]
    rates = total["rates"]
    fold2 = report["fold2_diagnostic"]
    lines = [
        "# P2 B0 三 Fold 联合错误分析",
        "",
        "## 固定评估结果",
        "",
        f"- Fold 平均 PQ：`{report['fold_mean_pq']:.9f} ± {report['fold_std_pq']:.9f}`；"
        f"全局合并 PQ：`{total['official_pq']:.9f}`。",
        f"- TP/FP/FN：`{total['hit_pairs']} / {total['false_positives']} / "
        f"{total['false_negatives']}`；物理图/标注记录："
        f"`{total['physical_images']} / {total['annotator_records']}`。",
        f"- 固定后处理：`{report['postprocess']}`。",
        "",
        "## 错误类型（非互斥）",
        "",
        f"- 小实例漏检：`{total['small_instance_misses']}`，占全部 FN "
        f"`{rates['small_miss_share_of_false_negatives']:.2%}`。",
        f"- 大面积共识 FP：`{total['large_false_positive_instances']}`，占共识 FP "
        f"`{rates['large_fp_share_of_consensus_false_positives']:.2%}`。",
        f"- 断裂关系：`{total['fragmented_gt_instances']}`，占标注实例 "
        f"`{rates['fragmented_share_of_truth_instances']:.2%}`。",
        f"- 粘连关系：`{total['merged_prediction_instances']}`，占预测×标注者机会 "
        f"`{rates['merged_share_of_prediction_annotator_opportunities']:.2%}`。",
        f"- 多标注物理图占比：`{rates['multi_annotator_image_share']:.2%}`；其中 "
        f"PQ 极差至少 0.1 的占比：`{rates['unstable_multi_annotator_image_share']:.2%}`。",
        "",
        "这些计数可能在同一图像/实例上重叠，不应相加为 100%。",
        "",
        "## Fold2 额外 FP",
        "",
        f"- Fold2 每标注记录 FP：`{fold2['fp_per_annotator_record']:.4f}`；"
        f"Fold0/1 均值：`{fold2['fold01_mean_fp_per_annotator_record']:.4f}`；"
        f"差值：`{fold2['fp_rate_excess']:+.4f}`。",
        f"- Fold2 每图预测实例：`{fold2['prediction_instances_per_image']:.4f}`；"
        f"Fold0/1 均值：`{fold2['fold01_mean_prediction_instances_per_image']:.4f}`。",
        f"- Fold2 每图大面积共识 FP：`{fold2['large_consensus_fp_per_image']:.4f}`；"
        f"Fold0/1 均值：`{fold2['fold01_mean_large_consensus_fp_per_image']:.4f}`。",
        "",
        "## 候选优先级",
        "",
    ]
    for item in report["candidate_priorities"]:
        verdict = "建议实现" if item["recommended"] else "暂不实现"
        lines.append(
            f"{item['rank']}. `{item['candidate']}` — **{verdict}**：{item['reason']}。"
        )
    lines.extend(["", "## 最差图像", ""])
    for row in report["worst_images"]:
        path = f"fold{row['fold']}/images/{Path(row['file_name']).stem}.png"
        lines.append(
            f"- [{row['file_name']}]({path})：PQ `{row['official_pq']:.4f}`，"
            f"TP/FP/FN `{row['hit_pairs']}/{row['false_positives']}/{row['false_negatives']}`"
        )
    lines.extend(["", "## 最好图像", ""])
    for row in report["best_images"]:
        path = f"fold{row['fold']}/images/{Path(row['file_name']).stem}.png"
        lines.append(
            f"- [{row['file_name']}]({path})：PQ `{row['official_pq']:.4f}`，"
            f"TP/FP/FN `{row['hit_pairs']}/{row['false_positives']}/{row['false_negatives']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-report", action="append", required=True, type=Path)
    parser.add_argument(
        "--output-dir", default="artifacts/analysis/stage_p2/b0_multifold", type=Path
    )
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.fold_report]
    result = summarize_multifold_error_reports(reports)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "index.json", result)
    atomic_write_text(args.output_dir / "README.md", _markdown(result))
    fields = [
        "file_name", "fold", "annotator_batches", "official_pq", "hit_pairs",
        "false_positives", "false_negatives", "prediction_instances",
        "truth_instances_total", "small_instance_misses",
        "large_false_positive_instances", "fragmented_gt_instances",
        "merged_prediction_instances", "annotator_pq_std", "annotator_pq_range",
        "categories",
    ]
    rows = [
        {**{field: row.get(field) for field in fields}, "categories": ";".join(row["categories"])}
        for row in result["per_image"]
    ]
    atomic_write_csv(args.output_dir / "image_metrics.csv", fields, rows)
    print(f"wrote multifold YOLO error summary to {args.output_dir}")


if __name__ == "__main__":
    main()
