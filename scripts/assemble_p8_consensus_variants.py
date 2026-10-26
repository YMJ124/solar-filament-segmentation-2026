#!/usr/bin/env python3
"""Assemble conservative P8 variants from immutable completed member caches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.instance.p8_consensus import compose_exclusive_consensus
from src.instance.yolo_cache import sha256_file
from src.io_utils import atomic_write_json
from src.metrics.pq import pairwise_rle_iou
from src.submission.rle import (
    decode_rle,
    label_map_to_rles,
    rle_counts,
    rle_from_counts,
    write_submission,
)
from src.submission.validate import audit_submission_rows


def _load_submission(path: Path) -> dict[str, list[dict[str, str]]]:
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"unexpected submission columns: {path}")
        for row in reader:
            image_id = str(row["filament_id"]).rsplit("_", 1)[0]
            by_image[image_id].append(dict(row))
    return dict(by_image)


def _rows_from_rles(image_id: str, rles: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "filament_id": f"{image_id}_{index}",
            "segmentation_rle": rle_counts(rle),
        }
        for index, rle in enumerate(rles, start=1)
    ]


def _submission_overlap(
    reference: dict[str, list[dict[str, str]]],
    candidate: dict[str, list[dict[str, str]]],
) -> dict:
    matched_reference = 0
    novel_candidate = 0
    best_reference = []
    best_candidate = []
    images_more = images_less = images_equal = 0
    for image_id in sorted(reference):
        reference_rles = [
            rle_from_counts(row["segmentation_rle"])
            for row in reference[image_id]
        ]
        candidate_rles = [
            rle_from_counts(row["segmentation_rle"])
            for row in candidate[image_id]
        ]
        matrix = pairwise_rle_iou(reference_rles, candidate_rles)
        reference_best = (
            matrix.max(axis=1) if len(reference_rles) and len(candidate_rles)
            else np.zeros(len(reference_rles), dtype=np.float64)
        )
        candidate_best = (
            matrix.max(axis=0) if len(reference_rles) and len(candidate_rles)
            else np.zeros(len(candidate_rles), dtype=np.float64)
        )
        matched_reference += int((reference_best > 0.5).sum())
        novel_candidate += int((candidate_best <= 0.5).sum())
        best_reference.extend(reference_best.tolist())
        best_candidate.extend(candidate_best.tolist())
        delta = len(candidate_rles) - len(reference_rles)
        images_more += int(delta > 0)
        images_less += int(delta < 0)
        images_equal += int(delta == 0)
    reference_instances = sum(map(len, reference.values()))
    candidate_instances = sum(map(len, candidate.values()))
    return {
        "reference_instances": reference_instances,
        "candidate_instances": candidate_instances,
        "instance_delta": candidate_instances - reference_instances,
        "matched_reference_iou_gt_0_5": matched_reference,
        "candidate_novel_or_shifted": novel_candidate,
        "mean_best_iou_reference_to_candidate": float(np.mean(best_reference)),
        "mean_best_iou_candidate_to_reference": float(np.mean(best_candidate)),
        "images_candidate_more": images_more,
        "images_candidate_less": images_less,
        "images_equal": images_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=Path(
            "artifacts/submissions/"
            "p8_e1_threefold_4tta_support2_maxarea_1536.audit.json"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "artifacts/instance/stage_p8/p8_e1_threefold_4tta_consensus/"
            "member_cache/manifest.json"
        ),
    )
    parser.add_argument(
        "--p5-submission",
        type=Path,
        default=Path("artifacts/submissions/p5_c2_t1_fold0_1536.csv"),
    )
    parser.add_argument(
        "--p6-submission",
        type=Path,
        default=Path("artifacts/submissions/p6_c2_full_fit_1536.csv"),
    )
    parser.add_argument(
        "--support3-output",
        type=Path,
        default=Path("artifacts/submissions/p8_e2_threefold_unanimous_1536.csv"),
    )
    parser.add_argument(
        "--anchored-output",
        type=Path,
        default=Path(
            "artifacts/submissions/"
            "p8_e3_p5_plus_threefold_unanimous_novel_1536.csv"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/submissions/p8_conservative_variants.audit.json"),
    )
    parser.add_argument(
        "--four_model_output",
        type=Path,
        default=Path(
            "artifacts/submissions/"
            "p8_e4_p5_plus_threefold_unanimous_p6_agreement_1536.csv"
        ),
    )
    args = parser.parse_args()

    source_audit = json.loads(args.source_audit.read_text())
    source_manifest = json.loads(args.source_manifest.read_text())
    if not source_audit["summary"]["complete"] or not source_manifest["complete"]:
        raise ValueError("P8 source inference is incomplete")
    if source_audit["config_sha256"] != source_manifest["config_sha256"]:
        raise ValueError("P8 source audit/manifest config mismatch")
    if source_audit["runtime_sha256"] != source_manifest["runtime_sha256"]:
        raise ValueError("P8 source audit/manifest runtime mismatch")
    if source_audit["cache_manifest_sha256"] != sha256_file(args.source_manifest):
        raise ValueError("P8 source cache manifest hash mismatch")

    entries = {}
    for row in source_manifest["entries"]:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"P8 member cache hash mismatch: {path}")
        entries[(str(row["file_name"]), int(row["fold"]))] = row
    folds = [int(value) for value in source_manifest["members"]]
    p5 = _load_submission(args.p5_submission)
    p6 = _load_submission(args.p6_submission)
    image_reports = {row["image_id"]: row for row in source_audit["images"]}
    expected_ids = set(image_reports)
    if set(p5) != expected_ids:
        raise ValueError("P5 and P8 image coverage differs")
    if set(p6) != expected_ids:
        raise ValueError("P6 and P8 image coverage differs")

    support3_rows = []
    anchored_rows = []
    support3_by_image = {}
    anchored_by_image = {}
    support3_fallback_images = []
    support3_clusters = 0
    support3_dropped_by_deoverlap = 0
    anchored_novel_clusters = 0
    anchored_added_instances = 0
    anchored_area_drops = 0
    per_image = []
    for image_id in sorted(expected_ids):
        image_name = f"{image_id}.jpeg"
        records = []
        for fold in folds:
            entry = entries[(image_name, fold)]
            payload = json.loads(Path(entry["path"]).read_text())
            if int(payload["candidate_count"]) != int(entry["candidate_count"]):
                raise ValueError(f"candidate count mismatch: {entry['path']}")
            records.extend(payload["candidates"])
        clusters = image_reports[image_id]["cluster_audit"]
        selected = []
        for cluster in clusters:
            if int(cluster["fold_support"]) != len(folds):
                continue
            record = dict(records[int(cluster["representative_index"])])
            record.update(cluster)
            selected.append(record)
        selected.sort(
            key=lambda row: (
                -int(row["fold_support"]),
                -float(row["cluster_score"]),
                -int(row["area"]),
            )
        )
        support3_clusters += len(selected)
        _, unanimous_rles, unanimous_meta = compose_exclusive_consensus(
            selected,
            height=2048,
            width=2048,
            min_area=600,
            max_instances=20,
        )
        support3_dropped_by_deoverlap += len(selected) - len(unanimous_rles)
        if unanimous_rles:
            image_support3_rows = _rows_from_rles(image_id, unanimous_rles)
        else:
            image_support3_rows = p5[image_id]
            support3_fallback_images.append(image_id)
        support3_rows.extend(image_support3_rows)
        support3_by_image[image_id] = image_support3_rows

        base_rles = [
            rle_from_counts(row["segmentation_rle"]) for row in p5[image_id]
        ]
        selected_rles = [
            rle_from_counts(row["segmentation_rle"]) for row in selected
        ]
        overlap = pairwise_rle_iou(base_rles, selected_rles)
        selected_best = (
            overlap.max(axis=0)
            if len(base_rles) and len(selected_rles)
            else np.zeros(len(selected_rles), dtype=np.float64)
        )
        novel = [
            row for row, best in zip(selected, selected_best) if float(best) <= 0.5
        ]
        anchored_novel_clusters += len(novel)
        labels = np.zeros((2048, 2048), dtype=np.int32)
        for index, row in enumerate(p5[image_id], start=1):
            mask = decode_rle(row["segmentation_rle"])
            if np.any(labels[mask] != 0):
                raise RuntimeError(f"P5 source masks overlap: {image_id}")
            labels[mask] = index
        added = 0
        for record in novel:
            if int(labels.max()) >= 20:
                break
            mask = decode_rle(record["segmentation_rle"])
            available = mask & (labels == 0)
            if int(available.sum()) < 600:
                anchored_area_drops += 1
                continue
            labels[available] = int(labels.max()) + 1
            added += 1
        anchored_added_instances += added
        anchored_rles = label_map_to_rles(labels)
        image_anchored_rows = _rows_from_rles(image_id, anchored_rles)
        anchored_rows.extend(image_anchored_rows)
        anchored_by_image[image_id] = image_anchored_rows
        per_image.append(
            {
                "image_id": image_id,
                "p5_instances": len(p5[image_id]),
                "support3_clusters": len(selected),
                "support3_instances": len(image_support3_rows),
                "support3_fallback": not unanimous_rles,
                "support3_meta": unanimous_meta,
                "p5_novel_support3_clusters": len(novel),
                "p5_anchored_added_instances": added,
                "p5_anchored_instances": len(image_anchored_rows),
            }
        )

    support3_summary = audit_submission_rows(
        support3_rows, expected_ids, validate_rles=True
    )
    anchored_summary = audit_submission_rows(
        anchored_rows, expected_ids, validate_rles=True
    )
    four_model_rows = []
    four_model_by_image = {}
    four_model_added = 0
    four_model_rejected_without_p6 = 0
    for image_id in sorted(expected_ids):
        base_rles = [
            rle_from_counts(row["segmentation_rle"]) for row in p5[image_id]
        ]
        p6_rles = [
            rle_from_counts(row["segmentation_rle"]) for row in p6[image_id]
        ]
        anchored_rles = [
            rle_from_counts(row["segmentation_rle"])
            for row in anchored_by_image[image_id]
        ]
        base_overlap = pairwise_rle_iou(base_rles, anchored_rles)
        anchored_best = (
            base_overlap.max(axis=0)
            if len(base_rles) and len(anchored_rles)
            else np.zeros(len(anchored_rles), dtype=np.float64)
        )
        novel_rles = [
            rle for rle, best in zip(anchored_rles, anchored_best) if best <= 0.5
        ]
        p6_overlap = pairwise_rle_iou(p6_rles, novel_rles)
        p6_best = (
            p6_overlap.max(axis=0)
            if len(p6_rles) and len(novel_rles)
            else np.zeros(len(novel_rles), dtype=np.float64)
        )
        accepted_novel = [
            rle for rle, best in zip(novel_rles, p6_best) if best > 0.5
        ]
        four_model_added += len(accepted_novel)
        four_model_rejected_without_p6 += len(novel_rles) - len(accepted_novel)
        combined_rles = base_rles + accepted_novel
        image_rows = _rows_from_rles(image_id, combined_rles)
        four_model_rows.extend(image_rows)
        four_model_by_image[image_id] = image_rows
    four_model_summary = audit_submission_rows(
        four_model_rows, expected_ids, validate_rles=True
    )
    support3_path = write_submission(support3_rows, args.support3_output)
    anchored_path = write_submission(anchored_rows, args.anchored_output)
    four_model_path = write_submission(four_model_rows, args.four_model_output)
    report = {
        "report_type": "p8_conservative_assembly_audit_v1",
        "source": {
            "audit": str(args.source_audit),
            "audit_sha256": sha256_file(args.source_audit),
            "manifest": str(args.source_manifest),
            "manifest_sha256": sha256_file(args.source_manifest),
            "config_sha256": source_audit["config_sha256"],
            "runtime_sha256": source_audit["runtime_sha256"],
            "p5_submission": str(args.p5_submission),
            "p5_submission_sha256": sha256_file(args.p5_submission),
            "p6_submission": str(args.p6_submission),
            "p6_submission_sha256": sha256_file(args.p6_submission),
        },
        "support3_only": {
            "submission": str(support3_path),
            "submission_sha256": sha256_file(support3_path),
            "summary": support3_summary,
            "clusters_before_deoverlap": support3_clusters,
            "clusters_dropped_by_deoverlap_or_cap": support3_dropped_by_deoverlap,
            "fallback_images": support3_fallback_images,
            "comparison_vs_p5": _submission_overlap(p5, support3_by_image),
        },
        "p5_plus_support3_novel": {
            "submission": str(anchored_path),
            "submission_sha256": sha256_file(anchored_path),
            "summary": anchored_summary,
            "novel_clusters_before_deoverlap": anchored_novel_clusters,
            "added_instances": anchored_added_instances,
            "area_or_overlap_drops": anchored_area_drops,
            "comparison_vs_p5": _submission_overlap(p5, anchored_by_image),
        },
        "p5_plus_support3_novel_with_p6_agreement": {
            "submission": str(four_model_path),
            "submission_sha256": sha256_file(four_model_path),
            "summary": four_model_summary,
            "added_instances": four_model_added,
            "unanimous_novel_rejected_without_p6_agreement": (
                four_model_rejected_without_p6
            ),
            "comparison_vs_p5": _submission_overlap(p5, four_model_by_image),
            "comparison_vs_p6": _submission_overlap(p6, four_model_by_image),
        },
        "per_image": per_image,
        "reversibility": {
            "source_member_caches_unchanged": True,
            "support2_submission_unchanged": True,
            "p5_submission_unchanged": True,
            "assembly_is_cpu_only": True,
        },
    }
    atomic_write_json(args.audit_output, report)
    print(json.dumps({
        "support3_only": report["support3_only"],
        "p5_plus_support3_novel": report["p5_plus_support3_novel"],
        "p5_plus_support3_novel_with_p6_agreement": report[
            "p5_plus_support3_novel_with_p6_agreement"
        ],
    }, indent=2))
    print(f"wrote conservative P8 audit to {args.audit_output}")


if __name__ == "__main__":
    main()
