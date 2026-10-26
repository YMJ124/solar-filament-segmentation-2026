import pytest

from src.instance.yolo_multifold_error_analysis import summarize_multifold_error_reports


def _row(fold: int, index: int) -> dict:
    return {
        "file_name": f"fold{fold}-{index}.jpeg",
        "fold": fold,
        "annotator_batches": 2,
        "official_pq": 0.2 + fold * 0.01,
        "hit_pairs": 2,
        "false_positives": 2 + fold,
        "false_negatives": 2,
        "hit_iou_sum": 1.2,
        "prediction_instances": 3 + fold,
        "truth_instances_total": 4,
        "consensus_false_positive_instances": 1,
        "large_false_positive_instances": int(fold == 2),
        "small_instance_misses": 1,
        "fragmented_gt_instances": 1,
        "merged_prediction_instances": 1,
        "annotator_pq_std": 0.1,
        "annotator_pq_range": 0.2,
        "categories": ["small_instance_miss"],
    }


def _report(fold: int) -> dict:
    return {
        "fold": fold,
        "postprocess": {"confidence": 0.1, "min_area": 600, "max_instances": 20, "force_one": True},
        "diagnostics": {"small_instance_area": 400},
        "annotation_sha256": "annotations",
        "fold_manifest_sha256": "folds",
        "training_signature_sha256": "training",
        "cache_dir": f"cache{fold}",
        "cache_run_config_sha256": f"run{fold}",
        "checkpoint_sha256": f"checkpoint{fold}",
        "per_image": [_row(fold, 0)],
    }


def test_multifold_error_summary_validates_and_aggregates():
    result = summarize_multifold_error_reports([_report(2), _report(0), _report(1)])
    assert result["folds"] == [0, 1, 2]
    assert result["global"]["physical_images"] == 3
    assert result["global"]["false_positives"] == 9
    assert result["global"]["rates"]["small_miss_share_of_false_negatives"] == pytest.approx(0.5)
    assert result["candidate_priorities"][1]["recommended"] is True
    assert result["fold2_diagnostic"]["fp_rate_excess"] > 0


def test_multifold_error_summary_rejects_incompatible_reports():
    reports = [_report(0), _report(1), _report(2)]
    reports[2]["postprocess"] = {**reports[2]["postprocess"], "min_area": 400}
    with pytest.raises(ValueError, match="different post-processing"):
        summarize_multifold_error_reports(reports)
