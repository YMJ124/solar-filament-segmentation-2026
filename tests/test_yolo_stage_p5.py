import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from src.instance.p5_diagnostics import (
    raw_iou_band,
    recoverability_route,
    summarize_recoverability_rows,
)
from src.instance.p5_refiner import (
    BoundaryWeightedDiceLoss,
    combine_support_preserving_instances,
    crop_bounds,
    fuse_refinement_in_boundary_band,
    instances_to_label_map,
    physical_image_sampling_weights,
    postprocessed_instances_from_rles,
    resize_refiner_crop,
    restore_refined_mask,
)
from src.instance.p5_multifold import (
    aggregate_refiner_folds,
    compare_refiner_policy_folds,
    formal_protocol_sha256,
    refinement_policy_payload,
    refinement_policy_sha256,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig, predictions_for_config
from src.submission.rle import mask_to_rle


def test_raw_iou_band_and_recoverability_routes():
    config = YoloPostprocessConfig(0.1, 600, 20, True)
    assert raw_iou_band(0.0) == "no_overlap"
    assert raw_iou_band(0.25) == "low_overlap_le_0_25"
    assert raw_iou_band(0.5) == "near_threshold_gt_0_25_to_0_5"
    assert raw_iou_band(0.5001) == "recoverable_gt_0_5"

    assert recoverability_route(
        np.array([0.2]), np.array([0.9]), np.array([1000]), config
    ) == "not_recoverable_from_cached_candidates"
    assert recoverability_route(
        np.array([0.6]), np.array([0.05]), np.array([1000]), config
    ) == "below_confidence_threshold"

    ious = np.zeros(21)
    ious[20] = 0.7
    assert recoverability_route(
        ious, np.linspace(0.9, 0.1, 21), np.full(21, 1000), config
    ) == "beyond_max_instances_cap"
    assert recoverability_route(
        np.array([0.7]), np.array([0.9]), np.array([500]), config
    ) == "raw_area_below_min_area"
    assert recoverability_route(
        np.array([0.7]), np.array([0.9]), np.array([700]), config
    ) == "postprocess_interaction_after_preselection"


def _row(
    fold: int,
    final_outcome: str,
    raw_band: str,
    route: str,
    *,
    cache_limit_hit: bool = False,
):
    return {
        "fold": fold,
        "area_band": "400_1599",
        "representation_band": "lt_1_proto_cell",
        "bbox_short_side_band": "0_31",
        "final_outcome": final_outcome,
        "raw_iou_band": raw_band,
        "recoverability_route": route,
        "cache_limit_hit": cache_limit_hit,
    }


def test_recoverability_summary_routes_ranking_and_refiner():
    rows = []
    for fold in range(3):
        rows.extend(
            [
                _row(
                    fold,
                    "no_overlap",
                    "recoverable_gt_0_5",
                    "below_confidence_threshold",
                ),
                _row(
                    fold,
                    "no_overlap",
                    "no_overlap",
                    "not_recoverable_from_cached_candidates",
                    cache_limit_hit=True,
                ),
                _row(
                    fold,
                    "near_threshold_gt_0_25_to_0_5",
                    "near_threshold_gt_0_25_to_0_5",
                    "not_recoverable_from_cached_candidates",
                ),
            ]
        )
    summary = summarize_recoverability_rows(rows)
    assert summary["global"]["final_no_overlap_instances"] == 6
    assert summary["global"]["final_no_overlap_raw_recoverable_rate"] == pytest.approx(
        0.5
    )
    assert summary["routing_checks"]["mask_quality_reranking_supported"] is True
    assert summary["routing_checks"]["candidate_generation_supported"] is True
    assert summary["routing_checks"]["crop_refiner_supported"] is True
    assert summary["recommended_recall_route"] == "mask_quality_reranking"
    assert summary["recommended_mask_route"] == "crop_refiner"


def test_refiner_postprocess_reproduces_existing_label_map():
    masks = np.zeros((3, 16, 16), dtype=bool)
    masks[0, 1:8, 1:8] = True
    masks[1, 5:12, 5:12] = True
    masks[2, 13:15, 13:15] = True
    scores = np.asarray([0.9, 0.8, 0.7])
    config = YoloPostprocessConfig(0.1, 8, 2, True)
    expected, _ = predictions_for_config(masks, scores, config)
    refined, retained_scores, retained_indices = postprocessed_instances_from_rles(
        [mask_to_rle(mask) for mask in masks],
        scores,
        height=16,
        width=16,
        config=config,
    )
    actual = instances_to_label_map(refined, height=16, width=16)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(retained_scores, [0.9, 0.8])
    np.testing.assert_array_equal(retained_indices, [0, 1])


def test_refiner_crop_resize_and_restore_stays_inside_bounds():
    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32).astype(np.uint8)
    coarse = np.zeros((32, 32), dtype=bool)
    coarse[10:14, 4:20] = True
    target = coarse.copy()
    bounds = crop_bounds(coarse, context_scale=1.5, minimum_side=8)
    image_crop, coarse_crop, target_crop = resize_refiner_crop(
        image, coarse, target, bounds, output_size=32
    )
    assert image_crop.shape == coarse_crop.shape == target_crop.shape == (32, 32)
    restored = restore_refined_mask(
        target_crop.astype(np.float32),
        bounds,
        height=32,
        width=32,
        threshold=0.5,
    )
    x0, y0, x1, y1 = bounds
    outside = restored.copy()
    outside[y0:y1, x0:x1] = False
    assert not outside.any()
    assert restored.any()


def test_boundary_band_fusion_preserves_deep_coarse_interior():
    coarse = np.zeros((32, 32), dtype=bool)
    coarse[8:24, 8:24] = True
    refined = np.zeros_like(coarse)
    fused = fuse_refinement_in_boundary_band(
        coarse,
        refined,
        coarse.astype(np.uint8),
        (0, 0, 32, 32),
        boundary_kernel=3,
    )
    assert fused[16, 16]
    assert not fused[8, 8]
    np.testing.assert_array_equal(fused[10:22, 10:22], coarse[10:22, 10:22])


def test_support_preserving_combination_restores_each_source_slot():
    coarse_a = np.zeros((24, 24), dtype=bool)
    coarse_b = np.zeros((24, 24), dtype=bool)
    coarse_a[2:8, 2:8] = True
    coarse_b[14:20, 14:20] = True
    refined_a = coarse_b.copy()
    refined_b = np.zeros_like(coarse_b)
    labels, rles, fallback_count = combine_support_preserving_instances(
        [refined_a, refined_b],
        [coarse_a, coarse_b],
        height=24,
        width=24,
        min_area=20,
    )
    assert len(rles) == 2
    assert fallback_count == 2
    assert np.all(labels[coarse_a] == 1)
    assert np.all(labels[coarse_b] == 2)


def test_boundary_policy_is_fixed_from_existing_protocol():
    config = yaml.safe_load(
        Path("configs/instance/stage_p5/p5_c1_formal_fold0.yaml").read_text()
    )
    payload = refinement_policy_payload(
        config, "boundary_band_preserve_support"
    )
    assert payload["boundary_kernel"] == 7
    assert payload["per_source_min_area_fallback"] == 600
    assert payload["support_preserving"] is True
    assert refinement_policy_sha256(
        config, "boundary_band_preserve_support"
    ) == refinement_policy_sha256(config, "boundary_band_preserve_support")


def test_boundary_weighted_loss_and_physical_sampling_weights():
    loss = BoundaryWeightedDiceLoss(boundary_weight=4.0, boundary_kernel=3)
    logits = torch.zeros((2, 1, 16, 16), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, :, 4:12, 4:12] = 1
    value = loss(logits, target)
    assert torch.isfinite(value)
    value.backward()
    assert logits.grad is not None
    weights = physical_image_sampling_weights(
        [{"file_name": "a"}, {"file_name": "a"}, {"file_name": "b"}]
    )
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.5])


def test_formal_refiner_configs_share_one_protocol():
    paths = [
        Path(f"configs/instance/stage_p5/p5_c1_formal_fold{fold}.yaml")
        for fold in range(3)
    ]
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    assert len({formal_protocol_sha256(config) for config in configs}) == 1
    assert [config["experiment"]["heldout_fold"] for config in configs] == [0, 1, 2]
    for fold, config in enumerate(configs):
        assert set(config["experiment"]["training_folds"]) == {0, 1, 2} - {fold}
        assert config["postprocess"] == {
            "confidence": 0.1,
            "min_area": 600,
            "max_instances": 20,
            "force_one": True,
        }


def test_p6_final_fit_refiner_locks_promoted_protocol():
    config = yaml.safe_load(
        Path("configs/instance/stage_p6/p6_c2_full_fit_refiner.yaml").read_text()
    )
    assert config["experiment"]["final_fit"] is True
    assert config["experiment"]["heldout_fold"] is None
    assert config["experiment"]["training_folds"] == [0, 1, 2]
    assert config["postprocess"] == {
        "confidence": 0.1,
        "min_area": 600,
        "max_instances": 20,
        "force_one": True,
    }
    assert config["training"]["epochs"] == 12
    assert config["loss"]["boundary_kernel"] == 7
    assert config["evaluation"]["refinement_policy"] == (
        "boundary_band_preserve_support"
    )


def _metric(pq: float, *, hits: int = 10, predictions: int = 8):
    return {
        "physical_images": 1,
        "annotator_records": 1,
        "hit_pairs": hits,
        "false_positives": 2,
        "false_negatives": 3,
        "hit_iou_sum": pq * (hits + 2.5),
        "official_pq": pq,
        "prediction_instances": predictions,
    }


def test_aggregate_refiner_folds_applies_gate_and_risk_counts():
    baseline_pq = [0.30, 0.31, 0.32]
    baseline = {
        "fold_reports": {
            str(fold): _metric(pq) for fold, pq in enumerate(baseline_pq)
        }
    }
    reports = {}
    for fold, pq in enumerate(baseline_pq):
        control = _metric(pq)
        candidate = _metric(pq + 0.01, hits=11, predictions=7)
        reports[fold] = {
            "complete": True,
            "status": "complete_fold",
            "heldout_fold": fold,
            "control": control,
            "candidate": candidate,
            "per_image": [
                {
                    "control_instances": 8,
                    "candidate_instances": 7,
                    "control": control,
                    "candidate": candidate,
                }
            ],
        }
    gate = {
        "minimum_mean_pq_delta": 0.008,
        "minimum_positive_folds": 2,
        "minimum_fold_delta": -0.005,
        "maximum_std_increase": 0.01,
    }
    result = aggregate_refiner_folds(reports, baseline, gate)
    assert result["comparison"]["pass"] is True
    assert result["comparison"]["positive_folds"] == 3
    assert result["comparison"]["mean_pq_delta"] == pytest.approx(0.01)
    assert result["risk"]["improved_images"] == 3
    assert result["risk"]["images_with_instance_removal"] == 3


def test_compare_refiner_policy_requires_score_and_reversibility():
    baseline_pq = [0.30, 0.31, 0.32]
    baseline = {
        "fold_reports": {
            str(fold): _metric(pq) for fold, pq in enumerate(baseline_pq)
        }
    }
    candidate_reports = {}
    reference_reports = {}
    for fold, pq in enumerate(baseline_pq):
        control = _metric(pq)
        reference_candidate = _metric(pq + 0.010, hits=10, predictions=7)
        policy_candidate = _metric(pq + 0.009, hits=11, predictions=8)
        shared = {
            "complete": True,
            "status": "complete_fold",
            "heldout_fold": fold,
            "control": control,
        }
        reference_reports[fold] = {
            **shared,
            "candidate": reference_candidate,
            "per_image": [
                {
                    "control_instances": 8,
                    "candidate_instances": 7,
                    "control": control,
                    "candidate": reference_candidate,
                }
            ],
        }
        candidate_reports[fold] = {
            **shared,
            "candidate": policy_candidate,
            "reversibility": {
                "source_cache_immutable": True,
                "checkpoint_immutable": True,
                "branch_output_only": True,
                "source_instance_count_preserved": True,
            },
            "per_image": [
                {
                    "control_instances": 8,
                    "candidate_instances": 8,
                    "control": control,
                    "candidate": policy_candidate,
                }
            ],
        }
    result = compare_refiner_policy_folds(
        candidate_reports,
        reference_reports,
        baseline,
        {
            "minimum_mean_pq_delta": 0.008,
            "minimum_positive_folds": 2,
            "minimum_fold_delta": -0.005,
            "maximum_std_increase": 0.01,
        },
        {
            "maximum_mean_pq_drop": 0.003,
            "maximum_fold_pq_drop": 0.005,
            "minimum_hit_pairs_delta": 0,
        },
    )
    assert result["pass"] is True
    assert result["comparison_vs_c1"]["hit_pairs_delta"] == 3
    assert all(result["comparison_vs_c1"]["checks"].values())
