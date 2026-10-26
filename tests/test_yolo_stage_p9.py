import numpy as np

from src.instance.p9_incremental import (
    AdditiveCandidate,
    FEATURE_NAMES,
    build_additive_candidates,
    extract_candidate_features,
    marginal_counts_and_utility,
)
from src.instance.p9_refinement import (
    aggregate_image_reports,
    compare_refined_folds,
    paired_image_bootstrap,
    registered_candidate_keys,
    select_registered_additions,
)
from src.instance.p9_deployment import (
    deployment_fingerprint,
    fit_final_selector,
    selected_additions_from_features,
)
from src.submission.rle import mask_to_rle


def test_additive_candidates_preserve_parent_and_reject_duplicate():
    t1 = np.zeros((32, 32), dtype=bool)
    t1[2:12, 2:12] = True
    duplicate = t1.copy()
    novel = np.zeros_like(t1)
    novel[20:30, 20:30] = True
    original_t1 = t1.copy()
    candidates, audit = build_additive_candidates(
        [t1], [duplicate, novel], [0.9, 0.8], [4, 7],
        min_area=20, max_instances=5, duplicate_iou=0.5,
    )
    assert np.array_equal(t1, original_t1)
    assert audit["duplicates"] == 1
    assert len(candidates) == 1
    assert candidates[0].source_index == 7
    assert not np.any(candidates[0].mask & t1)


def test_marginal_utility_distinguishes_recovery_from_false_positive():
    base = [np.asarray([[0.2]])]
    recovered = marginal_counts_and_utility(
        base, [np.asarray([0.8])], reference_pq=0.32
    )
    false_positive = marginal_counts_and_utility(
        base, [np.asarray([0.1])], reference_pq=0.32
    )
    assert recovered["delta_false_negatives"] == -1
    assert recovered["marginal_utility"] > 0
    assert false_positive["delta_false_positives"] == 1
    assert false_positive["marginal_utility"] < 0


def test_feature_vector_is_finite_and_registered():
    mask = np.zeros((32, 32), dtype=bool)
    mask[20:28, 3:14] = True
    candidate = build_additive_candidates(
        [], [mask], [0.7], [2], min_area=20, max_instances=5,
        duplicate_iou=0.5,
    )[0][0]
    image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    features = extract_candidate_features(
        candidate,
        image=image,
        i1_output_count=1,
        t1_output_count=0,
        raw_t1_rles=[mask_to_rle(mask)],
        raw_t1_scores=[0.2],
        distance_to_t1=np.full((32, 32), 5, dtype=np.float32),
    )
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_d2_reconstructs_registered_d1_selection_exactly():
    report = {
        "report_type": "p9_t1_i1_incremental_oof_v1",
        "complete": True,
        "comparison": {"pass": True},
        "target_summary": {"candidates": 2},
        "selection_summary": {"accepted": 1},
        "candidate_rows": [
            {
                "row_index": 0,
                "fold": 0,
                "file_name": "a.jpeg",
                "i1_source_index": 4,
                "i1_rank": 1,
                "accepted": False,
            },
            {
                "row_index": 1,
                "fold": 0,
                "file_name": "a.jpeg",
                "i1_source_index": 7,
                "i1_rank": 2,
                "accepted": True,
            },
        ],
    }
    all_keys, accepted_keys = registered_candidate_keys(report)
    mask = np.ones((8, 8), dtype=bool)
    additions = [
        AdditiveCandidate(mask, mask, 0.8, 4, 1, 0.0, 1.0),
        AdditiveCandidate(mask, mask, 0.7, 7, 2, 0.0, 1.0),
    ]
    selected = select_registered_additions(
        additions,
        expected_all=all_keys[(0, "a.jpeg")],
        expected_accepted=accepted_keys[(0, "a.jpeg")],
    )
    assert [candidate.source_index for candidate in selected] == [7]


def test_d2_gate_compares_against_promoted_c2_parent():
    reference = {
        0: {"official_pq": 0.34},
        1: {"official_pq": 0.33},
        2: {"official_pq": 0.32},
    }
    candidate = {
        0: {"official_pq": 0.345},
        1: {"official_pq": 0.334},
        2: {"official_pq": 0.321},
    }
    gate = {
        "minimum_mean_fold_pq_delta": 0.002,
        "minimum_positive_folds": 2,
        "minimum_fold_delta": -0.002,
        "maximum_fold_std_increase": 0.005,
    }
    result = compare_refined_folds(reference, candidate, gate)
    assert result["positive_folds"] == 3
    assert result["mean_fold_pq_delta"] > 0.002
    assert result["pass"] is True


def test_d2_paired_bootstrap_uses_whole_physical_images():
    rows = [
        {
            "reference": {
                "hit_pairs": 1,
                "false_positives": 1,
                "false_negatives": 1,
                "hit_iou_sum": 0.6,
            },
            "candidate": {
                "hit_pairs": 2,
                "false_positives": 0,
                "false_negatives": 0,
                "hit_iou_sum": 1.2,
            },
        },
        {
            "reference": {
                "hit_pairs": 1,
                "false_positives": 1,
                "false_negatives": 1,
                "hit_iou_sum": 0.6,
            },
            "candidate": {
                "hit_pairs": 1,
                "false_positives": 1,
                "false_negatives": 1,
                "hit_iou_sum": 0.6,
            },
        },
    ]
    result = paired_image_bootstrap(rows, iterations=200, seed=2026)
    assert result["unit"] == "physical_image_with_all_annotator_records"
    assert result["observed_global_pq_delta"] > 0
    assert result["ci95"][0] >= 0


def test_d2_image_report_aggregation_preserves_annotator_records():
    metadata = {
        "metric_name": "official_pq_v6",
        "source_version": 6,
        "source_url": "https://example.test/evaluator",
        "iou_threshold": 0.5,
        "matching": "all_pairs_strict_greater",
    }
    rows = [
        {
            "physical_images": 1,
            "annotator_records": 2,
            "hit_pairs": 3,
            "false_positives": 2,
            "false_negatives": 1,
            "hit_iou_sum": 1.8,
            "official_pq": 0.4,
            **metadata,
        },
        {
            "physical_images": 1,
            "annotator_records": 3,
            "hit_pairs": 4,
            "false_positives": 1,
            "false_negatives": 2,
            "hit_iou_sum": 2.4,
            "official_pq": 0.4,
            **metadata,
        },
    ]
    result = aggregate_image_reports(rows, prediction_instances=9)
    assert result["physical_images"] == 2
    assert result["annotator_records"] == 5
    assert result["hit_pairs"] == 7
    assert result["prediction_instances"] == 9


def test_d3_selector_uses_strict_frozen_threshold_in_candidate_order():
    mask = np.ones((8, 8), dtype=bool)
    additions = [
        AdditiveCandidate(mask, mask, 0.8, 4, 1, 0.0, 1.0),
        AdditiveCandidate(mask, mask, 0.7, 7, 2, 0.0, 1.0),
    ]

    class FixedModel:
        def predict(self, matrix):
            assert matrix.shape == (2, len(FEATURE_NAMES))
            return np.asarray([0.0, 0.25])

    selected, predictions = selected_additions_from_features(
        FixedModel(),
        additions,
        [np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES))],
        threshold=0.0,
    )
    assert predictions.tolist() == [0.0, 0.25]
    assert [candidate.source_index for candidate in selected] == [7]


def test_d3_final_selector_and_cache_fingerprint_are_deterministic():
    x = np.arange(24 * len(FEATURE_NAMES), dtype=np.float64).reshape(
        24, len(FEATURE_NAMES)
    )
    y = np.linspace(-1.0, 1.0, len(x))
    config = {
        "type": "extra_trees_regressor",
        "n_estimators": 8,
        "min_samples_leaf": 2,
        "max_features": 0.75,
    }
    first = fit_final_selector(x, y, config, seed=2026).predict(x)
    second = fit_final_selector(x, y, config, seed=2026).predict(x)
    assert np.array_equal(first, second)
    payload = {"model": "abc", "threshold": 0.0, "parents": ["d1", "d2"]}
    assert deployment_fingerprint(payload) == deployment_fingerprint(payload)
