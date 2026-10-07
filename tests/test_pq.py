import numpy as np
import pytest

from src.metrics.pq import (
    aggregate_counts,
    pairwise_rle_iou,
    score_iou_matrix,
    score_rles,
)
from src.submission.rle import mask_to_rle

pycocotools = pytest.importorskip("pycocotools")


def mask(x0, y0, x1, y1):
    value = np.zeros((16, 16), dtype=np.uint8)
    value[y0:y1, x0:x1] = 1
    return value


def test_perfect_match():
    rle = mask_to_rle(mask(1, 1, 5, 5))
    result = score_rles([rle], [rle])
    assert result.pq == 1.0
    assert (result.hit_pairs, result.false_positives, result.false_negatives) == (
        1,
        0,
        0,
    )


def test_empty_cases():
    rle = mask_to_rle(mask(1, 1, 5, 5))
    missed = score_rles([rle], [])
    false_alarm = score_rles([], [rle])
    both_empty = score_rles([], [])
    assert (missed.hit_pairs, missed.false_positives, missed.false_negatives) == (
        0,
        0,
        1,
    )
    assert (
        false_alarm.hit_pairs,
        false_alarm.false_positives,
        false_alarm.false_negatives,
    ) == (0, 1, 0)
    assert both_empty.pq == 0.0


def test_iou_equal_to_threshold_is_not_a_hit():
    result = score_iou_matrix(np.array([[0.5]]))
    assert result.hit_pairs == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.pq == 0.0


def test_one_truth_can_hit_two_predictions():
    result = score_iou_matrix(np.array([[1.0, 1.0]]))
    assert result.hit_pairs == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.pq == 1.0


def test_two_truths_can_hit_one_prediction():
    result = score_iou_matrix(np.array([[1.0], [1.0]]))
    assert result.hit_pairs == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.pq == 1.0


def test_official_notebook_v6_reference_semantics():
    """Parity fixture for the all-pairs code in the official v6 notebook."""
    matrix = np.array([[0.80, 0.51, 0.00], [0.00, 0.50, 0.70]])
    result = score_iou_matrix(matrix)
    hits = matrix > 0.5
    reference_iou_sum = float(matrix[hits].sum())
    reference_tp = int(hits.sum())
    reference_fp = int((hits.sum(axis=0) == 0).sum())
    reference_fn = int((hits.sum(axis=1) == 0).sum())
    reference_pq = reference_iou_sum / (
        reference_tp + 0.5 * reference_fp + 0.5 * reference_fn
    )
    assert result.hit_pairs == reference_tp
    assert result.false_positives == reference_fp
    assert result.false_negatives == reference_fn
    assert result.hit_iou_sum == pytest.approx(reference_iou_sum, abs=1e-12)
    assert result.pq == pytest.approx(reference_pq, abs=1e-12)


def test_rle_iou_matches_direct_boolean_iou():
    truth = mask(1, 1, 7, 7)
    prediction = mask(4, 4, 10, 10)
    matrix = pairwise_rle_iou(
        [mask_to_rle(truth)], [mask_to_rle(prediction)]
    )
    expected = np.logical_and(truth, prediction).sum() / np.logical_or(
        truth, prediction
    ).sum()
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(expected, abs=1e-12)


def test_global_aggregation_is_not_macro_average():
    perfect = score_iou_matrix(np.array([[1.0]]))
    missed = score_iou_matrix(np.zeros((1, 1)))
    report = aggregate_counts([perfect, missed], physical_images=1)
    assert report.annotator_records == 2
    assert report.hit_pairs == 1
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.official_pq == 0.5
