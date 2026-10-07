"""Official competition PQ semantics from Self Evaluation Notebook v6.

Unlike standard Panoptic Quality, every GT/prediction pair whose IoU is
strictly greater than 0.5 is counted. There is no one-to-one matching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

OFFICIAL_METRIC_NAME = "official_pq_v6"
OFFICIAL_METRIC_SOURCE = (
    "https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook"
)
OFFICIAL_IOU_THRESHOLD = 0.5
OFFICIAL_MATCHING = "all_pairs_strict_greater"


@dataclass(frozen=True)
class PQCounts:
    """Additive sufficient statistics for the competition PQ."""

    hit_pairs: int
    false_positives: int
    false_negatives: int
    hit_iou_sum: float

    @property
    def denominator(self) -> float:
        return self.hit_pairs + 0.5 * self.false_positives + 0.5 * self.false_negatives

    @property
    def pq(self) -> float:
        return float(self.hit_iou_sum / self.denominator) if self.denominator else 0.0


@dataclass(frozen=True)
class OfficialPQReport:
    """Traceable aggregate report for one evaluation run."""

    physical_images: int
    annotator_records: int
    hit_pairs: int
    false_positives: int
    false_negatives: int
    hit_iou_sum: float
    official_pq: float
    metric_name: str = OFFICIAL_METRIC_NAME
    source_version: int = 6
    source_url: str = OFFICIAL_METRIC_SOURCE
    iou_threshold: float = OFFICIAL_IOU_THRESHOLD
    matching: str = OFFICIAL_MATCHING

    def to_dict(self) -> dict:
        return asdict(self)


def pairwise_rle_iou(truth_rles: list[dict], prediction_rles: list[dict]) -> np.ndarray:
    """Return a [n_truth, n_prediction] IoU matrix using pycocotools C code."""
    n_truth, n_prediction = len(truth_rles), len(prediction_rles)
    if not n_truth or not n_prediction:
        return np.zeros((n_truth, n_prediction), dtype=np.float64)
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("pycocotools is required for official PQ") from exc
    # pycocotools returns [n_prediction, n_truth] for iou(dt, gt, iscrowd).
    matrix = mask_utils.iou(
        prediction_rles,
        truth_rles,
        [0] * n_truth,
    )
    return np.asarray(matrix, dtype=np.float64).T


def score_iou_matrix(
    iou_matrix: np.ndarray,
    n_truth: int | None = None,
    n_prediction: int | None = None,
) -> PQCounts:
    """Score one annotator-image record with the official all-pairs rule."""
    matrix = np.asarray(iou_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("iou_matrix must be two-dimensional")
    n_truth = matrix.shape[0] if n_truth is None else int(n_truth)
    n_prediction = matrix.shape[1] if n_prediction is None else int(n_prediction)
    if matrix.shape != (n_truth, n_prediction):
        raise ValueError(
            f"iou_matrix shape {matrix.shape} does not match "
            f"({n_truth}, {n_prediction})"
        )
    if n_truth == 0:
        return PQCounts(0, n_prediction, 0, 0.0)
    if n_prediction == 0:
        return PQCounts(0, 0, n_truth, 0.0)

    hits = matrix > OFFICIAL_IOU_THRESHOLD
    return PQCounts(
        hit_pairs=int(hits.sum()),
        false_positives=int((hits.sum(axis=0) == 0).sum()),
        false_negatives=int((hits.sum(axis=1) == 0).sum()),
        hit_iou_sum=float(matrix[hits].sum()),
    )


def score_rles(
    truth_rles: list[dict],
    prediction_rles: list[dict],
) -> PQCounts:
    """Score one annotator-image record represented as COCO RLE objects."""
    matrix = pairwise_rle_iou(truth_rles, prediction_rles)
    return score_iou_matrix(
        matrix,
        n_truth=len(truth_rles),
        n_prediction=len(prediction_rles),
    )


def aggregate_counts(
    counts: Iterable[PQCounts], physical_images: int
) -> OfficialPQReport:
    """Aggregate annotator-record counts exactly once into the official PQ."""
    rows = list(counts)
    hit_pairs = sum(row.hit_pairs for row in rows)
    false_positives = sum(row.false_positives for row in rows)
    false_negatives = sum(row.false_negatives for row in rows)
    hit_iou_sum = float(sum(row.hit_iou_sum for row in rows))
    denominator = hit_pairs + 0.5 * false_positives + 0.5 * false_negatives
    official_pq = float(hit_iou_sum / denominator) if denominator else 0.0
    return OfficialPQReport(
        physical_images=int(physical_images),
        annotator_records=len(rows),
        hit_pairs=hit_pairs,
        false_positives=false_positives,
        false_negatives=false_negatives,
        hit_iou_sum=hit_iou_sum,
        official_pq=official_pq,
    )
