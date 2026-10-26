import numpy as np

from src.instance.p8_consensus import (
    TTA_NAMES,
    apply_tta,
    cluster_instance_records,
    compose_exclusive_consensus,
    invert_tta_masks,
    select_consensus_records,
)
from src.submission.rle import mask_to_rle, rle_counts


def _record(mask, *, fold, transform="identity", score=0.8):
    return {
        "segmentation_rle": rle_counts(mask_to_rle(mask)),
        "score": score,
        "area": int(mask.sum()),
        "fold": fold,
        "transform": transform,
    }


def test_tta_round_trip_for_all_registered_transforms():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    mask = np.zeros((1, 6, 8), dtype=bool)
    mask[0, 1:4, 2:7] = True
    for name in TTA_NAMES:
        transformed_mask = apply_tta(mask[0], name)[None]
        assert np.array_equal(invert_tta_masks(transformed_mask, name), mask)
        assert apply_tta(image, name).flags.c_contiguous


def test_consensus_counts_distinct_folds_not_tta_copies():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:9, 3:12] = True
    records = [
        _record(mask, fold=0, transform="identity", score=0.9),
        _record(mask, fold=0, transform="hflip", score=0.8),
    ]
    clusters = cluster_instance_records(records, height=16, width=16, iou_threshold=0.5)
    selected, audit = select_consensus_records(
        records, clusters, minimum_fold_support=2
    )
    assert selected == []
    assert audit[0]["fold_support"] == 1


def test_consensus_selects_max_area_and_deoverlaps():
    small = np.zeros((20, 20), dtype=bool)
    small[2:10, 2:10] = True
    large = np.zeros((20, 20), dtype=bool)
    large[1:11, 1:11] = True
    other = np.zeros((20, 20), dtype=bool)
    other[12:18, 12:18] = True
    records = [
        _record(small, fold=0, score=0.95),
        _record(large, fold=1, score=0.75),
        _record(other, fold=0, score=0.85),
        _record(other, fold=2, score=0.80),
    ]
    clusters = cluster_instance_records(records, height=20, width=20, iou_threshold=0.5)
    selected, audit = select_consensus_records(
        records, clusters, minimum_fold_support=2, representative="max_area"
    )
    assert len(selected) == 2
    assert sorted(row["fold_support"] for row in selected) == [2, 2]
    assert any(row["representative_area"] == 100 for row in audit if row["accepted"])
    labels, rles, meta = compose_exclusive_consensus(
        selected, height=20, width=20, min_area=10, max_instances=5
    )
    assert len(rles) == len(meta) == 2
    assert labels.max() == 2
