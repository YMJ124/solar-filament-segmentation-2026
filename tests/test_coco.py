from pathlib import Path

from src.data.coco import CocoAnnotations


def test_physical_grouping():
    path = Path("MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    coco = CocoAnnotations(path)
    assert len(coco.images) == 1154
    assert len(coco.physical_files) == 707
    assert any(len(coco.batches_for_file(name)) > 1 for name in coco.physical_files)
