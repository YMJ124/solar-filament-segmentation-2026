# P6 locked full-data final fit

## Decision

The first P5-C2 submission scored approximately `0.31` on the public PQ
leaderboard.  Its local Fold0 score was `0.357274523`, while the three-fold
global OOF score was `0.339564359`.  Only one of 180 test images used the
coverage fallback, so that fallback cannot explain the gap.  The submitted
detector and refiner were both Fold0 artifacts trained on only their non-heldout
training folds.

P6 changes one factor only: train the already-promoted T1 detector and C2
refiner on all legal competition training data.  It does not search confidence,
minimum area, instance cap, mask threshold, boundary width, loss, architecture,
or epoch count.  P5-C2 remains the immutable rollback submission.

The test set has 21/180 (11.67%) observations from 2021--2022, compared with
40/707 (5.66%) in the full training set.  Full-data training therefore adds all
recent legal observations without introducing a test-tuned year weight.

## External comparison, 2026-10-18

- The competition now ranks submissions with Panoptic Quality and states that
  PQ above 0.35 is scientifically valuable:
  <https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking>.
- Recent public, post-metric-update notebooks expose detector/crop-refiner or
  Mask R-CNN pipelines around 0.36--0.37.  They support keeping the current
  detection-plus-local-refinement decomposition, but their validation and data
  provenance are not sufficiently exposed to copy their settings:
  <https://www.kaggle.com/competitions/filament-segmentation-2026/code?competitionId=143678&excludeNonAccessedDatasources=true&sortBy=scoreDescending>.
- The organizer reiterates that design justification and reproducible baseline
  comparisons matter more than chasing a confusing leaderboard:
  <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/742143>.
- Public MAGFiLO annotations overlapping the competition test observations are
  explicitly forbidden; P6 uses only the supplied training annotation JSON:
  <https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396>.

The external evidence supports retaining the two-stage architecture and making
the standard post-selection final fit.  It does not provide evidence for
changing local thresholds or using leaked annotations.

## Reproducible preparation

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

conda run --no-capture-output -n solar-filament \
  python -m scripts.build_yolo_dataset \
  --all-data \
  --fold-manifest data/manifests/train_folds.csv \
  --output data/processed/yolo_stage_p6/medoid_all \
  --target-policy medoid

conda run --no-capture-output -n solar-filament \
  python -m scripts.build_yolo_sampling_manifest \
  --all-data \
  --dataset-yaml data/processed/yolo_stage_p6/medoid_all/dataset.yaml \
  --output data/processed/yolo_stage_p6/sampling/p6_t1_all.json

conda run --no-capture-output -n solar-filament \
  python -m scripts.build_p5_refiner_dataset \
  --config configs/instance/stage_p6/p6_c2_full_fit_refiner.yaml
```

## Host GPU training

These two runs together can exceed five minutes and must be run by the user on
the host CUDA device:

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

conda run --no-capture-output -n solar-filament \
  python -m scripts.train_yolo \
  --config configs/instance/stage_p6/p6_t1_full_fit_1024.yaml

conda run --no-capture-output -n solar-filament \
  python -m scripts.train_p5_refiner \
  --config configs/instance/stage_p6/p6_c2_full_fit_refiner.yaml \
  --device 0
```

The final submission config is created only after both checkpoints exist and
their SHA256 hashes have been recorded.  Test inference remains `1536`, and the
fixed P5-C2 boundary-support policy remains unchanged.

## Completed result, 2026-10-18

Both locked training jobs completed on an NVIDIA GeForce RTX 5070 Laptop GPU.
The detector completed five epochs with exactly 707 scheduled draws in every
epoch.  The refiner completed 12 epochs and 3612 optimizer steps; its mean epoch
loss changed from `1.08194` to `0.819322`.  The final checkpoints are:

- detector: `artifacts/instance/stage_p6/p6_t1_full_fit_1024/weights/best.pt`,
  SHA256 `abb340ed04df7e29c50a6f1b6599ca2f448b637a4f7da27535dfc08574bfe59c`;
- refiner:
  `artifacts/instance/stage_p6/p6_c2_full_fit_refiner/training/final.pt`,
  SHA256 `60275a5ec2a40062132dfa615fe165790e6233972f12d1ef4b3adc6ab5786dfe`.

The detector's final-fit validation mapping points to its training data, so its
reported mAP is only a training health check.  It must not be interpreted as an
OOF improvement or used for model selection.

Full 1536 CUDA inference and two independent submission audits passed.  The
submission candidate is:

```text
artifacts/submissions/p6_c2_full_fit_1536.csv
SHA256 ab01d2da7a25ac82003a9ccfaaa1cf0358660776925c09744f5b634f07ed989e
```

It contains 1463 non-overlapping instances and covers all 180 test images.
There are no coverage-fallback images, all source instances are preserved, and
88 instances used the per-source C2 support fallback.  The full run took
293.44 seconds and peaked at 784 MB GPU memory.

## Pre-submission risk review

P5 contained 1240 instances while P6 contains 1463 (`+223`, or `+18.0%`).  At
IoU greater than 0.5, P6 matches 1011 P5 instances; 452 P6 masks are novel or
materially shifted relative to P5.  Of those 452 masks, 306 have area below
1600 pixels, matching the historical `[400,1600)` false-negative concentration.
This supports the direction of the change, but it does not prove the masks are
true positives because test ground truth is unavailable.

P5's OOF prediction density was 2912/422 = 6.90 instances per image and its
test density was 1240/180 = 6.89.  P6 rises to 1463/180 = 8.13.  Therefore P6
is a meaningful challenger rather than a mechanical final-fit replacement.
Submit it once as a leaderboard calibration point; keep P5-C2 as the default
and rollback until the exact public P6 score is recorded.  Do not tune a new
threshold from the result of this single submission.

## Public result and closure

The user reported a rounded public PQ of `0.29`, below P5's approximately
`0.31`.  P6 is therefore closed as `scored_not_promoted`; P5-C2 remains the
default.  The checkpoints and inference artifacts are retained as reproducible
negative evidence, but no P6 confidence/area/cap search is allowed because the
full-data model has no independent calibration holdout.

The failure analysis and post-experiment external comparison are recorded in
`docs/stage_p6_failure_review.md`.  The next candidate returns to the validated
P5 detector and tests only higher local-refiner resolution.
