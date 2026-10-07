# Competition compliance snapshot

Checked on 2026-09-24 against the official Kaggle competition pages and the
organizer's `Self_Evaluation_Notebook`, version 6.

## Competition contract

- Competition: `filament-segmentation-2026`.
- Input: only the provided 8-bit grayscale H-alpha JPEG images, all
  `2048 x 2048`.
- Task: instance segmentation of individual filaments. Category, spine and
  bounding-box metadata are not prediction targets.
- Submission columns must be exactly `filament_id,segmentation_rle`.
- Every row represents one non-empty filament instance. `filament_id` keeps the
  test image stem unchanged and adds a unique suffix. `segmentation_rle` stores
  compressed COCO RLE counts only; size is implicitly `2048 x 2048`.
- Metric: organizer PQ notebook v6. Every GT/prediction pair with IoU strictly
  greater than `0.5` is a TP pair; there is no one-to-one matching. FP and FN
  are unmatched prediction columns and GT rows. Counts are aggregated globally.

Official references:

- <https://www.kaggle.com/competitions/filament-segmentation-2026/overview>
- <https://www.kaggle.com/competitions/filament-segmentation-2026/data>
- <https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook>

## Data provenance and leakage policy

The workspace contains only the competition-provided annotation JSON:

- 707 physical training images;
- 180 test images;
- 1,154 annotator-image records;
- 8,199 instance annotations;
- 296 physical images with multiple annotator batches;
- annotation SHA256:
  `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`.

No external MAGFiLO annotations are permitted. In particular, public MAGFiLO
releases contain labels for competition test observations; those labels must not
be downloaded, joined, inspected or used. The organizer has stated that only
ground truth distributed through this competition may be used.

ImageNet-pretrained ResNet34 weights are used only for encoder initialization.
They are disclosed in the resolved config and run manifest. Final test inference
uses only the provided H-alpha test images.

## Local validation policy

- Folds group all observations from the same `YYYYMMDD` date.
- The fold manifest is fixed with seed 2026 and has no date crossing folds.
- A representative medoid annotator batch is used as the training target; no
  annotator masks are unioned.
- Full-image validation predicts each physical image once, then scores that
  prediction independently against every annotator batch, matching notebook v6.
- Validation reports the official positive-pair IoU/Dice histograms and the
  IoU>0 one-to-many/many-to-one degree distributions, in addition to PQ.
- Thresholds, minimum area and any future ensemble weights must be selected from OOF
  validation predictions only, never from test results.

## Reproducibility status

Implemented:

- pinned direct dependencies in `requirements.txt`;
- immutable fold and target-cache fingerprints;
- resolved run config and run manifest with package/hardware/data hashes;
- atomic `best.pt`, `last.pt`, OOF, JSON and submission writes;
- strict resume config fingerprint and full-state validation;
- full RLE decode, shape, non-empty, uniqueness and overlap audit.
- exact preflight checks for fold coverage/date isolation and all 707 cached targets;
- bounded label-map postprocessing to avoid per-instance 2048² mask retention.

Still required before final competition handoff, but not before the local stress
test:

- publish the initialized Git repository at competition close;
- create the required end-to-end Jupyter notebook;
- prepare the four-page technical report and final Google-form submission;
- submit a model-produced CSV to Kaggle and archive its score receipt.
