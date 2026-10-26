# P8-E1: three-fold × four-view consensus submission

## Purpose

P7-H1 showed that changing the refiner raster from 256 to 384 adds only
`+0.000861 PQ` on the locked Fold0 panel. P8 therefore returns to the validated
P5-C2 256 refiners and spends compute on recognition stability: three existing
fold detector/refiner pairs each infer identity, horizontal flip, vertical flip
and 180-degree rotation views.

Masks are clustered at IoU `>0.50`. A cluster is retained only when at least two
distinct folds support it; TTA copies from one fold never count as multiple
model votes. The largest member mask represents an accepted cluster, followed
by confidence/support ordered greedy de-overlap. The immutable P5 Fold0
submission is used only if an image has zero retained consensus instances.

Every fold/transform refined mask, score, source index, area and checkpoint is
stored before consensus. P8 cannot overwrite P5 outputs or upstream caches.

## Validation limitation

The current three fold models provide only one non-leaky model for each OOF
image, so the combined three-model rule cannot be honestly scored OOF without
leakage. P8 is therefore labelled
`leaderboard_calibration_no_clean_ensemble_oof`. It is the highest-evidence
submission challenger, not a replacement research baseline until externally
scored.

## Smoke result

On 2026-10-18, a one-image RTX 5070 Laptop GPU smoke test completed in 12.24 s:

- three members × four transforms completed;
- each member emitted six transform-level candidates;
- the 18 records formed two accepted clusters, both supported by all 3 folds;
- two non-overlapping instances were written, with no fallback;
- CUDA peak allocation was 812 MiB and peak RSS was approximately 3.74 GiB;
- an immediate rerun reused all three member caches and produced a byte-identical CSV.

Config SHA256:
`5f6dfa106ac05854a8f6b5a5d337a0b7cfc30d4c369ccb4b83d7382f0592f565`.

## Full command

Estimated runtime is approximately 37 minutes from the one-image measurement;
allow 40--50 minutes for image-dependent candidate density and I/O. Run on the
host terminal:

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

conda run --no-capture-output -n solar-filament \
  python -m scripts.run_p8_consensus_submission \
  --config configs/submission/p8_e1_threefold_4tta_consensus.yaml \
  --device 0
```

The run is resumable. If interrupted, execute the exact same command; completed
per-image member caches are reused only when the config and P8 source-code
fingerprint match.

Expected outputs:

- `artifacts/submissions/p8_e1_threefold_4tta_support2_maxarea_1536.csv`
- `artifacts/submissions/p8_e1_threefold_4tta_support2_maxarea_1536.audit.json`
- `artifacts/instance/stage_p8/p8_e1_threefold_4tta_consensus/member_cache/manifest.json`

Do not submit the CSV until the audit confirms 180/180 image coverage, valid
non-overlapping RLEs, sensible instance density, and no unexpected fallback
concentration.

## Full result and corrected submission choice

The full run completed in 55.0 minutes and passed structural audit, but raw E1
contains 1609 instances (+29.8% vs P5), repeating the failed P6 density pattern.
Do not submit the raw E1 CSV.

CPU-only conservative assembly produced E2/E3/E4. The recommended file is E4,
which preserves all 1240 P5 instances and adds only 128 masks supported by all
three fold models and P6 full-fit:

`artifacts/submissions/p8_e4_p5_plus_threefold_unanimous_p6_agreement_1536.csv`

See `artifacts/instance/stage_p8/stage_report.json` and
`docs/stage_p8_result_review.md` for the decision evidence.
