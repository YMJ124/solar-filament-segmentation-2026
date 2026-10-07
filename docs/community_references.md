# Community references and adopted decisions

Checked on 2026-10-07. Public results are treated as design evidence, not as a
replacement for this repository's date-grouped OOF validation.

## Primary evidence

- [Wzy633/solar-filament-segmentation-2026](https://github.com/Wzy633/solar-filament-segmentation-2026)
  reports legal Fold-0 PQ `0.35259` with a single YOLO26n-seg model trained at
  1024 px and inferred at 1536 px. Native instance prediction substantially
  raises recognition quality above connected components from a semantic mask.
- [HeShen-1/filament-segmentation-2026](https://github.com/HeShen-1/filament-segmentation-2026)
  reports five-fold mean PQ `0.3593` for a YOLO11m-seg detector plus refiner and
  finds that same-family fusion does not beat the strongest member. This
  supports the current single-model-first policy.
- [bcap52/solar-filament-segmentation-2026](https://github.com/bcap52/solar-filament-segmentation-2026)
  reports that native-resolution tiled ResNet34 U-Net plus boundary-weighted BCE
  improves PQ. Boundary supervision remains a fallback if mask quality, rather
  than recognition, becomes the measured bottleneck.
- [Ultralytics instance-segmentation documentation](https://docs.ultralytics.com/tasks/segment/)
  defines the supported custom-data and `yolo26n-seg.pt` training interface.

## Adopted now

- A single native instance-segmentation model; no ensemble.
- Existing date-grouped folds and one medoid annotator per physical JPEG.
- Direct polygon labels, preserving thin boundaries without rasterize-resize.
- 1024 px training, 1536 px validation inference, rotations and flips.
- Confidence/max-instance calibration using the official local PQ.
- Greedy confidence-ordered pixel ownership so output instances never overlap.

## Explicitly not adopted

- Labels from the public MAGFiLO release, because they overlap competition test
  observations and would leak test ground truth.
- Random image folds or duplicated annotator versions across train/validation.
- Fusion, TTA or a second-stage refiner before the single model clears its OOF
  gate.
- Public leaderboard values as local model-selection targets.

## License boundary

The conversion, calibration and exclusivity code in this repository was
implemented independently; no community repository source was copied.
Ultralytics itself is an optional external dependency distributed under
AGPL-3.0. Before final public release, the project license and distribution
method must be made compatible with that dependency. AGPL-3.0 is OSI approved,
but the user must still accept its copyleft obligations.
