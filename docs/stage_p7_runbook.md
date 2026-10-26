# P7-H1: 384 local-refiner resolution

## Decision

P6 scored approximately `0.29`, below P5's approximately `0.31`, after its test
instance density rose by 18%.  P7 therefore returns to the validated P5/T1
detector and does not add, delete, or re-rank detector instances.

P7-H1 changes one variable: the local refiner crop raster increases from
`256x256` to `384x384`.  The model architecture, context bounds, loss, epoch
budget, threshold, and reversible C2 boundary-support policy remain fixed.

The locked 256-C2 Fold0 panel baseline is:

- PQ `0.3730990959266039`;
- TP/FP/FN `189/130/170`;
- 211 source and output instances;
- report SHA256
  `9d90aa397e92893c33283e8533e3ab0b306d9c043ab0c1a942f8dfa1313af120`.

P7-H1 must reach at least `0.3830990959266039` (`+0.010`) on exactly the same
panel and preserve all 211 source slots.  Otherwise it is retired immediately.

## Preparation

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

conda run --no-capture-output -n solar-filament \
  python -m scripts.build_p5_refiner_dataset \
  --config configs/instance/stage_p7/p7_h1_refiner_384_fold0.yaml
```

## Locked Fold0 training

Preparation completed with 284 physical images and 3148 crops.  The dataset
index SHA256 is
`0510562f6f3c559c87c2f1d7fa26b48f6415404c16d00b0e9bbe84961cda1235`;
the config SHA256 is
`c7fb45af96ddc1aaeb105db1be0513e75255e13dbd1f0190bc62cb8551cd1bfc`.

A one-step and a 20-step CUDA smoke test passed on the RTX 5070 Laptop GPU.
The stable short benchmark completed 20 steps in 4.44 seconds.  With 2364 full
steps, startup/data overhead gives an estimated 7.5--9 minute run, so the user
must execute this command on host CUDA:

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

conda run --no-capture-output -n solar-filament \
  python -m scripts.train_p5_refiner \
  --config configs/instance/stage_p7/p7_h1_refiner_384_fold0.yaml \
  --device 0
```

## Fixed panel evaluation

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

conda run --no-capture-output -n solar-filament \
  python -m scripts.evaluate_p5_refiner \
  --config configs/instance/stage_p7/p7_h1_refiner_384_fold0.yaml \
  --checkpoint artifacts/instance/stage_p7/p7_h1_refiner_384_fold0/training/final.pt \
  --device 0 \
  --refinement-policy boundary_band_preserve_support \
  --output artifacts/instance/stage_p7/p7_h1_refiner_384_fold0/panel32_c2.json
```

Do not alter batch size, epoch count, threshold, boundary kernel, or panel after
seeing the result.  P5-C2 remains the submission default until P7 passes both
the panel and the subsequent three-fold OOF gate.

## Result (2026-10-18)

Training completed on CUDA for all 12 epochs / 2364 steps. The final checkpoint
SHA256 is `a934ad41d38b2230ab11641950a03aff7ae31c98f0eb9192b0d551c4c3959e7e`.

The 384-C2 panel result was `PQ=0.373960570`, TP/FP/FN
`191/128/168`, with all 211 source slots preserved. Against the locked 256-C2
control (`PQ=0.373099096`, `189/130/170`) the true delta is only
`+0.000861474`, below the required `+0.010`. Therefore P7-H1 failed the panel
gate and Fold1/2 training is cancelled. See `artifacts/instance/stage_p7/stage_report.json`
and `docs/stage_p7_failure_review.md`.

The raw evaluator report's built-in `pass=true` is against the unrefined P4
control, not the preregistered 256-C2 control; it must not be used as the P7
promotion decision.
