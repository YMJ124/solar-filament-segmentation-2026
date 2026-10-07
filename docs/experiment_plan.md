# Single-model staged optimization plan

## Fixed controls

- Semantic control: `resnet34_256`, fixed-panel PQ `0.122442`.
- Primary metric: organizer PQ v6 on date-isolated Fold 0.
- Training target: one medoid annotator batch per physical JPEG.
- No model fusion, test-label access, threshold tuning on test, or cross-fold
  duplicate observations.

Retired candidates and reconstructable reports are stored in
`artifacts/archive/retired_screen_candidates/`.

## Stage I0: native-instance data contract — complete

`scripts.build_yolo_dataset` converts the selected COCO polygons directly into
one-class YOLO segmentation labels. It creates image symlinks rather than a
second multi-gigabyte image copy and records source hashes and selected image
IDs. Expected Fold-0 result:

- train: 569 physical images, 4,160 instances;
- validation: 138 physical images, 1,071 instances;
- empty targets: 0.

## Stage I1: five-epoch architecture screen — user run

Train only the small pretrained YOLO26n-seg at 1024 px. This is the first
meaningful gate, not the final model.

```bash
python -m scripts.train_yolo \
  --config configs/instance/yolo26n_fold0_proxy.yaml
```

Then calibrate on the same fixed 32-image screen used by the semantic control:

```bash
python -m scripts.evaluate_yolo \
  --checkpoint artifacts/instance/yolo26n_fold0_proxy/weights/best.pt \
  --validation-file-list data/manifests/screen_fold0_32.json \
  --imgsz 1536 \
  --confidences 0.10 0.20 0.30 \
  --max-instances 5 10 20 \
  --output artifacts/instance/yolo26n_fold0_proxy/pq_screen32.json
```

Gate:

- PQ `< 0.12`: stop and audit labels/predictions before adding epochs;
- PQ `0.12–0.20`: retain only if epoch trend and TP/FP balance are improving;
- PQ `>= 0.20`: proceed to Stage I2;
- PQ `>= 0.30`: strong primary-model candidate.

## Stage I2: 30-epoch Fold-0 confirmation — conditional user run

Only run after Stage I1 passes `0.20`:

```bash
python -m scripts.train_yolo \
  --config configs/instance/yolo26n_fold0_full.yaml

python -m scripts.evaluate_yolo \
  --checkpoint artifacts/instance/yolo26n_fold0_full/weights/best.pt \
  --imgsz 1536 \
  --confidences 0.10 0.15 0.20 0.25 0.30 \
  --max-instances 5 10 15 20 \
  --output artifacts/instance/yolo26n_fold0_full/pq_fold0.json
```

The full run must beat the semantic control by at least `0.05` PQ. A target of
`>= 0.30` is realistic based on public, leakage-safe implementations, but is not
assumed until reproduced locally.

## Later stages

1. If recognition quality is good but matched IoU is weak, add a crop refiner or
   boundary-weighted objective.
2. If small filaments are missed, compare 1280 training or 1536 tiled inference
   before increasing model size.
3. Train more date folds only after Fold 0 is stable.
4. Consider an ensemble only after a single model is strong and an OOF error
   study identifies genuinely complementary failures.
