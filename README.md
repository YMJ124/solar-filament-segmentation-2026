# Filament Segmentation 2026

面向 Solar Filament Segmentation 2026 的单模型实验仓库。当前唯一默认主线是：

`YOLO 原生实例分割 -> 原始候选缓存 -> CPU 后处理搜索 -> 官方 PQ -> 错误图谱`

旧语义分割路线已从默认流程中剔除；代码仅作为显式 legacy/control 保留。
`artifacts/analysis/errors_fold0` 是旧语义模型产物，不能解释当前 YOLO。详见
[`docs/legacy_semantic_control.md`](docs/legacy_semantic_control.md)。

## 环境与数据

```bash
conda activate solar-filament
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"
python -m pip install -r requirements-yolo.txt
python -m scripts.audit_dataset
python -m scripts.build_folds --n-splits 5 --seed 2026
python -m scripts.build_screen_manifest
python -m scripts.build_yolo_dataset --output data/processed/yolo_fold0
```

YOLO 数据集使用 fold manifest 指定的 medoid annotator 训练。验证时同一张物理图像的
预测会对所有标注者 batch 分别计分，不会合并多标注者 GT。

## P0 评估闭环

第一步是唯一需要 GPU 的长时步骤：对 checkpoint 执行一次推理，将原始候选 mask
以 COCO 压缩 RLE 按图保存。缓存 manifest 记录 checkpoint/config SHA256、推理分辨率、
原始尺寸、图像/标注批次 ID、候选置信度和完整运行指纹。

```bash
python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/yolo26n_fold0_full/weights/best.pt \
  --fold 0 --imgsz 1024 --quantize 16 --no-retina-masks \
  --candidate-confidence 0.01 --max-candidates 50 \
  --output-dir artifacts/instance/yolo26n_fold0_full/candidates_fold0
```

中断后使用同一命令并添加 `--resume`；只会复用运行指纹完全一致的逐图缓存。

之后的参数搜索全在 CPU 上运行，不加载 checkpoint：

```bash
python -m scripts.evaluate_yolo \
  --cache-dir artifacts/instance/yolo26n_fold0_full/candidates_fold0 \
  --confidences 0.05 0.10 0.15 0.20 0.25 \
  --min-areas 200 300 400 600 800 \
  --max-instances 5 10 15 20 \
  --output artifacts/instance/yolo26n_fold0_full/postprocess_fold0.json
```

固定 32 图面板可在同一份 Fold-0 缓存上离线筛选，无需重新推理：

```bash
python -m scripts.evaluate_yolo \
  --cache-dir artifacts/instance/yolo26n_fold0_full/candidates_fold0 \
  --validation-file-list data/manifests/screen_fold0_32.json \
  --confidences 0.05 0.10 0.15 0.20 \
  --min-areas 200 400 600 --max-instances 5 10 20 \
  --output artifacts/instance/yolo26n_fold0_full/postprocess_screen32.json
```

选定后处理参数后生成当前 YOLO 的错误分析：

```bash
python -m scripts.analyze_yolo_errors \
  --cache-dir artifacts/instance/yolo26n_fold0_full/candidates_fold0 \
  --confidence 0.10 --min-area 400 --max-instances 20 \
  --small-instance-area 400 --large-fp-area 10000 \
  --output-dir artifacts/analysis/yolo26n_fold0_full_errors
```

输出包括 `image_metrics.csv`、`annotator_metrics.csv`、`index.json`、最好/最差及各错误类型图像的
四联可视化，以及可直接浏览的 `README.md`。

## 训练与提交

检查训练参数而不启动训练：

```bash
python -m scripts.train_yolo \
  --config configs/instance/yolo26n_fold0_proxy.yaml --dry-run
```

YOLO 单模型提交文件由独立入口生成，写入前会审计 RLE、重叠、图像覆盖和实例 ID：

```bash
python -m scripts.run_yolo_submission \
  --checkpoint artifacts/instance/yolo26n_fold0_full/weights/best.pt \
  --imgsz 1024 --quantize 16 --no-retina-masks \
  --confidence 0.10 --max-instances 20 --min-area 400 \
  --output artifacts/submissions/yolo26n_fold0_full.csv
```

详细的分阶段门槛见 [`docs/experiment_plan.md`](docs/experiment_plan.md)，比赛边界与交付要求见
[`docs/competition_compliance.md`](docs/competition_compliance.md)。

当前 B0/C1/C2 三 Fold 阶段的完整执行顺序与命令见
[`docs/stage_p1_runbook.md`](docs/stage_p1_runbook.md)。
