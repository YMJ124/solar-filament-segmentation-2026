# Filament Segmentation 2026

面向 Solar Filament Segmentation 2026 的单模型实验仓库。当前主线是：

`灰度图 -> 单个原生实例分割模型 -> 互斥实例 -> 官方 PQ -> submission`

`resnet34_256` 以固定 32 图 PQ `0.122442` 作为语义控制组。Model B、集成、
Flat U-Net、整图 512 缩放和其他失败候选已退出活跃代码；报告和权重哈希保存在
`artifacts/archive/`。

## 环境与数据准备

```bash
conda activate solar-filament
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"
python -m scripts.audit_dataset
python -m scripts.build_folds --n-splits 5 --seed 2026
python -m scripts.build_mask_cache --output data/cache/medoid_semantic_v2
python -m scripts.build_screen_manifest
python -m scripts.build_yolo_dataset --output data/processed/yolo_fold0
```

缓存仅包含固定 medoid annotator 的 semantic target 和实例框。训练验证严格按观测日期分组；整图预测会分别与每个标注者 batch 评分，不能把多标注者掩膜合并。

## 当前主线：原生实例分割

公开的比赛方案显示，当地高分的主要增益来自原生实例建模，而不是将语义概率图
继续拆成连通域。当前只引入单个 YOLO26n-seg 候选，不做集成。参考来源与取舍见
[`docs/community_references.md`](docs/community_references.md)。

安装可选实例分割依赖：

```bash
python -m pip install -r requirements-yolo.txt
```

检查已解析的训练参数，不启动训练：

```bash
python -m scripts.train_yolo \
  --config configs/instance/yolo26n_fold0_proxy.yaml --dry-run
```

五轮 proxy 训练和官方 PQ 验收命令见
[`docs/experiment_plan.md`](docs/experiment_plan.md)。只有固定面板 PQ 达到 `0.20` 才进入 30 epoch 确认。

## 保留的语义控制组

```bash
python -m scripts.train --config configs/global_resnet34_fold0.yaml

python -m scripts.evaluate_model \
  --checkpoint artifacts/runs/global_resnet34_fold0/checkpoints/best.pt \
  --fold 0 --tile-size 256 --threshold 0.3 --min-area 200 \
  --output artifacts/validation/global_fold0_metrics.json

python -m scripts.run_submission \
  --checkpoint artifacts/runs/global_resnet34_fold0/checkpoints/best.pt \
  --tile-size 256 --threshold 0.3 --min-area 200
```

`last.pt` 含 optimizer、scheduler、AMP scaler、RNG 和完整配置，可严格续训；`best.pt` 是较小的推理 checkpoint。续训配置指纹必须一致：

```bash
python -m scripts.train --config configs/global_resnet34_fold0.yaml \
  --resume artifacts/runs/global_resnet34_fold0/checkpoints/last.pt
```

阈值和最小面积只能在 OOF 上选择。当前 `0.3/200` 来自旧 A-only Fold-0 搜索，是起点而非跨模型通用最优值。

比赛边界与交付要求见 [`docs/competition_compliance.md`](docs/competition_compliance.md)。
