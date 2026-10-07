# Filament Segmentation 2026

面向 Solar Filament Segmentation 2026 的单模型实验仓库。当前主线只保留：

`灰度图 -> 单个语义分割模型 -> 滑窗/整图缩放 -> 连通域实例 -> 官方 PQ -> submission`

Model B、概率融合和多任务 boundary/distance 分支已经退出活跃代码。历史结论、指标与权重哈希保存在
`artifacts/archive/retired_model_b/`，不再参与训练或推理。

## 环境与数据准备

```bash
conda activate solar-filament
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"
python -m scripts.audit_dataset
python -m scripts.build_folds --n-splits 5 --seed 2026
python -m scripts.build_mask_cache --output data/cache/medoid_semantic_v2
python -m scripts.build_screen_manifest
```

缓存仅包含固定 medoid annotator 的 semantic target 和实例框。训练验证严格按观测日期分组；整图预测会分别与每个标注者 batch 评分，不能把多标注者掩膜合并。

## 单模型候选筛选漏斗

第一轮 ResNet34/ResNet101 与 256/512 crop 的 2×2 对照已完成。第二轮专门验证
「整图上下文 + 类别平衡损失 + 轻量 Flat U-Net」，不包含集成：

- `resnet34_full512_balanced.yaml`：保留 ImageNet ResNet34，只更换整图输入和损失；
- `flatunet_c32_full512_balanced.yaml`：用 25.8 万参数的 Flat U-Net 验证结构假设。

完整流程、淘汰门槛和长训练命令见 [`docs/experiment_plan.md`](docs/experiment_plan.md)。

单个候选先过固定 batch 学习能力检查：

```bash
python -m scripts.overfit_batch \
  --config configs/screen/resnet34_256.yaml --steps 20 \
  --output artifacts/screens/resnet34_256/overfit.json
```

再运行等预算 proxy training；每轮只在固定 32 图 Fold-0 面板计算官方 PQ：

```bash
python -m scripts.train --config configs/screen/resnet34_full512_balanced.yaml
python -m scripts.train --config configs/screen/flatunet_c32_full512_balanced.yaml
python -m scripts.summarize_screens
```

筛选成绩仅用于排序，不作为最终成绩。胜者必须移除 `validation_file_list`、`max_steps`、`max_validation_batches` 与 `max_images` 限制，在完整 Fold 0 上确认。

## 正式训练、评估与提交

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
引入的 Flat U-Net 来源和 MIT 许可证见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
