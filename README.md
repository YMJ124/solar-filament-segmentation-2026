# Filament Segmentation 2026

模块化的 Solar Filament Segmentation 2026 研究仓库。原始 Kaggle 数据位于
`MAGFiLO_1.0_Kaggle_2026/`，保持只读；代码、清单、fold、评估结果和提交文件分别存放在独立目录。

## 当前能力

仓库目前包含以下可复现链路：

1. 审计原始数据和 COCO 标注；
2. 按观测日期分组并平衡图像、标注者、实例、年份和站点生成固定 folds；
3. 选择 medoid annotator 并构建可校验的 semantic、boundary 和 distance targets；
4. 用配置文件训练预训练 ResNet34-U-Net（Model A）和使用 A 概率图的 EfficientNet-B0 U-Net（Model B）；
5. 用完整图像滑窗计算官方 PQ、IoU/Dice 分布和 1:n/n:1 关系；
6. 融合模型、分离实例并生成带审计报告的 submission CSV。

## 快速开始

```bash
conda activate solar-filament
cd /path/to/Solar\\ Filament\\ Segmentation/filament-segmentation-2026
pip install -r requirements.txt

python -m scripts.audit_dataset
python -m scripts.build_manifest
python -m scripts.build_folds --n-splits 5 --seed 2026
python -m scripts.run_baseline_submission
```

默认输出在 `artifacts/` 下。原始数据路径也可以通过 `--data-root` 覆盖。

## 深度学习流程

先生成 canonical fold/medoid manifest，再构建带指纹的压缩 targets：

```bash
python -m scripts.build_folds --n-splits 5 --seed 2026
python -m scripts.build_mask_cache --output data/cache/medoid_v1
python -m scripts.train --config configs/global_resnet34_fold0.yaml \
  --epochs 1 --max-steps 20 --max-validation-batches 2 \
  --full-pq-max-images 2
```

正式配置默认使用 `256×256`、batch size 4、梯度累积 2、AMP、3 epoch
warmup 和 50 epoch。短训练稳定后移除所有 smoke 限制再启动正式训练。
训练每 5 epoch（以及最后一轮）执行完整验证图滑窗推理，按官方 PQ 而不是
裁剪块 loss 保存 `best.pt`，并将对应 float16 OOF 概率保存到
`artifacts/runs/global_resnet34_fold0/oof_best/`。

实例后处理使用单张 `int32` label map，并默认限制每张图最多 128 个预测实例；
训练 GT 的单 annotator 图像最多 26 个实例。该上限用于阻止早期噪声模型在
2048×2048 图像上生成数百张稠密 mask 导致 RAM 激增，正式调参时应结合 OOF
报告中的 `images_at_safety_cap` 判断是否需要提高。

`last.pt` 采用原子写入，并包含 optimizer、scheduler、AMP scaler、随机数状态、
global step、最佳指标和完整配置；`best.pt` 只保留推理所需权重和溯源信息，
以减少磁盘占用。中断后应从 `last.pt` 继续：

```bash
python -m scripts.train --config configs/global_resnet34_fold0.yaml \
  --resume artifacts/runs/global_resnet34_fold0/checkpoints/last.pt
```

resume 时配置指纹必须一致，以防意外把不同实验接在同一个 checkpoint 上。
使用 `scripts.evaluate_model` 可独立复核整图官方 PQ，使用
`scripts.run_ensemble_submission` 生成并审计提交文件。

### Model B（连续性/细节修复）

Model B 以原始灰度图和冻结 Model A 的 sigmoid 概率图为两通道输入。
图像通道在模型内使用 Fold 0 mean/std 归一化，A 概率通道在模型内
从 `[0, 1]` 映射至 `[-1, 1]`。教师永远处于 `eval` 且不参与反向传播；
训练时可用轻量 dropout/模糊模拟教师误差，验证始终使用干净概率。

正式 Fold 0 训练命令（默认 40 epoch）：

```bash
python -m scripts.train --config configs/detail_efficientnet_b0_fold0.yaml
```

训练每 5 epoch 将以 `A+B` 前向但 `B` 单独权重 `(0, 1, 0)` 执行整图官方
PQ，按 PQ 保存 `best.pt`。对应 OOF 同时保存 `probability`、
`global_probability` 和 `detail_probability`，供后续 A+B 权重搜索。

只验证数据流而不进行正式训练时，可显式关闭预训练权重下载并限制一步：

```bash
python -m scripts.train --config configs/detail_efficientnet_b0_fold0.yaml \
  --encoder-weights none --epochs 1 --max-steps 1 \
  --max-validation-batches 1 --skip-full-pq --num-workers 0 \
  --run-dir /tmp/solar-filament-model-b-smoke
```

独立复核 B 的整图 PQ：

```bash
python -m scripts.evaluate_model \
  --global-checkpoint artifacts/runs/global_resnet34_fold0/checkpoints/best.pt \
  --detail-checkpoint artifacts/runs/detail_efficientnet_b0_fold0/checkpoints/best.pt \
  --fold 0 --threshold 0.325 --min-area 160
```

完整训练前可重复优化同一个固定 batch，排除 loss、反向传播或 target
对齐故障：

```bash
python -m scripts.overfit_batch --steps 20 \
  --output artifacts/smoke/overfit_batch.json
```

fold 0 训练部分的精确灰度统计为 `mean=0.3267324753`、
`std=0.2230217208`，配置中保留 6 位小数。其他 fold 应运行
`scripts.compute_normalization --fold N` 后写入各自配置，不能直接复用 fold 0。

比赛数据来源、官方指标、禁止使用的测试集公开标签以及最终交付缺口记录在
[`docs/competition_compliance.md`](docs/competition_compliance.md)。

## 官方 PQ 评估

项目的主指标对齐 Kaggle `Self Evaluation Notebook` v6。它与标准一对一
Panoptic Quality 不同：每个 IoU 严格大于 `0.5` 的 GT/预测配对都计入
hit pair。同一物理图像只推理一次，但该预测会分别与每个 annotator batch
计分，最后在全部 batch 上汇总 IoU、FP 和 FN。

评估比赛格式 CSV：

```bash
python -m scripts.evaluate_coco \
  --predictions artifacts/validation/predictions.csv \
  --fold 0 \
  --output artifacts/validation/official_metrics.json
```

评估模型 checkpoint：

```bash
python -m scripts.evaluate_model \
  --global-checkpoint artifacts/runs/global_resnet34_fold0/checkpoints/best.pt \
  --fold 0 \
  --output artifacts/validation/global_fold0_metrics.json
```

指标报告会记录 `metric_name=official_pq_v6`、官方 notebook 地址、IoU
阈值、匹配规则、物理图像数、annotator 记录数、正配对 IoU/Dice 直方图、
1:n/n:1 分布、推理配置、耗时和 CPU/GPU 内存峰值。

任何待提交 CSV 都应逐行解码审计：

```bash
python -m scripts.audit_submission \
  --submission artifacts/submissions/ensemble.csv \
  --output artifacts/submissions/ensemble_reaudit.json
```

## 数据约定

JSON 的 `image.id` 是 annotator batch 级别。官方评分会保留所有 batch，
不得把同一物理图像的多标注者 GT union 后评分。

训练时，单标注图像使用该 batch；多标注图像选择与其他标注者语义
mask 平均 IoU 最高的 medoid batch。并列时按 `image.id` 字典序选择。
Dataset 只从 `data/manifests/train_folds.csv` 中读取固定 fold 和
`selected_image_id`，不再在训练时重算 fold 或 union 多标注者标签。

fold group 为文件名中的 `YYYYMMDD`，因此同一天的所有观测不会跨 fold。
`train_folds.meta.json` 保存分组策略、seed、schema 版本和原始标注 SHA256。

## Target cache

target cache 策略名为 `medoid_annotator_v1`。每个缓存项包含：

- selected annotator image ID；
- semantic、boundary 和 distance target；
- 实例 bounding boxes，用于均匀选择实例后再裁剪正样本。

cache manifest 会校验标注 JSON SHA256、fold manifest SHA256、target policy
和 schema version。任何一项变化都会拒绝旧缓存，必须用新目录重建。
