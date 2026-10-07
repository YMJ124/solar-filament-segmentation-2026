# 单模型优先实验计划

## 原则

- 第一目标是把一个模型的完整 Fold-0 PQ 明显提高；集成只在主模型成熟后补漏。
- 每轮只改变一个模块：骨干容量、输入上下文、训练策略或后处理。
- proxy PQ 只负责淘汰和排序，最终判断必须使用完整 Fold 0 和官方 PQ。
- 不根据 test/Kaggle 反馈反向选择阈值、模型或权重。

## Gate 0：结构与资源（约 1 分钟/候选）

加载模型，检查参数量、输入输出形状和显存。任何 OOM、NaN 或输出尺寸错误直接淘汰。

## Gate 1：固定 batch 可学习性（约 2–5 分钟/候选）

每个候选运行 `scripts.overfit_batch --steps 20`。末三步平均 loss 必须低于第一步；若下降很弱，先排查学习率、归一化和标签，不进入长训练。

## 已完成的第一轮

crop 筛选结果为：ResNet34/256 `0.122442`、ResNet101/256 `0.116156`、
ResNet101/512 `0.009173`、ResNet34/512 `0.000253`。这不能证明 512 分辨率无效：
旧 512 试验在相同 batch 上限下有更少的优化步数，且 crop 丢掉了日面全局上下文。

## Gate 2：整图上下文/损失/结构拆分

参考 Flat U-Net 方案后，只借鉴三个可独立验证的思路：

1. 将完整 `2048×2048` 日面缩放到 `512×512`，训练和验证保持一致；
2. 用每 batch 的 `sqrt(negative/positive)` 作正类 BCE 权重，上限 20，再加 Dice；
3. 加入等通道、通道注意力的 Flat U-Net，但不引入其随机文件分折或多标注者直接复制。

两条候选都使用固定 32 图面板、3 epoch、每 epoch 最多 80 mini-batch、
batch 2 与梯度累积 2，即每 epoch 最多 40 次 optimizer update。

```bash
conda activate solar-filament
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

python -m scripts.train \
  --config configs/screen/resnet34_full512_balanced.yaml

python -m scripts.train \
  --config configs/screen/flatunet_c32_full512_balanced.yaml

python -m scripts.summarize_screens
```

这是模块化因果对照：

- `ResNet34 full512 balanced` 对比旧 `ResNet34/256 crop`，主要测整图上下文和类别平衡；
- `FlatUNet C32 full512 balanced` 对比前者，主要测结构本身。

按以下顺序排序：

1. proxy 官方 PQ；
2. TP 增加且 FP/FN 没有异常恶化；
3. 显存和每图推理时间；
4. 三轮趋势是否仍在上升。

不要用绝对 proxy PQ 预测最终 PQ。以已知 `0.122442` 为同面板参考：

- `< 0.12`：淘汰，先不加长训练；
- `0.12–0.15`：只在 TP/FP/FN 或上升趋势明显更好时保留；
- `>= 0.15`：进入 5–10 epoch 中程确认；
- `>= 0.20`：作为优先主模型候选。

## Gate 3：完整 Fold 0 确认

只保留 proxy 前两名。复制胜者配置并取消四项限制：

- `data.validation_file_list: null`
- `training.max_steps: null`
- `training.max_validation_batches: null`
- `validation.max_images: null`

先做 5–10 epoch 中程训练；若完整 Fold-0 PQ 无稳定上升或仍明显低于旧基线，则停止。只有中程有效才运行 40–80 epoch，并按完整 Fold-0 PQ 保存最佳模型。

## 后续模块顺序

1. 先完成当前整图输入/平衡损失/Flat U-Net 对照。
2. 从胜者出发，一次只比较一项：损失、采样、容量、分辨率。
3. 若语义 Dice 上升而 PQ 仍被粘连/断裂限制，再引入边界监督或直接实例分割；不先恢复模型 B。
4. 若 512 整图明显获益，再评估 768/1024；若细节损失明显，则比较高分辨率实例方案。
5. 只对胜者的 OOF 搜索 threshold/min_area/closing。
6. 最后才考虑第二模型；它必须在明确失败子集上提供互补，且完整 OOF 增益达到预设门槛。

公开方案显示高分辨率直接实例分割值得作为后续独立候选，但不能仅凭公开分数替换本地验证协议：

- Wzy633 的公开实现报告 Fold-0 PQ 0.35259，采用 YOLO 分割与更高训练/推理分辨率。
- HeShen 的公开两阶段方案报告 Fold-0 PQ 0.3661。

这些结果说明当前 0.14 并非数据集上限，同时提示“实例建模与分辨率”可能比简单增加语义 U-Net 参数更重要。
