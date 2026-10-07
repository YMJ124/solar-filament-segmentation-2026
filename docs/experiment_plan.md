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

## Gate 2：等预算 proxy（目标不超过 30 分钟/候选）

四个 `configs/screen/*.yaml` 都使用 3 epoch × 最多 80 batch、相同有效 batch 8，以及固定 32 图面板。按以下顺序排序：

1. proxy 官方 PQ；
2. TP 增加且 FP/FN 没有异常恶化；
3. 显存和每图推理时间；
4. 三轮趋势是否仍在上升。

不要用绝对 proxy PQ 预测最终 PQ；优先保留比 ResNet34/256 高至少 0.02 PQ，或在相近 PQ 下显著更快的候选。

## Gate 3：完整 Fold 0 确认

只保留 proxy 前两名。复制胜者配置并取消四项限制：

- `data.validation_file_list: null`
- `training.max_steps: null`
- `training.max_validation_batches: null`
- `validation.max_images: null`

先做 5–10 epoch 中程训练；若完整 Fold-0 PQ 无稳定上升或仍明显低于旧基线，则停止。只有中程有效才运行 40–80 epoch，并按完整 Fold-0 PQ 保存最佳模型。

## 后续模块顺序

1. 容量与上下文：当前 2×2。
2. 监督/裁剪：正样本比例、实例均匀采样、难负样本。
3. 损失：Dice+BCE 基线，再单独比较 focal/Tversky。
4. 分辨率：若 512 明显获益，再评估 768/1024 或高分辨率实例方案。
5. 后处理：只对胜者 OOF 搜索 threshold/min_area/closing。
6. 最后才考虑第二模型；它必须在明确失败子集上提供互补，且完整 OOF 增益达到预设门槛。

公开方案显示高分辨率直接实例分割值得作为后续独立候选，但不能仅凭公开分数替换本地验证协议：

- Wzy633 的公开实现报告 Fold-0 PQ 0.35259，采用 YOLO 分割与更高训练/推理分辨率。
- HeShen 的公开两阶段方案报告 Fold-0 PQ 0.3661。

这些结果说明当前 0.14 并非数据集上限，同时提示“实例建模与分辨率”可能比简单增加语义 U-Net 参数更重要。
