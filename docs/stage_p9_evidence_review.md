# P9 实验后外部对照复盘

检索日期：2026-10-18。

## 本地问题描述

P9-D1 在不可变 T1 上筛选 I1 独占增量，三折平均 PQ 提升
`+0.009883609`，但机制主要是 RQ/召回改善：全局 hit/FN 为 `+208/-208`，
同时增加 284 FP；SQ 从 `0.647245` 略降到 `0.645452`。选择器接纳的真实
正效用 precision 为 53.3%，Fold2 仅 45.0%。待确认问题是：冻结的 P5-C2
能否把新增 I1 coarse mask 修到更高质量，而不因训练分布或集合占用差异损伤
原 T1+C2 分支。

检索词：

- `solar filament YOLO11m 1536 refiner 256 five fold PQ`
- `Solar Filament Segmentation Challenge detector refiner false positives`
- `instance segmentation confidence mask quality score`
- `boundary patch refinement instance segmentation`
- `ExtraTrees impurity feature importance held-out permutation importance`
- `MAGFiLO leaderboard overlap evaluation issue`

## 直接同域社区对照

1. He, *Solar Filament Segmentation 2026 — Two-Stage Pipeline & Fusion Studies*：
   https://github.com/HeShen-1/filament-segmentation-2026

   - 相同点：同一 MAGFiLO/比赛、官方 all-pairs PQ、date-grouped Fold、
     detector→256 crop refiner；报告 YOLO11m@1536 + refiner 的五折均值约
     0.3593。
   - 差异：其 detector 为 YOLO11m、50 epoch，refiner 使用
     `[gray, CLAHE, seed-box]`、25% hard negatives、4-view TTA；本项目 D2
     冻结 YOLO26n/T1 与二通道 256 C2，不可直接比较绝对分数。
   - 对齐结论：支持先完成冻结两阶段链路，也支持在 C2 对新增 mask 失败时，
     只为新增候选改变 refiner 训练来源/负样本，而不是调 D1 阈值。该项目还
     报告同源融合引入相关 FP，故拒绝回到 P8 式投票。

2. Pawar, *Solar filament segmentation (IEEE BigData 2026 Cup)*：
   https://github.com/ameyypawar/filament-seg

   - 相同点：同一数据和 pooled PQ；同样把候选识别与像素边界质量分开，
     cross-fitted validation 报告为 0.4214。
   - 差异：用 full-resolution tiled U-Net 提供像素概率，YOLO11s/m 只负责
     实例归属，并做 limb-darkening correction；验证划分和计算预算不同。
   - 对齐结论：其“detector mask 太粗、语义概率负责像素”的观察支持 D1→C2
     职责拆分，但不采纳为 D2 变量；若未来研究 semantic support，必须作为
     新隔离 challenger，不能恢复旧语义 control 为默认流程。

3. Kaggle 社区关于测试集公开重叠与异常空预测得分的报告：
   https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062
   和 https://www.kaggle.com/discussions/product-feedback/729163

   这些是社区报告，不视为组织方已确认结论；但它们进一步说明 leaderboard
   不适合用来搜索 D1 阈值或 D2 参数。本项目继续只用预注册 OOF 决策。

## 通用论文与官方文档对照

1. Huang et al., *Mask Scoring R-CNN*, arXiv:1903.00241：
   https://arxiv.org/abs/1903.00241

   论文指出分类 confidence 与实际 mask IoU 不对齐，单独学习 mask quality
   score 可提升实例分割。P9-D1 的目标更接近本比赛需求：回归相对不可变
   T1 集合的边际官方-PQ效用。采纳“质量评分应独立于检测 confidence”的方向，
   不照搬 COCO AP 的 MaskIoU 目标。

2. Tang et al., *Boundary Patch Refinement for Instance Segmentation*, CVPR 2021：
   https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html

   论文用高分辨率局部 boundary patches 改善 coarse mask。它与 C2 的边界带
   局部更新一致，但论文评估为 Cityscapes/AP，不能保证对 MAGFiLO/PQ 有相同
   收益。采纳为 D2 使用既有 C2 的外部合理性，不据此增加新结构或带宽搜索。

3. scikit-learn, *Permutation feature importance*：
   https://scikit-learn.org/stable/modules/permutation_importance.html

   官方文档说明树的 impurity importance 可能偏向连续高基数特征并反映训练
   过拟合；held-out permutation importance 更能检查泛化。D1 报告中的特征
   排名仅作解释线索，不能据此删特征或设置新门槛。D2 可报告 held-out 诊断，
   但不得用它回调已经看到结果的 D1。

## 决策

- 已验证并采纳：冻结 D1 选择集，将同一 306 个 source 接入现有折对应 C2；
  D2 相对 P5-C2 三折平均 `+0.010062719`、三折全正，物理图 bootstrap
  95% CI 全部高于 0，3218 个 source 全部保留。
- 部署决策：用全部 OOF 候选/效用训练一次 final selector，固定接到已计分的
  P5 Fold0 T1+C2 测试分支。逐图先精确复现 P5 base-only RLE，再追加并联合
  精修；生成一个可恢复、可审计的官网 challenger。
- 暂缓：YOLO11m@1536 新 detector 和 full-resolution semantic support；
  它们有直接同域证据，但属于 D3 官网回执之后的独立候选。
- 拒绝：围绕 ExtraTrees threshold、tree count、duplicate IoU、追加数量或
  leaderboard 做细调；也不回到同源模型投票。

## D3 部署审计回收

冻结部署在 180 张测试图上选择 82 个增量，P5 的 1240 个实例全部保留
IoU>0.5 对应，最终 1322 行。新增密度为 `+6.61%`，低于此前 P6 的
`+18.0%` 和 P8-E4 的 `+10.3%`；82 个新增实例在年份和观测站上无异常集中。
该结果符合“独立质量评分后保守追加”的研究方向，允许作为一次性官网
challenger；官网回执不得反向用于阈值或数量搜索。
