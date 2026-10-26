# P5 极细实例跨领域技术复盘

> 检索日期：2026-10-17
> 本地决策指标：三折官方 PQ v6，不使用公开榜分代替 OOF
> 范围：遥感小目标、超高分辨率分割、细血管/神经/道路/裂缝、显微镜细长实例与通用实例分割

## 1. 本地问题与 P5-D0 结果

P4-T1 的主要残差不是单一问题：

- 2526 个 FN 中，1422 个（56.29%）与最终预测完全无重叠，更像候选缺失/排名问题；
- 873 个（34.56%）落在 `IoU (0.25, 0.5]`，更像高分辨率边界和连通性问题；
- 1536 推理下宽度仍不足一个 stride-4 prototype cell 的 GT 漏检率为 91.80%。

P5-D0 只将 `max_instances=20` 改为 `50`，其他条件不变：

| 指标 | cap=20 | cap=50 | 增量 |
|---|---:|---:|---:|
| 三折平均 PQ | 0.325102087 | 0.326106973 | +0.001004887 |
| 全局官方 PQ | 0.324845179 | 0.325787049 | +0.000941870 |
| Fold 0 PQ | 0.337522715 | 0.339145025 | +0.001622310 |
| Fold 1 PQ | 0.323752928 | 0.325321206 | +0.001568278 |
| Fold 2 PQ | 0.314030616 | 0.313854689 | -0.000175928 |
| Hit pairs | 2458 | 2516 | +58 |
| FP | 2353 | 2487 | +134 |
| FN | 2526 | 2468 | -58 |
| 预测实例 | 2912 | 3038 | +126 |

它通过了2/3 Fold 正增益、最差 Fold 和方差约束，但平均仅
`+0.001004887 < +0.002` 预注册门槛，因此**不采纳 cap=50**。额外
候选确实救回了一部分 FN，但 FP 增长更快；而且只救回 58/2526
个当前 FN，候选数截断不是主瓶颈。这将问题转向：

1. 低置信原始候选是否已经覆盖 GT；
2. 候选排名是否错把掩膜质量与分类置信当成同一件事；
3. 真正没有原始候选的 GT 是否需要高分辨率局部识别。

## 2. 检索词

- `tiny object detection high resolution feature pyramid P2`
- `slicing aided inference small object detection SAHI`
- `coarse to fine high resolution object detection chips`
- `high resolution local crop segmentation refinement`
- `boundary patch refinement instance segmentation`
- `tubular structure segmentation topology connectivity loss`
- `skeleton recall clDice cbDice vessel road crack segmentation`
- `distance transform tubular structure segmentation`
- `elongated branched instance segmentation Omnipose`
- `mask quality score confidence ranking instance segmentation`

## 3. 候选召回：小目标与粗到细局部计算

### 3.1 SAHI / 重叠切片推理

SAHI（ICIP 2022）将高分辨率图像切成重叠块，分别推理后聚合。
论文在 VisDrone/xView 上报告，仅切片推理就使 FCOS/VFNet/TOOD 的 AP
分别提升 6.8/5.1/5.3 个点，加上切片微调后累计提升
12.7/13.4/14.5 个点。

- 论文：<https://arxiv.org/abs/2202.06934>
- 官方实现：<https://github.com/Small-Object-Detection/SAHI>
- 与本项目相同：原图尺寸大，全图缩放后细节损失，可先用现有
  checkpoint 做无训练消融。
- 不同：遥感小目标通常是小包围框；太阳细丝可能“很窄但很长”，
  切片会截断同一实例，必须解决重叠块之间的 mask 拼接和去重。
- 决策：**保留为召回路线，但不直接照搬 SAHI 框聚合**。先完成原始
  候选 oracle 审计；若大部分无重叠 GT 在 `conf>=0.01` 原始候选中仍
  不存在，再测重叠切片。

### 3.2 AutoFocus / QueryDet / SNIPER

- AutoFocus（ICCV 2019）先在粗尺度预测 FocusPixels，只对可能含小目标的
  FocusChips 做更细尺度推理；其处理像素可减少 5 倍，mAP 只下降
  约 1 个点。
  <https://openaccess.thecvf.com/content_ICCV_2019/html/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.html>
- QueryDet（CVPR 2022）用粗层 query 稀疏启用高分辨率检测，COCO 上
  `AP_small +2.0`，高分辨率推理平均加速 3 倍。
  <https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html>
- SNIPER（NeurIPS 2018）在训练时只处理 GT 周围合适尺度的 chips，并配合负样本
  chip mining。<https://papers.nips.cc/paper/2018/hash/166cee72e93a992007a89b39eb29628b-Abstract.html>

这三项的共同价值是“将昂贵的高分辨率计算放到少数区域”。但本项目
只有一类目标且细丝可横跨大区域，不值得立即移植 QueryDet/SNIPER 整套
检测器。采纳其原则，实现上优先使用更小的“全图粗定位 + 局部细化”。

### 3.3 P2/FPN 高分辨率检测头

FPN 通过自顶向下与横向连接为高分辨率浅层注入语义，是小目标检测的
通用基础：
<https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html>。

但对本项目必须先区分：

- 若漏检 GT 的包围框也很小，stride-4 P2 检测头有直接机理支持；
- 若包围框很大而掩膜只是极窄，P2 检测头未必解决 prototype 下采样和
  掩膜分类问题。

因此 **P2 是条件候选，不是当前默认首选**。需先报告无重叠 GT 的
box 短边、mask 有效宽度与原始候选可恢复率。

## 4. 高分辨率局部精修

### 4.1 粗掩膜 + 原图 crop 的级联精修

- CascadePSP（CVPR 2020）把任意粗掩膜与高分辨率图像组合，由粗到细修正
  4K 图像边界。
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_CascadePSP_Toward_Class-Agnostic_and_Very_High-Resolution_Segmentation_via_Global_and_CVPR_2020_paper.html>
- Boundary Patch Refinement（CVPR 2021）只裁剪粗掩膜边界周围的高分辨率
  patch，Mask R-CNN 在 Cityscapes 上报告 `+4.3 AP`。
  <https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html>
- PointRend（CVPR 2020）只在不确定点逐级细化；RefineMask（CVPR 2021）多阶段融合
  细粒度特征，分别支持“稀疏精修”和“边界特征融合”：
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html>，
  <https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html>。

对本项目最合适的不是修改 YOLO prototype head，而是保持 T1 识别结果，
为每个候选构造带上下文的局部 crop，输入原图、粗掩膜，可选局部对比度
通道，输出实例精修掩膜。这与比赛中 YOLO + crop U-Net 的成功结构一致，
且可冻结 T1，把变量限制在 refiner。

限制是：它只能修正已有候选，不能单独救回真正没有原始候选的 GT。

### 4.2 全局-局部融合

GLNet 为超高分辨率分割同时处理低分辨率全图与高分辨率局部块，用融合
避免“全图丢细节、切块丢上下文”：<https://arxiv.org/abs/1905.06368>。

这对横跨大范围的细丝很有启发，但完整移植为新主模型成本过高。建议只吸收
一个设计：refiner 除局部 crop 外，还接收来自 T1/全图的粗 mask 或位置先验，
不让局部模型把一条长细丝切成多个实例。

## 5. 边界、中心线与拓扑约束

### 5.1 soft-clDice

clDice（CVPR 2021）用预测/真值的软骨架与掩膜交叉度量中心线完整性，
在血管、道路和神经数据上改善连通性和体积指标：
<https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html>。

它与细丝连通性高度匹配，但不应单独使用：只强调连通可能让附近实例粘连，
而比赛 PQ 同时惩罚合并与 FP。因此只在**单实例 crop refiner** 中与 BCE/Dice/
边界项组合。

### 5.2 Skeleton Recall Loss

Skeleton Recall Loss（ECCV 2024）预先对 GT 做骨架管状化，训练时只优化预测对
GT 骨架的软召回。论文在五个血管/道路/裂缝数据集上验证，相对其他拓扑
损失计算开销减少 90% 以上。

- 论文：<https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9904_ECCV_2024_paper.php>
- 官方实现：<https://github.com/MIC-DKFZ/Skeleton-Recall>

它是本项目**首选的轻量拓扑对照**，因为 GT 骨架可离线计算，不需要
每步做可微骨架化。但它本质上只优化 recall，必须与 BCE/Dice 同用来控制
过生长和 FP。

### 5.3 cbDice

cbDice（MICCAI 2024）在 clDice 上同时引入边界与中心线半径，目标是避免普通
Dice/clDice 偏向粗血管，并平衡不同直径分支。

- 论文：<https://papers.miccai.org/miccai-2024/129-Paper0458.html>
- 官方实现：<https://github.com/PengchengShi1220/cbDice>

它在机理上比 clDice 更贴合“细丝宽度变化大”，但实现和超参更复杂。
决策为第二个拓扑候选：只有轻量 Skeleton Recall 已证明方向有效后再测，
不与第一版 refiner 同时引入。

### 5.4 距离场与细长实例表示

Deep Distance Transform（CVPR 2020）让网络同时预测掩膜和到边界的距离场，
隐式编码中心线和截面半径；在其胰管实验中相对 3D U-Net 报告超过
13% Dice 提升：
<https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Deep_Distance_Transform_for_Tubular_Structure_Segmentation_in_CT_Scans_CVPR_2020_paper.html>。

Omnipose（Nature Methods 2022）预测距离场梯度并重建实例，专门解决 Cellpose 对
细长、分支形态过分割的问题：
<https://www.nature.com/articles/s41592-022-01639-4>，
<https://github.com/kevinjohncutler/omnipose>。

这两项对“细长实例不适合单中心收缩”的启发很强，但是重做实例表示会
改变主模型和实例聚合逻辑。近期只采纳**距离场辅助头**这个可分离部分，
不直接用 Omnipose 替换 T1。

## 6. 候选排名：D0 新暴露的路线

Mask Scoring R-CNN（CVPR 2019）指出，分类置信与掩膜 IoU 并不充分相关，用实例
特征和预测 mask 回归 mask IoU，再用质量分数排序：
<https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html>。

D0 中 cap=50 的低排名候选带来 58 个额外 hit，同时带来 134 FP，这与
“候选存在，但分类置信不能代表掩膜质量”的机理相容，但尚不是证明。
先做无训练 oracle 审计；只有在大量 GT 已被低置信原始 mask 高 IoU 覆盖时，
才开发轻量 mask-quality 重排名器，避免无证据增加新训练分支。

## 7. P5-D1 本地结果对外部证据的校正

P5-D1 实测显示，1422 个最终无重叠 GT 中有 874 个（61.46%）已经被
原始 mask 以 `IoU>0.5` 覆盖；三个 Fold 均在 58.77%–64.25%。这使
Mask Scoring R-CNN 的“分类置信不等于 mask 质量”从机理类比升级为
当前最直接的本地解释：526 个好候选低于 confidence=0.10，另有 271 个
被面积过滤、77 个在 top-20 之外。

相反，原始候选仍 `IoU<=0.25` 的无重叠 GT 只有 274/1422（19.27%），
未达到预注册的新候选生成门槛。因此 SAHI、P2、AutoFocus/QueryDet 等
路线仍有价值，但不再位于当前第一顺位。缓存达到 50 上限的比例很高，
所以 19.27% 只能视为当前缓存范围内的下界，不能声称检测器绝对没有候选。

固定网格与 D0 已分别排除“简单降低全局阈值”和“简单增大 cap”作为
合格解法。P5-R1 将只用测试时可得的置信度、面积、包围框、形状和排名
特征做 OOF 质量重排；这比直接加入检测头更贴合当前证据，也能保持
单变量、可回退和官方 PQ 可验证。

## 8. 结论与本地采纳顺序

### 第 0 优先级：原始候选可恢复性审计（已完成）

对 1422 个最终无重叠 GT，使用现有 `conf>=0.01`、`max_candidates=50`
缓存，在不应用 `.10/600/20` 截断前计算最大 mask IoU，并按以下两类分流：

- 原始候选 `IoU>0.5` 但被过滤/排名丢掉：候选质量重排名/校准问题；
- 原始候选仍无重叠或 `IoU<=0.25`：真正识别/特征分辨率问题。

压缩 RLE 实现避免了全图 dense 解码，完整审计实测约 2.2 秒。

### 第 1 优先级：OOF 候选质量重排

每个验证 Fold 的评分器只用另两个 Fold 训练；第一版固定概率阈值，不在
验证 Fold 搜索阈值。保持 `min_area=600/max_instances=20/force_one=true`，
只改变候选质量评分与排序。该顺序直接对应本地 874 个已存在的好候选。

首版 R1 固定 `quality>=0.5` 后平均 PQ 下降 `0.011067`。外部对照暴露了
实现差异：Mask Scoring R-CNN 使用实例图像特征回归连续 MaskIoU 并与
分类分数组合，Rank & Sort 则直接优化相对排序；R1 的几何二分类分数不应
被假定为跨 Fold 的绝对 0.5 决策概率。由于其 held-out AUC/AP 仍在每折
优于原 confidence，只允许再做一次固定基线配额的纯重排 R2；若失败，
停止该支线，不再进行阈值细调。

R2 在固定每图候选配额下仍下降 `0.005347`。本地失败机制是独立候选
质量标签不能表达重复候选之间的集合效应：重排后独占像素和面积过滤使
最终预测少 361。Learning NMS 与 Beyond mAP 都支持必须联合处理重复
候选；但 MMDetection 的 mask-level NMS 讨论也表明框重叠不能代表 mask
重复。对长而相邻的细丝，引入 learned/matrix NMS 已是新的复杂支线，
因此不继续 R3，按原顺序进入局部 refiner。

### 第 2 优先级：冻结 T1 的局部 refiner

若 oracle 表明近阈值/有候选残差占比可观，第一版采用：

- 带 1.5x 上下文的实例 crop，固定到 256²；
- 输入原图 + 粗 mask，可选增加 CLAHE 通道；
- 轻量 U-Net/ResNet-18 U-Net，冻结 T1；
- 基线损失先用 BCE + Dice + 边界加权；
- 只在该基线通过 Fold0 固定面板后，单变量增加 Skeleton Recall；
- cbDice 和距离场辅助头留作后续二选一，不与首版同时堆叠。

### 第 3 优先级：真正候选缺失时提高召回

- 若漏检主要是小 box：再测 P2/stride-4 检测头或更大检测器；
- 若主要是大 box + 极窄 mask：优先重叠切片推理/训练，不盲目加 P2；
- 切片必须同时保留全图分支，并对跨块同一实例做 mask 拼接，不能只用框 NMS。

### 明确不采纳

- 不因 cap=50 的微小正增益搜索 cap=25/30/35/40；
- 不把 `retina_masks` 当成召回方案，它只保留为近阈值边界对照；
- 不同时堆叠切片、新检测头、refiner 和拓扑损失，否则无法归因；
- 不直接用 Cellpose/Omnipose 预训练模型；其领域和实例密度与太阳细丝差异大，
  只借鉴距离场/流场表示。

## 8. 证据强度

- **A（直接且可近期验证）**：比赛的 boundary-weighted U-Net/裁剪 refiner；
  本地 D0；SAHI 切片原理；Skeleton Recall/soft-clDice。
- **B（机理相关但需改造）**：AutoFocus/QueryDet/SNIPER；PointRend/RefineMask/
  CascadePSP/BPR；cbDice/DDT。
- **C（长期架构候选）**：Omnipose 距离场实例重建、完整 GLNet、完整
  QueryDet 移植。

外部论文的 AP/Dice 与本项目 PQ 不可直接比较；上述数值只用于评估方法在
原数据集上是否有实验支持，本地采纳仍须通过固定面板与三折门槛。

## 9. P5-C1 Fold0 面板后的外部对照复盘

> 复盘日期：2026-10-17
> 本地现象：局部精修 `PQ +0.031561`、hit `+7`、FP `-28`、FN `-7`，
> 但 10/32 图退化，3 图因实例被删各损失一个 hit

### 9.1 检索描述与范围

- `local crop mask refinement coarse mask boundary instance segmentation`
- `PointRend custom dataset masks not reaching edge inconsistent`
- `RefineMask fine grained multi stage boundary refinement`
- `Mask Transfiner error prone pixels boundary AP`
- `instance mask paste threshold interpolation quantization error`

检索覆盖 PointRend、RefineMask、Mask Transfiner 的论文/官方实现，以及
Detectron2 mask paste 源码和相关 GitHub issue。未找到太阳细丝、官方
PQ v6、二通道 crop U-Net 与 `min_area` 联动的直接同类实验，因此下述结论
是机制对照，不是外部结果的数值复现。

### 9.2 与论文/社区结果的相同点和差异

- PointRend 通过在不确定位置逐级细化改善被低分辨率模型过度平滑的边界，
  说明“冻结实例识别、只修 mask”是已有实验支持的方向：
  <https://arxiv.org/abs/1912.08193>。与本项目不同，它使用主干的点特征和
  自适应 subdivision；P5-C1 对整个 256² crop 做密集预测。
- RefineMask 多阶段融合细粒度特征，在 COCO/LVIS/Cityscapes 相对
  Mask R-CNN 报告 `+2.6/+3.4/+3.8 AP`：
  <https://arxiv.org/abs/2104.08569>，
  <https://github.com/zhanggang001/RefineMask>。本地 hit IoU 均值
  `0.657687 -> 0.668903` 与其“边界质量提升”机理一致，但本项目的 PQ、
  灰度太阳图和固定 T1 crop 与这些自然图像 AP 不可直接比较。
- Mask Transfiner 只处理 error-prone quadtree nodes，并报告 COCO/BDD100K
  mask AP `+3.0`、Cityscapes boundary AP `+6.6`：
  <https://arxiv.org/abs/2111.13673>。它支持下一步优先诊断局部错误，而非
  无证据扩大整个 refiner；但其 transformer/quadtree 成本不适合作为当前
  单变量直接移植。
- Detectron2 官方 mask paste 实现明确把连续概率经插值后再用固定 threshold
  二值化，并说明旧实现存在更大 quantization error：
  <https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/mask_ops.py>。
  这与本地“精修后 14 个实例低于 min_area，3 图因此丢 hit”的风险直接相关：
  threshold、resize 与面积过滤是一个联动系统，不能把删除都解释为学会了背景。
- Detectron2 issue #1352 有用户报告 PointRend 在自定义数据上部分 mask
  无法延伸到实例边缘且表现不一致：
  <https://github.com/facebookresearch/detectron2/issues/1352>。该 issue 没有
  权威解决方案且被标为 invalid/unrelated，只能证明社区出现过相似现象，
  不能作为调参依据。

### 9.3 本地归因与决策

P5-C1 的 hit IoU sum 增加 6.701065，官方分母从 339.0 降至 328.5；若只
替换分母而保持原 hit IoU sum，PQ 约增加 0.011162；若只替换分子而保持
原分母，约增加 0.019767。因此当前 `+0.031561` 同时来自更好的边界/新增
hit 和更少 FP，不是纯粹通过删除预测抬分。

风险仍然明确：收益分布不均，21 图提升、1 图不变、10 图下降；最差三张
删掉实例后各出现 `hit -1/FN +1`。由于 Fold0 是筛选面板，不能用这些
退化图选择 threshold、min_area、epoch 或 fallback 规则。

决定：

1. 保持 P5-C1 架构、loss、0.5 threshold 和固定 T1 候选语义不变，进入
   三折 OOF 验证；不因 Fold0 成功立即加入 Skeleton Recall。
2. 三折报告除 PQ 门槛外，必须统计“精修后实例删除、删除后 hit 丢失、
   同实例数下 hit/IoU 改变”，验证收益是否跨折稳定。
3. 若三折因实例删除失败，下一候选才考虑单变量的 coarse-mask fallback；
   若实例数不变但狭长边界仍退化，再考虑 Skeleton Recall。两者不得同时改。

## 10. P5-C1 正式三折后的外部对照复盘

> 复盘日期：2026-10-17
> 本地现象：三折 PQ 全部提升，全局 `0.324845179 -> 0.337869755`
>（`+0.013024576`），但 hit `-182`、FP `-592`、FN `+182`；422 张图中
> 274 张发生实例删除，144 张在删除后损失至少一个原命中

### 10.1 检索描述与范围

- `segmentation refinement preserve coarse mask interior boundary only`
- `SegFix boundary pixels reliable interior instance segmentation`
- `Boundary Patch Refinement instance segmentation coarse mask`
- `RefineMask boundary-aware refinement complete mask ablation`
- `CascadePSP segmentation refinement failure case initial mask`

检索覆盖 SegFix、Boundary Patch Refinement、RefineMask、PointRend、
CascadePSP 的论文与官方实现。仍未找到太阳细丝、官方 PQ v6、固定
`min_area=600` 与 crop U-Net 联动的直接同类实验；下面是机制对照，不能把
自然图像 AP/mIoU 数值直接迁移为本项目的预期增益。

### 10.2 外部方法如何避免全掩膜覆盖风险

- SegFix 的出发点是内部像素预测比边界可靠，只定位边界像素，并用指向
  内部的偏移替换边界标签；它可作为已有语义或实例分割器的模型无关后处理：
  <https://arxiv.org/abs/2007.04269>，
  <https://github.com/openseg-group/openseg.pytorch>。这与本地“实例数不变时
  精修稳定增益、整实例被覆盖删除时召回受损”最直接对齐。
- Boundary Patch Refinement 根据粗分割抽取跨越前景/背景边界的小块，只在
  这些局部块内做高分辨率修正，而不是重新预测整个实例：
  <https://github.com/chenhang98/BPR>。其 Cityscapes boundary 指标与本项目
  PQ 不同，但“限制精修作用域”的设计可直接用于降低 C1 的内部/拓扑破坏。
- RefineMask 用多阶段、边界感知方式融合细粒度特征；官方配置显式采用
  `boundary_width=2`，而不是只靠一次完整 mask 覆盖：
  <https://arxiv.org/abs/2104.08569>，
  <https://github.com/zhanggang001/RefineMask>。本项目没有其主干细粒度特征，
  所以不照搬网络，只借鉴“边界带内更新、稳定内部”的约束。
- PointRend 只在不确定点进行自适应细化，CascadePSP 则同时采用全局和局部
  步骤；CascadePSP 官方仓库也展示了局部结构被错误修改的失败例：
  <https://arxiv.org/abs/1912.08193>，
  <https://github.com/hkchengrex/CascadePSP>。它们共同说明高分辨率精修并不
  自动保证拓扑或召回，必须限制修改位置或保留粗预测先验。

### 10.3 本地分解与方案取舍

按精修前后实例数分组重新聚合官方 PQ：

| 分组 | 图像数 | PQ（T1 -> C1） | hit Δ | FP Δ | FN Δ | hit IoU sum Δ |
|---|---:|---:|---:|---:|---:|---:|
| 实例减少 | 274 | `0.316487 -> 0.326420` | -203 | -571 | +203 | -90.4383 |
| 实例不变 | 148 | `0.348354 -> 0.366626` | +21 | -21 | -21 | +23.4705 |
| 实例增加 | 0 | - | 0 | 0 | 0 | 0 |

这证明两种效应同时存在：C1 在不删除实例时确实改善匹配与边界；删除分支
则用显著降低 FP 换取部分 TP。若对所有“实例减少”的整张图直接回退 T1，
可由现有逐图报告精确重算得到 `PQ=0.329637527`，相对 T1 仅
`+0.004792348`，低于 C1 正式门槛 `+0.008`。因此拒绝“只要实例减少就整图
回退”的粗粒度方案，它会同时撤销大量有效的 FP 清理。

决定：

1. P5-C1 按预注册门槛正式晋级；其定位是提高官方 PQ，不宣称提高召回。
2. 不事后修改 C1 的 threshold、`min_area` 或晋级门槛，也不把 144 张丢 hit
   图当作新规则的训练标签。
3. 下一候选只允许一个推理时变量：把 C1 的全 crop 覆盖改为
   **边界带局部融合**，粗掩膜深内部保持不变，边界带宽直接复用已固定的
   boundary kernel 半径，不新增阈值网格、不重训。先预注册相对 C1 的
   PQ/召回门槛，再在同一三折 OOF 报告上评估。
4. 暂不同时加入 Skeleton Recall、置信度门控或逐实例 fallback；若边界带
   融合无法在基本保留 C1 PQ 的同时恢复 hit，再单独转入检测召回阶段。

## 11. P5-C2 正式三折后的外部对照复盘

> 复盘日期：2026-10-17
> 本地现象：边界带融合相对 T1 全局 PQ `+0.014719180`、hit `+55`、
> FP `-55`、FN `-55`，2912 个来源实例全部保留；相对 C1 平均 PQ 再
> `+0.001740729`，恢复 237 个 hit

### 11.1 本轮检索

- `Boundary Patch Refinement instance segmentation boundary patches`
- `SegFix reliable interior pixels boundary refinement`
- `segmentation refinement preserve topology boundary only`
- `boundary preserving mask instance segmentation official implementation`

检索再次覆盖 BPR 的 CVPR 论文/官方实现、SegFix 的 ECCV 论文，以及边界
保持实例分割工作。没有找到太阳细丝、官方 PQ v6、固定来源实例槽位保护的
直接同类报告，因此只比较机制和方向，不直接比较绝对数值。

### 11.2 与外部证据的对应关系

- BPR 只抽取预测实例边界附近的小 patch 进行高分辨率修正，并在 Cityscapes
  的 boundary-aware 指标上获得主要收益：
  <https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html>，
  <https://github.com/chenhang98/BPR>。C2 同样把 C1 的作用域从完整 crop
  缩到固定边界带，本地 300/422 图提升、93 图下降，优于 C1 的
  241/422 提升、154 图下降；方向一致，但网络、数据和指标不同。
- SegFix 明确以“内部像素通常比边界可靠”为动机，只定位并替换边界像素：
  <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1633_ECCV_2020_paper.php>。
  C2 的深内部保持不变后，SQ 从 T1 的 `0.647245` 升至 `0.661765`，同时
  RQ 从 `0.501889` 升至 `0.513119`；这比 C1 的高 SQ/低召回权衡更均衡。
- BPR 论文还报告其边界模块可叠加到已有精修结果上。这支持把边界精修作为
  可插拔末端分支，而不是覆盖检测器或原始候选资产；本项目进一步加入来源
  粗支撑域预留和逐来源 min-area 恢复，属于本地可逆性约束，并非论文原方法。

### 11.3 本地结论

C2 的 214 个逐来源 fallback 分布在 158 张图上，但它们只恢复对应 T1 粗
mask，没有整图回退。最终三折每一张图的实例数都与 T1 相同，且：

- T1：PQ `0.324845179`，TP/FP/FN `2458/2353/2526`；
- C1：PQ `0.337869755`，TP/FP/FN `2276/1761/2708`；
- C2：PQ `0.339564359`，TP/FP/FN `2513/2298/2471`。

因此 C2 同时支配 T1 的 PQ、TP、FP、FN，并在全局 PQ 与 hit 上优于 C1；
Fold2 相对 C1 的 `-0.001967233` 是唯一折级退化，但小于预注册容忍值
`0.005`。所有 T1/C1 双重门槛和可逆性检查均通过。

决定：

1. P5-C2 取代 C1 作为默认 submission 分支；C1 报告和 checkpoint 保留，
   不删除、不覆盖。
2. T1 原始 cache 仍是后续研究基线和候选入口；C2 的最终二值结果不得成为
   下一阶段唯一输入。
3. P5 到此结束，不搜索 boundary width、threshold 或 fallback 条件。
4. 下一阶段若提高检测召回，必须从 T1 过滤前候选池或新检测器分支进入，
   保留 source ID 后再接 C2；不能在 C2 输出后凭空恢复候选。
