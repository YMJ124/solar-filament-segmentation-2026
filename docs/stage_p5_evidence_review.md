# P5 候选前的本地残差与外部证据对照

> 检索日期：2026-10-17

## 本地问题定义

P4-T1 与 P4-I1 在同一 1536 推理、`.10/600/20/force_one` 后处理下
按 GT annotation 配对。T1 从 I1 救回 460 个 GT，但同时退化 407 个，
净救回只有 53。净收益完全来自面积 `[400,1600)` 的 `+75`；
`<400`、`[1600,6400)` 和 `6400+` 分别为 `-8/-13/-1`。

T1 仍有 2526 个 FN：1422 个（56.29%）与任何预测无重叠，873 个
（34.56%）位于 `IoU (0.25,0.5]`，231 个（9.14%）位于 `(0,0.25]`。
1536 下仍小于一个 stride-4 prototype cell 的 183 个 GT 中漏掉 168 个
（91.80%），且 T1 相对 I1 净退化 4 个。

T1 在 98/422 图（23.22%）上有超过 20 个 `confidence>=0.10` 候选；
87 图在 top-20 之外仍有原始面积至少 600 的候选，共 385 个。
当前算法先截取 top-20，再执行重叠去除和最小面积过滤，因此最终
输出最多只有 15 个并不能证明 cap 没有截断有效候选。在 T1 cap 压力图上
净救回为 `+52`，其他 324 图只有 `+1`；这只说明增益集中于密集图，
尚不能将其解释为 cap 的因果效应。

## 检索词与来源

检索词包括：

- `Ultralytics YOLO segmentation thin objects retina_masks mask_ratio small objects`
- `YOLO26 mask prototype imgsz/4 mask_ratio`
- `instance segmentation high resolution mask boundary refinement`
- `PointRend RefineMask Mask Transfiner fine-grained boundaries`
- `Ultralytics segmentation max_det missing objects`

主要对照：

1. Ultralytics issue #26436：
   <https://github.com/ultralytics/ultralytics/issues/26436>
2. Ultralytics issue #24084：
   <https://github.com/ultralytics/ultralytics/issues/24084>
3. Ultralytics `retina_masks` 预测文档：
   <https://docs.ultralytics.com/modes/predict/>
4. Ultralytics issue #4536：
   <https://github.com/ultralytics/ultralytics/issues/4536>
5. PointRend, CVPR 2020：
   <https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html>
6. RefineMask, CVPR 2021：
   <https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html>
7. Mask Transfiner, CVPR 2022：
   <https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html>

进一步扩展到遥感小目标、超高分辨率图像、细血管/神经/道路/裂缝与
显微镜细长实例的完整检索、条件差异和采纳判断见：
`docs/stage_p5_cross_domain_review.md`。

## 适用性对照

### `mask_ratio=2`：拒绝作为预测分辨率修复

Ultralytics #26436 的复现和维护者回复都区分了训练 target 分辨率与
prototype 原生输出分辨率：`mask_ratio` 可改变前者，但后者仍是
`imgsz/4`。本项目安装的 Ultralytics 8.4.138 也已本地核对：YOLO26
`Proto26` 融合 P3/P4/P5 后只做一次 2× 上采样，而 loss 在 target 比
prototype 更细时会插值 prototype。因此将 `mask_ratio=2` 解释为“让模型
原生预测更细”是错的。它仍可作为更细监督实验，但必须另立假设，
不作为 P5 首选。

### `retina_masks=true`：只作为近阈值边界消融

官方文档将 `retina_masks` 定义为返回高分辨率 mask；#4536 有用户报告
改变先缩放还是先裁剪的路径后精度提升。但本地 8.4.138 源码显示
`process_mask` 与 `process_mask_native` 都从同一 prototype logits 开始，所以它
不会补回 1422 个完全无重叠 FN 中缺失的候选。它与 873 个
`IoU (0.25,0.5]` 残差的机制更匹配，因此只应在 cap 审计后作固定
Fold0 面板的单变量消融。

### 边界精修论文：机制相符，但现阶段过重

PointRend 在不确定点上自适应加密；RefineMask 逐级融合细粒度特征并
聚焦边界；Mask Transfiner 只精修下采样中丢失信息的稀疏区域。
这些方法支持“不必给整个 backbone 继续加分辨率，而应将计算放在
边界/高频位置”。但它们需要新 mask head 或 refiner，且主要解决粗糙边界；
本项目剩余 FN 中 56.29% 是无重叠候选，直接上这些模块还不足以解决
主要召回问题，暂不实现。

### cap 截断：未找到与本地顺序完全相同的外部对照

本项目的具体问题是自定义后处理先按 confidence 取 top-k，后做独占像素
与面积过滤。没有找到可直接归因到同一顺序的 Ultralytics issue 或论文。
因此不把它写成社区共识，只依据本地 87 图/385 个 cap 外面积合格候选
做可回退的因果测试。由于 `confidence>=0.10` 的候选最多为 40，虽然
原始 `conf>=0.01` 缓存常达 50 上限，当前缓存对测试 `max_instances=50`
仍然是完整的。

## P5-D0 实测结论

P5-D0 已于 2026-10-17 完成。`max_instances=50` 相对 20 的三折平均
PQ 仅提升 `+0.001004887`，未达预注册 `+0.002` 门槛，因此不采纳。
Fold0/1/2 分别为 `+0.001622310/+0.001568278/-0.000175928`，标准差
增加 `+0.000701735`。总计新增 126 个预测实例，hit pairs `+58`、
FN `-58`，但 FP `+134`。

这证明 cap 外确有有效候选，但直接放宽 cap 的信噪比不足，而且只恢复
58/2526 个当前 FN。后续不搜索 cap=25/30/35/40；先在现有缓存中审计
1422 个无重叠 GT 对低置信原始候选的可恢复性，再区分重排名与真正候选缺失。

## P5-D1 实测结论

P5-D1 已于 2026-10-17 完成。对 2526 个最终 miss 审计全部
`conf>=0.01` 原始缓存后，1288 个（50.99%）已有 `IoU>0.5` 候选。
更关键的是，1422 个最终无重叠 GT 中有 874 个（61.46%）已有可匹配
候选；Fold0/1/2 分别为 62.26%/58.77%/64.25%。这直接否定了“无重叠
主要等于检测器完全没看见”的工作假设。

这 874 个候选分别被原置信度阈值（526）、原始面积阈值（271）和
top-20（77）挡掉。固定 100 组合网格已显示单独把 confidence 从 0.10
降到 0.05 会把平均 PQ 从 0.325102 降到 0.320435；把 min_area 从 600
降到 400 也降到 0.323406。D0 单独扩大 cap 同样未过门槛。因此不能把
874 当成“放宽阈值即可取得的收益”，而应学习区分低分真阳性和大量
低分假阳性。

Mask Scoring R-CNN 对“分类置信与 mask IoU 不充分相关”的结论与本地
现象直接一致。本地第一版不移植其新网络头，而采用更轻量、可回退的
OOF 候选质量重排：验证 Fold 只接受其他两个 Fold 训练的评分器，固定
概率阈值，避免使用验证标签调参。SAHI/P2/更大检测器暂缓，因为当前
已有好候选的选择问题优先于候选生成。

另有 873/2526（34.56%）最终 miss 位于 `IoU (0.25,0.5]`，三个 Fold
均超过预注册 refiner 支持率，故 crop refiner 保留为 R1 后的并行机制方向。

缓存删失仍须保留限制：321/397 张相关图达到 50 候选上限，92.00% 的
miss 位于这些图上，因此“缓存内不可恢复”不能外推为模型绝对无候选。

## 修改后的 P5 顺序

1. **P5-D0：cap 因果审计（已完成，不采纳）**。保留 cap=20，不做 cap 细网格。
2. **P5-D1：原始候选可恢复性审计（已完成）**。无重叠 GT 中 61.46%
   已有高 IoU 原始候选，跨折一致；新候选生成未达路由门槛。
3. **P5-R1：OOF mask-quality 固定 0.5 筛选（已完成，不采纳）**。平均
   PQ `-0.011067`，三个 Fold 全降；FP 减少但 FN 增加 332。
4. **P5-R2：固定基线配额的纯重排（当前）**。held-out AUC/AP 均优于原
   confidence，但 R1 概率校准跨折不稳；R2 不设质量阈值，只在每图与基线
   相同的候选预算内按质量排序替换。
5. **冻结 T1 的 crop refiner**。先测 BCE + Dice + 边界加权，再单变量测
   Skeleton Recall；`retina_masks=true` 只作边界机制对照。
6. **候选生成后移**。只有重排和 refiner 仍留下经确认的原始候选缺失，
   才在 SAHI/全局-局部切片、P2 或更大检测器中选一个单变量候选。

## P5-R1 失败复盘（检索日期：2026-10-17）

R1 相对 T1 平均 PQ `-0.011067144`，Fold0/1/2 均下降；它少输出 598 个
实例、FP 减少 654，但 hit 减少 332、FN 增加 332。固定 0.5 阈值的
candidate recall 在三个 Fold 为 44.03%/32.79%/59.06%，说明跨折校准
不稳定。

但质量分数的 held-out ROC-AUC 为 0.7654/0.7467/0.7168，均明显高于
原 confidence 的 0.6335/0.6081/0.6236；AP 也在每折更高。因此问题更像
“有排序信号但固定绝对阈值不适配”，而不是几何特征完全无效。

外部方法与本地差异：Mask Scoring R-CNN 用 RoI 实例特征和预测 mask
回归连续 MaskIoU，并以 `classification × predicted IoU` 校准最终分数；
本地 R1 只有缓存几何特征、使用二元标签，并以 0.5 直接替代原筛选。
Rank & Sort Loss 和 Learning to Rank Proposals 均将重点放在候选相对排序，
也不要求分数在固定 0.5 上跨数据折校准。官方实现和论文：

- <https://github.com/zjhuang22/maskscoring_rcnn>
- <https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html>
- <https://github.com/kemaloksuz/RankSortLoss>
- <https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html>
- <https://openaccess.thecvf.com/content_ICCV_2019/html/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.html>

采纳修改：只追加一次“固定基线配额、质量分数纯重排”的 P5-R2，不扫
质量阈值、不把 COCO AP 增益外推到本地 PQ。若 R2 仍失败，则停止缓存
几何重排，转入能读取原图 crop/粗 mask 的 refiner。

## P5-R2 失败复盘（检索日期：2026-10-17）

R2 保持每图与 T1 完全相同的预面积过滤候选配额，仅改变候选身份和顺序，
但平均 PQ 仍 `-0.005347219`。Fold1 近似持平，Fold0/Fold2 分别下降
`-0.002901479/-0.013288170`；最终预测少 361、hit 少 193。

本地模型给“与更高分候选的最大 IoU”稳定的正系数，说明单候选标签会把
同一 GT 周围的多个重复 mask 一起视为正例。进入独占像素后，晚到候选
面积缩水并被过滤，导致“候选级 AUC 提高但集合 PQ 降低”。

外部对照与该现象直接相符：Learning NMS 认为独立候选训练本身不会奖励
“每个对象只留一个”，需要联合邻近检测并加入重复惩罚；Beyond mAP
专门度量实例分割的空间重复，并使用像素占用排序/NMS。MMDetection
#5992 同时提醒，实例的框可以高度重叠而 mask 不重叠，长细丝场景不能用
普通框 NMS 粗暴替代集合模型。

- <https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html>
- <https://openaccess.thecvf.com/content/CVPR2023/html/Jena_Beyond_mAP_Towards_Better_Evaluation_of_Instance_Segmentation_CVPR_2023_paper.html>
- <https://github.com/open-mmlab/mmdetection/issues/5992>
- <https://github.com/WXinlong/SOLO/blob/master/mmdet/core/post_processing/matrix_nms.py>

采纳判断：不新增 R3 learned NMS。它需要集合级目标、mask 关系建模和新的
去重门槛，复杂度与误杀相邻细丝的风险均过高。R1/R2 支线结束，下一步
进入 D1 已独立支持的冻结 T1 crop refiner。
