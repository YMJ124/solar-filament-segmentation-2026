# P5 定向残差优化运行手册

## 1. P5-D0：`max_instances` cap 因果审计（已完成）

### 目的

P4-T1 在 98/422 张 OOF 图上有超过 20 个 `confidence>=0.10` 候选，
87 图的 top-20 之外仍有原始面积至少 600 的候选，共 385 个。
当前后处理顺序是先取 top-k，再执行独占像素与面积过滤，因此需要
直接测量 cap 是否损失有效实例。

本实验只改变 `max_instances`：

| 项目 | 对照 | 候选 |
|---|---:|---:|
| checkpoint | P4-T1 | P4-T1 |
| inference | 1536, `retina_masks=false` | 相同 |
| confidence | 0.10 | 0.10 |
| min_area | 600 | 600 |
| force_one | true | true |
| max_instances | 20 | 50 |

不搜索 confidence/min_area，不混入 `retina_masks`，不使用 I1/B0 缓存。
当前缓存在 `confidence>=0.10` 时每图最多 40 个候选，因此虽然 GPU
缓存的 `max_candidates=50`，候选 `max_instances=50` 仍是完整评估。

### 预注册判定

`max_instances=50` 仅在同时满足以下条件时取代 T1 的 20：

1. 三折平均 PQ 相对 cap=20 至少 `+0.002`；
2. 至少 2/3 Fold 提升为正；
3. 任一 Fold 下降不超过 `-0.002`；
4. Fold 标准差增加不超过 `+0.005`。

若平均提升为正但不足 `+0.002`，则记为“cap 有轻微影响但不足以
修改正式后处理”。若 FN 下降但 FP 导致 PQ 下降，则记为“候选截断存在，
但直接放宽 cap 不是合格解法”，不进行结果驱动的 cap 细搜索。

### 运行命令

这是离线 CPU 任务，但需要解码 422 张图的 RLE 候选并评分两个配置，
预计超过 5 分钟，由用户在宿主终端执行：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.evaluate_yolo_multifold \
  --cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold0 \
  --cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold1 \
  --cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold2 \
  --folds 0 1 2 \
  --confidences 0.10 \
  --min-areas 600 \
  --max-instances 20 50 \
  --force-one \
  --output artifacts/instance/stage_p5/p5_d0_cap20_vs50.json
```

完成标志为 `artifacts/instance/stage_p5/p5_d0_cap20_vs50.json`。

### 实测结论

`max_instances=50` 的三折平均 PQ 为 `0.326106973`，相对 cap=20 的
`0.325102087` 仅提升 `+0.001004887`，未达预注册 `+0.002`。
Fold0/1/2 增量为 `+0.001622310/+0.001568278/-0.000175928`；新增
126 个预测实例带来 hit pairs `+58`、FN `-58`和 FP `+134`。

判定：**不采纳 cap=50，保留 cap=20，不进行 cap 细网格。**

## 2. P5-D1：原始候选可恢复性审计（已完成）

本实验只读取 P4-T1 的三折 OOF 原始候选缓存，使用压缩 RLE 直接计算
IoU，不重新训练或推理。完整 397 张物理图、2526 个 miss 实例耗时约
2.2 秒。

权威输出：

- `artifacts/analysis/stage_p5/raw_candidate_recoverability/index.json`
- `artifacts/analysis/stage_p5/raw_candidate_recoverability/instance_recoverability.csv`
- `artifacts/analysis/stage_p5/raw_candidate_recoverability/image_cache_censoring.csv`

最终无重叠的 1422 个 GT 中，874 个（61.46%）在 `conf>=0.01` 的原始
缓存内已有 `IoU>0.5` 候选。丢失路径为：

| 路径 | GT |
|---|---:|
| 原置信度低于 0.10 | 526 |
| 原始面积低于 600 | 271 |
| 排名在 top-20 之外 | 77 |
| 当前缓存内不可恢复 | 548 |

Fold0/1/2 的无重叠可恢复率分别为 62.26%/58.77%/64.25%，方向跨折
一致。最终 misses 中另有 873 个（34.56%）是 `IoU (0.25,0.5]` 的
近阈值残差，支持后续 crop refiner。

需要注意：397 张有 miss 的图中有 321 张达到缓存上限 50；2526 个
miss 记录中 2324 个（92.00%）位于这些图上。因此“当前缓存不可恢复”
只是 `conf>=0.01/max_candidates=50` 范围内的下界，不能写成检测器绝对
没有候选。

判定：**先做 mask-quality 候选重排；随后做冻结 T1 的 crop refiner。**
暂缓 SAHI/P2/更大检测器，因为当前首要损失是已有好候选的选择，而非
候选生成。固定 100 组合网格已证明简单降低全局阈值会让 FP 增长快于
hit；D0 也已证明单独增大 cap 不够，因此不重复这些搜索。

## 3. P5-R1：候选质量重排（当前步骤）

### 单变量定义

- checkpoint、1536 推理缓存、`min_area=600`、`max_instances=20` 和
  `force_one=true` 保持不变；
- 候选池从缓存下限 `conf>=0.01` 读取；
- 只用推理时可获得的置信度和 mask 几何特征预测候选是否可能达到
  官方 `IoU>0.5`，替换原始 confidence 的筛选/排序；
- 每个验证 Fold 的重排器只能用另外两个 Fold 的 GT 标签训练，禁止用
  当前验证 Fold 标签拟合或选阈值；
- 第一版固定概率阈值，不做结果驱动阈值网格，确保相对 T1 只改变
  “候选质量评分”。

### 预注册晋级门槛

相对固定 T1 `.10/600/20/force_one`，必须同时满足：

1. 三折平均 PQ 至少 `+0.002`；
2. 至少 2/3 Fold 提升；
3. 最差 Fold 不低于 `-0.002`；
4. Fold 标准差增加不超过 `+0.005`。

若 R1 未通过，不围绕概率阈值做大网格；先检查是否因为仅靠几何特征
无法预测 mask 质量，再决定加入图像/crop 特征还是转入局部 refiner。

## 4. P5-R1 实测与外部对照（已完成，不采纳）

R1 的三折平均 PQ 为 `0.314034942`，相对 T1 `-0.011067144`；
Fold0/1/2 分别下降 `-0.005830430/-0.014392465/-0.012978539`。
它减少 654 个 FP，但同时减少 332 个 hit、增加 332 个 FN，固定
`probability>=0.5` 过于保守。判定：**R1 不采纳。**

这次失败没有否定质量排序本身。held-out candidate 诊断中，质量分数相对
原 confidence 的 ROC-AUC/AP 分别为：

| Fold | quality AUC/AP | 原 confidence AUC/AP |
|---|---:|---:|
| 0 | 0.7654 / 0.6785 | 0.6335 / 0.5581 |
| 1 | 0.7467 / 0.5901 | 0.6081 / 0.4563 |
| 2 | 0.7168 / 0.5704 | 0.6236 / 0.4933 |

失败点是固定 0.5 阈值的跨折选中率不稳（17.56%–39.90%），而不是分数
完全没有排序信号。Mask Scoring R-CNN 使用实例特征和预测 mask 回归
MaskIoU，并将分类分数与质量分数组合；Rank & Sort Loss 与 Learning to
Rank Proposals 都直接优化/使用排序。它们均不支持把几何二分类概率 0.5
当作可迁移的固定决策阈值。因此下一版不做阈值细网格，而只测排序。

对照来源：

- Mask Scoring R-CNN：<https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html>
- 官方实现：<https://github.com/zjhuang22/maskscoring_rcnn>
- Rank & Sort Loss：<https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html>
- 官方实现：<https://github.com/kemaloksuz/RankSortLoss>
- Learning to Rank Proposals：<https://openaccess.thecvf.com/content_ICCV_2019/html/Tan_Learning_to_Rank_Proposals_for_Object_Detection_ICCV_2019_paper.html>

## 5. P5-R2：固定基线配额的纯重排（当前步骤）

- OOF 评分器、特征、标签和训练 Fold 隔离与 R1 完全相同；
- 对每张图先按原 T1 规则计算候选配额：`confidence>=0.10` 的数量，最多
  20；若为 0 则按 `force_one` 取 1；
- 从完整 `conf>=0.01` 缓存中按质量分数选取相同配额；
- 保持 `min_area=600` 和独占像素顺序不变；
- 不使用质量概率硬阈值，不搜索阈值，因此只检验“更好的排序能否在相同
  候选预算下替换掉低质量高置信候选”。

R2 沿用 R1 的四项晋级门槛。若 R2 失败，几何质量分数路线停止，不再
做阈值/融合细调；转入包含原图 crop 特征的局部 refiner。

## 6. P5-R2 实测与外部对照（已完成，不采纳）

R2 三折平均 PQ 为 `0.319754868`，相对 T1 `-0.005347219`；Fold0/1/2
增量为 `-0.002901479/+0.000147993/-0.013288170`。即使每图预面积过滤
配额与 T1 完全相同，最终仍少 361 个实例，hit `-193`、FN `+193`、
FP `-410`。判定：**不采纳，停止缓存几何重排。**

三个模型对 `max_iou_with_higher_score` 的系数均为正（0.443–0.537），
说明逐候选标签把“与同一 GT 重复匹配的候选”也学成了高质量。排序后，
这些候选互相抢占像素，再被 `min_area=600` 过滤；R2 的独立候选目标没有
表达集合级去重/覆盖约束。

Learning NMS（CVPR 2017）明确指出，独立窗口分类会鼓励同一对象的多个
相似候选同时高分；解决它需要联合处理邻近检测并显式惩罚重复。Beyond
mAP（CVPR 2023）也指出实例分割的 duplicate predictions 需要像素占用级
排序/NMS。MMDetection #5992 则展示了框重叠与 mask 重叠可能完全不同，
这对长细丝尤其重要，不能直接追加框 NMS。

- Learning NMS：<https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html>
- Beyond mAP：<https://openaccess.thecvf.com/content/CVPR2023/html/Jena_Beyond_mAP_Towards_Better_Evaluation_of_Instance_Segmentation_CVPR_2023_paper.html>
- MMDetection mask-level NMS 讨论：<https://github.com/open-mmlab/mmdetection/issues/5992>
- SOLO Matrix NMS 实现：<https://github.com/WXinlong/SOLO/blob/master/mmdet/core/post_processing/matrix_nms.py>

集合级 learned NMS/像素占用排序需要新目标与新选择器，且可能误杀相邻
细丝，已超出“轻量质量重排”的单变量范围。当前不追加 R3。

## 7. 后续路由

- D0 未通过，保留 cap=20，不追加 cap 细网格。
- D1 已完成，候选重排与 crop refiner 均获跨折支持；新候选生成未达门槛。
- R1/R2 均已淘汰，相关活动实现清理，只保留小型决策报告。
- 下一步执行冻结 T1 的 crop refiner，首版为 BCE + Dice + 边界加权；
  只有基线有效后再单变量加入 Skeleton Recall。
- `retina_masks=true` 仅保留为近阈值边界机制对照，不作为召回方案。

跨领域证据、方法差异和 D1 后的候选选择见
`docs/stage_p5_cross_domain_review.md`。

## 8. P5-C1：冻结 T1 的局部裁剪精修基线（Fold0 面板已通过）

P5-C1 不是独立检测器，也不替换 T1 权重。它冻结已经晋级的 T1，严格复现
Fold0 面板后处理 `.10/400/12/force_one`，再对每个保留实例的原图灰度
crop 与 T1 coarse mask 做二通道局部精修。首版只检验 BCE + Dice + 固定
边界加权是否能把近阈值实例推过官方严格 `IoU>0.5`，不同时加入 Skeleton
Recall、PointRend、retina mask 或阈值搜索。

### 固定设计与隔离

- Held-out Fold0 只用于最终 32 图固定面板评分，不参与训练、选 epoch 或选阈值。
- 训练数据只来自 Fold1/2 各自的 OOF T1 缓存；配置哈希、T1 训练签名、
  cache manifest 哈希和标注哈希均写入索引。
- crop 为 `256x256`，上下文比例 `1.5`，最小边长 64；模型为 base=16 的
  2-channel compact U-Net，共约 194 万参数。
- 固定 12 epoch、batch 16、AdamW `lr=0.001`，每张物理图总采样质量相等。
  同一物理图有多个标注批次时会生成多个目标，但不会因此获得更高总采样概率。
- 空间增强只做同步水平/垂直翻转；不做会改变太阳图像强度语义的额外增强。

完整裁剪数据已离线构建完成：284 张 Fold1/2 图像、3148 个样本，其中
281 张产生 T1 候选；Fold1/2 样本数为 1486/1662。目标组成如下：

| 目标类别 | 样本数 |
|---|---:|
| 无重叠，背景监督 | 807 |
| 低重叠 `(0,0.25]` | 272 |
| 近阈值 `(0.25,0.5]` | 542 |
| 已匹配 `>0.5` | 1527 |

索引：
`artifacts/instance/stage_p5/p5_c1_crop_refiner_fold0/train_crops_folds1_2/index.json`
（SHA256 `d5563e51dc044e06078cbb37e0e0dc356640fd40d481013a03677e1c90f9e07c`）。
构建耗时 107.89 秒，样本压缩文件共约 90.5 MiB。

### 面板门槛

正式评估必须先精确复现 T1 Fold0 control `PQ=0.3492144945974552`，然后
候选至少达到 `0.3592144945974552`（绝对 `+0.010`）才可进入三折验证。
若失败，不搜索 mask threshold 或 epoch；先按 AGENTS.md 要求将失败模式与
GitHub/论文中的局部精修问题对照，再决定是否单变量加入 Skeleton Recall。

### 正式训练命令

训练共 12 epoch、约 2364 个优化步，预计超过 5 分钟，必须由用户在宿主
GPU 终端执行：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.train_p5_refiner \
  --config configs/instance/stage_p5/p5_c1_crop_refiner_fold0.yaml \
  --device 0
```

成功标志为
`artifacts/instance/stage_p5/p5_c1_crop_refiner_fold0/training/final.pt`。
训练完成后，再运行固定 Fold0 面板：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.evaluate_p5_refiner \
  --config configs/instance/stage_p5/p5_c1_crop_refiner_fold0.yaml \
  --checkpoint artifacts/instance/stage_p5/p5_c1_crop_refiner_fold0/training/final.pt \
  --device 0
```

完成标志为
`artifacts/instance/stage_p5/p5_c1_crop_refiner_fold0/panel32.json`。

### 实测结果与判定

训练在 RTX 5070 Laptop GPU 上完成 12/12 epoch、2364 steps，用时
158.44 秒。loss 从 1.1743 降到 0.8446，约第 6 轮后进入平台期；无
NaN/Inf，`final.pt` 与 `last.pt` 权重一致。checkpoint SHA256 为
`62e3207926586325a35d4772184a633eefd2cb230db59cbe03c51ca5079ad491`。

正式面板在同一宿主 GPU 上运行，报告内记录
`device=cuda:0`、`NVIDIA GeForce RTX 5070 Laptop GPU`。control 精确复现
T1 `PQ=0.3492144945974551`，候选为 `0.3807755809654660`，增量
`+0.0315610863680109`，通过 `+0.010` 门槛：

| 指标 | T1 control | P5-C1 | 增量 |
|---|---:|---:|---:|
| PQ | 0.349214495 | 0.380775581 | +0.031561086 |
| Hit pairs | 180 | 187 | +7 |
| FP | 139 | 111 | -28 |
| FN | 179 | 172 | -7 |
| 预测实例 | 211 | 197 | -14 |
| Hit IoU sum | 118.383714 | 125.084778 | +6.701065 |

收益不是单纯删除实例：19/32 张图预测实例数未变，其中 12 张 PQ 提升；
平均命中 IoU 也从 0.657687 提高到 0.668903。不过 10/32 张图退化，且
3 张图因精修后面积不足删除实例而各损失一个 hit。这是三折阶段必须审计
的主要风险，不允许在 Fold0 搜索 mask threshold 或 min_area 修补。

判定：**P5-C1 通过 Fold0 固定面板，进入正式三折验证；暂不加入 Skeleton
Recall 或其他新变量。** 外部对照复盘见
`docs/stage_p5_cross_domain_review.md` 第 9 节。

## 9. P5-C1 正式三折 OOF 验证（已完成并晋级）

Fold0 面板只负责筛选方法方向；正式三折不复用面板的
`.10/400/12/force_one` checkpoint。三折统一恢复 P4-T1 正式后处理
`.10/600/20/force_one`，每个 held-out Fold 都从另外两折重新生成训练 crop
并从头训练同一 refiner。

正式协议 SHA256：
`6102036f37ccca307b660f8f2cdbc22b28994ec4c1ee9d4c0c6a453ec5a6cbf3`。
三个配置只允许 held-out/training Fold 和对应输出路径不同：

- `configs/instance/stage_p5/p5_c1_formal_fold0.yaml`
- `configs/instance/stage_p5/p5_c1_formal_fold1.yaml`
- `configs/instance/stage_p5/p5_c1_formal_fold2.yaml`

### 固定晋级门槛

相对 P4-T1 正式三折 `.10/600/20/force_one`，必须同时满足：

1. 三折平均 PQ 至少 `+0.008`；
2. 至少 2/3 Fold 提升为正；
3. 任一 Fold 下降不得超过 `-0.005`；
4. 候选 Fold 标准差不得超过 T1 标准差 `+0.010`。

mask threshold 固定 0.5，不搜索 epoch、min_area、threshold 或 fallback。
此外报告实例删除与 hit 丢失风险，但不在看到结果后追加新门槛。

### 已完成的正式训练数据

| Held-out Fold | 训练 Fold | 图像 | 非空图像 | crop 样本 | 12 epoch steps | index SHA256 |
|---:|---|---:|---:|---:|---:|---|
| 0 | 1,2 | 284 | 274 | 3245 | 2436 | `d5970334...52954d0` |
| 1 | 0,2 | 281 | 275 | 3404 | 2556 | `05b52ad2...8cf7ef7` |
| 2 | 0,1 | 279 | 271 | 2971 | 2232 | `8e1864f1...446339` |

同一来源 Fold 在不同组合中的样本数完全一致：Fold0/1/2 分别为
1565/1406/1839，证明生成过程没有因 held-out 配置改变同一来源数据。
三个数据集共约 285.5 MiB。

### 三折正式训练命令

三模型串行预计约 8–10 分钟，超过 5 分钟，由用户在宿主 GPU 执行：

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

set -euo pipefail
for fold in 0 1 2; do
  conda run --no-capture-output -n solar-filament python -m scripts.train_p5_refiner \
    --config "configs/instance/stage_p5/p5_c1_formal_fold${fold}.yaml" \
    --device 0
done
```

成功后应同时存在：

- `artifacts/instance/stage_p5/p5_c1_formal_multifold/fold0/training/final.pt`
- `artifacts/instance/stage_p5/p5_c1_formal_multifold/fold1/training/final.pt`
- `artifacts/instance/stage_p5/p5_c1_formal_multifold/fold2/training/final.pt`

训练完成后由 Codex 核验三个 checkpoint，再在宿主 GPU 运行完整三折评估；
不得在训练与正式评估之间修改任何配置。

### 正式结果

三折训练均由宿主 `NVIDIA GeForce RTX 5070 Laptop GPU` 完成，固定 12 epoch，
无 smoke 截断，Fold0/1/2 分别为 2436/2556/2232 steps。三个 checkpoint 的
120 个模型张量均为有限值，且逐折 `final.pt == last.pt`。

| Held-out Fold | T1 PQ | P5-C1 PQ | PQ Δ |
|---:|---:|---:|---:|
| 0 | 0.337522715 | 0.350844682 | +0.013321967 |
| 1 | 0.323752928 | 0.335169818 | +0.011416889 |
| 2 | 0.314030616 | 0.328253778 | +0.014223162 |

- 三折平均：`0.325102087 -> 0.338089426`，`+0.012987339`；
- 全局官方 PQ：`0.324845179 -> 0.337869755`，`+0.013024576`；
- Fold 标准差：`0.009637941 -> 0.009450937`，变化 `-0.000187003`；
- 3/3 Fold 为正，最差 Fold 增量 `+0.011416889`；四项预注册门槛全部通过；
- 全局 hit/FP/FN：`2458/2353/2526 -> 2276/1761/2708`；
- SQ：`0.647245429 -> 0.669578880`，RQ：`0.501888719 -> 0.504600377`；
- 241/422 图提升、27 图不变、154 图下降；274 图发生实例删除，其中
  144 图删除后损失至少一个原命中。

汇总报告：
`artifacts/instance/stage_p5/p5_c1_formal_multifold/multifold.json`，SHA256
`17f18571e1ce2a58d3c1c439e13c2a06c76d4c43dccaf0a1c590149381cf6cff`。

判定：**P5-C1 正式晋级**。该方案提高官方 PQ 和边界质量，但不是召回方案；
实例删除风险作为下一候选的优化对象，不能用事后新增门槛推翻本次晋级。
实验后外部对照与下一单变量见
`docs/stage_p5_cross_domain_review.md` 第 10 节。

## 10. P5-C2：保留来源支撑域的边界带局部融合（预注册）

### 目的与唯一实验变量

C1 的模型、checkpoint、crop、threshold、T1 后处理及 Fold 全部冻结，不训练。
唯一实验变量是把“精修 mask 覆盖整个 crop”改为：只在 C1 loss 已固定的
`boundary_kernel=7` 所定义边界带内采用精修结果，边界带外严格保留 T1 粗
mask。边界带先在 `256x256` refiner 坐标生成，再映射回原图；不得搜索带宽。

以下是所有未来方案共用的可恢复性不变量，不计作调参变量：

1. T1 cache、C1 checkpoint/正式报告只读，C2 写入独立目录；
2. 保留每个来源实例的粗 mask、score 与 source index；
3. 一个实例的精修扩张不得占用其他来源实例的 T1 粗支撑域；
4. 若融合后的来源实例在独占像素后小于 `min_area=600`，只恢复该实例的
   原始 T1 粗 mask，而不是整图回退；
5. 每折 C2 最终实例数必须与 T1 control 完全相同，否则报告无效；
6. 即使 C2 未晋级，也保留 T1/C1 双路径，不允许后续只读取删实例后的 C1
   二值结果。

### 固定门槛

C2 必须同时满足：

- 相对 T1：沿用 C1 正式门槛，即平均 PQ `>= +0.008`、至少 2/3 Fold 为正、
  最差 Fold `>= -0.005`、标准差增量 `<= +0.010`；
- 相对 C1：三折平均 PQ 下降不超过 `0.003`，任一 Fold 下降不超过 `0.005`；
- hit 总数不得低于 C1 的 2276；
- 三折每折及全局 `prediction_instances_delta_vs_t1 == 0`；
- `images_with_instance_removal == 0`、
  `images_losing_hits_after_instance_removal == 0`。

这些门槛在看到 C2 结果前固定。通过则 C2 取代 C1 作为默认 submission 分支；
未通过则保留 C1 为得分分支、T1 为未损伤研究基线，并转入检测召回优化，
不得继续搜索边界宽度、threshold 或 fallback 条件。

预注册配置：
`configs/instance/stage_p5/p5_c2_boundary_policy.yaml`，SHA256
`9f18a21442ad1d343c1f338474fdf8a77143ba18cbe570f5b37f6b7e73d17a5c`；
三折共同 refinement-policy SHA256：
`a8366427aafc0817a94c3bdbaa7af3bf4163816220e360169820f3436ded1882`。

精确重放命令（复用现有 C1 权重，不训练）：

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"

set -euo pipefail
for fold in 0 1 2; do
  conda run --no-capture-output -n solar-filament \
    python -m scripts.evaluate_p5_refiner \
    --config "configs/instance/stage_p5/p5_c1_formal_fold${fold}.yaml" \
    --checkpoint "artifacts/instance/stage_p5/p5_c1_formal_multifold/fold${fold}/training/final.pt" \
    --device 0 \
    --refinement-policy boundary_band_preserve_support \
    --output "artifacts/instance/stage_p5/p5_c2_boundary_policy/fold${fold}/full_fold.json"
done

conda run --no-capture-output -n solar-filament \
  python -m scripts.summarize_p5_refiner_policy \
  --config configs/instance/stage_p5/p5_c2_boundary_policy.yaml
```

本机正式运行约 3 分钟，低于 5 分钟；若其他机器预估超过 5 分钟，则按
`AGENTS.md` 规则交由用户在宿主终端执行。

### 正式结果（已完成并晋级）

| Fold | T1 PQ | C1 PQ | C2 PQ | C2 - T1 | C2 - C1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.337522715 | 0.350844682 | 0.357274523 | +0.019751807 | +0.006429841 |
| 1 | 0.323752928 | 0.335169818 | 0.335929398 | +0.012176470 | +0.000759581 |
| 2 | 0.314030616 | 0.328253778 | 0.326286545 | +0.012255929 | -0.001967233 |

- C2 三折平均 PQ `0.339830155`，相对 T1 `+0.014728069`，相对 C1
  `+0.001740729`；
- 全局官方 PQ `0.339564359`，相对 T1 `+0.014719180`；
- TP/FP/FN `2513/2298/2471`，相对 T1 为 `+55/-55/-55`；
- 2912 个来源实例全部保留，三折每张图的实例数均未减少；
- 214 个逐来源 fallback，分布于 158 张图；没有整图 fallback；
- 300/422 图提升、29 图不变、93 图下降；
- 相对 T1、相对 C1 和全部可逆性门槛均通过。

正式汇总：
`artifacts/instance/stage_p5/p5_c2_boundary_policy/multifold.json`。
判定：**P5-C2 正式晋级为默认 submission 分支，P5 阶段结束**。T1 仍是
未损伤研究基线；后续召回优化必须从 T1 原始候选层进入，再接 C2，不能只
读取 C2 最终二值输出。实验后外部对照见跨领域复盘第 11 节。
