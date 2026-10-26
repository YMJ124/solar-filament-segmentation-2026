# P2 定向目标候选运行记录

## 错误证据

权威联合分析位于 `artifacts/analysis/stage_p2/b0_multifold/`，固定使用
`confidence=0.10`、`min_area=600`、`max_instances=20`、`force_one=true`。

- B0 Fold 平均 PQ 为 `0.287873009 ± 0.013008320`，全局合并 PQ 为
  `0.287295567`，精确复现 P1 基线。
- 小实例漏检 445 个，占 FN 的 15.94%，未达到预先声明的 30% 主因门槛；
  P2 暂不实现小实例定向候选。
- 大面积共识 FP 只有 5 个，占共识 FP 的 0.43%，不能解释 Fold2 的额外
  FP；P2 暂不实现更激进形状/面积过滤。
- 42.65% 的物理图有多个标注者，其中 56.11% 的图像跨标注者 PQ 极差
  至少 0.1。P2 只实现多标注批次展开候选。

## 候选 C1：展开标注者、物理图等权

训练集将每个标注批次展开成独立 YOLO 样本；每个样本权重为该物理图
标注者数量的倒数，因此每张物理图的总抽样权重严格为 1。每 epoch 抽取
的样本数等于训练物理图数，避免仅因展开数据而增加优化步数。验证集继续
使用 medoid 目标，不做 union。

- target policy：`expanded_annotator_equal_physical_weight_v1`
- 采样：带放回加权随机采样，seed 2026，单 GPU，`workers=0`
- 模型和训练数学保持 B0：YOLO26n-seg、1024、batch 2、5 epoch、
  `mask_ratio=4`
- Fold0 配置（归档）：
  `artifacts/instance/archive/stage_p2/p2_c1_multi_annotator_equal_weight_1024/config.yaml`
- Fold0 展开数据已在淘汰后清理。

历史训练命令如下；候选已淘汰，不得重跑：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.train_yolo \
  --config configs/instance/stage_p2/c1_multi_annotator_equal_weight_1024_fold0.yaml
```

训练开始后必须在日志中确认 CUDA/`device=0`，不得静默回退 CPU。训练完成
后先使用固定 32 图面板 `.10/400/12/force_one`；相对 B0 至少 `+0.010 PQ`
才构建 Fold1/2 数据与配置。

## 结果与决策

训练已在 `device=0` 完成 5/5 epoch，`best.pt` SHA256 为
`8c68a1df56a61e051562f2a118e75c6e5d392ab0d863d33c1d652d9483d9cbf6`。
固定面板使用 `data/manifests/screen_fold0_32.json`；该清单已通过 B0 精确
复算，PQ、TP/FP/FN、IoU 和预测数均与 P1 权威报告一致。

| 方案 | PQ | TP | FP | FN | 预测实例 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.300213036 | 150 | 121 | 209 | 176 |
| P2 C1 | 0.278764674 | 143 | 145 | 216 | 189 |

候选相对 B0 为 `-0.021448362 PQ`，低于要求的 `+0.010`，面板淘汰。
与已退役的 P1 random-annotator 候选相比改善 `+0.005684216 PQ`，说明物理图
等权抑制了一部分额外 FP，但仍同时损失 TP、增加 FP 和 FN，不能晋级。

结论：不训练 Fold1/2，不执行多 Fold 网格，也不进入长训练。B0 继续作为
活动基线。下一轮应先细分非小实例 FN 和 Fold2 的非大面积 FP；当前证据仍
不支持小实例专项或更激进面积/形状过滤。

淘汰后已清理活动 checkpoint、候选缓存、展开训练数据、专用配置入口和专用
训练适配代码，合计约 164 MB，均移动到系统回收站。仅保留面板报告、训练
指标 CSV、运行参数、配置和数据/采样清单作为决策证据：
`artifacts/instance/archive/stage_p2/p2_c1_multi_annotator_equal_weight_1024/`。
