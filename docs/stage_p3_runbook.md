# P3 中等面积实例定向重采样

## 证据与候选定义

P3 细分了 B0 三 Fold 的固定后处理错误，权威报告位于
`artifacts/analysis/stage_p3/b0_failure_modes/index.json`。三 Fold 共计 2792
个 FN，其中面积 `[400,1600)` 的 FN 为 1538 个，占 55.09%；该面积段共有
2528 个 GT，漏检率为 60.84%。另外 41.58% 的 FN 与某个预测最大 IoU 位于
`(0.25,0.5]`，说明主要问题不是只缺极小实例。

唯一候选 C1 使用 medoid 目标，不改变标注、不做 union。训练分区中，若一张
物理图的 medoid 标注含至少 6 个面积 `[400,1600)` 的实例，则增加一个相同
图像/标签样本；验证分区不重复。策略名固化为
`medoid_medium_area_400_1600_count6_repeat1_v1`，阈值不可在同名策略下修改。

| Fold | 训练物理图 | 重复图 | 实际训练样本 | 增幅 |
|---|---:|---:|---:|---:|
| 0 | 569 | 138 | 707 | 24.25% |
| 1 | 566 | 145 | 711 | 25.62% |
| 2 | 564 | 141 | 705 | 25.00% |

三 Fold 的验证目标仍为 `medoid_annotator_v1`。模型和训练数学保持 B0：
YOLO26n-seg、1024、batch 2、5 epoch、mask ratio 4、seed 2026。

## Fold0 固定面板

Fold0 已在 CUDA:0 上完成，5 epoch 累计 190.271 秒。固定 32 图面板仅使用
`.10/400/12/force_one`，没有搜索参数。

| 方案 | PQ | TP | FP | FN | 预测实例 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.300213036 | 150 | 121 | 209 | 176 |
| P3 C1 | 0.333165129 | 176 | 137 | 183 | 208 |

C1 相对 B0 为 `+0.032952093 PQ`，通过 `+0.010` 面板门槛。因此 C1 当前是
`pending_multifold`，不是最终晋级；只有完成 Fold1/2 和固定 100 组合共同
网格后才能决定是否保留。

## Fold1/2 训练结果

Fold1/2 已完成 5/5 epoch，训练参数与 dry-run 一致。Ultralytics Mask
mAP50-95 仅作为健康检查，三折相对同折 B0 均为正：

| Fold | B0 Mask mAP50-95 | P3 C1 | 增量 | 累计秒数 |
|---|---:|---:|---:|---:|
| 0 | 0.15187 | 0.17800 | +0.02613 | 190.271 |
| 1 | 0.13697 | 0.15371 | +0.01674 | 176.935 |
| 2 | 0.14979 | 0.16388 | +0.01409 | 175.153 |

这些指标不能代替官方 PQ；完整官方评估的最终结论见下节。

## 最终多折结果

固定 100 组合共同网格耗时 7022.552 秒，最优参数仍为
`.10/600/20/force_one`。

| Fold | B0 PQ | P3 C1 PQ | 增量 |
|---|---:|---:|---:|
| 0 | 0.305391483 | 0.324388361 | +0.018996878 |
| 1 | 0.283977090 | 0.270311589 | -0.013665501 |
| 2 | 0.274250454 | 0.303698149 | +0.029447696 |
| 均值 | 0.287873009 | 0.299466033 | +0.011593024 |

候选通过平均提升、2/3 Fold 正提升和标准差门槛，但 Fold1 下降超过允许的
`-0.005`，因此以 `failed:worst_fold` 淘汰。全局合并 PQ 为 0.299960762，
相对 B0 提高 0.012665195；它不能覆盖预先声明的最差 Fold 约束。

P3 C1 的主要计数变化为 TP `+126`、FN `-126`、FP `+63`。问题集中在
Fold1：TP `-21`、FN `+21`、FP `+72`；而 Fold2 同时减少 148 个 FP 和 51
个 FN。静态计数阈值重采样具有明显 Fold 依赖，不能进入长训练，也不应通过
调整阈值或后处理继续追分。

## 历史训练命令

Fold1/2 已完成，以下命令仅作 provenance 记录，不要重复执行：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.train_yolo \
  --config configs/instance/stage_p3/c1_medium_rich_repeat_1024_fold1.yaml

conda run --no-capture-output -n solar-filament python -m scripts.train_yolo \
  --config configs/instance/stage_p3/c1_medium_rich_repeat_1024_fold2.yaml
```

下一步生成三折完整候选缓存，再运行固定 100 组合共同网格；不要直接进入
长训练。

## 历史完整多折评估命令

本节命令已经完成，仅作为 provenance 记录；候选已淘汰，不要重跑。

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024_fold0/weights/best.pt \
  --model-config configs/instance/stage_p3/c1_medium_rich_repeat_1024_fold0.yaml \
  --fold 0 \
  --output-dir artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024/candidates_fold0 \
  --imgsz 1024 --device 0 --candidate-confidence 0.01 --max-candidates 50 --resume
```

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024_fold1/weights/best.pt \
  --model-config configs/instance/stage_p3/c1_medium_rich_repeat_1024_fold1.yaml \
  --fold 1 \
  --output-dir artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024/candidates_fold1 \
  --imgsz 1024 --device 0 --candidate-confidence 0.01 --max-candidates 50 --resume
```

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024_fold2/weights/best.pt \
  --model-config configs/instance/stage_p3/c1_medium_rich_repeat_1024_fold2.yaml \
  --fold 2 \
  --output-dir artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024/candidates_fold2 \
  --imgsz 1024 --device 0 --candidate-confidence 0.01 --max-candidates 50 --resume
```

缓存完成后执行固定共同网格；参数不要增删：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.evaluate_yolo_multifold \
  --cache-dir artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024/candidates_fold0 \
  --cache-dir artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024/candidates_fold1 \
  --cache-dir artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024/candidates_fold2 \
  --folds 0 1 2 \
  --confidences 0.05 0.10 0.15 0.20 0.25 \
  --min-areas 200 400 600 800 \
  --max-instances 8 10 12 15 20 \
  --force-one \
  --output artifacts/instance/stage_p3/p3_c1_medium_rich_repeat_1024/multifold_search.json
```

## 清理边界

P2 与 P3 已淘汰候选的活动训练目录、checkpoint、候选缓存、专用数据、配置
入口和专用构建策略均已移入系统回收站或从活动代码移除。P3 仅保留约 784 KB
的审计证据：`artifacts/instance/archive/stage_p3/`。P0/P1 B0 权威产物、P3
错误分析和历史 control 不得删除。
