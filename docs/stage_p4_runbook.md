# P4 分辨率消融与固定预算采样

## 阶段目标

P4 只验证两个可分离假设，不改模型、标注目标或评估规则：

1. **P4-I1**：对现有 B0 1024 checkpoint 仅把推理分辨率提到 1536，检验
   细长实例在 mask prototype 网格上表示不足是否为主要瓶颈。
2. **P4-T1**：保持 B0 每 epoch 恰好 `N` 次抽样，在每个观测站内对含面积
   `[400,1600)` 实例的物理图做连续、有上限的重采样，用于分离 P3
   的样本分布效应和额外优化步数效应。

两个候选不同时改动。P4-I1 先行；P4-T1 已准备好可复现配置，但必须
在 I1 面板结果回收后再运行。

## 准备性诊断

权威产物为 `artifacts/analysis/stage_p4/readiness/index.json`。诊断复用 B0 三折
候选缓存，没有重新 GPU 推理。

| 1024 推理下的等效宽度 | GT | FN | 漏检率 | 其中面积 `[400,1600)` |
|---|---:|---:|---:|---:|
| `<1` 个 stride-4 prototype cell | 1132 | 952 | 84.10% | 743 |
| `1–<2` cells | 2773 | 1426 | 51.42% | 1591 |
| `>=2` cells | 1079 | 414 | 38.37% | 194 |

全局 4984 个 GT 中有 2792 个 FN，漏检率 56.02%。等效宽度小于一个
prototype cell 的实例漏检率达 84.10%，且三折都存在，因此先做 1536
推理消融有直接数据依据。

P3 的固定重复把每 epoch 样本数分别从 569/566/564 提到
707/711/705，优化步数增加 24.25%/25.62%/25.00%。`6+` 中等实例图的
暴露占比同时增加约 14.8–15.2 个百分点，并轻微改变了观测站构成。
因此 P3 的 Fold 依赖结果不能归因于单一采样因素。

## P4-I1：B0 checkpoint 的 1536 推理

只使用 Fold0 的固定 32 图面板首筛，后处理固定为
`.10/400/12/force_one`，不搜索参数。缓存显式记录训练分辨率 1024、
推理分辨率 1536、比例 1.5 以及显式许可的 mismatch。

首先在宿主终端运行 GPU 候选缓存：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p1/b0_medoid_1024_fold0/weights/best.pt \
  --model-config configs/instance/stage_p1/b0_medoid_1024_fold0.yaml \
  --fold 0 \
  --validation-file-list data/manifests/screen_fold0_32.json \
  --output-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/panel_candidates_fold0 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

缓存完成后运行单组合 CPU 评估：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.evaluate_yolo \
  --cache-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/panel_candidates_fold0 \
  --validation-file-list data/manifests/screen_fold0_32.json \
  --confidences 0.10 --min-areas 400 --max-instances 12 --force-one \
  --output artifacts/instance/stage_p4/p4_i1_b0_infer1536/panel32.json
```

面板基线为 `0.3002130359`，只有候选 `PQ >= 0.3102130359`
才通过。若失败，不再生成三折 1536 缓存；若通过，再对 Fold0/1/2
生成完整缓存，并只运行 AGENTS.md 中固定的 100 组合共同网格。

### Fold0 面板结果

P4-I1 得到 `PQ=0.318934758`，相对 B0 提升 `+0.018721722`，通过
`+0.010` 门槛。TP/FP/FN 从 B0 的 150/121/209 变为 160/132/199：
多匹配 10 个实例、少漏检 10 个，同时增加 11 个 FP。收益方向与表征宽度诊断
一致，但仍需三折共同网格确认稳定性。

检查缓存时修复了两项仅影响 provenance 的向后兼容问题：无采样策略的历史
B0 不再把 `null` 采样字段加入训练签名；YAML 省略的 Ultralytics 默认
`mask_ratio=4` 会正规化为 4。修正后训练签名重新精确匹配 B0：
`73d12b41f3a3090552035ffbbcdfbf5251800bc554b78d63fc836560f2f62465`。

下一步在宿主 GPU 分别生成三折完整缓存：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p1/b0_medoid_1024_fold0/weights/best.pt \
  --model-config configs/instance/stage_p1/b0_medoid_1024_fold0.yaml \
  --fold 0 \
  --output-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold0 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p1/b0_medoid_1024_fold1/weights/best.pt \
  --model-config configs/instance/stage_p1/b0_medoid_1024_fold1.yaml \
  --fold 1 \
  --output-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold1 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p1/b0_medoid_1024_fold2/weights/best.pt \
  --model-config configs/instance/stage_p1/b0_medoid_1024_fold2.yaml \
  --fold 2 \
  --output-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold2 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

### 三折完整缓存快速检查

Fold0/1/2 的 138/141/143 张缓存均已通过完整性、Fold 唯一性、B0 训练签名和
1536 推理签名检查。在不搜索参数、直接使用 B0 最优后处理
`.10/600/20/force_one` 时：

| Fold | B0 PQ | P4-I1 PQ | 增量 |
|---|---:|---:|---:|
| 0 | 0.305391483 | 0.336617502 | +0.031226019 |
| 1 | 0.283977090 | 0.315134696 | +0.031157606 |
| 2 | 0.274250454 | 0.306676940 | +0.032426486 |
| 均值 | 0.287873009 | 0.319476379 | +0.031603370 |

标准差从 0.013008320 降为 0.012602830。总计 TP `+213`、FN `-213`、FP
仅 `+25`。这个单组合快速检查已经通过平均提升、3/3 Fold 正提升、最差 Fold
和稳定性四项门槛。以下预注册的固定 100 组合共同网格已完成：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.evaluate_yolo_multifold \
  --cache-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold0 \
  --cache-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold1 \
  --cache-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold2 \
  --folds 0 1 2 \
  --confidences 0.05 0.10 0.15 0.20 0.25 \
  --min-areas 200 400 600 800 \
  --max-instances 8 10 12 15 20 \
  --force-one \
  --output artifacts/instance/stage_p4/p4_i1_b0_infer1536/multifold_search.json
```

### P4-I1 正式多折结论

100/100 个共同组合全部完成，耗时 7354.013 秒（122.57 分钟）。共同最优
仍为 `.10/600/20/force_one`，与 B0 和快速检查完全相同，因此
提升不是由重新调参造成。均值提升 `+0.031603370`，3/3 Fold 为正，
最差 Fold 仍提升 `+0.031157606`，标准差下降 `0.000405490`，正式通过
全部多折门槛。P4-I1 状态为 `promoted`。

最优邻域也有支撑：`.10/400/20` 均值 0.318854414，仅低 0.000622；
`.05/600/20` 均值 0.317900797，仅低 0.001576。`max_instances=20`
位于预注册网格上边界，但本阶段不扩展网格；15 时的最好结果仍在最优值
0.005 之内，不属于单一孤立点。

## P4-T1：观测站守恒的固定预算采样

采样策略固定为
`station_preserving_medium_density_400_1600_cap1p5_v1`。对每张训练物理图，
先计算 `raw_weight = 1 + 0.5 * min(medium_count / 6, 1)`，然后在每个观测站
内归一化，使该站的期望抽样总量严格等于原物理图数。每 epoch 有放回
抽取恰好 569 次，与 Fold0 B0 预算相同。

Fold0 清单中权重范围为 `[0.7656, 1.1955]`。各站期望暴露仍为
Bh/Ch/Lh/Mh/Th/Uh = 97/95/106/100/88/83。`6+` 组的期望暴露从
138 增到 161.17，远低于 P3 的 276；同时不增加优化步数。

权威输入：

- `data/processed/yolo_stage_p4/sampling/p4_t1_fold0.json`
- `configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold0.yaml`

训练会在输出目录记录 `sampling_trace.json`，包含每 epoch 的实际抽样序列统计。
只支持单 GPU、`workers=0`。

P4-I1 通过时，T1 训练后的面板推理使用 1536；I1 失败时使用 1024。
不允许在看到 T1 结果后再改变这一决策。

P4-I1 已正式通过，因此 T1 Fold0 训练完成后必须使用 1536 面板推理。
训练命令：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.train_yolo \
  --config configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold0.yaml
```

### P4-T1 Fold0 训练审计

Fold0 已完成 5/5 epoch，累计 153.582 秒，best 和 last 均对应第 5 轮。
Ultralytics Mask mAP50-95 从同 Fold B0 的 0.15187 提高到 0.17463
（`+0.02276`），Mask mAP50 从 0.43046 提高到 0.48091（`+0.05045`）。
这些只作为训练健康检查，不代替官方 PQ。

`sampling_trace.json` 记录完整 5 轮，每轮严格 569 次抽样，且每轮
`draw_counts` 都能由 seed 2026 精确重放。每轮覆盖 342–370 张唯一物理图；
5 轮合计覆盖 564/569 张，有 5 张因有放回抽样未被抽到，属于已预先固定的
采样语义，需在结果解释中保留。
`0–2/3–5/6+` 三个中等实例分组实际平均抽样数为
186.8/225.8/156.4。P3 的 `6+` 固定暴露是 276 次/轮，因此 T1 已将重采样
强度和优化预算分离。

下一步使用预先锁定的 1536 推理分辨率生成 Fold0 固定 32 图面板缓存。
对照为 P4-I1 在同一 1536 面板上的 `PQ=0.318934758`，T1 需要至少
`PQ=0.328934758` 才通过训练改动的 `+0.010` 面板门槛。

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold0/weights/best.pt \
  --model-config configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold0.yaml \
  --fold 0 \
  --validation-file-list data/manifests/screen_fold0_32.json \
  --output-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/panel_candidates_fold0 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

### P4-T1 Fold0 面板结果

固定 `.10/400/12/force_one` 下，T1 的 1536 面板 PQ 为
`0.349214495`，相对同分辨率 P4-I1 对照 `+0.030279737`，远超
`+0.010` 门槛。TP/FP/FN 从 I1 的 160/132/199 变为 180/139/179：
TP `+20`、FN `-20`，仅增加 7 个 FP。相对原始 1024 B0 面板总提升为
`+0.049001459`。T1 通过 Fold0 面板，进入 Fold1/2。

跨折 provenance 检查同时固化为：共同训练签名只包含共同采样策略，
Fold 专属采样清单 SHA256 单独保留为数据 provenance。修正后三折共同训练
签名为 `1283ecf3...608fa3f`；模型、预测和 PQ 未修改或重算。

Fold1/2 清单和配置已生成并通过 dry-run。分别运行：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.train_yolo \
  --config configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold1.yaml
```

```bash
conda run --no-capture-output -n solar-filament python -m scripts.train_yolo \
  --config configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold2.yaml
```

### P4-T1 Fold1/2 训练审计

Fold1/2 均完成 5/5 epoch，分别耗时 145.142/146.522 秒。相对同 Fold B0，
Mask mAP50-95 在 Fold1 为 `+0.00197`，在 Fold2 为 `-0.00397`。这是混合
健康信号，但 P3 已证明 Ultralytics mAP 不能替代官方 PQ，因此不在此提前
淘汰。

Fold1/2 每轮分别严格抽样 566/564 次，5 轮全部可由 seed 2026 精确
重放。单轮唯一物理图覆盖分别为 340–369 和 338–362；5 轮合计分别覆盖
562/566 和 558/564 张。采样预算与 provenance 均合格。

下一步用三个 T1 best checkpoint 生成完整 1536 缓存：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold0/weights/best.pt \
  --model-config configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold0.yaml \
  --fold 0 \
  --output-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold0 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold1/weights/best.pt \
  --model-config configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold1.yaml \
  --fold 1 \
  --output-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold1 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

```bash
conda run --no-capture-output -n solar-filament python -m scripts.cache_yolo_candidates \
  --checkpoint artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold2/weights/best.pt \
  --model-config configs/instance/stage_p4/p4_t1_station_preserving_medium_1024_fold2.yaml \
  --fold 2 \
  --output-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold2 \
  --imgsz 1536 --allow-inference-imgsz-mismatch --device 0 \
  --candidate-confidence 0.01 --max-candidates 50 --resume
```

### P4-T1 三折快速检查

三折完整缓存已通过身份检查。直接沿用 P4-I1 正式最优的
`.10/600/20/force_one` 时：

| Fold | P4-I1 PQ | P4-T1 PQ | T1 增量 |
|---|---:|---:|---:|
| 0 | 0.336617502 | 0.337522715 | +0.000905214 |
| 1 | 0.315134696 | 0.323752928 | +0.008618232 |
| 2 | 0.306676940 | 0.314030616 | +0.007353676 |
| 均值 | 0.319476379 | 0.325102087 | +0.005625707 |

T1 在 3/3 Fold 都比 I1 高，标准差从 0.012602830 降至 0.009637941。
相对 I1，TP `+53`、FN `-53`、FP `+37`。相对原始 B0，三折均值提升
`+0.037229078`，已满足 AGENTS.md 预注册的所有正式门槛。

若将 `+0.008` 额外作为相对已晋级 I1 的更严格增量门槛，当前
`+0.005626` 还差 0.002374。该额外门槛不是 AGENTS.md 中的 B0 晋级规则，
但用于判断 T1 是否值得取代 I1。因此仍执行预注册的固定 100 组合：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.evaluate_yolo_multifold \
  --cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold0 \
  --cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold1 \
  --cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold2 \
  --folds 0 1 2 \
  --confidences 0.05 0.10 0.15 0.20 0.25 \
  --min-areas 200 400 600 800 \
  --max-instances 8 10 12 15 20 \
  --force-one \
  --output artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/multifold_search.json
```

### P4-T1 固定 100 组共同网格结果

共同最优仍是 `.10/600/20/force_one`，与快速检查完全一致：

| Fold | 原始 B0 PQ | P4-I1 PQ | P4-T1 PQ | T1 vs B0 | T1 vs I1 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.305391483 | 0.336617502 | 0.337522715 | +0.032131232 | +0.000905214 |
| 1 | 0.283977090 | 0.315134696 | 0.323752928 | +0.039775839 | +0.008618232 |
| 2 | 0.274250454 | 0.306676940 | 0.314030616 | +0.039780163 | +0.007353676 |
| 均值 | 0.287873009 | 0.319476379 | 0.325102087 | +0.037229078 | +0.005625707 |

标准差为 `0.009637941`，相对原始 B0 减少 `0.003370379`，相对 I1
减少 `0.002964889`。相对 B0 的四项预注册门槛全部通过，因此 T1 正式
晋级。相对 I1 的额外 `+0.008` 严格增量参考线未达到，但它不是
AGENTS.md 中的正式门槛。由于 T1 在 3/3 Fold 都提升且方差更低，
最大化 PQ 时选 T1；如优先避免重训，则保留 I1 作为简化备选。

网格局部稳定性：第二名 `.10/400/20` 的均值为 `0.323406430`，
距最优 `0.001695657`；在最优 `0.002` 内有 2 组，`0.005` 内有 8 组。
`max_instances=20` 仍处于预注册网格边界，本阶段不追加网格；下一步先做
T1 与 I1 的配对残差归因，确认剩余误差是否仍集中在细丝尺度、分割
边界还是候选截断。

### T1/I1 配对残差归因

审计入口已实现；项目全量回归为 `76 passed`。它只读取已有的 1536 候选缓存，不重新
训练或推理；它按同一 GT annotation 配对 I1/T1，统计救回、退化、
1536 等效宽度分层、剩余 FN IoU 分层，以及 `max_instances=20` 的真实
截断压力。实测 30 秒完成 9/422 图，预计约 20–25 分钟，由用户运行：

```bash
conda run --no-capture-output -n solar-filament python -m scripts.analyze_p4_residuals \
  --i1-cache-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold0 \
  --i1-cache-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold1 \
  --i1-cache-dir artifacts/instance/stage_p4/p4_i1_b0_infer1536/candidates_fold2 \
  --t1-cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold0 \
  --t1-cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold1 \
  --t1-cache-dir artifacts/instance/stage_p4/p4_t1_station_preserving_medium_1024/candidates_fold2 \
  --confidence 0.10 --min-area 600 --max-instances 20 --force-one \
  --output-dir artifacts/analysis/stage_p4/t1_vs_i1_residuals
```

完成标志为 `index.json`、`instance_transitions.csv`、
`image_cap_pressure.csv` 和 `README.md` 四个文件都存在。

### 配对残差结果与 P5 路由

审计已完成，4984 个 GT 记录中，T1 从 I1 救回 460 个、退化 407 个，
净救回 53。净收益完全来自面积 `[400,1600)` 的 `+75`；其他三个
面积段合计为 `-22`。T1 剩余 2526 个 FN 中，`no_overlap=1422`、
`near_threshold=873`、`low_overlap=231`。1536 下仍小于一个 prototype cell
的 183 个 GT 漏掉 168 个，且相对 I1 净退化 4 个。

`max_instances=20` 在 T1 的 98/422 图上会先于面积和重叠过滤发生截断；
87 图在 cap 外仍有原始面积合格候选，共 385 个。T1 的 `+53` 净救回中
`+52` 来自这 98 张密集图，因此 P5 先做固定 `.10/600/force_one` 下
`max_instances=20` 对 `50` 的缓存因果审计，不扩展其他网格。

外部对照和采纳/拒绝理由记录于 `docs/stage_p5_evidence_review.md`。

## 本阶段不做

- 不同时改变训练和推理分辨率。
- 不恢复 P3 的静态重复、不调重复阈值。
- 不加 soft label、boundary loss、TTA、refiner 或更大模型。
- 不使用新后处理网格抵消候选的失败。
- 不利用公开榜单替代本地分组 OOF 决策。
