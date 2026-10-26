# P9-D1: P5 路线的 OOF 增量候选选择

## 目的

P8-E4 在官网仍为约 `0.31`，证明同源模型投票不能可靠判断缺失实例。
P9 回到 P5-D1 已确定的路线：从 T1 过滤前/旁路候选改善召回，再接已经
晋级的 C2。第一步只检验已有 P4-I1 是否能作为互补候选源，不训练检测器。

P4 的逐 GT 审计显示，I1 能匹配 407 个被 T1 漏掉的标注实例，但也有 460
个反向差异。直接追加所有非重复 I1 候选的初步 OOF 对照为
`0.324845 -> 0.326094`（`+0.001249`），低于固定 `+0.002` 门槛，因此不晋级。

## 单变量和隔离

- T1 `.10/600/20/force_one` 的全部实例按原 mask、顺序和 source ID 保留；
- I1 只提供与 T1 IoU 不超过 0.5 的增量实例，独占像素后仍须面积至少 600；
- 评分目标是候选相对不可变 T1 的边际官方-PQ效用，不是独立 mask 标签；
- 每个验证 Fold 的 ExtraTrees 只用另外两个 Fold 的候选和 GT 拟合；
- 29 个特征全部来自推理时可见的 confidence、mask 几何、原图灰度与
  T1 原始候选支持，不使用验证 Fold GT；
- 固定预测效用阈值为 0，不搜索 confidence、数量、面积或模型阈值；
- 本阶段只做 detector-level OOF 门控。通过后才允许把相同选中实例接入
  现有折对应 C2 refiner；失败则不做 C2、不生成提交。

P9 是追加分支，绝不覆盖 T1/I1 cache、P5 checkpoint 或 P5 submission。

## 固定门槛

相对同一次运行精确复现的 T1：三折平均 PQ 至少 `+0.002`、至少 2/3 Fold
提升、最差 Fold 不低于 `-0.002`、Fold 标准差增加不超过 `+0.005`。

## 验证命令

短单元测试：

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"
conda run -n solar-filament pytest -q tests/test_yolo_stage_p9.py
```

完整三折需要读取并校验两套 422 图缓存，预计约 6--8 分钟，由用户执行：

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"
conda run --no-capture-output -n solar-filament python -m scripts.evaluate_p9_incremental_oof \
  --config configs/instance/stage_p9/p9_d1_t1_i1_incremental_oof.yaml
```

完成后回收：
`artifacts/instance/stage_p9/p9_d1_t1_i1_incremental_oof.json`。

## D1 正式结果

D1 已完成并通过 detector gate。T1 三折平均 PQ
`0.325102087 -> 0.334985696`（`+0.009883609`），Fold0/1/2 分别
`+0.012219658/+0.012940759/+0.004490410`，标准差增加
`+0.003165211`；四项预注册门槛全部通过。全局官方 PQ
`0.324845179 -> 0.334553518`，hit/FP/FN 变化为
`+208/+284/-208`。609 个候选中固定阈值接纳 306 个，不得根据结果再改
ExtraTrees、效用阈值、IoU、面积或候选数量。

正式报告 SHA256：
`e6f3e7da4c5c4f0ff8e6f1c33d80ddc07a5264bfd0b9475064b7e1008ee402e0`。

## P9-D2：把登记的 D1 增量接入冻结 C2

### 单变量与不可变量

- D1 的 306 个 OOF 选择按 `(fold, file_name, i1_source_index, i1_rank)`
  原样重建，不重新训练或打分；完整运行必须精确复现 D1 control。
- Fold0/1/2 只使用各自已有的 held-out C2 checkpoint；checkpoint、256 crop、
  threshold=0.5、boundary kernel=7 与逐来源 fallback 全部冻结。
- 每张图把全部 T1 source 放在前面，再按 D1 登记顺序追加 I1 source；所有
  coarse mask 保持互斥，精修扩张不得占用其他 source 的 coarse 支撑域。
- `2912 + 306 = 3218` 个登记 source 必须全部保留。任一 source 因独占像素
  不足 600 时只恢复自身 coarse mask，不得删实例或整图回退。
- 本阶段仍是 OOF 评估；通过前不得训练 final-fit、生成测试缓存或提交。

预注册配置：
`configs/instance/stage_p9/p9_d2_selected_additions_c2_oof.yaml`，SHA256
`073d5514e8022fe3e9b38a34ff03716626530285a42fedca1e93e37decdc88d1`。

### 固定门槛

正式候选相对已晋级 P5-C2：三折平均 PQ 至少 `+0.002`、至少 2/3 Fold
为正、最差 Fold 不低于 `-0.002`、Fold 标准差增加不超过 `+0.005`；此外
必须保持全部 3218 个登记 source。逐图物理图 bootstrap 只报告不确定性，
不替代或修改上述门槛。

### 验证状态与正式命令

P9 定向测试现为 `9 passed`，项目全量回归为 `102 passed`（22 条既有
`pycocotools`/NumPy-2 DeprecationWarning）。每折 1 图 CPU smoke 已完成：
11 个 T1 source、2 个登记增量共 13 个输出全部保留，0 fallback；smoke
分数不作决策证据。

完整评估需加载两套 422 图缓存并运行三套 refiner，预计接近或超过 5 分钟，
由用户在宿主 GPU 执行：

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"
conda run --no-capture-output -n solar-filament \
  python -m scripts.evaluate_p9_refined_incremental_oof \
  --config configs/instance/stage_p9/p9_d2_selected_additions_c2_oof.yaml \
  --device 0
```

运行开始时必须确认 CUDA 可见；脚本在请求 `--device 0` 而 CUDA 不可用时会
直接报错，不会静默退回 CPU。完成后回收：
`artifacts/instance/stage_p9/p9_d2_selected_additions_c2_oof.json`。

## D2 正式结果

D2 已完成并通过全部预注册门槛。相对 P5-C2，三折平均 PQ
`0.339830155 -> 0.349892874`（`+0.010062719`），Fold0/1/2 分别
`+0.013525137/+0.012709640/+0.003953380`，标准差增加
`+0.003634172`。全局 PQ `0.339564359 -> 0.349438782`，hit/FP/FN
变化为 `+211/+281/-211`，3218 个登记 source 全部保留。物理图配对
bootstrap 95% CI 为 `[+0.006296472,+0.013578264]`，正增益概率为 1.0。

正式报告 SHA256：
`93a01327e787341eac1959cc4fd367f476c5dd64f7d96af48a149252ef620dbe`。

## P9-D3：无泄漏 selector 与测试 challenger

### 固定部署协议

- final selector 只用 D1 的 609 个 OOF detector 候选及其 OOF GT 边际效用；
  不读取测试标签、测试统计或 leaderboard，ExtraTrees 与阈值 0 全部冻结；
- 测试主干固定为已计分的 P5 Fold0 T1+C2；I1 只提供增量候选，不替换或
  重排任何 T1 source；
- 每张图先验证同一次推理在无增量时能逐 RLE 精确复现 P5，再执行 D2 已验证
  的 T1-first 联合 C2；
- P5 唯一 coverage-fallback 图保持原提交不变，不把该特殊情况扩展为未经
  OOF 验证的新分支；
- 每张图的完整结果按配置、脚本、权重、selector、P5 parent 与图像 SHA256
  原子缓存。中断后只复用指纹完全一致的缓存；
- 只生成一次 challenger，不根据候选数量或官网结果回调阈值。

final selector 已生成：

- 配置：`configs/instance/stage_p9/p9_d3_final_selector.yaml`，SHA256
  `3d7be906e7c170b3f8a1669ea9c62070d962044377c60005ea29e8c5221fec6d`；
- 模型：`artifacts/instance/stage_p9/p9_d3_final_selector.joblib`，SHA256
  `e202193f594b000a3bcc8126275a8e17913c3d7d07c763f4350725f9258e4741`；
- 报告：`artifacts/instance/stage_p9/p9_d3_final_selector.json`，SHA256
  `b7b2a7668044a47585f3be534280659d74afd38419c6b9b820c6482f565ae28a`。

训练拟合诊断在冻结阈值下选中 268/609；它不是 OOF 估计，不用于再次决策。

### 验证状态与正式命令

10 图 GPU smoke 已通过：66 个 T1 source、12 个未筛增量中选择 4 个，输出
70 个实例；3 图实际进入增量路径，9 个逐来源 fallback，所有 source 保留，
base-only RLE 逐字复现 P5。复跑时 10/10 命中缓存，输出 SHA256 完全一致。
P9 定向测试为 `9 passed`，项目全量回归为 `102 passed`（22 条既有警告）。

完整 180 图包含两次 detector 推理与一次 C2，预计超过 5 分钟，由用户执行：

```bash
cd "/home/yu/桌面/Solar Filament Segmentation/filament-segmentation-2026"
conda run --no-capture-output -n solar-filament \
  python -m scripts.run_p9_submission \
  --config configs/submission/p9_d3_p5_fold0_incremental_1536.yaml \
  --device 0
```

输出：

- `artifacts/submissions/p9_d3_p5_fold0_incremental_1536.csv`
- `artifacts/submissions/p9_d3_p5_fold0_incremental_1536.audit.json`
- `artifacts/instance/stage_p9/p9_d3_test_cache/`（可恢复逐图缓存）

正式审计通过后只允许提交一次。官网结果只决定是否替换 P5，不得用于搜索
selector threshold、duplicate IoU、面积、实例上限或 C2 参数。

### D3 正式测试推理结果

完整 180 图已完成并通过审计：P5 的 1238 个 T1 source 与 2 个 coverage
fallback 全部保留，从 194 个未筛增量中选择 82 个，65 图获得增量，最终
1322 行。RLE 全部合法、无缺图、最大每图 16 个实例，121 个逐来源 fallback。
相对 P5 的 1240 行增加 82 行（`+6.61%`），低于 P6 的 `+18.0%` 和 P8-E4
的 `+10.3%` 密度扩张。

跨提交 IoU 审计显示：P5 的 1240 个实例全部在 D3 中有 IoU>0.5 对应；82
个新增实例全部相对 P5 IoU<=0.5。1238 个 P5 RLE 逐字不变，仅 2 个在联合
支撑保护下发生轻微边界变化，IoU 分别为 `0.993774` 和 `0.978322`。82 个
新增实例面积中位数 1789.5，38 个小于 1600；各年份和六个观测站均无异常
集中。完整 180 图缓存复跑 180/180 命中，重建提交 SHA256 完全一致。

正式提交 SHA256：
`4ffc68e2dd92367cf72b6dab53c49092566285c330d290f7d51f52e00712ac11`。
正式审计 SHA256：
`9a3570bdb98be48487b2ed3985c74e01b037708dc295757c6a859aed53593c2a`。

结论：该文件可作为唯一一次 P9-D3 官网 challenger。提交后只记录官方回执；
无论结果如何，都不得据此修改已冻结的 selector 或 C2 参数。

用户进一步规定：P9-D3 是当前阶段最后一次以官网分数作为验证。回执完成后
停止 leaderboard-driven experiment，不再提交不同阈值、数量、融合或架构
变体。下一阶段先完成 OOF→官网迁移差、Fold2 分布漂移、utility 校准、候选
召回上限和 C2 误差的本地分层诊断，再检索同赛题社区方案及相关论文；新的
方法必须先走完整 OOF 和固定门槛。以后官网提交只能用于最终交付，不作为
模型选择或参数搜索工具。
