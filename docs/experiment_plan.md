# YOLO 分阶段优化计划

## P0：评估闭环

当前实现已将评估拆分为三个独立阶段：

1. `scripts.cache_yolo_candidates`：每个 checkpoint/分辨率只执行一次 GPU 推理，按图保存
   mask RLE 与 confidence，同时冻结 checkpoint、训练配置、推理配置和数据指纹。
2. `scripts.evaluate_yolo`：只读缓存，在 CPU 上网格搜索
   `confidence / min_area / max_instances / force_one`，使用官方 PQ v6。
3. `scripts.analyze_yolo_errors`：对选定参数生成逐图、逐标注者指标和错误可视化。

缓存会拒绝两类不完整搜索：搜索 confidence 低于缓存下限，或搜索
`max_instances` 超过推理时 `max_candidates`。中断恢复也会校验每张图的运行指纹，
防止混入另一 checkpoint 或配置的候选。

错误分析必须至少输出：

- 每张图的 PQ、TP、FP、FN，预测实例数和标注实例数；
- 跨标注者都未匹配的大面积 FP，以及小 GT 漏检；
- 同一 GT 对应多个预测的断裂关系，以及同一预测覆盖多个 GT 的粘连关系；
- 每个标注者的 PQ 及每图标注者 PQ 的均值、标准差、极差；
- 最好/最差图像和各错误类型的四联图。

断裂与粘连属于诊断启发式，不改变官方 IoU > 0.5 的 TP/FP/FN 语义。

## P1：提高验证可信度

- 匹配基线 B0：YOLO26n、1024、medoid、batch 2、5 epoch。
- 候选 C1：YOLO26n、1024、每 epoch 随机标注者；候选 C2：YOLO26n、1280、medoid。
- 固定 32 图面板使用 `.10/400/12/force_one`，候选相对 B0 至少 `+0.010 PQ`。
- 仅通过面板的候选进入 Fold1/2；B0 始终完成 Fold0/1/2。
- 三 Fold 后处理统一搜索 100 组冻结网格，以平均 PQ 排序，再比较标准差和 FP。
- 晋级要求平均至少 `+0.008`、至少 2/3 Fold 提升、单 Fold最差不低于 `-0.005`、
  标准差增量不超过 `+0.010`。
- 长训练只允许在通过 32 图和多 Fold 两级门槛后启动。完整命令见
  [`stage_p1_runbook.md`](stage_p1_runbook.md)。

## P2：多标注者训练目标

依次比较 medoid 基线、每 epoch 随机选标注者批次、带采样权重的多标注批次展开。
不将所有标注直接 union。

随机标注实现保持每张物理图每 epoch 只出现一次，用 `seed + file_name` 生成确定性排列并按
epoch 循环，优先遍历所有标注版本；验证集仍使用 medoid，且强制 `workers=0`。恢复训练时
直接使用 trainer 的恢复 epoch 选择对应标注版本。

## P3：分辨率和模型

只在 P0/P1 稳定后进行，顺序为：

1. YOLO26n 1280，`batch=2, workers=0, epochs=5`；
2. YOLO26s 1024，`batch=2, workers=0, epochs=5`；
3. 前两者至少一个明显提升时，再运行 YOLO26s 1280 proxy；
4. 更大模型或 1536 最后考虑，本机不作默认命令。

## P4：按错误类型改后处理

- 小块 FP 为主：考虑面积、细长度、局部对比度过滤。
- 断裂为主：考虑同方向、低间距片段连接。
- 粘连为主：研究边界或骨架辅助拆分。
- FN 为主：优先转向分辨率、模型或训练目标，不继续激进提高 confidence/min_area。

## P5：工程归档

实验产物分为 `active / accepted / archive`；安全默认为 `workers=0`、`cache=false`、小 batch 和
周期 checkpoint。旧语义模型只作为 legacy/control，不出现在默认训练与评估流程中。
