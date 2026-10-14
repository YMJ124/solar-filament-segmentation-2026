# P1 三 Fold 可信验证记录

本阶段已完成。最终决策以
`artifacts/instance/stage_p1/stage_report.json` 为准，不再执行本阶段候选训练。

## 已接受基线

- B0：YOLO26n、1024、medoid、batch 2、5 epoch、seed 2026。
- Fold0/1/2 平均 PQ：`0.28787300897915613`。
- Fold 标准差：`0.01300831955120201`。
- 共同后处理参数：`confidence=0.10`、`min_area=600`、
  `max_instances=20`、`force_one=true`。

保留产物：

- `artifacts/instance/stage_p1/b0_medoid_1024_fold0/`
- `artifacts/instance/stage_p1/b0_medoid_1024_fold1/`
- `artifacts/instance/stage_p1/b0_medoid_1024_fold2/`
- `artifacts/instance/stage_p1/b0_medoid_1024/`
- `data/processed/yolo_stage_p1/medoid_fold0/`
- `data/processed/yolo_stage_p1/medoid_fold1/`
- `data/processed/yolo_stage_p1/medoid_fold2/`

## 已退役候选

- C1：random annotator per epoch、1024，固定 32 图面板相对 B0
  `-0.027132578309500555 PQ`，未通过 `+0.010` 门槛。
- C2：medoid、1280，固定 32 图面板相对 B0
  `+0.005768808992543839 PQ`，未通过 `+0.010` 门槛。

两者的训练产物、候选缓存、专用数据和运行配置已从活动工作区清理。
仅保留两份小型面板证据：

- `artifacts/instance/archive/stage_p1/c1_random_annotator_1024_panel32.json`
- `artifacts/instance/archive/stage_p1/c2_medoid_1280_panel32.json`

随机标注者调度与训练代码作为通用基础设施保留，但不在默认训练流程中启用。

## 验证与下一步

本阶段结论为 `all_failed`，无候选进入 30 epoch 长训练。下一步回到
训练目标和错误样例复核，优先分析小实例漏检、Fold2 假阳性和多标注者差异。

快速回归检查：

```bash
conda run -n solar-filament pytest -q tests/test_yolo_stage_p1.py
```
