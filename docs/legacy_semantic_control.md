# Legacy semantic control

旧 `ResNet34 + semantic mask + connected components` 路线只作为历史对照保留，不属于当前默认实验流程。

下列内容均为 legacy/control：

- `scripts/train.py`、`scripts/evaluate_model.py`、`scripts/run_submission.py`；
- `scripts/tune_oof_postprocess.py`、`scripts/analyze_oof_errors.py`、`src/oof_*`；
- `configs/global_resnet34_fold0.yaml`、`configs/screen/`；
- `artifacts/runs/global_resnet34_fold0`、`artifacts/postprocess/fold0`；
- `artifacts/analysis/errors_fold0`。

不得将这些语义模型 OOF 概率、后处理参数、错误分类或可视化当作当前 YOLO 的证据。
YOLO 只使用 `cache_yolo_candidates -> evaluate_yolo -> analyze_yolo_errors` 评估链。

如需恢复语义 control，必须显式使用上述 legacy 入口并添加
`--allow-legacy-semantic`，否则程序会立即停止。输出应写入独立的
`artifacts/legacy_semantic/` 目录，不覆盖 YOLO 缓存或报告。
