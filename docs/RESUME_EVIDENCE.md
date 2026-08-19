# Resume evidence matrix

Generated from machine-readable source audits. README prose is not accepted as source evidence.

| Claim | Status | Evidence boundary |
| --- | --- | --- |
| `dataset` | **verified** | Source artifacts and gates pass. |
| `yolo` | **implemented_unverified** | [{"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/baseline_yolo11n_512/results.csv", "purpose": "baseline_yolo11n_512 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp2_yolo11n_640/results.csv", "purpose": "exp2_yolo11n_640 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp3_yolo11s_512/results.csv", "purpose": "exp3_yolo11s_512 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp4_yolo11s_512_e50/results.csv", "purpose": "exp4_yolo11s_512_e50 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/baseline_yolo11n_512/test_evaluation/test_metrics.json", "purpose": "baseline independent test metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp4_yolo11s_512_e50/test_evaluation/test_metrics.json", "purpose": "selected-model independent test metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp4_yolo11s_512_e50/performance/performance.json", "purpose": "selected-model latency samples/summary", "exists": false}] |
| `review` | **verified** | "Local prototype; no authentication, multi-user locking study, or operator study." |
| `vlm` | **unsupported** | ["Selected project YOLO checkpoint is missing.", "No committed source prediction set exists for the historical Dev F1=0.600 statement.", "Recovered Dev labels have mixed reviewer provenance and cannot independently verify the resume claim.", "No grounded-Qwen3-VL real run exists because the selected project YOLO checkpoint is missing."] |

## Claims

### dataset - verified

> 构建数据清洗、去重、同源分组切分、训练评估与可视化流水线，处理 8,762 张图片、43,727 个标注、7 类 PPE 目标，使用数据指纹和 source-group 隔离控制数据泄漏。

Code: `src/visionguard/remapping.py`, `src/visionguard/finalization.py`, `src/visionguard/evidence.py`

Verification:

- `.venv/bin/python scripts/remap_dataset_classes.py --data training/source_downloads.yaml --output-dir training/datasets/safety_clean`
- `.venv/bin/python scripts/finalize_dataset.py --data training/datasets/safety_clean/data.yaml --output-dir training/datasets/safety_final --seed 42`
- `.venv/bin/python scripts/verify_resume_evidence.py --data training/datasets/safety_final/data.yaml --output docs/source-verification.json`

Artifacts: `docs/source-verification.json`, `docs/evidence/class_remap_report.json`, `docs/evidence/finalization_report.json`, `docs/evidence/freeze_manifest.json`

### yolo - implemented_unverified

> 完成 4 组 YOLO11 控制实验，最终模型独立测试集 mAP@50–95 0.4602、mAP@50 0.6615，较基线提升 0.0289，MPS 单图平均延迟 23.61 ms。

Code: `src/visionguard/training.py`, `src/visionguard/evaluation.py`, `src/visionguard/benchmarking.py`, `src/visionguard/evidence.py`

Verification:

- `.venv/bin/python scripts/verify_resume_evidence.py --require-verified`
- `make comparison`

Artifacts: `docs/source-verification.json`

Gap: [{"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/baseline_yolo11n_512/results.csv", "purpose": "baseline_yolo11n_512 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp2_yolo11n_640/results.csv", "purpose": "exp2_yolo11n_640 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp3_yolo11s_512/results.csv", "purpose": "exp3_yolo11s_512 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp4_yolo11s_512_e50/results.csv", "purpose": "exp4_yolo11s_512_e50 training metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/baseline_yolo11n_512/test_evaluation/test_metrics.json", "purpose": "baseline independent test metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp4_yolo11s_512_e50/test_evaluation/test_metrics.json", "purpose": "selected-model independent test metrics", "exists": false}, {"path": "/Users/wangkeyu/Documents/Codex/2026-08-12/visionguard-interview-alignment/work/VisionGuard/outputs/experiments/exp4_yolo11s_512_e50/performance/performance.json", "purpose": "selected-model latency samples/summary", "exists": false}]

### review - verified

> 自研人工审核 Web 平台，支持人物框编辑、违规确认、Test 盲审、脏数据排除和 JSONL 原子写入，并实现标注协议校验和评估数据门禁。

Code: `src/visionguard/review.py`, `src/visionguard/demo_server.py`, `demo/static/app.js`, `schemas/review_protocol.schema.json`

Verification:

- `.venv/bin/python -m pytest -q tests/test_review.py`
- `node --check demo/static/app.js`

Artifacts: `schemas/review_protocol.schema.json`, `docs/evidence/review_visual_qa.json`

Gap: "Local prototype; no authentication, multi-user locking study, or operator study."

### vlm - unsupported

> 完成 YOLO/Qwen3-VL/grounded-Qwen3-VL 消融实验；通过统一 IoU-grounded 指标定位 VLM 幻觉与 grounding 问题，Dev 最优方案 F1 0.600。

Code: `src/visionguard/vlm.py`, `src/visionguard/vlm_evaluation.py`, `scripts/evaluate_vlm.py`

Verification:

- `.venv/bin/python scripts/prepare_historical_vlm_dev.py`
- `.venv/bin/python scripts/evaluate_vlm.py --config configs/vlm_qwen_real_historical.yaml --cache-dir outputs/vlm/cache --output outputs/vlm/qwen_real_historical.json`
- `make vlm-fixture`
- `.venv/bin/python scripts/promote_vlm_diagnostic.py`

Artifacts: `docs/evidence/vlm_asset_audit.json`, `docs/evidence/vlm_dev_source_audit.json`, `docs/evidence/qwen_real_historical_result.json`, `docs/evidence/QWEN_DIAGNOSTIC_FAILURES.md`, `outputs/vlm/fixture_evaluation.json (fixture only)`

Gap: ["Selected project YOLO checkpoint is missing.", "No committed source prediction set exists for the historical Dev F1=0.600 statement.", "Recovered Dev labels have mixed reviewer provenance and cannot independently verify the resume claim.", "No grounded-Qwen3-VL real run exists because the selected project YOLO checkpoint is missing."]
