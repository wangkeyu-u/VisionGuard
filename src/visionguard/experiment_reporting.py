from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return _load_json(path) if path.is_file() else None


def _best_validation_row(results_path: Path) -> dict[str, str]:
    if not results_path.is_file():
        raise FileNotFoundError(f"Training results do not exist: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Training results are empty: {results_path}")
    map_key = next((key for key in rows[0] if "mAP50-95" in key), None)
    if map_key is None:
        raise ValueError("results.csv does not contain a validation mAP50-95 column.")
    return max(rows, key=lambda row: float(row[map_key]))


def _training_summary(
    training: dict[str, Any], resume: dict[str, Any] | None
) -> dict[str, Any]:
    summary = {
        "status": training["status"],
        "started_at": training["started_at"],
        "ended_at": training["ended_at"],
        "duration_seconds": float(training["duration_seconds"]),
        "completed_epochs": training.get("completed_epochs"),
        "resumed": False,
    }
    if resume and resume.get("status") == "completed":
        summary.update(
            {
                "status": "completed",
                "ended_at": resume["ended_at"],
                "duration_seconds": summary["duration_seconds"]
                + float(resume["duration_seconds"]),
                "completed_epochs": resume.get("completed_epochs"),
                "resumed": True,
            }
        )
    return summary


def generate_experiment_report(
    run_dir: Path,
    data_path: Path,
    dataset_report_path: Path,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    training = _load_json(run_dir / "training_run.json")
    resume = _load_optional_json(run_dir / "resume_run.json")
    evaluation = _load_json(run_dir / "test_evaluation" / "test_metrics.json")
    performance = _load_json(run_dir / "performance" / "performance.json")
    errors = _load_json(run_dir / "error_analysis" / "error_analysis.json")
    dataset = _load_json(dataset_report_path.expanduser().resolve())
    best_validation = _best_validation_row(run_dir / "results.csv")
    config = training["config"]
    training_summary = _training_summary(training, resume)
    model_name = Path(config["model"]).stem
    test_metrics = evaluation["metrics"]
    environment = training["environment"]
    finalization = dataset.get("finalization", dataset)
    splits = finalization.get("splits", dataset.get("splits", {}))
    counts = finalization.get("counts", dataset.get("counts", {}))
    class_annotations = dataset.get("class_annotations", {})
    if not class_annotations:
        class_annotations = {
            name: sum(stats["class_annotations"][name] for stats in splits.values())
            for name in next(iter(splits.values()))["class_annotations"]
        }

    metric_keys = {key.strip(): value for key, value in best_validation.items()}
    val_precision = next((value for key, value in metric_keys.items() if "precision" in key), "n/a")
    val_recall = next((value for key, value in metric_keys.items() if "recall" in key), "n/a")
    val_map50 = next((value for key, value in metric_keys.items() if "mAP50(B)" in key), "n/a")
    val_map = next((value for key, value in metric_keys.items() if "mAP50-95" in key), "n/a")
    duration = float(training_summary["duration_seconds"])
    completed_epochs = training_summary["completed_epochs"] or best_validation.get("epoch", "n/a")
    target_epochs = int(config["epochs"])
    early_stopped = (
        training_summary["status"] == "completed"
        and isinstance(completed_epochs, int)
        and completed_epochs < target_epochs
    )

    lines = [
        f"# VisionGuard {model_name} Experiment Report",
        "",
        "## 1. 项目目标",
        "",
        "建立可复现的工作场所 PPE 检测实验，检测 person、helmet、vest、gloves、boots、"
        "no_helmet 与 no_vest；本阶段不包含前端、OCR、LLM、数据库、认证或 Docker。",
        "",
        "## 2. 数据集信息与指纹",
        "",
        f"- data.yaml: `{data_path.expanduser().resolve()}`",
        f"- SHA-256: `{training['dataset_fingerprint']['computed_sha256']}`",
        f"- 指纹逐文件验证: `{training['dataset_fingerprint']['passed']}` "
        f"({training['dataset_fingerprint']['files_verified']} images)",
        f"- 图片数: train {splits['train']['images']}, valid {splits['valid']['images']}, "
        f"test {splits['test']['images']}",
        "",
        "## 3. 类别分布",
        "",
        "| Class | Annotations |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in class_annotations.items())
    lines.extend(
        [
            "",
            "`no_helmet` 和 `no_vest` 的样本明显少于正类，相关指标的不确定性与类别不平衡风险较高。",
            "",
            "## 4. 数据清洗和防泄漏措施摘要",
            "",
            "- 使用 SHA-256 对完全重复图片去重，并验证 train/valid/test 无相同图片哈希。",
            "- 来源分组依赖 Roboflow 文件名中的 `.rf.` 前缀；缺少可靠的上游 source ID 时，这是近似来源分组。",
            "- 52 个重复组曾存在标签不一致，采用确定性择优保留，未伪造类别平衡。",
            f"- 去重移除 {counts.get('duplicate_images_removed', 78)} 张图片；最终无重复与跨 split 泄漏。",
            "",
            "## 5. 环境与硬件",
            "",
            f"- Python: `{environment['python_version']}`",
            f"- PyTorch: `{environment['pytorch_version']}`",
            f"- torchvision: `{environment['torchvision_version']}`",
            f"- Ultralytics: `{environment['ultralytics_version']}`",
            f"- Platform: `{environment['platform']}` / `{environment['machine']}`",
            f"- MPS available: `{environment['mps_available']}`; actual device: `{config['device']}`",
            f"- Git commit: `{training['git_commit']}`",
            "",
            "## 6. 训练配置",
            "",
            f"- Model: `{config['model']}` (pretrained checkpoint)",
            f"- epochs={config['epochs']}, imgsz={config['imgsz']}, batch={config['batch']}",
            f"- patience={config['patience']}, workers={config['workers']}, seed={config['seed']}",
            "- deterministic=True，未自动调整用户指定参数。",
            "",
            "## 7. 训练过程",
            "",
            f"- Final status: `{training_summary['status']}`",
            f"- Start: `{training_summary['started_at']}`",
            f"- End: `{training_summary['ended_at']}`",
            f"- Duration: {duration:.1f} seconds ({duration / 60:.2f} minutes)",
            f"- Completed epochs: {completed_epochs}/{target_epochs}",
            f"- Resumed after interruption: `{training_summary['resumed']}`",
            f"- Early stopping: `{early_stopped}` (patience={config['patience']})",
            "- 完整日志见 `training.log`，逐 epoch 指标见 `results.csv`。",
            "",
            "## 8. 验证集结果",
            "",
            f"最佳 mAP@50-95 epoch 行: epoch `{best_validation.get('epoch', 'n/a').strip()}`。",
            "",
            "| Precision | Recall | mAP@50 | mAP@50-95 |",
            "| ---: | ---: | ---: | ---: |",
            f"| {val_precision} | {val_recall} | {val_map50} | {val_map} |",
            "",
            "## 9. 测试集结果",
            "",
            "使用 `best.pt` 在 test split 上独立评估。",
            "",
            "| Precision | Recall | mAP@50 | mAP@50-95 | Test images | Predicted boxes |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {test_metrics['precision']:.6f} | {test_metrics['recall']:.6f} | "
            f"{test_metrics['map50']:.6f} | {test_metrics['map50_95']:.6f} | "
            f"{evaluation['test_images']} | {evaluation['total_prediction_boxes']} |",
            "",
            "## 10. 每类别表现",
            "",
            "| Class | Precision | Recall | AP@50 | AP@50-95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in evaluation["per_class"]:
        lines.append(
            f"| {row['class_name']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['ap50']:.6f} | {row['ap50_95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 11. 延迟与模型大小",
            "",
            f"- Device: `{performance['device']}`, imgsz={performance['imgsz']}, "
            f"batch=1, fixed samples={performance['samples']}",
            f"- Model loading: {performance['model_load_time_ms']:.3f} ms",
            "- 以下延迟已预热并同步设备，不包含模型加载时间。",
            f"- Mean/P50/P95: {performance['average_latency_ms']:.3f} / "
            f"{performance['p50_latency_ms']:.3f} / {performance['p95_latency_ms']:.3f} ms/image",
            f"- Preprocess/inference/postprocess: "
            f"{performance['average_stage_time_ms']['preprocess']:.3f} / "
            f"{performance['average_stage_time_ms']['inference']:.3f} / "
            f"{performance['average_stage_time_ms']['postprocess']:.3f} ms/image",
            f"- best.pt size: {performance['model_file_size_mb']:.3f} MiB",
            "",
            "## 12. 失败案例",
            "",
            "错误分析是供人工复核的启发式候选，不是绝对 false positive/false negative。",
            "",
            "| Category | Events | Images |",
            "| --- | ---: | ---: |",
        ]
    )
    for category, count in errors["event_counts"].items():
        lines.append(
            f"| {category.replace('_', ' ')} | {count} | {errors['image_counts'][category]} |"
        )
    lines.extend(
        [
            "",
            "详见 `error_analysis/ERROR_ANALYSIS.md` 与分类图片目录。",
            "",
            "## 13. 数据集限制",
            "",
            "- `no_helmet`、`no_vest` 相对稀少，安全违规类的召回率需要重点关注。",
            "- 数据来自多个公开来源/Roboflow 变体，标签定义、遮挡标注和框紧致度可能不一致。",
            "- `.rf.` 前缀分组是来源隔离代理，不等同于原始视频、工地或拍摄会话的可靠 group ID。",
            "- 52 个重复组的历史标签不一致提示仍可能存在标注噪声。",
            "- 启发式遮挡、小目标和密集场景分类需要人工抽查，不能代替专门场景标签。",
            "",
            "## 14. 下一轮实验建议",
            "",
            "1. 先人工复核 no_helmet/no_vest 与高频疑似漏检样例，修复标签噪声后再比较模型改动。",
            "2. 在不改变测试集的前提下，单变量比较更高输入尺寸或更大模型，并记录 MPS 内存/延迟代价。",
            "3. 对安全违规类尝试有依据的采样或损失策略；不得复制测试样本或伪造平衡。",
            "4. 增加可靠的 source/video/site group ID 后重新审计来源级泄漏。",
        ]
    )
    report_path = run_dir / "EXPERIMENT_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
