from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionguard.dataset import collect_split_images, load_dataset_definition
from visionguard.inference import iter_batched_predictions, release_device_cache
from visionguard.training import resolve_device


@dataclass(frozen=True)
class EvaluationConfig:
    model: Path
    data: Path
    output_dir: Path
    imgsz: int = 512
    batch: int = 4
    workers: int = 0
    device: str = "auto"
    prediction_samples: int = 30

    def resolved(self) -> EvaluationConfig:
        model = self.model.expanduser().resolve()
        data = self.data.expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Best checkpoint does not exist: {model}")
        if not data.is_file():
            raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
        return EvaluationConfig(
            model=model,
            data=data,
            output_dir=self.output_dir.expanduser().resolve(),
            imgsz=self.imgsz,
            batch=self.batch,
            workers=self.workers,
            device=resolve_device(self.device),
            prediction_samples=self.prediction_samples,
        )


def build_metrics_report(
    names: dict[int, str],
    mean_metrics: list[float],
    per_class_rows: list[dict[str, Any]],
    test_images: int,
    prediction_boxes: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    precision, recall, map50, map50_95 = mean_metrics
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        **context,
        "test_images": test_images,
        "total_prediction_boxes": prediction_boxes,
        "prediction_box_confidence_threshold": 0.001,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        },
        "classes": [names[index] for index in sorted(names)],
        "per_class": per_class_rows,
    }


def write_metrics_exports(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "test_metrics.json"
    csv_path = output_dir / "test_metrics_per_class.csv"
    markdown_path = output_dir / "TEST_EVALUATION.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fieldnames = ["class_id", "class_name", "precision", "recall", "ap50", "ap50_95"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["per_class"])
    metrics = report["metrics"]
    lines = [
        "# Test-set evaluation",
        "",
        f"- Checkpoint: `{report['model']}`",
        f"- Dataset: `{report['data']}`",
        f"- Split: `test` ({report['test_images']} images)",
        f"- Device: `{report['device']}`",
        f"- Input size: `{report['imgsz']}`",
        f"- Total predicted boxes (confidence >= 0.001): {report['total_prediction_boxes']}",
        "",
        "## Overall metrics",
        "",
        "| Precision | Recall | mAP@50 | mAP@50-95 |",
        "| ---: | ---: | ---: | ---: |",
        f"| {metrics['precision']:.6f} | {metrics['recall']:.6f} | "
        f"{metrics['map50']:.6f} | {metrics['map50_95']:.6f} |",
        "",
        "## Per-class metrics",
        "",
        "| ID | Class | Precision | Recall | AP@50 | AP@50-95 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_class"]:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['ap50']:.6f} | {row['ap50_95']:.6f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def evaluate_yolo(config: EvaluationConfig) -> dict[str, Any]:
    config = config.resolved()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    definition = load_dataset_definition(config.data)
    test_images = collect_split_images(definition, "test")
    from ultralytics import YOLO

    model = YOLO(str(config.model))
    runtime_data = config.model.parent.parent / "resolved_data.yaml"
    if not runtime_data.is_file():
        raise FileNotFoundError(
            "Read-only runtime data YAML is missing beside the training run: " f"{runtime_data}"
        )
    metrics = model.val(
        data=str(runtime_data),
        split="test",
        imgsz=config.imgsz,
        batch=config.batch,
        workers=config.workers,
        device=config.device,
        plots=True,
        project=str(config.output_dir),
        name="plots",
        exist_ok=True,
        verbose=True,
    )
    class_indices = [int(value) for value in metrics.box.ap_class_index]
    result_index = {class_id: index for index, class_id in enumerate(class_indices)}
    per_class_rows: list[dict[str, Any]] = []
    for class_id, class_name in sorted(definition.class_names.items()):
        index = result_index.get(class_id)
        values = metrics.box.class_result(index) if index is not None else (0.0, 0.0, 0.0, 0.0)
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(values[0]),
                "recall": float(values[1]),
                "ap50": float(values[2]),
                "ap50_95": float(values[3]),
            }
        )

    mean_metrics = list(metrics.box.mean_results())
    speed = {key: float(value) for key, value in metrics.speed.items()}
    release_device_cache(config.device)

    prediction_boxes = 0
    predictions = iter_batched_predictions(
        model,
        test_images,
        config.batch,
        imgsz=config.imgsz,
        conf=0.001,
        device=config.device,
        verbose=False,
    )
    for result in predictions:
        prediction_boxes += len(result.boxes)

    release_device_cache(config.device)
    sample_images = test_images[: config.prediction_samples]
    if sample_images:
        for _ in iter_batched_predictions(
            model,
            sample_images,
            config.batch,
            imgsz=config.imgsz,
            conf=0.25,
            device=config.device,
            save=True,
            project=str(config.output_dir),
            name="prediction_samples",
            exist_ok=True,
            verbose=False,
        ):
            pass

    report = build_metrics_report(
        definition.class_names,
        mean_metrics,
        per_class_rows,
        len(test_images),
        prediction_boxes,
        {
            "model": str(config.model),
            "data": str(config.data),
            "device": config.device,
            "imgsz": config.imgsz,
            "ultralytics_speed_ms_per_image": speed,
            "plots_directory": str(config.output_dir / "plots"),
        },
    )
    report["exports"] = {
        key: str(value) for key, value in write_metrics_exports(report, config.output_dir).items()
    }
    (config.output_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
