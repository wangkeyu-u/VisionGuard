from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from visionguard.dataset import collect_split_images, load_dataset_definition
from visionguard.training import resolve_device


@dataclass(frozen=True)
class BenchmarkConfig:
    model: Path
    data: Path
    output_dir: Path
    imgsz: int = 512
    device: str = "auto"
    samples: int = 100
    warmup: int = 5


def _synchronize(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()
    elif device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def benchmark_yolo(config: BenchmarkConfig) -> dict[str, Any]:
    model_path = config.model.expanduser().resolve()
    data_path = config.data.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    device = resolve_device(config.device)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    test_images = collect_split_images(load_dataset_definition(data_path), "test")
    selected = test_images[: min(config.samples, len(test_images))]
    if not selected:
        raise ValueError("The test split contains no images to benchmark.")
    output_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    load_started = time.perf_counter()
    model = YOLO(str(model_path))
    _synchronize(device)
    load_ms = (time.perf_counter() - load_started) * 1000
    for index in range(config.warmup):
        model.predict(
            source=str(selected[index % len(selected)]),
            imgsz=config.imgsz,
            device=device,
            verbose=False,
        )
    _synchronize(device)

    wall_times: list[float] = []
    stage_times: dict[str, list[float]] = {"preprocess": [], "inference": [], "postprocess": []}
    for image_path in selected:
        _synchronize(device)
        started = time.perf_counter()
        result = model.predict(
            source=str(image_path), imgsz=config.imgsz, device=device, verbose=False
        )[0]
        _synchronize(device)
        wall_times.append((time.perf_counter() - started) * 1000)
        for stage in stage_times:
            stage_times[stage].append(float(result.speed[stage]))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": str(model_path),
        "data": str(data_path),
        "device": device,
        "imgsz": config.imgsz,
        "batch": 1,
        "samples": len(selected),
        "warmup_images": config.warmup,
        "model_load_time_ms": load_ms,
        "latency_includes_model_load": False,
        "latency_definition": "synchronized single-image wall time after warmup",
        "average_latency_ms": statistics.fmean(wall_times),
        "p50_latency_ms": float(np.percentile(wall_times, 50)),
        "p95_latency_ms": float(np.percentile(wall_times, 95)),
        "average_stage_time_ms": {
            stage: statistics.fmean(values) for stage, values in stage_times.items()
        },
        "model_file_size_bytes": model_path.stat().st_size,
        "model_file_size_mb": model_path.stat().st_size / (1024 * 1024),
        "sample_images": [str(path) for path in selected],
    }
    json_path = output_dir / "performance.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        "# Inference performance",
        "",
        f"- Device: `{device}`",
        f"- Input size: `{config.imgsz}`",
        f"- Fixed test images: {len(selected)}",
        f"- Warm-up images: {config.warmup}",
        "- Batch size: 1",
        "- Latency excludes model loading and uses synchronized wall time after warm-up.",
        f"- Model load: {load_ms:.3f} ms",
        f"- Mean latency: {report['average_latency_ms']:.3f} ms/image",
        f"- P50 latency: {report['p50_latency_ms']:.3f} ms/image",
        f"- P95 latency: {report['p95_latency_ms']:.3f} ms/image",
        f"- Mean preprocess: {report['average_stage_time_ms']['preprocess']:.3f} ms/image",
        f"- Mean inference: {report['average_stage_time_ms']['inference']:.3f} ms/image",
        f"- Mean postprocess: {report['average_stage_time_ms']['postprocess']:.3f} ms/image",
        f"- Model size: {report['model_file_size_mb']:.3f} MiB",
    ]
    (output_dir / "PERFORMANCE.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report
