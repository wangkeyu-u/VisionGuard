from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.comparison_reporting import build_comparison, generate_comparison_report
from visionguard.experiment_reporting import _training_summary


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_experiment(tmp_path: Path, name: str, map_value: float) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    config = {
        "model": "yolo11n.pt",
        "epochs": 2,
        "imgsz": 512,
        "batch": 1,
        "patience": 1,
        "workers": 0,
        "seed": 42,
        "device": "cpu",
    }
    _write_json(
        run_dir / "training_run.json",
        {
            "status": "completed",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:01:00Z",
            "duration_seconds": 60,
            "completed_epochs": 2,
            "config": config,
        },
    )
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        f"1,0.5,0.4,0.45,{map_value - 0.01}\n"
        f"2,0.6,0.5,0.55,{map_value}\n",
        encoding="utf-8",
    )
    per_class = [
        {
            "class_id": 5,
            "class_name": "no_helmet",
            "precision": 0.4,
            "recall": 0.3,
            "ap50": 0.3,
            "ap50_95": map_value / 2,
        },
        {
            "class_id": 6,
            "class_name": "no_vest",
            "precision": 0.5,
            "recall": 0.4,
            "ap50": 0.4,
            "ap50_95": map_value / 1.5,
        },
    ]
    _write_json(
        run_dir / "test_evaluation" / "test_metrics.json",
        {
            "test_images": 10,
            "metrics": {
                "precision": 0.6,
                "recall": 0.5,
                "map50": 0.55,
                "map50_95": map_value - 0.02,
            },
            "per_class": per_class,
        },
    )
    _write_json(
        run_dir / "performance" / "performance.json",
        {
            "average_latency_ms": 10.0,
            "p50_latency_ms": 9.0,
            "p95_latency_ms": 12.0,
            "model_file_size_mb": 5.0,
            "samples": 100,
            "device": "cpu",
        },
    )
    return run_dir


def test_training_summary_treats_completed_resume_as_final_status() -> None:
    training = {
        "status": "interrupted",
        "started_at": "start",
        "ended_at": "interrupt",
        "duration_seconds": 10,
        "completed_epochs": 1,
    }
    resume = {
        "status": "completed",
        "ended_at": "finish",
        "duration_seconds": 20,
        "completed_epochs": 3,
    }

    summary = _training_summary(training, resume)

    assert summary["status"] == "completed"
    assert summary["duration_seconds"] == 30
    assert summary["completed_epochs"] == 3
    assert summary["resumed"] is True


def test_comparison_selects_highest_validation_map_and_exports_reports(tmp_path: Path) -> None:
    baseline = _make_experiment(tmp_path, "baseline", 0.40)
    candidate = _make_experiment(tmp_path, "candidate", 0.50)

    comparison = build_comparison([baseline, candidate])

    assert comparison["selected_experiment"] == "candidate"
    assert comparison["validation_map50_95_delta"] == pytest.approx(0.1)
    assert comparison["test_map50_95_delta"] == pytest.approx(0.1)

    outputs = generate_comparison_report([baseline, candidate], tmp_path / "report")
    assert all(path.is_file() for path in outputs.values())
    assert "candidate" in outputs["markdown"].read_text(encoding="utf-8")
    artifact = json.loads(outputs["artifact"].read_text(encoding="utf-8"))
    assert artifact["surface"] == "report"
    assert len(artifact["manifest"]["charts"]) == 2
