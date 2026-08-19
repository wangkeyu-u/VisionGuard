from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from visionguard.evidence import EXPECTED_EXPERIMENTS, build_verification, verify_dataset


def _dataset(root: Path) -> Path:
    for split, color in (("train", "red"), ("valid", "green"), ("test", "blue")):
        image = root / split / "images" / f"source-{split}.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(image)
        label = root / split / "labels" / f"source-{split}.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data = root / "data.yaml"
    document = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": ["person"],
    }
    data.write_text(yaml.safe_dump(document), encoding="utf-8")
    return data


def _experiment_artifacts(root: Path) -> None:
    for index, name in enumerate(EXPECTED_EXPERIMENTS):
        path = root / name / "results.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "metrics/mAP50-95(B)"])
            writer.writeheader()
            writer.writerow({"epoch": 0, "metrics/mAP50-95(B)": 0.1 + index / 100})
    for name, score in ((EXPECTED_EXPERIMENTS[0], 0.21), (EXPECTED_EXPERIMENTS[-1], 0.25)):
        path = root / name / "test_evaluation" / "test_metrics.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"metrics": {"map50_95": score, "map50": score + 0.1}}), encoding="utf-8")
    performance = root / EXPECTED_EXPERIMENTS[-1] / "performance" / "performance.json"
    performance.parent.mkdir()
    performance.write_text(
        json.dumps(
            {
                "device": "test-device",
                "samples": 7,
                "warmup_images": 2,
                "average_latency_ms": 4.2,
                "p50_latency_ms": 4.0,
                "p95_latency_ms": 5.1,
            }
        ),
        encoding="utf-8",
    )


def test_verifier_recomputes_values_from_source_artifacts(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "dataset")
    experiments = tmp_path / "experiments"
    _experiment_artifacts(experiments)
    report = build_verification(data, experiments)
    assert report["dataset"]["status"] == "verified"
    assert report["dataset"]["images"] == 3
    assert report["dataset"]["annotations"] == 3
    assert report["yolo"]["status"] == "verified"
    assert report["yolo"]["derived"]["absolute_map50_95_improvement"] == pytest.approx(0.04)


def test_dataset_gate_detects_source_group_leakage_after_tamper(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "dataset")
    source = tmp_path / "dataset/train/images/source-train.jpg"
    target = tmp_path / "dataset/test/images/source-train.jpg"
    target.write_bytes(source.read_bytes())
    (tmp_path / "dataset/test/labels/source-train.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    report = verify_dataset(data)
    assert report["status"] == "failed"
    assert report["cross_split_source_groups"] == 1
    assert report["cross_split_sha256_groups"] == 1


def test_yolo_gate_fails_when_one_control_experiment_is_deleted(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "dataset")
    experiments = tmp_path / "experiments"
    _experiment_artifacts(experiments)
    (experiments / EXPECTED_EXPERIMENTS[2] / "results.csv").unlink()
    report = build_verification(data, experiments)
    assert report["yolo"]["status"] == "missing"
    assert any(EXPECTED_EXPERIMENTS[2] in item["path"] for item in report["yolo"]["missing"])
