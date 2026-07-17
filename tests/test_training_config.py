from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visionguard.training import (
    FROZEN_DATASET_SHA256,
    TrainingConfig,
    _resolved_ultralytics_data_yaml,
)


def test_baseline_training_config_resolves_paths_and_values(tmp_path: Path) -> None:
    config = TrainingConfig(data=tmp_path / "data.yaml", project=tmp_path / "runs").resolved()

    assert config.model == "yolo11n.pt"
    assert config.epochs == 30
    assert config.imgsz == 512
    assert config.batch == 4
    assert config.patience == 8
    assert config.workers == 0
    assert config.seed == 42
    assert config.device in {"mps", "cpu"}
    assert config.expected_fingerprint == FROZEN_DATASET_SHA256
    assert config.data.is_absolute()
    assert config.project.is_absolute()


def test_training_config_rejects_invalid_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch"):
        TrainingConfig(data=tmp_path / "data.yaml", batch=0).resolved()


def test_runtime_data_yaml_resolves_root_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "dataset" / "data.yaml"
    for split in ("train", "valid", "test"):
        (source.parent / split / "images").mkdir(parents=True)
        (source.parent / split / "labels").mkdir(parents=True)
        (source.parent / split / "images" / "sample.jpg").write_bytes(b"image")
        (source.parent / split / "labels" / "sample.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    original = (
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnames: [person]\n"
    )
    source.write_text(original, encoding="utf-8")

    destination = _resolved_ultralytics_data_yaml(source, tmp_path / "resolved.yaml")

    assert source.read_text(encoding="utf-8") == original
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    runtime_root = destination.parent / "runtime_dataset"
    assert payload["path"] == str(runtime_root)
    assert (runtime_root / "train" / "images" / "sample.jpg").is_file()
    assert (runtime_root / "valid" / "labels" / "sample.txt").is_file()
