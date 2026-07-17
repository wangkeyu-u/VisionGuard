from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from visionguard.config import DatasetValidationConfig
from visionguard.dataset import label_path_for_image, validate_dataset
from visionguard.reporting import write_dataset_reports


def _write_image(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def _build_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    _write_image(root / "images" / "train" / "valid.jpg")
    (root / "labels" / "train" / "valid.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    _write_image(root / "images" / "val" / "invalid_class.jpg", (80, 20, 20))
    (root / "labels" / "val" / "invalid_class.txt").write_text(
        "9 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    _write_image(root / "images" / "val" / "missing_label.jpg", (20, 80, 20))

    duplicate = root / "images" / "test" / "duplicate.jpg"
    duplicate.write_bytes((root / "images" / "train" / "valid.jpg").read_bytes())
    (root / "labels" / "test" / "duplicate.txt").write_text("", encoding="utf-8")

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["person", "helmet", "safety_vest"],
    }
    data_path = tmp_path / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path


def test_label_path_uses_mirrored_labels_directory(tmp_path: Path) -> None:
    image = tmp_path / "images" / "train" / "frame.jpg"
    assert label_path_for_image(image, tmp_path) == tmp_path / "labels" / "train" / "frame.txt"


def test_validator_detects_requested_issue_types_and_writes_reports(tmp_path: Path) -> None:
    data_path = _build_dataset(tmp_path)
    output_dir = tmp_path / "reports"

    report = validate_dataset(DatasetValidationConfig(data_path, output_dir))
    codes = {issue["code"] for issue in report["issues"]}

    assert report["valid"] is False
    assert report["summary"]["images"] == 4
    assert report["summary"]["images_by_class"]["person"] == 1
    assert report["summary"]["annotations_by_class"]["person"] == 1
    assert report["summary"]["duplicate_groups"] == 1
    assert {"invalid_class_id", "missing_label", "empty_label", "duplicate_image"} <= codes

    json_path, markdown_path = write_dataset_reports(report, output_dir)
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["images"] == 4
    assert "VisionGuard Dataset Validation Report" in markdown_path.read_text(encoding="utf-8")


def test_validator_detects_invalid_bbox_and_unreadable_image(tmp_path: Path) -> None:
    data_path = _build_dataset(tmp_path)
    root = tmp_path / "dataset"
    (root / "images" / "test" / "broken.jpg").write_text("not an image", encoding="utf-8")
    (root / "labels" / "test" / "broken.txt").write_text("1 1.2 0.5 0.2 0.2\n", encoding="utf-8")

    report = validate_dataset(DatasetValidationConfig(data_path, tmp_path / "reports"))
    codes = {issue["code"] for issue in report["issues"]}

    assert "unreadable_image" in codes
    assert "invalid_bbox" in codes


def test_valid_dataset_passes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    colors = {"train": (20, 30, 40), "val": (50, 60, 70), "test": (80, 90, 100)}
    for split, color in colors.items():
        image_path = root / "images" / split / f"{split}.jpg"
        _write_image(image_path, color)
        label_path = root / "labels" / split / f"{split}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("2 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "validation": "images/val",
                "test": "images/test",
                "names": ["person", "helmet", "safety_vest"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = validate_dataset(DatasetValidationConfig(data_path, tmp_path / "reports"))

    assert report["valid"] is True
    assert report["summary"]["errors"] == 0
    assert report["summary"]["images_by_class"]["safety_vest"] == 3
    assert report["summary"]["annotations_by_class"]["safety_vest"] == 3
