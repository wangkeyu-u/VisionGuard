from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from visionguard.config import SAFETY_TARGET_CLASSES, DatasetRemapConfig
from visionguard.remapping import remap_dataset

SOURCE_CLASSES = [
    "Gloves",
    "Helmet",
    "Human",
    "Safety Boot",
    "Safety Vest",
    "boots",
    "glasses",
    "gloves",
    "hat",
    "helmet",
    "no boot",
    "no boots",
    "no gloves",
    "no hat",
    "no vest",
    "vest",
]


def _write_sample(root: Path, split: str, name: str, labels: list[str]) -> None:
    image_path = root / split / "images" / f"{name}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), (30, 50, 70)).save(image_path)
    label_path = root / split / "labels" / f"{name}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def _build_source_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "safety"
    box = "0.5 0.5 0.4 0.4"
    _write_sample(root, "train", "mapped", [f"{class_id} {box}" for class_id in (2, 1, 9, 8, 6)])
    _write_sample(root, "train", "glasses_only", [f"6 {box}"])
    _write_sample(
        root, "valid", "mapped", [f"{class_id} {box}" for class_id in (0, 7, 3, 5, 10, 11)]
    )
    _write_sample(root, "test", "mapped", [f"{class_id} {box}" for class_id in (4, 15, 12, 13, 14)])
    data_path = root / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": 16,
                "names": SOURCE_CLASSES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def _class_ids(label_path: Path) -> list[int]:
    return [int(line.split()[0]) for line in label_path.read_text(encoding="utf-8").splitlines()]


def test_remapping_rewrites_ids_removes_glasses_and_preserves_source(tmp_path: Path) -> None:
    data_path = _build_source_dataset(tmp_path)
    source_label = data_path.parent / "train" / "labels" / "mapped.txt"
    original_label_content = source_label.read_text(encoding="utf-8")
    output_dir = tmp_path / "safety_clean"

    report = remap_dataset(DatasetRemapConfig(data_path, output_dir))

    assert _class_ids(output_dir / "train" / "labels" / "mapped.txt") == [0, 1, 1, 1]
    assert _class_ids(output_dir / "valid" / "labels" / "mapped.txt") == [3, 3, 4, 4, 8, 8]
    assert _class_ids(output_dir / "test" / "labels" / "mapped.txt") == [2, 2, 7, 5, 6]
    assert (output_dir / "train" / "labels" / "glasses_only.txt").read_text(encoding="utf-8") == ""
    assert (output_dir / "train" / "images" / "glasses_only.jpg").is_file()
    assert source_label.read_text(encoding="utf-8") == original_label_content

    assert report["removed_glasses_annotations"] == 2
    assert report["empty_label_images_count"] == 1
    assert report["splits"]["train"]["images"] == 2
    assert report["old_class_counts"]["helmet"] == 1
    assert report["new_class_counts"]["helmet"] == 3

    clean_yaml = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8"))
    assert clean_yaml["names"] == {index: name for index, name in enumerate(SAFETY_TARGET_CLASSES)}
    assert clean_yaml["nc"] == 9
    assert (
        json.loads((output_dir / "class_remap_report.json").read_text(encoding="utf-8"))[
            "source_dataset_modified"
        ]
        is False
    )
    assert (output_dir / "class_remap_report.md").is_file()


def test_remapping_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    data_path = _build_source_dataset(tmp_path)
    output_dir = tmp_path / "safety_clean"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        remap_dataset(DatasetRemapConfig(data_path, output_dir))
