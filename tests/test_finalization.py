from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from PIL import Image

from visionguard.config import DatasetFreezeConfig
from visionguard.finalization import finalize_dataset

CLEAN_CLASSES = [
    "person",
    "helmet",
    "vest",
    "gloves",
    "boots",
    "no_helmet",
    "no_vest",
    "no_gloves",
    "no_boots",
]


def _write_record(
    root: Path,
    split: str,
    filename: str,
    color: tuple[int, int, int],
    class_ids: list[int],
) -> Path:
    image_path = root / split / "images" / filename
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(image_path)
    label_path = root / split / "labels" / f"{image_path.stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "".join(f"{class_id} 0.5 0.5 0.4 0.4\n" for class_id in class_ids),
        encoding="utf-8",
    )
    return image_path


def _build_clean_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "safety_clean"
    original_images: list[Path] = []
    for index in range(30):
        split = ("train", "valid", "test")[index % 3]
        original_images.append(
            _write_record(
                root,
                split,
                f"source_{index}_jpg.rf.hash{index}.png",
                (index, 100 + index, 200 - index),
                [index % 7, 5, 6],
            )
        )
    _write_record(
        root,
        "valid",
        "source_0_jpg.rf.variant.png",
        (240, 10, 10),
        [0, 5, 6],
    )
    duplicate = _write_record(
        root,
        "test",
        "duplicate_jpg.rf.copy.png",
        (0, 0, 0),
        [1, 5, 6],
    )
    duplicate.write_bytes(original_images[1].read_bytes())
    _write_record(root, "train", "rare_only_jpg.rf.x.png", (250, 0, 0), [7])
    _write_record(root, "test", "mixed_rare_jpg.rf.x.png", (0, 250, 0), [0, 8])
    data_path = root / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": 9,
                "names": CLEAN_CLASSES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def test_finalization_deduplicates_filters_groups_and_freezes(tmp_path: Path) -> None:
    data_path = _build_clean_dataset(tmp_path)
    output_dir = tmp_path / "safety_final"

    report = finalize_dataset(DatasetFreezeConfig(data_path, output_dir, seed=42))

    assert report["counts"]["input_images"] == 34
    assert report["counts"]["duplicate_images_removed"] == 1
    assert report["counts"]["empty_label_images_removed"] == 1
    assert report["counts"]["final_images"] == 32
    assert report["counts"]["input_removed_class_annotations"] == {
        "no_boots": 1,
        "no_gloves": 1,
    }
    assert report["audit"]["passed"] is True
    assert report["audit"]["duplicate_sha256_groups"] == 0
    assert report["audit"]["cross_split_sha256_groups"] == 0
    assert report["audit"]["cross_split_source_groups"] == 0

    clean_yaml = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8"))
    assert clean_yaml["nc"] == 7
    assert list(clean_yaml["names"].values()) == CLEAN_CLASSES[:7]
    all_class_ids: set[int] = set()
    image_hashes: set[str] = set()
    source_zero_splits: set[str] = set()
    for split in ("train", "valid", "test"):
        images = list((output_dir / split / "images").glob("*"))
        labels = list((output_dir / split / "labels").glob("*.txt"))
        assert len(images) == len(labels)
        assert images
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            assert digest not in image_hashes
            image_hashes.add(digest)
            if image.name.startswith("source_0_jpg.rf."):
                source_zero_splits.add(split)
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                all_class_ids.add(int(line.split()[0]))
    assert all(0 <= class_id <= 6 for class_id in all_class_ids)
    assert len(source_zero_splits) == 1
    assert (output_dir / "freeze_manifest.json").is_file()
    assert (output_dir / "finalization_report.json").is_file()
    assert (output_dir / "finalization_report.md").is_file()

    rebuilt = finalize_dataset(DatasetFreezeConfig(data_path, output_dir, seed=42, overwrite=True))
    assert rebuilt["freeze"]["dataset_sha256"] == report["freeze"]["dataset_sha256"]
    assert not list(tmp_path.glob(".safety_final.backup-*"))


def test_finalization_refuses_existing_output(tmp_path: Path) -> None:
    data_path = _build_clean_dataset(tmp_path)
    output_dir = tmp_path / "safety_final"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        finalize_dataset(DatasetFreezeConfig(data_path, output_dir, seed=42))
