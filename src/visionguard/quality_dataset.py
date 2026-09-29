from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from visionguard.dataset import collect_split_images, label_path_for_image, load_dataset_definition
from visionguard.fingerprint import sha256_file, verify_frozen_dataset

RARE_CLASS_IDS = {5: "no_helmet", 6: "no_vest"}
SPLIT_DIRECTORIES = {"train": "train", "val": "valid", "test": "test"}


@dataclass(frozen=True)
class QualityDatasetConfig:
    data: Path
    output_dir: Path
    expected_fingerprint: str
    rare_multiplier: int = 3
    hard_multiplier: int = 4
    small_box_area: float = 0.01
    dense_annotations: int = 8
    overlap_iou: float = 0.10
    overwrite: bool = False

    def resolved(self) -> QualityDatasetConfig:
        if self.rare_multiplier < 1:
            raise ValueError("rare_multiplier must be at least 1")
        if self.hard_multiplier < self.rare_multiplier:
            raise ValueError("hard_multiplier must be at least rare_multiplier")
        if not 0 < self.small_box_area < 1:
            raise ValueError("small_box_area must be between 0 and 1")
        if self.dense_annotations < 1:
            raise ValueError("dense_annotations must be positive")
        if not 0 <= self.overlap_iou <= 1:
            raise ValueError("overlap_iou must be between 0 and 1")
        return QualityDatasetConfig(
            data=self.data.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            expected_fingerprint=self.expected_fingerprint,
            rare_multiplier=self.rare_multiplier,
            hard_multiplier=self.hard_multiplier,
            small_box_area=self.small_box_area,
            dense_annotations=self.dense_annotations,
            overlap_iou=self.overlap_iou,
            overwrite=self.overwrite,
        )


@dataclass(frozen=True)
class LabelBox:
    class_id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        )


def _read_boxes(label_path: Path) -> list[LabelBox]:
    boxes: list[LabelBox] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}")
        try:
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"invalid YOLO label at {label_path}:{line_number}") from exc
        boxes.append(LabelBox(class_id, x, y, width, height))
    return boxes


def _iou(first: LabelBox, second: LabelBox) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def sampling_decision(
    boxes: list[LabelBox],
    rare_multiplier: int,
    hard_multiplier: int,
    small_box_area: float,
    dense_annotations: int,
    overlap_iou: float,
) -> tuple[int, list[str]]:
    rare_indices = [index for index, box in enumerate(boxes) if box.class_id in RARE_CLASS_IDS]
    if not rare_indices:
        return 1, []
    reasons = ["rare_violation"]
    if any(boxes[index].area < small_box_area for index in rare_indices):
        reasons.append("small_rare_box")
    if len(boxes) >= dense_annotations:
        reasons.append("dense_scene")
    if any(
        _iou(boxes[index], other) >= overlap_iou
        for index in rare_indices
        for other_index, other in enumerate(boxes)
        if index != other_index
    ):
        reasons.append("overlapped_rare_box")
    multiplier = hard_multiplier if len(reasons) > 1 else rare_multiplier
    return multiplier, reasons


def _copy_or_link(source: Path, destination: Path, link_source: Path | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_source is not None:
        try:
            os.link(link_source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def _replica_name(path: Path, replica_index: int) -> str:
    return f"{path.stem}__vg_repeat_{replica_index}{path.suffix.lower()}"


def _write_data_yaml(staging_dir: Path, class_names: dict[int, str]) -> None:
    payload = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
        "derived": {
            "purpose": "train-only rare safety violation resampling",
            "validation_and_test_content": "byte-identical to the frozen source dataset",
        },
    }
    (staging_dir / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _freeze(staging_dir: Path, provenance: dict[str, dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for relative_image in sorted(
        path.relative_to(staging_dir)
        for split in SPLIT_DIRECTORIES.values()
        for path in (staging_dir / split / "images").iterdir()
        if path.is_file()
    ):
        image = staging_dir / relative_image
        label = staging_dir / relative_image.parent.parent / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise RuntimeError(f"derived image has no label: {image}")
        relative_label = label.relative_to(staging_dir)
        image_hash = sha256_file(image)
        label_hash = sha256_file(label)
        info = provenance[str(relative_image)]
        files.append(
            {
                "split": info["split"],
                "image": str(relative_image),
                "label": str(relative_label),
                "image_sha256": image_hash,
                "label_sha256": label_hash,
                "source_image": info["source_image"],
                "replica_index": info["replica_index"],
                "sampling_reasons": info["sampling_reasons"],
            }
        )
        digest.update(f"{relative_image}|{image_hash}|{relative_label}|{label_hash}\n".encode())
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "hash_algorithm": "sha256",
        "dataset_sha256": digest.hexdigest(),
        "files": files,
    }


def _publish(staging_dir: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        staging_dir.replace(output_dir)
        return
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    output_dir.replace(backup)
    try:
        staging_dir.replace(output_dir)
    except Exception:
        backup.replace(output_dir)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def build_quality_dataset(config: QualityDatasetConfig) -> dict[str, Any]:
    """Build a frozen train-only rare-class resampling dataset for safety fine-tuning."""
    config = config.resolved()
    source_verification = verify_frozen_dataset(config.data, config.expected_fingerprint)
    definition = load_dataset_definition(config.data)
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(f"output directory already exists: {config.output_dir}")
    if (
        config.output_dir == definition.root
        or config.output_dir in definition.root.parents
        or definition.root in config.output_dir.parents
    ):
        raise ValueError("quality dataset output and source dataset must not overlap")

    staging_dir = config.output_dir.parent / f".{config.output_dir.name}.tmp-{uuid.uuid4().hex}"
    provenance: dict[str, dict[str, Any]] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    reason_images: Counter[str] = Counter()
    link_modes: Counter[str] = Counter()
    try:
        for split, output_split in SPLIT_DIRECTORIES.items():
            original_class_counts: Counter[str] = Counter()
            derived_class_counts: Counter[str] = Counter()
            original_images = collect_split_images(definition, split)
            derived_images = 0
            for image in original_images:
                label = label_path_for_image(image, definition.root)
                boxes = _read_boxes(label)
                multiplier, reasons = (1, [])
                if split == "train":
                    multiplier, reasons = sampling_decision(
                        boxes,
                        config.rare_multiplier,
                        config.hard_multiplier,
                        config.small_box_area,
                        config.dense_annotations,
                        config.overlap_iou,
                    )
                    reason_images.update(reasons)
                for box in boxes:
                    original_class_counts[definition.class_names[box.class_id]] += 1
                base_image = staging_dir / output_split / "images" / image.name
                base_label = staging_dir / output_split / "labels" / f"{image.stem}.txt"
                link_modes[_copy_or_link(image, base_image)] += 1
                link_modes[_copy_or_link(label, base_label)] += 1
                for replica_index in range(multiplier):
                    if replica_index == 0:
                        destination_image = base_image
                    else:
                        destination_image = (
                            staging_dir
                            / output_split
                            / "images"
                            / _replica_name(image, replica_index)
                        )
                        destination_label = (
                            staging_dir
                            / output_split
                            / "labels"
                            / f"{destination_image.stem}.txt"
                        )
                        link_modes[_copy_or_link(image, destination_image, base_image)] += 1
                        link_modes[_copy_or_link(label, destination_label, base_label)] += 1
                    relative = destination_image.relative_to(staging_dir)
                    provenance[str(relative)] = {
                        "split": output_split,
                        "source_image": str(image),
                        "replica_index": replica_index,
                        "sampling_reasons": reasons,
                    }
                    derived_images += 1
                    for box in boxes:
                        derived_class_counts[definition.class_names[box.class_id]] += 1
            split_stats[output_split] = {
                "original_images": len(original_images),
                "derived_images": derived_images,
                "original_class_annotations": dict(original_class_counts),
                "derived_class_annotations": dict(derived_class_counts),
                "content_unchanged": None,
            }

        _write_data_yaml(staging_dir, definition.class_names)
        manifest = _freeze(staging_dir, provenance)
        source_manifest = json.loads(
            (definition.root / "freeze_manifest.json").read_text(encoding="utf-8")
        )
        source_hashes = {
            str(record["image"]): (record["image_sha256"], record["label_sha256"])
            for record in source_manifest["files"]
        }
        for output_split in ("valid", "test"):
            derived_records = [
                record for record in manifest["files"] if record["split"] == output_split
            ]
            unchanged = len(derived_records) == split_stats[output_split]["original_images"]
            unchanged = unchanged and all(
                source_hashes.get(record["image"])
                == (record["image_sha256"], record["label_sha256"])
                for record in derived_records
            )
            split_stats[output_split]["content_unchanged"] = unchanged
        split_stats["train"]["content_unchanged"] = False
        (staging_dir / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_data": str(config.data),
            "source_fingerprint": source_verification.to_dict(),
            "output_data": str(config.output_dir / "data.yaml"),
            "configuration": {
                "rare_multiplier": config.rare_multiplier,
                "hard_multiplier": config.hard_multiplier,
                "small_box_area": config.small_box_area,
                "dense_annotations": config.dense_annotations,
                "overlap_iou": config.overlap_iou,
                "rare_classes": RARE_CLASS_IDS,
            },
            "split_stats": split_stats,
            "train_sampling_reason_images": dict(reason_images),
            "storage_operations": dict(link_modes),
            "audit": {
                "source_fingerprint_verified": True,
                "only_train_was_resampled": True,
                "validation_content_hashes_match_source": split_stats["valid"][
                    "content_unchanged"
                ],
                "test_content_hashes_match_source": split_stats["test"]["content_unchanged"],
                "test_was_not_used_for_sampling": True,
            },
            "freeze": {
                "manifest": str(config.output_dir / "freeze_manifest.json"),
                "dataset_sha256": manifest["dataset_sha256"],
                "files": len(manifest["files"]),
            },
        }
        (staging_dir / "quality_dataset_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _publish(staging_dir, config.output_dir)
        return report
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
