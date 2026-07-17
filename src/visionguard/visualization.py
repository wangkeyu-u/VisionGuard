from __future__ import annotations

import colorsys
import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from visionguard.config import SPLIT_ALIASES, DatasetVisualizationConfig
from visionguard.dataset import (
    DatasetDefinition,
    bbox_validation_error,
    collect_split_images,
    label_path_for_image,
    load_dataset_definition,
)


@dataclass(frozen=True)
class YoloAnnotation:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class SampleCandidate:
    image_path: Path
    annotations: tuple[YoloAnnotation, ...]


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable, visually separated RGB color for a class ID."""
    hue = (class_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)


def _read_annotations(
    label_path: Path, class_names: dict[int, str]
) -> tuple[tuple[YoloAnnotation, ...], int]:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return (), 1

    annotations: list[YoloAnnotation] = []
    invalid_count = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            invalid_count += 1
            continue
        try:
            class_id = int(fields[0])
            bbox = [float(value) for value in fields[1:]]
        except ValueError:
            invalid_count += 1
            continue
        if class_id not in class_names or bbox_validation_error(bbox) is not None:
            invalid_count += 1
            continue
        annotations.append(YoloAnnotation(class_id, *bbox))
    return tuple(annotations), invalid_count


def _image_can_be_opened(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.load()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def _sample_candidates(
    definition: DatasetDefinition, split: str
) -> tuple[list[SampleCandidate], int, int]:
    candidates: list[SampleCandidate] = []
    skipped_images = 0
    invalid_annotations = 0
    for image_path in collect_split_images(definition, split):
        if not image_path.is_file() or not _image_can_be_opened(image_path):
            skipped_images += 1
            continue
        label_path = label_path_for_image(image_path, definition.root)
        if not label_path.is_file():
            skipped_images += 1
            continue
        annotations, invalid_count = _read_annotations(label_path, definition.class_names)
        invalid_annotations += invalid_count
        if not annotations:
            skipped_images += 1
            continue
        candidates.append(SampleCandidate(image_path, annotations))
    return candidates, skipped_images, invalid_annotations


def _box_pixels(annotation: YoloAnnotation, image_width: int, image_height: int) -> tuple[int, ...]:
    left = round((annotation.x_center - annotation.width / 2) * image_width)
    top = round((annotation.y_center - annotation.height / 2) * image_height)
    right = round((annotation.x_center + annotation.width / 2) * image_width)
    bottom = round((annotation.y_center + annotation.height / 2) * image_height)
    return (
        max(0, min(image_width - 1, left)),
        max(0, min(image_height - 1, top)),
        max(0, min(image_width - 1, right)),
        max(0, min(image_height - 1, bottom)),
    )


def _draw_candidate(
    candidate: SampleCandidate,
    output_path: Path,
    class_names: dict[int, str],
) -> None:
    with Image.open(candidate.image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) * 0.004))

    for annotation in candidate.annotations:
        color = class_color(annotation.class_id)
        box = _box_pixels(annotation, image.width, image.height)
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{class_names[annotation.class_id]} | conf: N/A"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = box[0]
        text_y = box[1] - text_height - 6
        if text_y < 0:
            text_y = box[1] + 2
        background = (
            text_x,
            text_y,
            min(image.width - 1, text_x + text_width + 6),
            min(image.height - 1, text_y + text_height + 4),
        )
        draw.rectangle(background, fill=color)
        draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def _output_filename(index: int, image_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem).strip("._") or "image"
    digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:8]
    return f"sample_{index:03d}_{safe_stem}_{digest}.png"


def _empty_class_distribution(class_names: dict[int, str]) -> dict[str, int]:
    return {name: 0 for name in class_names.values()}


def visualize_dataset(config: DatasetVisualizationConfig) -> dict[str, Any]:
    config = config.resolved()
    definition = load_dataset_definition(config.data_path)
    rng = random.Random(config.seed)
    split_reports: dict[str, dict[str, Any]] = {}
    total_distribution: Counter[str] = Counter()
    warnings: list[str] = []

    for split in SPLIT_ALIASES:
        split_dir = config.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for previous_sample in split_dir.glob("sample_*.png"):
            previous_sample.unlink()

        candidates, skipped_images, invalid_annotations = _sample_candidates(definition, split)
        sample_count = min(config.samples_per_split, len(candidates))
        selected = rng.sample(candidates, sample_count)
        class_distribution: Counter[str] = Counter()
        samples: list[dict[str, Any]] = []

        if sample_count < config.samples_per_split:
            warnings.append(
                f"{split}: requested {config.samples_per_split} samples, but only "
                f"{sample_count} annotated images were available."
            )

        for index, candidate in enumerate(selected, 1):
            output_path = split_dir / _output_filename(index, candidate.image_path)
            _draw_candidate(candidate, output_path, definition.class_names)
            sample_distribution = Counter(
                definition.class_names[annotation.class_id] for annotation in candidate.annotations
            )
            class_distribution.update(sample_distribution)
            samples.append(
                {
                    "source_image": str(candidate.image_path),
                    "output_image": str(output_path),
                    "objects": len(candidate.annotations),
                    "class_distribution": {
                        name: sample_distribution[name] for name in definition.class_names.values()
                    },
                }
            )

        total_objects = sum(class_distribution.values())
        complete_distribution = {
            name: class_distribution[name] for name in definition.class_names.values()
        }
        total_distribution.update(class_distribution)
        split_reports[split] = {
            "requested": config.samples_per_split,
            "available_annotated_images": len(candidates),
            "images_visualized": len(samples),
            "skipped_images": skipped_images,
            "invalid_annotations_skipped": invalid_annotations,
            "objects_visualized": total_objects,
            "average_objects_per_image": round(total_objects / len(samples), 4) if samples else 0.0,
            "class_distribution": complete_distribution,
            "samples": samples,
        }

    total_images = sum(split["images_visualized"] for split in split_reports.values())
    total_objects = sum(split["objects_visualized"] for split in split_reports.values())
    color_map = {
        name: "#{:02x}{:02x}{:02x}".format(*class_color(class_id))
        for class_id, name in definition.class_names.items()
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_yaml": str(definition.data_path),
        "dataset_root": str(definition.root),
        "output_dir": str(config.output_dir),
        "configuration": {
            "samples_per_split": config.samples_per_split,
            "seed": config.seed,
        },
        "class_names": {str(class_id): name for class_id, name in definition.class_names.items()},
        "class_colors": color_map,
        "summary": {
            "images_visualized": total_images,
            "objects_visualized": total_objects,
            "average_objects_per_image": round(total_objects / total_images, 4)
            if total_images
            else 0.0,
            "class_distribution": {
                name: total_distribution[name] for name in definition.class_names.values()
            },
        },
        "splits": split_reports,
        "warnings": warnings,
    }
