from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from visionguard.config import IMAGE_EXTENSIONS, SPLIT_ALIASES, DatasetValidationConfig


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    code: str
    message: str
    split: str | None = None
    image_path: str | None = None
    label_path: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class DatasetDefinition:
    data_path: Path
    root: Path
    class_names: dict[int, str]
    split_sources: dict[str, tuple[Path, ...]]


def _read_yaml(data_path: Path) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")
    try:
        content = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {data_path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"Dataset YAML must contain a mapping at its root: {data_path}")
    return content


def _parse_class_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Every key in data.yaml 'names' must be an integer class ID.") from exc
    else:
        raise ValueError("data.yaml must define 'names' as a list or an ID-to-name mapping.")

    if not names:
        raise ValueError("data.yaml 'names' must contain at least one class.")
    if any(index < 0 for index in names):
        raise ValueError("Class IDs in data.yaml cannot be negative.")
    return dict(sorted(names.items()))


def _dataset_root(data_path: Path, yaml_data: dict[str, Any]) -> Path:
    raw_root = yaml_data.get("path")
    if raw_root in (None, ""):
        return data_path.parent
    root = Path(str(raw_root)).expanduser()
    return root.resolve() if root.is_absolute() else (data_path.parent / root).resolve()


def _split_value(yaml_data: dict[str, Any], split: str) -> Any:
    for alias in SPLIT_ALIASES[split]:
        if alias in yaml_data:
            return yaml_data[alias]
    return None


def _resolve_sources(raw_value: Any, root: Path) -> list[Path]:
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    sources: list[Path] = []
    for value in values:
        path = Path(str(value)).expanduser()
        sources.append(path.resolve() if path.is_absolute() else (root / path).resolve())
    return sources


def _images_from_manifest(manifest: Path, root: Path) -> list[Path]:
    images: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if candidate.is_absolute():
            images.append(candidate.resolve())
            continue
        root_candidate = (root / candidate).resolve()
        manifest_candidate = (manifest.parent / candidate).resolve()
        images.append(root_candidate if root_candidate.exists() else manifest_candidate)
    return images


def _collect_images(source: Path, root: Path) -> list[Path]:
    if source.is_dir():
        return sorted(
            path.resolve()
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if source.is_file() and source.suffix.lower() == ".txt":
        return _images_from_manifest(source, root)
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source.resolve()]
    return []


def load_dataset_definition(data_path: Path) -> DatasetDefinition:
    """Load a complete YOLO dataset definition for downstream pipeline steps."""
    resolved_data_path = data_path.expanduser().resolve()
    yaml_data = _read_yaml(resolved_data_path)
    class_names = _parse_class_names(yaml_data.get("names"))
    root = _dataset_root(resolved_data_path, yaml_data)
    split_sources: dict[str, tuple[Path, ...]] = {}
    for split in SPLIT_ALIASES:
        raw_value = _split_value(yaml_data, split)
        if raw_value in (None, ""):
            aliases = " or ".join(f"'{alias}'" for alias in SPLIT_ALIASES[split])
            raise ValueError(f"data.yaml must define the {split} split using {aliases}.")
        split_sources[split] = tuple(_resolve_sources(raw_value, root))
    return DatasetDefinition(resolved_data_path, root, class_names, split_sources)


def collect_split_images(definition: DatasetDefinition, split: str) -> list[Path]:
    if split not in definition.split_sources:
        raise ValueError(f"Unknown dataset split '{split}'. Expected one of: train, val, test.")
    images: list[Path] = []
    for source in definition.split_sources[split]:
        if not source.exists():
            raise FileNotFoundError(f"Dataset split path does not exist for '{split}': {source}")
        images.extend(_collect_images(source, definition.root))
    return sorted(set(images))


def label_path_for_image(image_path: Path, dataset_root: Path) -> Path:
    """Map an image path to the conventional mirrored YOLO label path."""
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")

    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        relative = Path(image_path.name)
    return (dataset_root / "labels" / relative).with_suffix(".txt")


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def bbox_validation_error(values: list[float]) -> str | None:
    x_center, y_center, width, height = values
    if not all(math.isfinite(value) for value in values):
        return "bounding-box values must be finite numbers"
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        return "x_center and y_center must be within [0, 1]"
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        return "width and height must be within (0, 1]"
    tolerance = 1e-6
    if (
        x_center - width / 2 < -tolerance
        or x_center + width / 2 > 1.0 + tolerance
        or y_center - height / 2 < -tolerance
        or y_center + height / 2 > 1.0 + tolerance
    ):
        return "bounding box extends outside normalized image boundaries"
    return None


def _validate_label(
    label_path: Path,
    image_path: Path,
    split: str,
    class_names: dict[int, str],
) -> tuple[list[DatasetIssue], Counter[int], int, int]:
    issues: list[DatasetIssue] = []
    class_counts: Counter[int] = Counter()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return (
            [
                DatasetIssue(
                    "error",
                    "unreadable_label",
                    f"Label file cannot be read as UTF-8 text: {exc}",
                    split,
                    str(image_path),
                    str(label_path),
                )
            ],
            class_counts,
            0,
            0,
        )

    nonempty_lines = [
        (number, line.strip()) for number, line in enumerate(lines, 1) if line.strip()
    ]
    if not nonempty_lines:
        issues.append(
            DatasetIssue(
                "warning",
                "empty_label",
                "Label file contains no annotations; this may be a valid negative image.",
                split,
                str(image_path),
                str(label_path),
            )
        )
        return issues, class_counts, 0, 0

    valid_count = 0
    for line_number, line in nonempty_lines:
        fields = line.split()
        if len(fields) != 5:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_label_format",
                    f"Expected 5 values, found {len(fields)}.",
                    split,
                    str(image_path),
                    str(label_path),
                    line_number,
                )
            )
            continue
        try:
            class_id = int(fields[0])
        except ValueError:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_class_id",
                    f"Class ID must be an integer, found '{fields[0]}'.",
                    split,
                    str(image_path),
                    str(label_path),
                    line_number,
                )
            )
            continue
        if class_id not in class_names:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_class_id",
                    f"Class ID {class_id} is not declared in data.yaml.",
                    split,
                    str(image_path),
                    str(label_path),
                    line_number,
                )
            )
            continue
        try:
            bbox = [float(value) for value in fields[1:]]
        except ValueError:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_bbox",
                    "Bounding-box values must be numeric.",
                    split,
                    str(image_path),
                    str(label_path),
                    line_number,
                )
            )
            continue
        bbox_error = bbox_validation_error(bbox)
        if bbox_error:
            issues.append(
                DatasetIssue(
                    "error",
                    "invalid_bbox",
                    bbox_error,
                    split,
                    str(image_path),
                    str(label_path),
                    line_number,
                )
            )
            continue
        class_counts[class_id] += 1
        valid_count += 1

    return issues, class_counts, len(nonempty_lines), valid_count


def _empty_split_stats(class_names: dict[int, str]) -> dict[str, Any]:
    return {
        "sources": [],
        "images": 0,
        "label_files": 0,
        "annotation_lines": 0,
        "valid_annotations": 0,
        "images_by_class": {name: 0 for name in class_names.values()},
        "annotations_by_class": {name: 0 for name in class_names.values()},
    }


def _duplicate_groups(hash_records: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    return [
        {"sha256": digest, "occurrences": records}
        for digest, records in sorted(hash_records.items())
        if len(records) > 1
    ]


def validate_dataset(config: DatasetValidationConfig) -> dict[str, Any]:
    config = config.resolved()
    yaml_data = _read_yaml(config.data_path)
    class_names = _parse_class_names(yaml_data.get("names"))
    root = _dataset_root(config.data_path, yaml_data)
    issues: list[DatasetIssue] = []
    split_stats = {split: _empty_split_stats(class_names) for split in SPLIT_ALIASES}
    hash_records: dict[str, list[dict[str, str]]] = defaultdict(list)

    for split in SPLIT_ALIASES:
        raw_value = _split_value(yaml_data, split)
        if raw_value in (None, ""):
            issues.append(
                DatasetIssue("error", "missing_split", f"No '{split}' split is defined.", split)
            )
            continue

        sources = _resolve_sources(raw_value, root)
        split_stats[split]["sources"] = [str(source) for source in sources]
        images: list[Path] = []
        for source in sources:
            if not source.exists():
                issues.append(
                    DatasetIssue(
                        "error", "missing_split_path", f"Split path does not exist: {source}", split
                    )
                )
                continue
            collected = _collect_images(source, root)
            if not collected:
                issues.append(
                    DatasetIssue(
                        "warning", "empty_split", f"No supported images found in: {source}", split
                    )
                )
            images.extend(collected)

        images = sorted(set(images))
        split_stats[split]["images"] = len(images)
        class_counter: Counter[int] = Counter()
        class_image_counter: Counter[int] = Counter()
        for image_path in images:
            image_text = str(image_path)
            if not image_path.is_file():
                issues.append(
                    DatasetIssue(
                        "error",
                        "missing_image",
                        "Image listed in split does not exist.",
                        split,
                        image_text,
                    )
                )
                continue
            if not _image_is_readable(image_path):
                issues.append(
                    DatasetIssue(
                        "error", "unreadable_image", "Image cannot be decoded.", split, image_text
                    )
                )
            try:
                digest = _file_hash(image_path, config.hash_algorithm)
                hash_records[digest].append({"split": split, "path": image_text})
            except OSError as exc:
                issues.append(
                    DatasetIssue(
                        "error",
                        "unreadable_image",
                        f"Image cannot be read: {exc}",
                        split,
                        image_text,
                    )
                )

            label_path = label_path_for_image(image_path, root)
            if not label_path.is_file():
                issues.append(
                    DatasetIssue(
                        "error",
                        "missing_label",
                        "No matching YOLO label file was found.",
                        split,
                        image_text,
                        str(label_path),
                    )
                )
                continue
            split_stats[split]["label_files"] += 1
            label_issues, label_counts, total_lines, valid_lines = _validate_label(
                label_path, image_path, split, class_names
            )
            issues.extend(label_issues)
            class_counter.update(label_counts)
            class_image_counter.update(label_counts.keys())
            split_stats[split]["annotation_lines"] += total_lines
            split_stats[split]["valid_annotations"] += valid_lines

        split_stats[split]["images_by_class"] = {
            class_names[class_id]: class_image_counter[class_id] for class_id in class_names
        }
        split_stats[split]["annotations_by_class"] = {
            class_names[class_id]: class_counter[class_id] for class_id in class_names
        }

    duplicates = _duplicate_groups(hash_records)
    for group in duplicates:
        occurrences = group["occurrences"]
        issues.append(
            DatasetIssue(
                "warning",
                "duplicate_image",
                f"Identical image content appears {len(occurrences)} times (SHA-256: {group['sha256']}).",
            )
        )

    issue_counts = Counter(issue.severity for issue in issues)
    totals_by_class = {
        name: sum(stats["annotations_by_class"][name] for stats in split_stats.values())
        for name in class_names.values()
    }
    images_by_class = {
        name: sum(stats["images_by_class"][name] for stats in split_stats.values())
        for name in class_names.values()
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "valid": issue_counts["error"] == 0,
        "data_yaml": str(config.data_path),
        "dataset_root": str(root),
        "class_names": {str(class_id): name for class_id, name in class_names.items()},
        "summary": {
            "images": sum(stats["images"] for stats in split_stats.values()),
            "label_files": sum(stats["label_files"] for stats in split_stats.values()),
            "annotation_lines": sum(stats["annotation_lines"] for stats in split_stats.values()),
            "valid_annotations": sum(stats["valid_annotations"] for stats in split_stats.values()),
            "images_by_class": images_by_class,
            "annotations_by_class": totals_by_class,
            "errors": issue_counts["error"],
            "warnings": issue_counts["warning"],
            "duplicate_groups": len(duplicates),
        },
        "splits": split_stats,
        "duplicates": duplicates,
        "issues": [issue.to_dict() for issue in issues],
    }
