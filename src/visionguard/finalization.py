from __future__ import annotations

import hashlib
import json
import random
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from visionguard.config import (
    FINAL_REMOVED_CLASSES,
    FINAL_TARGET_CLASSES,
    SPLIT_ALIASES,
    DatasetFreezeConfig,
)
from visionguard.dataset import (
    DatasetDefinition,
    bbox_validation_error,
    collect_split_images,
    label_path_for_image,
    load_dataset_definition,
)
from visionguard.reporting import write_finalization_reports

OUTPUT_SPLITS = ("train", "valid", "test")
CANONICAL_SPLITS = {"train": "train", "val": "valid", "test": "test"}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    original_split: str
    source_key: str
    image_hash: str
    output_lines: tuple[str, ...]
    class_counts: dict[int, int]
    removed_counts: dict[str, int]

    @property
    def object_count(self) -> int:
        return sum(self.class_counts.values())


@dataclass(frozen=True)
class SourceGroup:
    key: str
    records: tuple[ImageRecord, ...]
    class_counts: dict[int, int]
    class_image_counts: dict[int, int]

    @property
    def size(self) -> int:
        return len(self.records)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(image_path: Path) -> str:
    name = image_path.name
    return name.split(".rf.", 1)[0] if ".rf." in name else image_path.stem


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _parse_record(
    image_path: Path,
    label_path: Path,
    original_split: str,
    class_names: dict[int, str],
    target_ids: dict[str, int],
) -> ImageRecord:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read label as UTF-8: {label_path}: {exc}") from exc

    output_lines: list[str] = []
    class_counts: Counter[int] = Counter()
    removed_counts: Counter[str] = Counter()
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"Invalid YOLO label at {label_path}:{line_number}; expected 5 values."
            )
        try:
            old_id = int(fields[0])
            bbox = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value at {label_path}:{line_number}.") from exc
        if old_id not in class_names:
            raise ValueError(f"Undeclared class ID {old_id} at {label_path}:{line_number}.")
        bbox_error = bbox_validation_error(bbox)
        if bbox_error:
            raise ValueError(f"Invalid box at {label_path}:{line_number}: {bbox_error}.")
        old_name = class_names[old_id]
        if old_name in FINAL_REMOVED_CLASSES:
            removed_counts[old_name] += 1
            continue
        if old_name not in target_ids:
            raise ValueError(f"No final class rule exists for '{old_name}' in {label_path}.")
        new_id = target_ids[old_name]
        class_counts[new_id] += 1
        output_lines.append(" ".join((str(new_id), *fields[1:])))

    return ImageRecord(
        image_path=image_path,
        label_path=label_path,
        original_split=original_split,
        source_key=_source_key(image_path),
        image_hash=_hash_file(image_path),
        output_lines=tuple(output_lines),
        class_counts=dict(class_counts),
        removed_counts=dict(removed_counts),
    )


def _load_records(definition: DatasetDefinition) -> list[ImageRecord]:
    source_names = set(definition.class_names.values())
    expected_names = set(FINAL_TARGET_CLASSES) | set(FINAL_REMOVED_CLASSES)
    if source_names != expected_names:
        missing = sorted(expected_names - source_names)
        unexpected = sorted(source_names - expected_names)
        raise ValueError(
            f"Clean taxonomy mismatch. Missing={missing or 'none'}, unexpected={unexpected or 'none'}."
        )
    target_ids = {name: class_id for class_id, name in enumerate(FINAL_TARGET_CLASSES)}
    records: list[ImageRecord] = []
    for split in SPLIT_ALIASES:
        for image_path in collect_split_images(definition, split):
            label_path = label_path_for_image(image_path, definition.root)
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for source image: {image_path}")
            records.append(
                _parse_record(
                    image_path,
                    label_path,
                    split,
                    definition.class_names,
                    target_ids,
                )
            )
    return sorted(records, key=lambda record: str(record.image_path))


def _deduplicate(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.image_hash].append(record)

    kept_records: list[ImageRecord] = []
    duplicate_groups: list[dict[str, Any]] = []
    removed_files: list[dict[str, Any]] = []
    for image_hash, group in sorted(by_hash.items()):
        ranked = sorted(group, key=lambda record: (-record.object_count, str(record.image_path)))
        kept = ranked[0]
        kept_records.append(kept)
        if len(group) == 1:
            continue
        removed = ranked[1:]
        label_signatures = {record.output_lines for record in group}
        duplicate_groups.append(
            {
                "sha256": image_hash,
                "kept_image": str(kept.image_path),
                "kept_label": str(kept.label_path),
                "removed_images": [str(record.image_path) for record in removed],
                "removed_labels": [str(record.label_path) for record in removed],
                "label_annotations_consistent": len(label_signatures) == 1,
                "selection_rule": "most retained objects, then lexicographically smallest path",
            }
        )
        for record in removed:
            removed_files.append(
                {
                    "reason": "duplicate_sha256",
                    "image": str(record.image_path),
                    "label": str(record.label_path),
                    "sha256": image_hash,
                    "kept_image": str(kept.image_path),
                }
            )
    return (
        sorted(kept_records, key=lambda record: str(record.image_path)),
        duplicate_groups,
        removed_files,
    )


def _remove_empty_records(
    records: list[ImageRecord], removed_files: list[dict[str, Any]]
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    retained: list[ImageRecord] = []
    empty: list[ImageRecord] = []
    for record in records:
        if record.output_lines:
            retained.append(record)
            continue
        empty.append(record)
        removed_files.append(
            {
                "reason": "empty_after_class_filter",
                "image": str(record.image_path),
                "label": str(record.label_path),
                "sha256": record.image_hash,
            }
        )
    return retained, empty


def _make_source_groups(records: list[ImageRecord]) -> list[SourceGroup]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_key].append(record)
    result: list[SourceGroup] = []
    for key, source_records in sorted(grouped.items()):
        class_counts: Counter[int] = Counter()
        class_image_counts: Counter[int] = Counter()
        for record in source_records:
            class_counts.update(record.class_counts)
            class_image_counts.update(record.class_counts.keys())
        result.append(
            SourceGroup(
                key,
                tuple(sorted(source_records, key=lambda record: str(record.image_path))),
                dict(class_counts),
                dict(class_image_counts),
            )
        )
    return result


def _ratio_targets(total: int, ratios: dict[str, float]) -> dict[str, float]:
    return {split: total * ratio for split, ratio in ratios.items()}


def _empty_assignment_stats() -> dict[str, Any]:
    return {
        "images": 0,
        "class_counts": Counter(),
        "class_image_counts": Counter(),
    }


def _assignment_objective(
    stats: dict[str, dict[str, Any]],
    image_targets: dict[str, float],
    class_targets: dict[str, dict[int, float]],
    class_image_targets: dict[str, dict[int, float]],
) -> float:
    score = 0.0
    for split in OUTPUT_SPLITS:
        image_target = max(image_targets[split], 1.0)
        score += 8.0 * ((stats[split]["images"] - image_target) / image_target) ** 2
        for class_id in range(len(FINAL_TARGET_CLASSES)):
            annotation_target = max(class_targets[split][class_id], 1.0)
            coverage_target = max(class_image_targets[split][class_id], 1.0)
            score += (
                (stats[split]["class_counts"][class_id] - annotation_target) / annotation_target
            ) ** 2
            score += (
                4.0
                * (
                    (stats[split]["class_image_counts"][class_id] - coverage_target)
                    / coverage_target
                )
                ** 2
            )
    return score


def _apply_group(stats: dict[str, Any], group: SourceGroup, direction: int) -> None:
    stats["images"] += direction * group.size
    for class_id, count in group.class_counts.items():
        stats["class_counts"][class_id] += direction * count
    for class_id, count in group.class_image_counts.items():
        stats["class_image_counts"][class_id] += direction * count


def _stratified_group_assignment(
    groups: list[SourceGroup], ratios: dict[str, float], seed: int
) -> tuple[dict[str, list[SourceGroup]], dict[str, float]]:
    total_images = sum(group.size for group in groups)
    total_class_counts: Counter[int] = Counter()
    total_class_images: Counter[int] = Counter()
    for group in groups:
        total_class_counts.update(group.class_counts)
        total_class_images.update(group.class_image_counts)

    image_targets = _ratio_targets(total_images, ratios)
    class_targets = {
        split: {class_id: total_class_counts[class_id] * ratios[split] for class_id in range(7)}
        for split in OUTPUT_SPLITS
    }
    class_image_targets = {
        split: {class_id: total_class_images[class_id] * ratios[split] for class_id in range(7)}
        for split in OUTPUT_SPLITS
    }
    stats = {split: _empty_assignment_stats() for split in OUTPUT_SPLITS}
    assignments = {split: [] for split in OUTPUT_SPLITS}
    group_to_split: dict[str, str] = {}
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)

    def rarity(group: SourceGroup) -> int:
        present = group.class_image_counts
        return min((total_class_images[class_id] for class_id in present), default=total_images)

    ordered.sort(key=lambda group: (rarity(group), -group.size))
    split_tiebreak = list(OUTPUT_SPLITS)
    rng.shuffle(split_tiebreak)
    for group in ordered:
        choices: list[tuple[float, int, str]] = []
        for tie_index, split in enumerate(split_tiebreak):
            before = _assignment_objective(stats, image_targets, class_targets, class_image_targets)
            _apply_group(stats[split], group, 1)
            after = _assignment_objective(stats, image_targets, class_targets, class_image_targets)
            _apply_group(stats[split], group, -1)
            choices.append((after - before, tie_index, split))
        selected_split = min(choices)[2]
        assignments[selected_split].append(group)
        group_to_split[group.key] = selected_split
        _apply_group(stats[selected_split], group, 1)

    for _ in range(5):
        improved = False
        current_score = _assignment_objective(
            stats, image_targets, class_targets, class_image_targets
        )
        candidates = list(groups)
        rng.shuffle(candidates)
        for group in candidates:
            old_split = group_to_split[group.key]
            best_split = old_split
            best_score = current_score
            _apply_group(stats[old_split], group, -1)
            for new_split in OUTPUT_SPLITS:
                if new_split == old_split:
                    continue
                _apply_group(stats[new_split], group, 1)
                score = _assignment_objective(
                    stats, image_targets, class_targets, class_image_targets
                )
                _apply_group(stats[new_split], group, -1)
                if score + 1e-12 < best_score:
                    best_score = score
                    best_split = new_split
            if best_split == old_split:
                _apply_group(stats[old_split], group, 1)
                continue
            assignments[old_split].remove(group)
            assignments[best_split].append(group)
            group_to_split[group.key] = best_split
            _apply_group(stats[best_split], group, 1)
            current_score = best_score
            improved = True
        if not improved:
            break

    for class_name in ("no_helmet", "no_vest"):
        class_id = FINAL_TARGET_CLASSES.index(class_name)
        missing_splits = [
            split for split in OUTPUT_SPLITS if stats[split]["class_image_counts"][class_id] == 0
        ]
        if missing_splits:
            raise RuntimeError(
                f"Grouped stratification could not place '{class_name}' in: {', '.join(missing_splits)}."
            )

    final_score = _assignment_objective(stats, image_targets, class_targets, class_image_targets)
    return assignments, {"objective": final_score, **image_targets}


def _write_data_yaml(staging_dir: Path, source_data_path: Path, seed: int) -> None:
    data = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(FINAL_TARGET_CLASSES),
        "names": {class_id: name for class_id, name in enumerate(FINAL_TARGET_CLASSES)},
        "frozen": {
            "seed": seed,
            "source_data_yaml": str(source_data_path),
            "split_strategy": "SHA-256 deduplication + grouped approximate multilabel stratification",
        },
    }
    (staging_dir / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _copy_metadata(source_root: Path, staging_dir: Path) -> list[str]:
    copied: list[str] = []
    for source_file in sorted(source_root.glob("README*")):
        if source_file.is_file():
            shutil.copy2(source_file, staging_dir / source_file.name)
            copied.append(source_file.name)
    return copied


def _write_final_files(
    assignments: dict[str, list[SourceGroup]], staging_dir: Path
) -> dict[str, dict[str, str]]:
    provenance: dict[str, dict[str, str]] = {}
    seen_names: set[tuple[str, str]] = set()
    for split in OUTPUT_SPLITS:
        for group in assignments[split]:
            for record in group.records:
                key = (split, record.image_path.name)
                if key in seen_names:
                    raise ValueError(
                        f"Output filename collision in {split}: {record.image_path.name}"
                    )
                seen_names.add(key)
                destination_image = staging_dir / split / "images" / record.image_path.name
                destination_label = staging_dir / split / "labels" / f"{record.image_path.stem}.txt"
                destination_image.parent.mkdir(parents=True, exist_ok=True)
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record.image_path, destination_image)
                destination_label.write_text(
                    "\n".join(record.output_lines) + "\n", encoding="utf-8"
                )
                provenance[str(destination_image)] = {
                    "source_image": str(record.image_path),
                    "source_key": record.source_key,
                }
    return provenance


def _audit_and_freeze(
    staging_dir: Path,
    provenance: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_splits: dict[str, set[str]] = defaultdict(set)
    split_stats: dict[str, dict[str, Any]] = {}
    manifest_files: list[dict[str, str]] = []
    dataset_digest = hashlib.sha256()
    total_images = 0

    for split in OUTPUT_SPLITS:
        images = sorted((staging_dir / split / "images").glob("*"))
        labels = sorted((staging_dir / split / "labels").glob("*.txt"))
        label_names = {label.stem for label in labels}
        image_stems = {image.stem for image in images}
        if image_stems != label_names:
            raise RuntimeError(f"Image/label mismatch in final {split} split.")
        class_counts: Counter[str] = Counter()
        class_images: Counter[str] = Counter()
        for image_path in images:
            label_path = staging_dir / split / "labels" / f"{image_path.stem}.txt"
            image_hash = _hash_file(image_path)
            label_hash = _hash_file(label_path)
            info = provenance[str(image_path)]
            image_hashes[image_hash].append({"split": split, "path": str(image_path)})
            source_splits[info["source_key"]].add(split)
            present_classes: set[str] = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                fields = line.split()
                if len(fields) != 5:
                    raise RuntimeError(f"Invalid final label: {label_path}:{line_number}")
                class_id = int(fields[0])
                if not 0 <= class_id < len(FINAL_TARGET_CLASSES):
                    raise RuntimeError(f"Final class ID outside 0..6: {label_path}:{line_number}")
                class_name = FINAL_TARGET_CLASSES[class_id]
                class_counts[class_name] += 1
                present_classes.add(class_name)
            if not present_classes:
                raise RuntimeError(
                    f"Empty label unexpectedly written to final dataset: {label_path}"
                )
            class_images.update(present_classes)
            relative_image = image_path.relative_to(staging_dir)
            relative_label = label_path.relative_to(staging_dir)
            manifest_files.append(
                {
                    "split": split,
                    "image": str(relative_image),
                    "label": str(relative_label),
                    "image_sha256": image_hash,
                    "label_sha256": label_hash,
                    "source_image": info["source_image"],
                    "source_key": info["source_key"],
                }
            )
            dataset_digest.update(
                f"{relative_image}|{image_hash}|{relative_label}|{label_hash}\n".encode()
            )
        total_images += len(images)
        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "annotations": sum(class_counts.values()),
            "class_annotations": {name: class_counts[name] for name in FINAL_TARGET_CLASSES},
            "class_image_coverage": {name: class_images[name] for name in FINAL_TARGET_CLASSES},
        }

    duplicate_groups = [records for records in image_hashes.values() if len(records) > 1]
    cross_split_duplicates = [
        records for records in duplicate_groups if len({record["split"] for record in records}) > 1
    ]
    source_leakage = {
        source_key: sorted(splits)
        for source_key, splits in source_splits.items()
        if len(splits) > 1
    }
    audit = {
        "passed": not duplicate_groups and not cross_split_duplicates and not source_leakage,
        "total_images": total_images,
        "class_id_range": [0, 6],
        "image_label_pairs_valid": True,
        "duplicate_sha256_groups": len(duplicate_groups),
        "cross_split_sha256_groups": len(cross_split_duplicates),
        "cross_split_source_groups": len(source_leakage),
        "source_leakage": source_leakage,
        "splits": split_stats,
    }
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hash_algorithm": "sha256",
        "dataset_sha256": dataset_digest.hexdigest(),
        "files": manifest_files,
    }
    return audit, manifest


def _publish_staging(staging_dir: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        staging_dir.replace(output_dir)
        return
    backup_dir = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    output_dir.replace(backup_dir)
    try:
        staging_dir.replace(output_dir)
    except Exception:
        backup_dir.replace(output_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def finalize_dataset(config: DatasetFreezeConfig) -> dict[str, Any]:
    """Deduplicate, group-stratify, audit, and freeze a final YOLO dataset copy."""
    config = config.resolved()
    definition = load_dataset_definition(config.data_path)
    if _paths_overlap(definition.root, config.output_dir):
        raise ValueError("Source and output dataset directories must not overlap.")
    if config.output_dir.exists() and not config.overwrite:
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")
    staging_dir = config.output_dir.parent / f".{config.output_dir.name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True)
    try:
        records = _load_records(definition)
        input_removed_counts: Counter[str] = Counter()
        for record in records:
            input_removed_counts.update(record.removed_counts)
        deduplicated, duplicate_groups, removed_files = _deduplicate(records)
        retained, empty_records = _remove_empty_records(deduplicated, removed_files)
        groups = _make_source_groups(retained)
        ratios = {"train": config.train_ratio, "valid": config.val_ratio, "test": config.test_ratio}
        assignments, stratification = _stratified_group_assignment(groups, ratios, config.seed)
        provenance = _write_final_files(assignments, staging_dir)
        _write_data_yaml(staging_dir, config.data_path, config.seed)
        metadata_files = _copy_metadata(definition.root, staging_dir)
        audit, manifest = _audit_and_freeze(staging_dir, provenance)
        if not audit["passed"]:
            raise RuntimeError("Final dataset audit failed; staged output will be removed.")
        (staging_dir / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        total_annotations = sum(split["annotations"] for split in audit["splits"].values())
        for split in OUTPUT_SPLITS:
            stats = audit["splits"][split]
            stats["actual_ratio"] = stats["images"] / audit["total_images"]
            stats["target_ratio"] = ratios[split]
            stats["ratio_deviation"] = stats["actual_ratio"] - ratios[split]
            stats["class_annotation_proportions"] = {
                name: stats["class_annotations"][name] / stats["annotations"]
                if stats["annotations"]
                else 0.0
                for name in FINAL_TARGET_CLASSES
            }

        total_class_annotations = {
            name: sum(audit["splits"][split]["class_annotations"][name] for split in OUTPUT_SPLITS)
            for name in FINAL_TARGET_CLASSES
        }
        total_class_images = {
            name: sum(
                audit["splits"][split]["class_image_coverage"][name] for split in OUTPUT_SPLITS
            )
            for name in FINAL_TARGET_CLASSES
        }
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_data_yaml": str(config.data_path),
            "source_dataset_root": str(definition.root),
            "output_dataset_root": str(config.output_dir),
            "source_dataset_modified": False,
            "configuration": {"seed": config.seed, "split_ratios": ratios},
            "strategy": {
                "deduplication": "exact image bytes using SHA-256",
                "source_grouping": "filename prefix before Roboflow .rf. export suffix",
                "split": "group-level approximate multilabel stratification with local improvement",
                "strict_stratification": False,
                "limitation": (
                    "Exact ratios are not guaranteed because all variants from one source are assigned "
                    "as an indivisible group. No oversampling or synthetic balancing was used."
                ),
                "objective": stratification["objective"],
            },
            "counts": {
                "input_images": len(records),
                "duplicate_groups": len(duplicate_groups),
                "duplicate_groups_with_inconsistent_labels": sum(
                    not group["label_annotations_consistent"] for group in duplicate_groups
                ),
                "duplicate_images_removed": len(records) - len(deduplicated),
                "empty_label_images_removed": len(empty_records),
                "final_images": audit["total_images"],
                "input_removed_class_annotations": {
                    name: input_removed_counts[name] for name in sorted(FINAL_REMOVED_CLASSES)
                },
                "final_annotations": total_annotations,
                "source_groups": len(groups),
            },
            "target_classes": {
                str(class_id): name for class_id, name in enumerate(FINAL_TARGET_CLASSES)
            },
            "removed_classes": sorted(FINAL_REMOVED_CLASSES),
            "total_class_annotations": total_class_annotations,
            "total_class_image_coverage": total_class_images,
            "splits": audit["splits"],
            "duplicate_groups": duplicate_groups,
            "removed_files": removed_files,
            "audit": {key: value for key, value in audit.items() if key != "splits"},
            "freeze": {
                "manifest": str(config.output_dir / "freeze_manifest.json"),
                "dataset_sha256": manifest["dataset_sha256"],
            },
            "copied_metadata_files": metadata_files,
            "final_data_yaml": {
                "path": ".",
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": 7,
                "names": {
                    str(class_id): name for class_id, name in enumerate(FINAL_TARGET_CLASSES)
                },
            },
        }
        write_finalization_reports(report, staging_dir)
        _publish_staging(staging_dir, config.output_dir)
        return report
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
