from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_EXPERIMENTS = (
    "baseline_yolo11n_512",
    "exp2_yolo11n_640",
    "exp3_yolo11s_512",
    "exp4_yolo11s_512_e50",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing(path: Path, purpose: str) -> dict[str, Any]:
    return {"path": str(path), "purpose": purpose, "exists": False}


def verify_dataset(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.is_file():
        return {
            "status": "missing",
            "missing": [_missing(data_yaml, "frozen dataset YAML")],
        }
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(document.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names_value = document.get("names", {})
    names = list(names_value.values()) if isinstance(names_value, dict) else list(names_value)
    split_keys = {"train": "train", "valid": "val", "test": "test"}
    splits: dict[str, Any] = {}
    image_hashes: dict[str, list[str]] = {}
    source_groups: dict[str, set[str]] = {}
    dataset_digest = hashlib.sha256()
    missing: list[dict[str, Any]] = []
    total_annotations = 0
    for split, yaml_key in split_keys.items():
        relative = document.get(yaml_key)
        if not isinstance(relative, str):
            missing.append(_missing(data_yaml, f"{yaml_key} split entry"))
            continue
        images_dir = Path(relative)
        if not images_dir.is_absolute():
            images_dir = (root / images_dir).resolve()
        if not images_dir.is_dir():
            missing.append(_missing(images_dir, f"{split} image directory"))
            continue
        images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        annotation_count = 0
        for image in images:
            try:
                relative_image = image.relative_to(root)
            except ValueError:
                relative_image = image
            parts = list(relative_image.parts)
            if "images" not in parts:
                raise ValueError(f"image path has no images component: {image}")
            parts[parts.index("images")] = "labels"
            label = root.joinpath(*parts).with_suffix(".txt")
            if not label.is_file():
                missing.append(_missing(label, "paired YOLO label"))
                continue
            image_hash, label_hash = sha256_file(image), sha256_file(label)
            image_hashes.setdefault(image_hash, []).append(split)
            source_key = image.name.split(".rf.", 1)[0] if ".rf." in image.name else image.stem
            source_groups.setdefault(source_key, set()).add(split)
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            for line in lines:
                fields = line.split()
                if len(fields) != 5 or not fields[0].isdigit() or int(fields[0]) >= len(names):
                    raise ValueError(f"invalid YOLO annotation: {label}: {line}")
            annotation_count += len(lines)
            dataset_digest.update(
                f"{relative_image}|{image_hash}|{label.relative_to(root)}|{label_hash}\n".encode()
            )
        total_annotations += annotation_count
        splits[split] = {"images": len(images), "annotations": annotation_count}
    duplicate_groups = sum(len(paths) > 1 for paths in image_hashes.values())
    cross_split_hashes = sum(len(set(paths)) > 1 for paths in image_hashes.values())
    cross_split_sources = {
        group: sorted(values) for group, values in source_groups.items() if len(values) > 1
    }
    passed = not missing and not duplicate_groups and not cross_split_hashes and not cross_split_sources
    return {
        "status": "verified" if passed else "failed",
        "source": str(data_yaml.resolve()),
        "source_sha256": sha256_file(data_yaml),
        "dataset_sha256": dataset_digest.hexdigest(),
        "classes": names,
        "class_count": len(names),
        "images": sum(row["images"] for row in splits.values()),
        "annotations": total_annotations,
        "splits": splits,
        "duplicate_sha256_groups": duplicate_groups,
        "cross_split_sha256_groups": cross_split_hashes,
        "cross_split_source_groups": len(cross_split_sources),
        "source_group_leakage": cross_split_sources,
        "missing": missing,
    }


def _best_validation_metric(path: Path) -> float:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty training results: {path}")
    key = next((key for key in rows[0] if "mAP50-95" in key), None)
    if key is None:
        raise ValueError(f"mAP50-95 column missing: {path}")
    return max(float(row[key]) for row in rows)


def verify_yolo_outputs(experiments_dir: Path) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    experiments: dict[str, Any] = {}
    for name in EXPECTED_EXPERIMENTS:
        results = experiments_dir / name / "results.csv"
        if not results.is_file():
            missing.append(_missing(results, f"{name} training metrics"))
            continue
        experiments[name] = {
            "results_csv": str(results),
            "results_sha256": sha256_file(results),
            "best_validation_map50_95": _best_validation_metric(results),
        }
    baseline_metrics = experiments_dir / EXPECTED_EXPERIMENTS[0] / "test_evaluation" / "test_metrics.json"
    final_metrics = experiments_dir / EXPECTED_EXPERIMENTS[-1] / "test_evaluation" / "test_metrics.json"
    performance = experiments_dir / EXPECTED_EXPERIMENTS[-1] / "performance" / "performance.json"
    for path, purpose in (
        (baseline_metrics, "baseline independent test metrics"),
        (final_metrics, "selected-model independent test metrics"),
        (performance, "selected-model latency samples/summary"),
    ):
        if not path.is_file():
            missing.append(_missing(path, purpose))
    derived: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    if baseline_metrics.is_file() and final_metrics.is_file():
        baseline = json.loads(baseline_metrics.read_text(encoding="utf-8"))
        final = json.loads(final_metrics.read_text(encoding="utf-8"))
        baseline_map = float(baseline["metrics"]["map50_95"])
        final_map = float(final["metrics"]["map50_95"])
        derived.update(
            {
                "baseline_test_map50_95": baseline_map,
                "selected_test_map50_95": final_map,
                "selected_test_map50": float(final["metrics"]["map50"]),
                "absolute_map50_95_improvement": final_map - baseline_map,
            }
        )
        sources["baseline_test"] = {"path": str(baseline_metrics), "sha256": sha256_file(baseline_metrics)}
        sources["selected_test"] = {"path": str(final_metrics), "sha256": sha256_file(final_metrics)}
    if performance.is_file():
        perf = json.loads(performance.read_text(encoding="utf-8"))
        derived["latency"] = {
            key: perf[key]
            for key in ("device", "samples", "warmup_images", "average_latency_ms", "p50_latency_ms", "p95_latency_ms")
        }
        sources["performance"] = {"path": str(performance), "sha256": sha256_file(performance)}
    return {
        "status": "verified" if not missing else "missing",
        "expected_experiments": list(EXPECTED_EXPERIMENTS),
        "experiments": experiments,
        "derived": derived,
        "sources": sources,
        "missing": missing,
    }


def build_verification(data_yaml: Path, experiments_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "scripts/verify_resume_evidence.py",
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "dataset": verify_dataset(data_yaml),
        "yolo": verify_yolo_outputs(experiments_dir),
    }


def write_verification(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
