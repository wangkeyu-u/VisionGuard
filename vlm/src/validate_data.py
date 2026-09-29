#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm import read_jsonl, validate_inspection  # noqa: E402

DATA_FILES = {
    "train": PROJECT_ROOT / "vlm" / "data" / "train.jsonl",
    "dev": PROJECT_ROOT / "vlm" / "data" / "dev.jsonl",
    "test": PROJECT_ROOT / "vlm" / "data" / "test.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VisionGuard VLM gold JSONL data.")
    parser.add_argument(
        "--require-all-splits",
        action="store_true",
        help="Fail when dev or test is empty; useful before reporting final metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    identifiers: dict[str, str] = {}
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    gold_images: set[str] = set()

    for split, path in DATA_FILES.items():
        records, issues = read_jsonl(path)
        errors.extend(str(issue) for issue in issues)
        counts[split] = len(records)
        if not records:
            message = f"{path.relative_to(PROJECT_ROOT)} is empty"
            if split == "train" or args.require_all_splits:
                errors.append(message)
            else:
                warnings.append(message)
        for index, record in enumerate(records, start=1):
            prefix = f"{path.relative_to(PROJECT_ROOT)}:{index}"
            required = {"id", "image", "source_group", "source_split", "target"}
            missing = required - set(record)
            if missing:
                errors.append(f"{prefix}: missing keys {sorted(missing)}")
                continue
            record_id = record["id"]
            if not isinstance(record_id, str) or not record_id:
                errors.append(f"{prefix}: id must be a non-empty string")
            elif record_id in identifiers:
                errors.append(f"{prefix}: duplicate id also used in {identifiers[record_id]}")
            else:
                identifiers[record_id] = prefix
            if record["source_split"] != split:
                errors.append(
                    f"{prefix}: source_split is {record['source_split']!r}, expected {split!r}"
                )
            source_group = record["source_group"]
            if not isinstance(source_group, str) or not source_group:
                errors.append(f"{prefix}: source_group must be a non-empty string")
            else:
                groups_by_split[split].add(source_group)

            image_value = record["image"]
            if not isinstance(image_value, str) or not image_value:
                errors.append(f"{prefix}: image must be a non-empty relative path")
            else:
                image_path = (PROJECT_ROOT / image_value).resolve()
                if PROJECT_ROOT not in image_path.parents:
                    errors.append(f"{prefix}: image path escapes the project root")
                elif not image_path.is_file():
                    errors.append(f"{prefix}: image does not exist: {image_value}")
                gold_images.add(image_value)

            target_errors = validate_inspection(record["target"])
            errors.extend(f"{prefix}: target: {message}" for message in target_errors)

    for first_index, first_split in enumerate(DATA_FILES):
        for second_split in list(DATA_FILES)[first_index + 1 :]:
            overlap = groups_by_split[first_split] & groups_by_split[second_split]
            if overlap:
                errors.append(
                    f"source-group leakage between {first_split} and {second_split}: "
                    f"{sorted(overlap)[:10]}"
                )

    review_path = PROJECT_ROOT / "vlm" / "data" / "review_log.jsonl"
    review_records, review_issues = read_jsonl(review_path)
    errors.extend(str(issue) for issue in review_issues)
    excluded_images = {
        record.get("image")
        for record in review_records
        if record.get("decision") == "exclude" and isinstance(record.get("image"), str)
    }
    leaked_exclusions = gold_images & excluded_images
    if leaked_exclusions:
        errors.append(f"excluded images appear in gold data: {sorted(leaked_exclusions)}")

    print("VisionGuard VLM data validation")
    print("=" * 36)
    for split in DATA_FILES:
        print(f"{split:>5}: {counts.get(split, 0)} records")
    print(f"review: {len(review_records)} decisions")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print("PASS" if not errors else f"FAIL ({len(errors)} errors)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
