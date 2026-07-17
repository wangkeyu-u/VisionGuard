#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.review_server import ANNOTATION_PROTOCOL_VERSION  # noqa: E402
from visionguard.vlm import read_jsonl, validate_inspection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block Test evaluation until independent, protocol-v2 gold data passes QA."
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "data_v2" / "test.jsonl",
    )
    parser.add_argument(
        "--history",
        type=Path,
        nargs="*",
        default=[
            PROJECT_ROOT / "vlm" / "data" / "train.jsonl",
            PROJECT_ROOT / "vlm" / "data" / "dev.jsonl",
            PROJECT_ROOT / "vlm" / "data" / "test.jsonl",
        ],
        help="Prior gold files whose source groups must not occur in Test v2.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "data" / "dev.jsonl",
        help="Reviewed reference split used for person-box geometry drift checks.",
    )
    parser.add_argument("--min-records", type=int, default=20)
    parser.add_argument("--min-median-area-ratio", type=float, default=0.4)
    parser.add_argument("--max-median-area-ratio", type=float, default=2.5)
    parser.add_argument(
        "--disallow-reviewer-substring",
        default="codex",
        help="Case-insensitive reviewer marker forbidden in Test; pass an empty string to disable.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in box)
        and all(0 <= value <= 1000 for value in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def _record_boxes(record: dict[str, Any]) -> list[list[int]]:
    review_boxes = record.get("review", {}).get("person_boxes", [])
    if isinstance(review_boxes, list) and review_boxes:
        return [box for box in review_boxes if _valid_box(box)]
    unique: list[list[int]] = []
    for finding in record.get("target", {}).get("findings", []):
        box = finding.get("person_box") if isinstance(finding, dict) else None
        if _valid_box(box) and box not in unique:
            unique.append(box)
    return unique


def _median_area(records: list[dict[str, Any]]) -> float | None:
    areas = [
        (box[2] - box[0]) * (box[3] - box[1]) / 1_000_000
        for record in records
        for box in _record_boxes(record)
    ]
    return statistics.median(areas) if areas else None


def _read(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records, issues = read_jsonl(path)
    errors.extend(str(issue) for issue in issues)
    return records


def audit_test_data(
    test_path: Path,
    history_paths: list[Path],
    reference_path: Path,
    min_records: int = 20,
    min_median_area_ratio: float = 0.4,
    max_median_area_ratio: float = 2.5,
    disallowed_reviewer_substring: str = "codex",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    test_path = test_path.expanduser().resolve()
    records = _read(test_path, errors)
    if len(records) < min_records:
        errors.append(f"Test has {len(records)} records; require at least {min_records}")

    history_groups: set[str] = set()
    for history_path in history_paths:
        for record in _read(history_path.expanduser().resolve(), errors):
            group = record.get("source_group")
            if isinstance(group, str) and group:
                history_groups.add(group)

    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    seen_groups: set[str] = set()
    all_boxes: list[list[int]] = []
    for index, record in enumerate(records, start=1):
        prefix = f"{test_path}:{index}"
        record_id = record.get("id")
        image = record.get("image")
        group = record.get("source_group")
        for value, seen, label in (
            (record_id, seen_ids, "id"),
            (image, seen_images, "image"),
            (group, seen_groups, "source_group"),
        ):
            if not isinstance(value, str) or not value:
                errors.append(f"{prefix}: {label} must be a non-empty string")
            elif value in seen:
                errors.append(f"{prefix}: duplicate {label} {value!r}")
            else:
                seen.add(value)
        if group in history_groups:
            errors.append(f"{prefix}: source_group {group!r} already exists in prior gold data")
        if record.get("source_split") != "test":
            errors.append(f"{prefix}: source_split must be 'test'")
        if isinstance(image, str):
            resolved_image = (PROJECT_ROOT / image).resolve()
            if PROJECT_ROOT not in resolved_image.parents or not resolved_image.is_file():
                errors.append(f"{prefix}: image is missing or outside the project: {image}")

        target_errors = validate_inspection(record.get("target"))
        errors.extend(f"{prefix}: target: {message}" for message in target_errors)
        review = record.get("review")
        if not isinstance(review, dict):
            errors.append(f"{prefix}: review metadata is required")
            continue
        if review.get("annotation_protocol_version") != ANNOTATION_PROTOCOL_VERSION:
            errors.append(f"{prefix}: annotation_protocol_version must be {ANNOTATION_PROTOCOL_VERSION!r}")
        if review.get("person_box_contract_confirmed") is not True:
            errors.append(f"{prefix}: full-person box contract was not confirmed")
        reviewer = review.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            errors.append(f"{prefix}: a human reviewer is required")
        elif disallowed_reviewer_substring and disallowed_reviewer_substring.casefold() in reviewer.casefold():
            errors.append(f"{prefix}: Test reviewer {reviewer!r} matches a disallowed marker")

        boxes = review.get("person_boxes")
        if not isinstance(boxes, list) or not boxes:
            errors.append(f"{prefix}: person_boxes must contain at least one full visible person")
            continue
        for box_index, box in enumerate(boxes):
            if not _valid_box(box):
                errors.append(f"{prefix}: person_boxes[{box_index}] is invalid")
            else:
                all_boxes.append(box)
        valid_boxes = [box for box in boxes if _valid_box(box)]
        if len({tuple(box) for box in valid_boxes}) != len(valid_boxes):
            errors.append(f"{prefix}: person_boxes contains duplicates")
        if valid_boxes != sorted(valid_boxes, key=lambda box: (box[0], box[1])):
            errors.append(f"{prefix}: person_boxes is not ordered from left to right")
        expected = {f"p{box_index}": box for box_index, box in enumerate(boxes, start=1)}
        for finding_index, finding in enumerate(record.get("target", {}).get("findings", [])):
            if not isinstance(finding, dict):
                continue
            person_id = finding.get("person_id")
            if person_id not in expected or finding.get("person_box") != expected.get(person_id):
                errors.append(f"{prefix}: finding {finding_index} is inconsistent with person_boxes")

    reference_errors: list[str] = []
    reference_records = _read(reference_path.expanduser().resolve(), reference_errors)
    if reference_errors:
        warnings.extend(f"reference: {message}" for message in reference_errors)
    test_median = _median_area(records)
    reference_median = _median_area(reference_records)
    median_ratio = None
    if test_median is None:
        errors.append("Test contains no valid person boxes")
    elif reference_median is None or reference_median <= 0:
        warnings.append("Reference contains no valid person boxes; geometry drift was not checked")
    else:
        median_ratio = test_median / reference_median
        if not min_median_area_ratio <= median_ratio <= max_median_area_ratio:
            errors.append(
                "person-box median area drift is outside the allowed range: "
                f"ratio={median_ratio:.3f}, allowed=[{min_median_area_ratio}, {max_median_area_ratio}]"
            )

    small_box_rate = (
        sum((box[2] - box[0]) * (box[3] - box[1]) < 10_000 for box in all_boxes) / len(all_boxes)
        if all_boxes
        else None
    )
    if small_box_rate is not None and small_box_rate > 0.5:
        warnings.append(f"{small_box_rate:.1%} of person boxes cover less than 1% of the image")

    return {
        "status": "PASS" if not errors else "BLOCKED",
        "test_path": str(test_path),
        "test_sha256": _sha256(test_path),
        "record_count": len(records),
        "person_box_count": len(all_boxes),
        "unique_source_group_count": len(seen_groups),
        "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "geometry": {
            "test_median_area_fraction": test_median,
            "reference_median_area_fraction": reference_median,
            "median_area_ratio": median_ratio,
            "small_box_rate_below_one_percent": small_box_rate,
        },
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    if args.min_records < 1:
        raise SystemExit("--min-records must be positive")
    if not 0 < args.min_median_area_ratio <= args.max_median_area_ratio:
        raise SystemExit("median area ratio bounds must satisfy 0 < min <= max")
    report = audit_test_data(
        args.test,
        args.history,
        args.reference,
        args.min_records,
        args.min_median_area_ratio,
        args.max_median_area_ratio,
        args.disallow_reviewer_substring,
    )
    output = args.output or args.test.with_name("TEST_DATA_QA.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"QA report saved to {output}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
