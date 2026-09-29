#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm import read_jsonl, write_jsonl  # noqa: E402

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "gloves",
    4: "boots",
    5: "no_helmet",
    6: "no_vest",
}
SPLIT_DIRS = {"train": "train", "dev": "valid", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, unreviewed VLM annotation candidate manifest."
    )
    parser.add_argument("--per-split", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_DIRS),
        default=list(SPLIT_DIRS),
        help="Build only the requested review queues.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "safety_final",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "data" / "candidates.jsonl",
    )
    parser.add_argument(
        "--history-data-dir",
        type=Path,
        action="append",
        help="Gold/review directory to exclude. Repeat to combine histories; defaults to vlm/data.",
    )
    return parser.parse_args()


def _source_group(path: Path) -> str:
    return path.stem.split(".rf.", maxsplit=1)[0]


def _image_for_label(label: Path) -> Path | None:
    image_dir = label.parent.parent / "images"
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = image_dir / f"{label.stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _class_ids(label: Path) -> list[int]:
    ids: list[int] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            ids.append(int(parts[0]))
        except ValueError:
            continue
    return ids


def _category(class_ids: list[int]) -> str:
    has_helmet = 5 in class_ids
    has_vest = 6 in class_ids
    if has_helmet and has_vest:
        return "both_violations"
    if has_helmet:
        return "no_helmet"
    if has_vest:
        return "no_vest"
    return "no_labeled_violation"


def _group_from_image(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).stem.split(".rf.", maxsplit=1)[0]


def _already_reviewed(data_dirs: list[Path]) -> tuple[set[str], set[str]]:
    used_groups: set[str] = set()
    used_images: set[str] = set()
    for data_dir in data_dirs:
        for name in ("train", "dev", "test"):
            records, _ = read_jsonl(data_dir / f"{name}.jsonl")
            for record in records:
                group = record.get("source_group") or _group_from_image(record.get("image"))
                if isinstance(group, str) and group:
                    used_groups.add(group)
                image = record.get("image")
                if isinstance(image, str) and image:
                    used_images.add(image)
        review_records, _ = read_jsonl(data_dir / "review_log.jsonl")
        for record in review_records:
            group = record.get("source_group") or _group_from_image(record.get("image"))
            if isinstance(group, str) and group:
                used_groups.add(group)
            image = record.get("image")
            if isinstance(image, str) and image:
                used_images.add(image)
    return used_groups, used_images


def main() -> int:
    args = parse_args()
    if args.per_split < 1:
        raise SystemExit("--per-split must be positive")
    rng = random.Random(args.seed)
    history_dirs = args.history_data_dir or [PROJECT_ROOT / "vlm" / "data"]
    used_groups, used_images = _already_reviewed([path.expanduser().resolve() for path in history_dirs])
    output: list[dict[str, object]] = []

    for target_split in args.splits:
        directory_name = SPLIT_DIRS[target_split]
        label_dir = args.dataset_root / directory_name / "labels"
        buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
        seen_groups: set[str] = set()
        for label in sorted(label_dir.glob("*.txt")):
            group = _source_group(label)
            if group in seen_groups or group in used_groups:
                continue
            image = _image_for_label(label)
            if image is None:
                continue
            relative_image = str(image.relative_to(PROJECT_ROOT))
            if relative_image in used_images:
                continue
            class_ids = _class_ids(label)
            record = {
                "id": image.stem,
                "image": relative_image,
                "label": str(label.relative_to(PROJECT_ROOT)),
                "source_group": group,
                "source_split": target_split,
                "category": _category(class_ids),
                "source_classes": [CLASS_NAMES.get(class_id, str(class_id)) for class_id in class_ids],
                "annotation_count": len(class_ids),
                "review_status": "pending",
            }
            buckets[str(record["category"])].append(record)
            seen_groups.add(group)
        for records in buckets.values():
            rng.shuffle(records)
            records.sort(key=lambda record: int(record["annotation_count"]))

        categories = ["both_violations", "no_helmet", "no_vest", "no_labeled_violation"]
        selected: list[dict[str, object]] = []
        while len(selected) < args.per_split and any(buckets[category] for category in categories):
            for category in categories:
                if buckets[category] and len(selected) < args.per_split:
                    selected.append(buckets[category].pop(0))
        output.extend(selected)
        used_groups.update(str(record["source_group"]) for record in selected)

    write_jsonl(args.output, output)
    counts: dict[str, int] = defaultdict(int)
    for record in output:
        counts[str(record["source_split"])] += 1
    print(f"Wrote {len(output)} unreviewed candidates to {args.output}")
    print(json.dumps(dict(sorted(counts.items())), indent=2))
    print("Candidates are not gold labels; manually review before copying into train/dev/test JSONL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
