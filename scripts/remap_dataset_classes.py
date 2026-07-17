#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.config import DatasetRemapConfig  # noqa: E402
from visionguard.remapping import remap_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive, class-remapped copy of a YOLO dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to the source data.yaml.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "safety_clean",
        help="New dataset directory. It must not already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = remap_dataset(DatasetRemapConfig(args.data, args.output_dir))
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Dataset remapping failed: {exc}", file=sys.stderr)
        return 2

    print(f"Clean dataset: {report['output_dataset_root']}")
    print(f"Images copied: {sum(split['images'] for split in report['splits'].values())}")
    print(f"Removed glasses annotations: {report['removed_glasses_annotations']}")
    print(f"Empty-label images: {report['empty_label_images_count']}")
    print(f"Markdown report: {Path(report['output_dataset_root']) / 'class_remap_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
