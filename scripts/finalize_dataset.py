#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.config import DatasetFreezeConfig  # noqa: E402
from visionguard.finalization import finalize_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate, re-split, audit, and freeze a final YOLO dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to clean data.yaml.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "safety_final",
        help="New final dataset directory. It must not already exist.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Safely replace an existing final output only after a new staged copy passes all audits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = finalize_dataset(
            DatasetFreezeConfig(args.data, args.output_dir, args.seed, overwrite=args.overwrite)
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Dataset finalization failed: {exc}", file=sys.stderr)
        return 2

    print(f"Final dataset: {report['output_dataset_root']}")
    print(f"Images: {report['counts']['input_images']} -> {report['counts']['final_images']}")
    print(f"Duplicate images removed: {report['counts']['duplicate_images_removed']}")
    print(f"Empty-label images removed: {report['counts']['empty_label_images_removed']}")
    for split, stats in report["splits"].items():
        print(f"{split}: {stats['images']} images ({stats['actual_ratio']:.2%})")
    print(f"Audit passed: {report['audit']['passed']}")
    print(f"Report: {Path(report['output_dataset_root']) / 'finalization_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
