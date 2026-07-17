#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.config import DatasetValidationConfig  # noqa: E402
from visionguard.dataset import validate_dataset  # noqa: E402
from visionguard.reporting import write_dataset_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to the Ultralytics data.yaml file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports",
        help="Directory for dataset_report.json and dataset_report.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DatasetValidationConfig(data_path=args.data, output_dir=args.output_dir)
    try:
        report = validate_dataset(config)
        json_path, markdown_path = write_dataset_reports(
            report, config.output_dir.expanduser().resolve()
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(f"Dataset status: {'PASS' if report['valid'] else 'FAIL'}")
    print(f"Images: {summary['images']}")
    print(f"Valid annotations: {summary['valid_annotations']} / {summary['annotation_lines']}")
    print(f"Errors: {summary['errors']} | Warnings: {summary['warnings']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
