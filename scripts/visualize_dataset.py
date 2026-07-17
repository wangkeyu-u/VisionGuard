#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.config import DatasetVisualizationConfig  # noqa: E402
from visionguard.reporting import write_dataset_preview_reports  # noqa: E402
from visionguard.visualization import visualize_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random annotated YOLO dataset samples.")
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to the Ultralytics data.yaml file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualization",
        help="Directory for rendered samples and preview reports.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=8,
        help="Maximum annotated images to sample per split.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used for repeatable sampling."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DatasetVisualizationConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
    )
    try:
        report = visualize_dataset(config)
        json_path, markdown_path = write_dataset_preview_reports(
            report, config.output_dir.expanduser().resolve()
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Dataset visualization failed: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(f"Images visualized: {summary['images_visualized']}")
    print(f"Objects visualized: {summary['objects_visualized']}")
    print(f"Average objects per image: {summary['average_objects_per_image']:.4f}")
    for warning in report["warnings"]:
        print(f"Warning: {warning}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
