#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.error_analysis import ErrorAnalysisConfig, analyze_yolo_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate heuristic YOLO test-set error analysis.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--low-confidence", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-examples-per-category", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    try:
        report = analyze_yolo_errors(ErrorAnalysisConfig(**vars(parse_args())))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Analyzed {report['test_images']} test images.")
    for category, count in report["image_counts"].items():
        print(f"{category}: {count} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
