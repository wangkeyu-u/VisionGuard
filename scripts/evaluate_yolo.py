#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.evaluation import EvaluationConfig, evaluate_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint on the test split.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prediction-samples", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    try:
        report = evaluate_yolo(EvaluationConfig(**vars(parse_args())))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Test images: {report['test_images']}")
    for name, value in report["metrics"].items():
        print(f"{name}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
