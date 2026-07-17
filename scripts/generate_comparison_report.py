#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.comparison_reporting import generate_comparison_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VisionGuard YOLO experiments.")
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "experiment_comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = generate_comparison_report(args.experiments, args.output_dir)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Comparison report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
