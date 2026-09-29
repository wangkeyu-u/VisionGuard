#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.quality_reporting import build_quality_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare safety-selected checkpoints on validation without touching test."
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-name", default="exp4_yolo11s_512_e50")
    parser.add_argument("--max-overall-regression", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "quality_upgrade",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_quality_comparison(
            args.runs,
            args.output_dir,
            args.baseline_name,
            args.max_overall_regression,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Quality report failed: {exc}", file=sys.stderr)
        return 2
    print(f"Selected on validation: {report['selected_run']}")
    print(f"Report: {args.output_dir / 'QUALITY_COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
