#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.experiment_reporting import generate_experiment_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a VisionGuard experiment report.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = generate_experiment_report(args.run_dir, args.data, args.dataset_report)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Report generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Experiment report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
