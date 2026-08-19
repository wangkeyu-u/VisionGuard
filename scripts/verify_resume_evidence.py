#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.evidence import build_verification, write_verification  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute VisionGuard resume evidence from source artifacts.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "training/datasets/safety_final/data.yaml")
    parser.add_argument("--dataset-root", type=Path, help="Explicit root for a relocated data.yaml")
    parser.add_argument("--experiments", type=Path, default=PROJECT_ROOT / "outputs/experiments")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/source-verification.json")
    parser.add_argument("--require-verified", action="store_true")
    args = parser.parse_args()
    try:
        report = build_verification(args.data, args.experiments, args.dataset_root)
        write_verification(report, args.output)
    except (KeyError, OSError, ValueError) as exc:
        print(f"Evidence verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    statuses = {"dataset": report["dataset"]["status"], "yolo": report["yolo"]["status"]}
    print(f"Wrote {args.output}: {statuses}")
    if args.require_verified and any(status != "verified" for status in statuses.values()):
        print("Evidence gate BLOCKED: required source artifacts are absent or invalid.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
