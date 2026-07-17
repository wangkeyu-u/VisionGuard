#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.training import FROZEN_DATASET_SHA256, resume_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume a stopped Ultralytics run with frozen-dataset verification.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to weights/last.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the frozen source data.yaml.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-fingerprint", default=FROZEN_DATASET_SHA256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = resume_yolo(
            args.checkpoint,
            args.data,
            args.expected_fingerprint,
            args.device,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Resume failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        f"Training resumed from epoch {result['resume_from_epoch']} and finished {result['completed_epochs']} epochs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
