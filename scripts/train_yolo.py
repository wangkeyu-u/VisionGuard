#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.training import FROZEN_DATASET_SHA256, TrainingConfig, train_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an auditable Ultralytics YOLO baseline.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, mps, cpu, or another Ultralytics device"
    )
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    parser.add_argument("--name", default="baseline_yolo11n_512")
    parser.add_argument("--expected-fingerprint", default=FROZEN_DATASET_SHA256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig(**vars(args))
    try:
        result = train_yolo(config, PROJECT_ROOT)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Training completed in {result['duration_seconds']:.1f} seconds.")
    print(f"Run directory: {result['run_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
