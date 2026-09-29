#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.safety_selection import (  # noqa: E402
    SafetySelectionConfig,
    select_safety_checkpoint,
)
from visionguard.training import FROZEN_DATASET_SHA256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a checkpoint using validation safety-class AP only."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "safety_final" / "data.yaml",
    )
    parser.add_argument("--expected-fingerprint", default=FROZEN_DATASET_SHA256)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    try:
        report = select_safety_checkpoint(SafetySelectionConfig(**vars(parse_args())))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Safety selection failed: {exc}", file=sys.stderr)
        return 2
    selected = report["selected"]
    print(f"Selected: {selected['checkpoint']}")
    print(f"Safety H-mean AP@50-95: {selected['safety_hmean_map50_95']:.4f}")
    print(f"Copied to: {selected['copied_to']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
