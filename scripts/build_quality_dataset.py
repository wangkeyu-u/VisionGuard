#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.quality_dataset import QualityDatasetConfig, build_quality_dataset  # noqa: E402
from visionguard.training import FROZEN_DATASET_SHA256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frozen train-only rare-class dataset for safety fine-tuning."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "safety_final" / "data.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "training" / "datasets" / "safety_quality_exp5",
    )
    parser.add_argument("--expected-fingerprint", default=FROZEN_DATASET_SHA256)
    parser.add_argument("--rare-multiplier", type=int, default=3)
    parser.add_argument("--hard-multiplier", type=int, default=4)
    parser.add_argument("--small-box-area", type=float, default=0.01)
    parser.add_argument("--dense-annotations", type=int, default=8)
    parser.add_argument("--overlap-iou", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        report = build_quality_dataset(QualityDatasetConfig(**vars(parse_args())))
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Quality dataset build failed: {exc}", file=sys.stderr)
        return 2
    print("Safety-focused quality dataset is ready.")
    print(json.dumps(report["split_stats"], indent=2, ensure_ascii=False))
    print(f"Fingerprint: {report['freeze']['dataset_sha256']}")
    print(f"Report: {Path(report['output_data']).parent / 'quality_dataset_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
