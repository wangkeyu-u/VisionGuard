#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.training import FROZEN_DATASET_SHA256, TrainingConfig, train_yolo  # noqa: E402

SOURCE_DATA = PROJECT_ROOT / "training" / "datasets" / "safety_final" / "data.yaml"
QUALITY_DATA = PROJECT_ROOT / "training" / "datasets" / "safety_quality_exp5" / "data.yaml"
SOURCE_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "exp4_yolo11s_512_e50"
    / "weights"
    / "best.pt"
)

RECIPES = {
    "rare_only": {
        "data": QUALITY_DATA,
        "imgsz": 512,
        "class_weight_power": 0.0,
        "name": "exp5a_rare_balance",
    },
    "weighted_only": {
        "data": SOURCE_DATA,
        "imgsz": 512,
        "class_weight_power": 0.5,
        "name": "exp5b_class_weighted",
    },
    "resolution_only": {
        "data": SOURCE_DATA,
        "imgsz": 640,
        "class_weight_power": 0.0,
        "name": "exp5c_resolution_640",
    },
    "safety_tune": {
        "data": QUALITY_DATA,
        "imgsz": 640,
        "class_weight_power": 0.5,
        "name": "exp5_safety_tune",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Exp4 with controlled safety-class quality recipes."
    )
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="safety_tune")
    parser.add_argument("--model", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "outputs" / "experiments")
    return parser.parse_args()


def _fingerprint_for(data: Path) -> str:
    if data.resolve() == SOURCE_DATA.resolve():
        return FROZEN_DATASET_SHA256
    manifest = data.parent / "freeze_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"quality dataset is missing; run scripts/build_quality_dataset.py first: {manifest}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    fingerprint = payload.get("dataset_sha256")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"quality dataset manifest has no fingerprint: {manifest}")
    return fingerprint


def main() -> int:
    args = parse_args()
    recipe = RECIPES[args.recipe]
    data = Path(recipe["data"])
    config = TrainingConfig(
        data=data,
        model=str(args.model.expanduser().resolve()),
        epochs=args.epochs,
        imgsz=int(recipe["imgsz"]),
        batch=args.batch,
        patience=6,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
        project=args.project,
        name=str(recipe["name"]),
        expected_fingerprint=_fingerprint_for(data),
        class_weight_power=float(recipe["class_weight_power"]),
        save_period=3,
    )
    try:
        result = train_yolo(config, PROJECT_ROOT)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Quality fine-tuning failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Recipe {args.recipe} completed in {result['duration_seconds']:.1f} seconds.")
    print(f"Run directory: {result['run_directory']}")
    print("Next: select the safety checkpoint on validation; do not inspect test yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
