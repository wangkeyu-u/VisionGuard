#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm import build_adapter, load_evaluation_config  # noqa: E402
from visionguard.vlm_evaluation import evaluate_adapter, write_evaluation_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11/Qwen3-VL ablations with grounded metrics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config, examples = load_evaluation_config(args.config)
        threshold = float(config.get("iou_threshold", 0.5))
        results = [
            evaluate_adapter(build_adapter(run, args.config.resolve().parent), examples, threshold)
            for run in config["runs"]
        ]
        report = write_evaluation_report(results, args.output, args.config, threshold)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"VLM evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output} ({', '.join(report['execution_modes'])})")
    if report["contains_fixture_results"]:
        print("FIXTURE NOTICE: synthetic pipeline verification; not a completed model ablation experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
