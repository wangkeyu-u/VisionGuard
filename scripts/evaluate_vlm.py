#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm import CachedAdapter, build_adapter, load_evaluation_config  # noqa: E402
from visionguard.vlm_evaluation import evaluate_adapter, write_evaluation_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11/Qwen3-VL ablations with grounded metrics.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Atomic per-run JSONL caches for resume.")
    args = parser.parse_args()
    try:
        config, examples = load_evaluation_config(args.config)
        threshold = float(config.get("iou_threshold", 0.5))
        results = []
        for run in config["runs"]:
            adapter = build_adapter(run, args.config.resolve().parent)
            if args.cache_dir:
                signature_payload = {
                    "run": run,
                    "manifest_ids": [example.id for example in examples],
                    "iou_threshold": threshold,
                }
                signature = hashlib.sha256(
                    json.dumps(signature_payload, sort_keys=True).encode()
                ).hexdigest()
                adapter = CachedAdapter(
                    adapter,
                    args.cache_dir / f"{run['name']}.jsonl",
                    signature,
                )
            results.append(evaluate_adapter(adapter, examples, threshold))
        report = write_evaluation_report(results, args.output, args.config, threshold)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"VLM evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output} ({', '.join(report['execution_modes'])})")
    if report["contains_fixture_results"]:
        print("FIXTURE NOTICE: synthetic pipeline verification; not a completed model ablation experiment.")
    failures = sum(run["metrics"]["runtime_error_count"] for run in report["runs"])
    if failures:
        print(f"FAILURE REPORT: {failures} examples failed; see runs[].runtime_errors.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
