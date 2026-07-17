#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm import read_jsonl, score_prediction_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VisionGuard VLM prediction JSONL.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _latency_summary(records: list[dict[str, object]]) -> dict[str, float | None]:
    values = sorted(
        float(record["latency_ms"]) for record in records if isinstance(record.get("latency_ms"), (int, float))
    )
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    p95_index = min(len(values) - 1, max(0, round(0.95 * len(values) + 0.5) - 1))
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(values[p95_index], 3),
    }


def _repair_summary(records: list[dict[str, object]]) -> dict[str, int | float]:
    attempted = sum(int(record.get("repair_attempts", 0)) > 0 for record in records)
    total_attempts = sum(int(record.get("repair_attempts", 0)) for record in records)
    successful = sum(int(record.get("repair_attempts", 0)) > 0 and not record.get("error") for record in records)
    return {
        "records_repaired": successful,
        "records_with_repair_attempt": attempted,
        "total_repair_attempts": total_attempts,
        "repair_success_rate": round(successful / attempted, 6) if attempted else 0.0,
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.iou_threshold <= 1:
        raise SystemExit("--iou-threshold must be in (0, 1]")
    gold, gold_issues = read_jsonl(args.gold.resolve())
    predictions, prediction_issues = read_jsonl(args.predictions.resolve())
    issues = [*gold_issues, *prediction_issues]
    if issues:
        raise SystemExit("\n".join(str(issue) for issue in issues))
    report = score_prediction_records(gold, predictions, args.iou_threshold)
    report["latency"] = _latency_summary(predictions)
    report["schema_repair"] = _repair_summary(predictions)
    report["runtime_error_count"] = sum(bool(record.get("error")) for record in predictions)
    output = args.output or args.predictions.with_name(f"{args.predictions.stem}_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Metrics saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
