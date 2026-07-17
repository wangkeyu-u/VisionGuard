#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.benchmarking import BenchmarkConfig, benchmark_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO on a fixed test-image set.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    try:
        report = benchmark_yolo(BenchmarkConfig(**vars(parse_args())))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Mean latency: {report['average_latency_ms']:.3f} ms/image")
    print(f"P95 latency: {report['p95_latency_ms']:.3f} ms/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
