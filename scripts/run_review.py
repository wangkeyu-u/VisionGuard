#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.review_server import ReviewServerConfig, run_review_server  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local VisionGuard annotation review desk.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "data" / "candidates.jsonl",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "data",
    )
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--max-request-kb", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ReviewServerConfig(project_root=PROJECT_ROOT, **vars(args))
    try:
        run_review_server(config)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Review desk failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nReview desk stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
