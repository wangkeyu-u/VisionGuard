#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.demo_server import DemoServerConfig, run_demo_server  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local VisionGuard image/video demo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "exp4_yolo11s_512_e50"
        / "weights"
        / "best.pt",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-upload-mb", type=int, default=150)
    parser.add_argument("--max-video-seconds", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "demo")
    return parser.parse_args()


def main() -> int:
    try:
        run_demo_server(DemoServerConfig(**vars(parse_args())))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
