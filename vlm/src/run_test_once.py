#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Dev-selected configuration on Test v2 exactly once after its QA gate passes."
    )
    parser.add_argument("--gold", type=Path, default=PROJECT_ROOT / "vlm" / "data_v2" / "test.jsonl")
    parser.add_argument(
        "--qa-report",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "data_v2" / "TEST_DATA_QA.json",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "outputs" / "test_final" / "FROZEN_SELECTION.json",
        help="Frozen Dev selection evidence. Test v1 consumption does not prevent a new Test v2 benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "outputs" / "test_v2_final",
    )
    parser.add_argument("--yolo-device", default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)


def main() -> int:
    args = parse_args()
    gold = args.gold.expanduser().resolve()
    qa_path = args.qa_report.expanduser().resolve()
    selection_path = args.selection.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prediction_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "RUN_MANIFEST.json"

    # The manifest is written before inference. Even a crash consumes the one allowed run.
    if manifest_path.exists():
        raise SystemExit(f"Test v2 has already been claimed; refusing a rerun: {manifest_path}")
    if prediction_path.exists() or metrics_path.exists():
        raise SystemExit(f"Test v2 output already exists; refusing to overwrite {output_dir}")

    qa = _load_json(qa_path, "Test v2 QA report")
    if qa.get("status") != "PASS":
        raise SystemExit("Test v2 QA status is not PASS; evaluation remains blocked")
    gold_sha256 = _sha256(gold)
    if qa.get("test_sha256") != gold_sha256:
        raise SystemExit("Test v2 changed after QA; rerun the QA gate before evaluation")

    selection = _load_json(selection_path, "frozen Dev selection")
    if selection.get("selected_on") != "dev" or selection.get("selected_run") != "YOLO-only":
        raise SystemExit("frozen selection must show that YOLO-only was selected on Dev")
    checkpoint_value = selection.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise SystemExit("frozen selection does not identify a checkpoint")
    checkpoint = (PROJECT_ROOT / checkpoint_value).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"selected checkpoint does not exist: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    if selection.get("checkpoint_sha256") != checkpoint_sha256:
        raise SystemExit("selected checkpoint hash differs from the frozen Dev selection")
    imgsz = int(selection.get("imgsz", 512))
    confidence = float(selection.get("confidence", 0.25))
    iou_threshold = float(selection.get("person_box_iou_threshold", 0.3))

    manifest: dict[str, Any] = {
        "status": "running",
        "claimed_at": datetime.now(UTC).isoformat(),
        "one_time_run_consumed": True,
        "gold": str(gold),
        "gold_sha256": gold_sha256,
        "qa_report": str(qa_path),
        "qa_report_sha256": _sha256(qa_path),
        "dev_selection": str(selection_path),
        "dev_selection_sha256": _sha256(selection_path),
        "mode": "yolo",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "imgsz": imgsz,
        "confidence": confidence,
        "person_box_iou_threshold": iou_threshold,
    }
    _atomic_write_json(manifest_path, manifest)

    baseline_command = [
        sys.executable,
        str(PROJECT_ROOT / "vlm" / "src" / "run_baseline.py"),
        "--split",
        "test",
        "--data-path",
        str(gold),
        "--mode",
        "yolo",
        "--yolo-model",
        str(checkpoint),
        "--yolo-device",
        args.yolo_device,
        "--imgsz",
        str(imgsz),
        "--confidence",
        str(confidence),
        "--output",
        str(prediction_path),
    ]
    evaluate_command = [
        sys.executable,
        str(PROJECT_ROOT / "vlm" / "src" / "evaluate.py"),
        "--gold",
        str(gold),
        "--predictions",
        str(prediction_path),
        "--iou-threshold",
        str(iou_threshold),
        "--output",
        str(metrics_path),
    ]
    baseline_result = _run(baseline_command)
    manifest["baseline_returncode"] = baseline_result.returncode
    if baseline_result.returncode != 0:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(manifest_path, manifest)
        print("Test inference failed. The run remains consumed; diagnose without rerunning Test.", file=sys.stderr)
        return baseline_result.returncode

    evaluate_result = _run(evaluate_command)
    manifest["evaluation_returncode"] = evaluate_result.returncode
    manifest["status"] = "completed" if evaluate_result.returncode == 0 else "failed"
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["prediction_sha256"] = _sha256(prediction_path) if prediction_path.is_file() else None
    manifest["metrics_sha256"] = _sha256(metrics_path) if metrics_path.is_file() else None
    _atomic_write_json(manifest_path, manifest)
    print(f"One-time Test v2 run recorded in {manifest_path}")
    return evaluate_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
