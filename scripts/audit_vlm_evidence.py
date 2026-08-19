#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit real VLM assets without inventing results.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/evidence/vlm_asset_audit.json")
    args = parser.parse_args()
    model_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct"
    revision = (
        (model_root / "refs/main").read_text(encoding="utf-8").strip() if (model_root / "refs/main").is_file() else None
    )
    snapshot = model_root / "snapshots" / str(revision)
    qwen_result = PROJECT_ROOT / "docs/evidence/qwen_real_historical_result.json"
    yolo_checkpoint = PROJECT_ROOT / "outputs/experiments/exp4_yolo11s_512_e50/weights/best.pt"
    dev_audit = PROJECT_ROOT / "docs/evidence/vlm_dev_source_audit.json"
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "runtime": platform.platform(),
        "qwen": {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": revision,
            "config": artifact(snapshot / "config.json"),
            "weights": artifact(snapshot / "model.safetensors"),
            "diagnostic_result": artifact(qwen_result),
        },
        "yolo": {"selected_checkpoint": artifact(yolo_checkpoint)},
        "dev_source_audit": artifact(dev_audit),
        "ablation_arms": {
            "yolo11": "missing_checkpoint",
            "qwen3_vl": "real_diagnostic_mixed_provenance" if qwen_result.is_file() else "missing",
            "grounded_qwen3_vl": "missing_run_and_yolo_checkpoint",
        },
        "full_three_model_ablation_verified": False,
        "resume_f1_0_600_verified": False,
        "blocking_reasons": [],
    }
    if not yolo_checkpoint.is_file():
        report["blocking_reasons"].append("Selected project YOLO checkpoint is missing.")
    if not qwen_result.is_file():
        report["blocking_reasons"].append("Promoted real Qwen diagnostic evidence is missing.")
    else:
        diagnostic = json.loads(qwen_result.read_text(encoding="utf-8"))
        report["qwen"]["diagnostic_recomputed"] = diagnostic["recomputed"]
        report["qwen"]["diagnostic_claim_support"] = diagnostic["claim_support"]
    report["blocking_reasons"].append(
        "No committed source prediction set exists for the historical Dev F1=0.600 statement."
    )
    report["blocking_reasons"].append(
        "Recovered Dev labels have mixed reviewer provenance and cannot independently verify the resume claim."
    )
    report["blocking_reasons"].append(
        "No grounded-Qwen3-VL real run exists because the selected project YOLO checkpoint is missing."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
