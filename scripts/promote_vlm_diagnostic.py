#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm_evidence import build_diagnostic_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a real Qwen diagnostic into checked evidence.")
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "outputs/vlm/qwen_real_historical.json"
    )
    parser.add_argument(
        "--dev-audit", type=Path, default=PROJECT_ROOT / "docs/evidence/vlm_dev_source_audit.json"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "docs/evidence/qwen_real_historical_result.json"
    )
    parser.add_argument(
        "--failures", type=Path, default=PROJECT_ROOT / "docs/evidence/QWEN_DIAGNOSTIC_FAILURES.md"
    )
    args = parser.parse_args()
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    revision = "89644892e4d85e24eaac8bacfd4f463576704203"
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots" / revision
    try:
        report = json.loads(args.input.read_text(encoding="utf-8"))
        dev_audit = json.loads(args.dev_audit.read_text(encoding="utf-8"))
        evidence = build_diagnostic_evidence(
            report,
            args.input,
            dev_audit,
            model_id,
            revision,
            snapshot / "model.safetensors",
            snapshot / "config.json",
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Diagnostic evidence promotion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    values = evidence["recomputed"]
    lines = [
        "# Qwen3-VL historical Dev diagnostic failures",
        "",
        "> Real-model diagnostic only. Mixed-provenance historical labels prevent this result from "
        "verifying the resume's three-model ablation or Dev F1=0.600 statement.",
        "",
        "## Recomputed result",
        "",
        f"- Examples: {values['examples']}",
        f"- Cache hits on the evidence rerun: {values['cache_hit_count']}",
        f"- Strict JSON parse failures: {values['strict_json_parse_failures']}",
        f"- IoU-grounded F1 at 0.5 IoU: {values['grounding_f1']:.4f}",
        f"- Unsupported finding rate: {values['unsupported_finding_rate']:.4f}",
        f"- Negative-scene hallucination rate: {values['negative_scene_hallucination_rate']:.4f}",
        "",
        "## Strict JSON parse failures",
        "",
    ]
    lines.extend(f"- `{example_id}`" for example_id in values["failure_ids"])
    lines.extend(
        [
            "",
            "The evaluator retains these failures as failed records instead of dropping them, "
            "so they reduce coverage and cannot improve the metric.",
            "",
        ]
    )
    args.failures.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(values, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
