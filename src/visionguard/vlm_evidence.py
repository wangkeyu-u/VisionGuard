from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute_run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    """Recompute grounded counts from record-level model outputs, never summary prose."""
    records = run.get("records")
    if not isinstance(records, list):
        raise ValueError("run.records must be a list")
    predicted = targets = matched = schema_valid = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each run record must be an object")
        target = record.get("target", {})
        target_findings = target.get("findings", []) if isinstance(target, dict) else []
        if not isinstance(target_findings, list):
            raise ValueError("record target findings must be a list")
        targets += len(target_findings)
        if record.get("schema_valid") is True:
            schema_valid += 1
            prediction = record.get("prediction", {})
            prediction_findings = prediction.get("findings", []) if isinstance(prediction, dict) else []
            if not isinstance(prediction_findings, list):
                raise ValueError("record prediction findings must be a list")
            predicted += len(prediction_findings)
        record_matches = record.get("matched_grounded_findings", 0)
        if not isinstance(record_matches, int) or record_matches < 0:
            raise ValueError("matched_grounded_findings must be a non-negative integer")
        matched += record_matches
    if matched > min(predicted, targets):
        raise ValueError("matched count exceeds prediction or target count")
    precision = matched / predicted if predicted else 1.0
    recall = matched / targets if targets else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "examples": len(records),
        "schema_valid_examples": schema_valid,
        "grounded_true_positives": matched,
        "predicted_findings": predicted,
        "target_findings": targets,
        "grounding_precision": precision,
        "grounding_recall": recall,
        "grounding_f1": f1,
    }


def verify_report_summary(run: dict[str, Any], recomputed: dict[str, Any], tolerance: float = 1e-12) -> None:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("run.metrics must be an object")
    for key in (
        "examples",
        "grounded_true_positives",
        "predicted_findings",
        "target_findings",
    ):
        if metrics.get(key) != recomputed[key]:
            raise ValueError(f"reported {key} does not match record-level recomputation")
    for key in ("grounding_precision", "grounding_recall", "grounding_f1"):
        reported = metrics.get(key)
        if not isinstance(reported, (int, float)) or abs(float(reported) - recomputed[key]) > tolerance:
            raise ValueError(f"reported {key} does not match record-level recomputation")


def build_diagnostic_evidence(
    report: dict[str, Any],
    report_path: Path,
    dev_audit: dict[str, Any],
    model_id: str,
    revision: str,
    weights_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise ValueError("Qwen diagnostic must contain exactly one run")
    run = runs[0]
    recomputed = recompute_run_metrics(run)
    verify_report_summary(run, recomputed)
    runtime_errors = run.get("runtime_errors", [])
    if not isinstance(runtime_errors, list):
        raise ValueError("runtime_errors must be a list")
    parse_failures = [row for row in runtime_errors if row.get("error_type") == "JSONDecodeError"]
    metrics = run["metrics"]
    cache_hits = metrics.get("cache_hit_count")
    if cache_hits != recomputed["examples"] - len(runtime_errors):
        raise ValueError("cache hit count must equal successful examples in the resumed evidence run")
    if metrics.get("runtime_error_count") != len(runtime_errors):
        raise ValueError("runtime error summary does not match runtime_errors")
    if len(parse_failures) != len(runtime_errors):
        raise ValueError("the recorded diagnostic failures are not all strict JSON parse failures")
    embedded_report = deepcopy(report)
    embedded_report["config"] = "configs/vlm_qwen_real_historical.yaml"
    return {
        "schema_version": 1,
        "evidence_class": "real_model_diagnostic_mixed_provenance",
        "claim_support": {
            "qwen_only_diagnostic": True,
            "full_three_model_ablation": False,
            "resume_dev_f1_0_600": False,
            "reason": (
                "This is one real Qwen3-VL arm on recovered mixed-provenance historical Dev labels. "
                "The project YOLO checkpoint, grounded-Qwen run, and original F1=0.600 predictions are absent."
            ),
        },
        "model": {
            "model_id": model_id,
            "revision": revision,
            "weights_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
        },
        "data": {
            "source_ref": dev_audit["source_ref"],
            "source_commit": dev_audit["source_commit"],
            "source_path": dev_audit["source_path"],
            "source_sha256": dev_audit["source_sha256"],
            "records": dev_audit["records"],
            "reviewer_provenance": dev_audit["reviewer_provenance"],
            "evidence_grade": dev_audit["evidence_grade"],
            "limitation": dev_audit["limitation"],
        },
        "source_report": {
            "sha256": sha256_file(report_path),
            "iou_threshold": report["iou_threshold"],
            "execution_modes": report["execution_modes"],
        },
        "recomputed": {
            **recomputed,
            "cache_hit_count": cache_hits,
            "strict_json_parse_failures": len(parse_failures),
            "failure_ids": [row["id"] for row in parse_failures],
            "unsupported_finding_rate": metrics["unsupported_finding_rate"],
            "negative_scene_hallucination_rate": metrics["negative_scene_hallucination_rate"],
            "mean_latency_ms": metrics["mean_latency_ms"],
            "deterministic_coordinate_repair_count": metrics[
                "deterministic_coordinate_repair_count"
            ],
        },
        "evaluation": embedded_report,
    }
