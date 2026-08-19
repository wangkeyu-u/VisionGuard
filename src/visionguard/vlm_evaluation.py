from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionguard.vlm import VisionAdapter, VLMExample, validate_finding_document


def normalize_qwen_coordinate_scale(document: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Convert Qwen's documented 0-1000 grounding scale to the 0-1 contract."""
    normalized = deepcopy(document)
    repairs = 0
    findings = normalized.get("findings", [])
    if not isinstance(findings, list):
        return normalized, repairs
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        box = finding.get("person_box")
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, (int, float)) for value in box)
            and all(0 <= float(value) <= 1000 for value in box)
            and any(float(value) > 1 for value in box)
        ):
            finding["person_box"] = [float(value) / 1000 for value in box]
            repairs += 1
    return normalized, repairs


def box_iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_findings(predictions: list[dict[str, Any]], targets: list[dict[str, Any]], threshold: float) -> int:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            if prediction["violation"] != target["violation"]:
                continue
            iou = box_iou(prediction["person_box"], target["person_box"])
            if iou >= threshold:
                candidates.append((iou, prediction_index, target_index))
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for _, prediction_index, target_index in sorted(candidates, reverse=True):
        if prediction_index not in used_predictions and target_index not in used_targets:
            used_predictions.add(prediction_index)
            used_targets.add(target_index)
    return len(used_predictions)


def evaluate_adapter(adapter: VisionAdapter, examples: list[VLMExample], iou_threshold: float) -> dict[str, Any]:
    records = []
    prediction_count = target_count = matched_count = schema_valid = negative_count = hallucinated_negative = 0
    latencies = []
    runtime_errors = []
    cache_hits = 0
    deterministic_repairs = 0
    for example in examples:
        try:
            result = adapter.predict(example)
        except Exception as exc:  # evaluation must preserve completed cache entries for resume
            runtime_errors.append(
                {"id": example.id, "error_type": type(exc).__name__, "message": str(exc)}
            )
            records.append(
                {
                    "id": example.id,
                    "schema_valid": False,
                    "schema_errors": [f"runtime error: {type(exc).__name__}: {exc}"],
                    "matched_grounded_findings": 0,
                    "prediction": None,
                    "target": example.target,
                    "latency_ms": None,
                }
            )
            target_count += len(example.target.get("findings", []))
            continue
        cache_hits += result.cache_hit
        document, repairs = normalize_qwen_coordinate_scale(result.document)
        deterministic_repairs += repairs
        errors = validate_finding_document(document)
        predictions = document.get("findings", []) if not errors else []
        targets = example.target.get("findings", [])
        matched = _match_findings(predictions, targets, iou_threshold) if not errors else 0
        schema_valid += not errors
        prediction_count += len(predictions)
        target_count += len(targets)
        matched_count += matched
        if not targets:
            negative_count += 1
            hallucinated_negative += bool(predictions)
        latencies.append(result.latency_ms)
        records.append(
            {
                "id": example.id,
                "schema_valid": not errors,
                "schema_errors": errors,
                "matched_grounded_findings": matched,
                "prediction": document,
                "raw_prediction": result.document if repairs else None,
                "deterministic_coordinate_repairs": repairs,
                "target": example.target,
                "latency_ms": round(result.latency_ms, 3),
            }
        )
    precision = matched_count / prediction_count if prediction_count else 1.0
    recall = matched_count / target_count if target_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "adapter": adapter.name,
        "execution_mode": adapter.execution_mode,
        "result_interpretation": (
            "Pipeline fixture only; metrics are synthetic and are not model-quality evidence."
            if adapter.execution_mode == "fixture"
            else "Real adapter execution on the configured images and weights."
        ),
        "runtime_metadata": getattr(adapter, "runtime_metadata", {}),
        "metrics": {
            "examples": len(examples),
            "schema_valid_rate": schema_valid / len(examples),
            "grounded_true_positives": matched_count,
            "predicted_findings": prediction_count,
            "target_findings": target_count,
            "grounding_precision": precision,
            "grounding_recall": recall,
            "grounding_f1": f1,
            "unsupported_finding_rate": (
                (prediction_count - matched_count) / prediction_count
                if prediction_count
                else 0.0
            ),
            "negative_scene_hallucination_rate": hallucinated_negative / negative_count if negative_count else 0.0,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "runtime_error_count": len(runtime_errors),
            "cache_hit_count": cache_hits,
            "deterministic_coordinate_repair_count": deterministic_repairs,
        },
        "runtime_errors": runtime_errors,
        "records": records,
    }


def write_evaluation_report(
    results: list[dict[str, Any]],
    output: Path,
    config_path: Path,
    iou_threshold: float,
) -> dict[str, Any]:
    modes = sorted({result["execution_mode"] for result in results})
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": str(config_path.resolve()),
        "iou_threshold": iou_threshold,
        "execution_modes": modes,
        "contains_real_model_results": "real" in modes,
        "contains_fixture_results": "fixture" in modes,
        "runs": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
