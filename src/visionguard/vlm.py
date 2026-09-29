from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIOLATION_CLASSES = ("no_helmet", "no_vest")
CONFIDENCE_LEVELS = {"low", "medium", "high"}
TOP_LEVEL_KEYS = {"findings", "uncertainties", "recommended_action"}
FINDING_KEYS = {"person_id", "person_box", "violation", "evidence", "confidence"}
PERSON_ID_PATTERN = re.compile(r"^p[1-9][0-9]*$")


@dataclass(frozen=True)
class JsonlIssue:
    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[JsonlIssue]]:
    records: list[dict[str, Any]] = []
    issues: list[JsonlIssue] = []
    if not path.is_file():
        return records, [JsonlIssue(path, 0, "file does not exist")]
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            issues.append(JsonlIssue(path, line_number, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(value, dict):
            issues.append(JsonlIssue(path, line_number, "record must be a JSON object"))
            continue
        records.append(value)
    return records, issues


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_inspection(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["inspection must be a JSON object"]
    keys = set(payload)
    if keys != TOP_LEVEL_KEYS:
        errors.append(
            f"top-level keys must be {sorted(TOP_LEVEL_KEYS)}; got {sorted(keys)}"
        )

    findings = payload.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be an array")
        findings = []
    seen: set[tuple[str, str]] = set()
    for index, finding in enumerate(findings):
        prefix = f"findings[{index}]"
        if not isinstance(finding, dict):
            errors.append(f"{prefix} must be an object")
            continue
        finding_keys = set(finding)
        if finding_keys != FINDING_KEYS:
            errors.append(
                f"{prefix} keys must be {sorted(FINDING_KEYS)}; got {sorted(finding_keys)}"
            )
        person_id = finding.get("person_id")
        if not isinstance(person_id, str) or not PERSON_ID_PATTERN.fullmatch(person_id):
            errors.append(f"{prefix}.person_id must look like p1, p2, ...")
        violation = finding.get("violation")
        if violation not in VIOLATION_CLASSES:
            errors.append(f"{prefix}.violation must be one of {list(VIOLATION_CLASSES)}")
        if isinstance(person_id, str) and isinstance(violation, str):
            pair = (person_id, violation)
            if pair in seen:
                errors.append(f"{prefix} duplicates {person_id}/{violation}")
            seen.add(pair)

        box = finding.get("person_box")
        if not isinstance(box, list) or len(box) != 4:
            errors.append(f"{prefix}.person_box must contain four integers")
        elif any(isinstance(value, bool) or not isinstance(value, int) for value in box):
            errors.append(f"{prefix}.person_box values must be integers")
        elif any(value < 0 or value > 1000 for value in box):
            errors.append(f"{prefix}.person_box values must be between 0 and 1000")
        elif box[0] >= box[2] or box[1] >= box[3]:
            errors.append(f"{prefix}.person_box must satisfy x1 < x2 and y1 < y2")

        evidence = finding.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"{prefix}.evidence must be a non-empty string")
        if finding.get("confidence") not in CONFIDENCE_LEVELS:
            errors.append(f"{prefix}.confidence must be low, medium, or high")

    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list) or any(
        not isinstance(item, str) or not item.strip() for item in uncertainties or []
    ):
        errors.append("uncertainties must be an array of non-empty strings")
        uncertainties = []

    action = payload.get("recommended_action")
    if action not in {"human_review", "no_action"}:
        errors.append("recommended_action must be human_review or no_action")
    expected_action = "human_review" if findings or uncertainties else "no_action"
    if action in {"human_review", "no_action"} and action != expected_action:
        errors.append(
            f"recommended_action must be {expected_action} for the supplied findings/uncertainties"
        )
    return errors


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, count=1, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate, count=1)
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output does not contain a valid JSON object")


def normalized_xywh_to_xyxy(box: Iterable[float]) -> list[int]:
    center_x, center_y, width, height = [float(value) for value in box]
    coordinates = [
        (center_x - width / 2) * 1000,
        (center_y - height / 2) * 1000,
        (center_x + width / 2) * 1000,
        (center_y + height / 2) * 1000,
    ]
    return [max(0, min(1000, math.floor(value + 0.5))) for value in coordinates]


def pixel_xyxy_to_normalized(box: Iterable[float], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        max(0, min(1000, round(x1 / width * 1000))),
        max(0, min(1000, round(y1 / height * 1000))),
        max(0, min(1000, round(x2 / width * 1000))),
        max(0, min(1000, round(y2 / height * 1000))),
    ]


def box_iou(first: list[int], second: list[int]) -> float:
    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _match_findings(
    gold: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, int, int, dict[str, tuple[int, int, int]]]:
    matched_gold: set[int] = set()
    true_positives = 0
    per_class: dict[str, list[int]] = {
        class_name: [0, 0, 0] for class_name in VIOLATION_CLASSES
    }
    for prediction in predicted:
        violation = prediction.get("violation")
        best_index = None
        best_iou = -1.0
        for index, target in enumerate(gold):
            if index in matched_gold or target.get("violation") != violation:
                continue
            overlap = box_iou(prediction.get("person_box", [0, 0, 0, 0]), target["person_box"])
            if overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index is not None and best_iou >= iou_threshold:
            matched_gold.add(best_index)
            true_positives += 1
            if violation in per_class:
                per_class[violation][0] += 1
        elif violation in per_class:
            per_class[violation][1] += 1
    false_positives = len(predicted) - true_positives
    false_negatives = len(gold) - true_positives
    for index, target in enumerate(gold):
        if index not in matched_gold and target.get("violation") in per_class:
            per_class[target["violation"]][2] += 1
    return (
        true_positives,
        false_positives,
        false_negatives,
        {key: tuple(value) for key, value in per_class.items()},
    )


def _prf(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, float]:
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def score_prediction_records(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    predictions_by_id = {record.get("id"): record for record in prediction_records}
    totals = [0, 0, 0]
    class_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    parsed = 0
    schema_valid = 0
    clear_images = 0
    hallucinated_clear_images = 0
    evaluated = 0
    missing_ids: list[str] = []

    for gold_record in gold_records:
        record_id = str(gold_record.get("id"))
        prediction_record = predictions_by_id.get(record_id)
        if prediction_record is None:
            missing_ids.append(record_id)
            continue
        evaluated += 1
        prediction = prediction_record.get("prediction")
        if isinstance(prediction, dict):
            parsed += 1
            if not validate_inspection(prediction):
                schema_valid += 1
        else:
            prediction = {"findings": [], "uncertainties": [], "recommended_action": "no_action"}
        gold = gold_record["target"]
        gold_findings = gold.get("findings", [])
        predicted_findings = prediction.get("findings", [])
        if not gold_findings:
            clear_images += 1
            if predicted_findings:
                hallucinated_clear_images += 1
        true_positive, false_positive, false_negative, per_class = _match_findings(
            gold_findings, predicted_findings, iou_threshold
        )
        totals[0] += true_positive
        totals[1] += false_positive
        totals[2] += false_negative
        for class_name, values in per_class.items():
            for index, value in enumerate(values):
                class_totals[class_name][index] += value

    return {
        "gold_records": len(gold_records),
        "prediction_records": len(prediction_records),
        "evaluated_records": evaluated,
        "missing_prediction_ids": missing_ids,
        "parsed_json_rate": round(parsed / evaluated, 6) if evaluated else 0.0,
        "schema_valid_rate": round(schema_valid / evaluated, 6) if evaluated else 0.0,
        "iou_match_threshold": iou_threshold,
        "overall": {**_prf(*totals), "tp": totals[0], "fp": totals[1], "fn": totals[2]},
        "per_class": {
            class_name: {
                **_prf(*class_totals[class_name]),
                "tp": class_totals[class_name][0],
                "fp": class_totals[class_name][1],
                "fn": class_totals[class_name][2],
            }
            for class_name in VIOLATION_CLASSES
        },
        "clear_images": clear_images,
        "hallucinated_clear_images": hallucinated_clear_images,
        "hallucination_rate_on_clear_images": (
            round(hallucinated_clear_images / clear_images, 6) if clear_images else None
        ),
    }
