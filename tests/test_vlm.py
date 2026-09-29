from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.vlm import (
    extract_json_object,
    normalized_xywh_to_xyxy,
    read_jsonl,
    score_prediction_records,
    validate_inspection,
    write_jsonl,
)


def valid_target() -> dict[str, object]:
    return {
        "findings": [
            {
                "person_id": "p1",
                "person_box": [100, 100, 500, 900],
                "violation": "no_helmet",
                "evidence": "The visible head has no helmet.",
                "confidence": "medium",
            }
        ],
        "uncertainties": [],
        "recommended_action": "human_review",
    }


def test_validate_inspection_accepts_strict_target() -> None:
    assert validate_inspection(valid_target()) == []


def test_validate_inspection_rejects_bad_box_and_inconsistent_action() -> None:
    target = valid_target()
    target["findings"][0]["person_box"] = [500, 100, 100, 900]  # type: ignore[index]
    target["recommended_action"] = "no_action"

    errors = validate_inspection(target)

    assert any("x1 < x2" in error for error in errors)
    assert any("must be human_review" in error for error in errors)


def test_extract_json_object_handles_markdown_fence_and_prefix() -> None:
    raw = "Model output:\n```json\n" + json.dumps(valid_target()) + "\n```"
    assert extract_json_object(raw) == valid_target()


def test_normalized_xywh_conversion_matches_reviewed_example() -> None:
    assert normalized_xywh_to_xyxy(
        [0.64375, 0.71328125, 0.09375, 0.1015625]
    ) == [597, 663, 691, 764]


def test_jsonl_round_trip_and_invalid_line_reporting(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [{"id": "one"}, {"id": "two"}]
    write_jsonl(path, records)
    assert read_jsonl(path) == (records, [])

    path.write_text('{"id":"one"}\nnot-json\n', encoding="utf-8")
    parsed, issues = read_jsonl(path)
    assert parsed == [{"id": "one"}]
    assert len(issues) == 1
    assert issues[0].line == 2


def test_scoring_matches_violation_and_person_box() -> None:
    gold = [{"id": "a", "target": valid_target()}]
    predictions = [{"id": "a", "prediction": valid_target(), "latency_ms": 10.0}]

    report = score_prediction_records(gold, predictions)

    assert report["schema_valid_rate"] == 1.0
    assert report["overall"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
    }


def test_scoring_reports_hallucination_on_clear_image() -> None:
    clear = {
        "findings": [],
        "uncertainties": [],
        "recommended_action": "no_action",
    }
    gold = [{"id": "clear", "target": clear}]
    predictions = [{"id": "clear", "prediction": valid_target()}]

    report = score_prediction_records(gold, predictions)

    assert report["hallucination_rate_on_clear_images"] == 1.0


def test_extract_json_object_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        extract_json_object("No violations detected.")
