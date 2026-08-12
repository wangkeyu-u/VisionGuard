from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.vlm import FixtureAdapter, load_evaluation_config, validate_finding_document
from visionguard.vlm_evaluation import box_iou, evaluate_adapter, write_evaluation_report

FIXTURES = Path(__file__).parent / "fixtures" / "vlm"


def test_box_iou_and_schema_validation() -> None:
    assert box_iou([0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]) == pytest.approx(0.25)
    assert validate_finding_document(
        {
            "scene_summary": "One worker.",
            "findings": [
                {
                    "person_box": [0.1, 0.1, 0.5, 0.9],
                    "violation": "no_helmet",
                    "evidence": "Visible head without helmet.",
                    "confidence": "medium",
                }
            ],
            "uncertainties": [],
            "recommended_action": "Review.",
        }
    ) == []
    assert "normalized xyxy" in " ".join(
        validate_finding_document(
            {
                "scene_summary": "Bad box.",
                "findings": [
                    {
                        "person_box": [0.8, 0.1, 0.2, 0.9],
                        "violation": "no_helmet",
                        "evidence": "fixture",
                        "confidence": "low",
                    }
                ],
                "uncertainties": [],
                "recommended_action": "Review.",
            }
        )
    )


def test_fixture_evaluation_marks_synthetic_results(tmp_path: Path) -> None:
    _, examples = load_evaluation_config(Path("configs/vlm_ablation.fixture.yaml"))
    adapter = FixtureAdapter("grounded_fixture", FIXTURES / "grounded_predictions.jsonl")

    result = evaluate_adapter(adapter, examples, iou_threshold=0.5)
    report_path = tmp_path / "result.json"
    report = write_evaluation_report([result], report_path, Path("fixture.yaml"), 0.5)

    assert result["execution_mode"] == "fixture"
    assert "not model-quality evidence" in result["result_interpretation"]
    assert result["metrics"]["grounding_f1"] == 1.0
    assert result["metrics"]["negative_scene_hallucination_rate"] == 0.0
    assert report["contains_real_model_results"] is False
    assert report["contains_fixture_results"] is True
    assert json.loads(report_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_fixture_ablation_exposes_hallucination_metric() -> None:
    _, examples = load_evaluation_config(Path("configs/vlm_ablation.fixture.yaml"))
    adapter = FixtureAdapter("qwen_fixture", FIXTURES / "qwen3_vl_predictions.jsonl")

    result = evaluate_adapter(adapter, examples, iou_threshold=0.5)

    assert result["metrics"]["grounded_true_positives"] == 0
    assert result["metrics"]["unsupported_finding_rate"] == 1.0
    assert result["metrics"]["negative_scene_hallucination_rate"] == 1.0
