from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.vlm import (
    AdapterResult,
    CachedAdapter,
    FixtureAdapter,
    VisionAdapter,
    load_evaluation_config,
    validate_finding_document,
)
from visionguard.vlm_evaluation import (
    box_iou,
    evaluate_adapter,
    normalize_qwen_coordinate_scale,
    write_evaluation_report,
)
from visionguard.vlm_evidence import recompute_run_metrics, verify_report_summary

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


def test_cached_adapter_resumes_without_repeating_prediction(tmp_path: Path) -> None:
    class CountingAdapter(VisionAdapter):
        name = "counting"
        execution_mode = "real"
        calls = 0

        def predict(self, example):
            self.calls += 1
            return AdapterResult(
                {
                    "scene_summary": "cached",
                    "findings": [],
                    "uncertainties": [],
                    "recommended_action": "review",
                },
                3.5,
            )

    _, examples = load_evaluation_config(Path("configs/vlm_ablation.fixture.yaml"))
    adapter = CountingAdapter()
    first = CachedAdapter(adapter, tmp_path / "cache.jsonl", "signature")
    assert first.predict(examples[0]).cache_hit is False
    second = CachedAdapter(adapter, tmp_path / "cache.jsonl", "signature")
    assert second.predict(examples[0]).cache_hit is True
    assert adapter.calls == 1


def test_runtime_failure_is_preserved_in_failure_report() -> None:
    class FailingAdapter(VisionAdapter):
        name = "failure"
        execution_mode = "real"

        def predict(self, example):
            raise RuntimeError("intentional failure")

    _, examples = load_evaluation_config(Path("configs/vlm_ablation.fixture.yaml"))
    result = evaluate_adapter(FailingAdapter(), examples[:1], iou_threshold=0.5)
    assert result["metrics"]["runtime_error_count"] == 1
    assert result["runtime_errors"][0]["error_type"] == "RuntimeError"


def test_qwen_coordinate_scale_repair_is_explicit_and_non_mutating() -> None:
    document = {
        "scene_summary": "worker",
        "findings": [
            {
                "person_box": [20, 100, 800, 950],
                "violation": "no_vest",
                "evidence": "visible",
                "confidence": "medium",
            }
        ],
        "uncertainties": [],
        "recommended_action": "review",
    }
    normalized, repairs = normalize_qwen_coordinate_scale(document)
    assert repairs == 1
    assert normalized["findings"][0]["person_box"] == [0.02, 0.1, 0.8, 0.95]
    assert document["findings"][0]["person_box"] == [20, 100, 800, 950]


def test_record_level_recomputation_rejects_tampered_metric() -> None:
    run = {
        "metrics": {
            "examples": 1,
            "grounded_true_positives": 1,
            "predicted_findings": 1,
            "target_findings": 1,
            "grounding_precision": 1.0,
            "grounding_recall": 1.0,
            "grounding_f1": 1.0,
        },
        "records": [
            {
                "schema_valid": True,
                "prediction": {"findings": [{"violation": "no_vest"}]},
                "target": {"findings": [{"violation": "no_vest"}]},
                "matched_grounded_findings": 1,
            }
        ],
    }
    recomputed = recompute_run_metrics(run)
    verify_report_summary(run, recomputed)
    run["metrics"]["grounding_f1"] = 0.6
    with pytest.raises(ValueError, match="record-level recomputation"):
        verify_report_summary(run, recomputed)
