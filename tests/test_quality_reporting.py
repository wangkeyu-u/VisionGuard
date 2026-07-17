from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionguard.quality_reporting import build_quality_comparison


def make_run(root: Path, name: str, helmet: float, vest: float, overall: float) -> Path:
    run = root / name
    run.mkdir()
    harmonic = 2 * helmet * vest / (helmet + vest)
    selection = {
        "selected": {
            "checkpoint": str(run / "weights" / "best_safety.pt"),
            "overall_map50_95": overall,
            "per_class": {
                "no_helmet": {"ap50_95": helmet},
                "no_vest": {"ap50_95": vest},
            },
            "safety_macro_map50_95": (helmet + vest) / 2,
            "safety_hmean_map50_95": harmonic,
        }
    }
    (run / "safety_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (run / "training_config.yaml").write_text(
        "imgsz: 512\nclass_weight_power: 0.5\nexpected_fingerprint: abc\n",
        encoding="utf-8",
    )
    return run


def test_quality_comparison_selects_balanced_safety_improvement(tmp_path: Path) -> None:
    baseline = make_run(tmp_path, "baseline", 0.20, 0.30, 0.50)
    candidate = make_run(tmp_path, "candidate", 0.25, 0.34, 0.49)

    report = build_quality_comparison(
        [baseline, candidate], tmp_path / "report", baseline_name="baseline"
    )

    assert report["selected_run"] == "candidate"
    assert report["test_data_used"] is False
    assert report["runs"][1]["delta_vs_baseline"]["no_helmet_map50_95"] == pytest.approx(
        0.05
    )
    assert (tmp_path / "report" / "QUALITY_COMPARISON.md").is_file()


def test_quality_comparison_rejects_safety_gain_that_breaks_global_guardrail(
    tmp_path: Path,
) -> None:
    baseline = make_run(tmp_path, "baseline", 0.20, 0.30, 0.50)
    collapsed = make_run(tmp_path, "collapsed", 0.40, 0.45, 0.45)

    report = build_quality_comparison(
        [baseline, collapsed], tmp_path / "report", baseline_name="baseline"
    )

    assert report["selected_run"] == "baseline"
    assert report["runs"][1]["passes_overall_guardrail"] is False
