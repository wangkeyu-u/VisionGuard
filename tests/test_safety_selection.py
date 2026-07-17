from __future__ import annotations

from visionguard.safety_selection import add_safety_scores, select_safety_record


def record(name: str, helmet: float, vest: float, overall: float) -> dict[str, object]:
    return {
        "checkpoint": name,
        "overall_map50_95": overall,
        "per_class": {
            "no_helmet": {"ap50_95": helmet},
            "no_vest": {"ap50_95": vest},
        },
    }


def test_safety_score_penalizes_collapsing_one_violation_class() -> None:
    collapsed = add_safety_scores(record("collapsed.pt", 0.40, 0.0, 0.60))
    balanced = add_safety_scores(record("balanced.pt", 0.20, 0.24, 0.52))

    assert collapsed["safety_hmean_map50_95"] == 0.0
    assert balanced["safety_hmean_map50_95"] > 0.0
    assert select_safety_record([collapsed, balanced])["checkpoint"] == "balanced.pt"


def test_safety_selection_uses_overall_map_only_as_final_tiebreak() -> None:
    lower_overall = record("safety-first.pt", 0.25, 0.25, 0.48)
    higher_overall = record("global-first.pt", 0.25, 0.25, 0.55)

    assert select_safety_record([lower_overall, higher_overall])["checkpoint"] == (
        "global-first.pt"
    )
