from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _run_record(run_dir: Path) -> dict[str, Any]:
    selection_path = run_dir / "safety_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"safety selection is missing for {run_dir}; run select_safety_checkpoint.py first"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"selection report has no selected checkpoint: {selection_path}")
    training_config: dict[str, Any] = {}
    config_path = run_dir / "training_config.yaml"
    if config_path.is_file():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            training_config = payload
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": selected["checkpoint"],
        "imgsz": training_config.get("imgsz", selection.get("imgsz")),
        "class_weight_power": training_config.get("class_weight_power", 0.0),
        "dataset_fingerprint": training_config.get("expected_fingerprint"),
        "overall_map50_95": selected["overall_map50_95"],
        "no_helmet_map50_95": selected["per_class"]["no_helmet"]["ap50_95"],
        "no_vest_map50_95": selected["per_class"]["no_vest"]["ap50_95"],
        "safety_macro_map50_95": selected["safety_macro_map50_95"],
        "safety_hmean_map50_95": selected["safety_hmean_map50_95"],
    }


def build_quality_comparison(
    run_dirs: list[Path],
    output_dir: Path,
    baseline_name: str | None = None,
    max_overall_regression: float = 0.01,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("at least two run directories are required for a quality comparison")
    if max_overall_regression < 0:
        raise ValueError("max_overall_regression cannot be negative")
    records = [_run_record(path.expanduser().resolve()) for path in run_dirs]
    baseline = next(
        (record for record in records if record["run"] == baseline_name), records[0]
    )
    for record in records:
        record["delta_vs_baseline"] = {
            metric: float(record[metric]) - float(baseline[metric])
            for metric in (
                "overall_map50_95",
                "no_helmet_map50_95",
                "no_vest_map50_95",
                "safety_hmean_map50_95",
            )
        }
        record["passes_overall_guardrail"] = (
            float(record["overall_map50_95"])
            >= float(baseline["overall_map50_95"]) - max_overall_regression
        )
    eligible = [record for record in records if record["passes_overall_guardrail"]]
    selected = max(
        eligible,
        key=lambda record: (
            record["safety_hmean_map50_95"],
            record["safety_macro_map50_95"],
            record["overall_map50_95"],
        ),
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_split": "validation",
        "test_data_used": False,
        "baseline": baseline["run"],
        "selection_metric": "safety_hmean_map50_95",
        "overall_map50_95_guardrail": {
            "maximum_allowed_regression": max_overall_regression,
            "minimum_eligible_value": float(baseline["overall_map50_95"])
            - max_overall_regression,
        },
        "runs": records,
        "selected_run": selected["run"],
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Safety-focused model quality comparison",
        "",
        "All selection metrics are from validation. Test data is not used.",
        "",
        "| Run | Input | cls_pw | Overall mAP | no_helmet AP | no_vest AP | Safety H-mean | Δ H-mean | Guardrail |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['run']} | {record['imgsz']} | {record['class_weight_power']:.2f} | "
            f"{record['overall_map50_95']:.4f} | {record['no_helmet_map50_95']:.4f} | "
            f"{record['no_vest_map50_95']:.4f} | {record['safety_hmean_map50_95']:.4f} | "
            f"{record['delta_vs_baseline']['safety_hmean_map50_95']:+.4f} | "
            f"{'pass' if record['passes_overall_guardrail'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            f"Selected on validation safety H-mean: `{selected['run']}`.",
            "",
            "Do not run the held-out test until this selection and its configuration are frozen.",
        ]
    )
    (output_dir / "QUALITY_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report
