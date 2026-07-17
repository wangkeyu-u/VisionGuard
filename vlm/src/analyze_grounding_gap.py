#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.vlm import box_iou, read_jsonl  # noqa: E402

CLASSES = ("no_helmet", "no_vest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose image-level classification versus person-box grounding gaps."
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="NAME=GOLD_JSONL,PREDICTIONS_JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _prf(values: list[int]) -> dict[str, float | int]:
    true_positive, false_positive, false_negative = values
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def _box_area(box: list[int]) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / 1_000_000


def diagnose(gold_path: Path, predictions_path: Path) -> dict[str, Any]:
    gold, gold_issues = read_jsonl(gold_path.resolve())
    predictions, prediction_issues = read_jsonl(predictions_path.resolve())
    issues = [*gold_issues, *prediction_issues]
    if issues:
        raise ValueError("; ".join(str(issue) for issue in issues))
    predictions_by_id = {str(record.get("id")): record for record in predictions}
    per_class = {class_name: [0, 0, 0] for class_name in CLASSES}
    any_violation = [0, 0, 0]
    max_ious: list[float] = []
    gold_areas: list[float] = []
    prediction_areas: list[float] = []

    for gold_record in gold:
        prediction_record = predictions_by_id.get(str(gold_record.get("id")), {})
        prediction = prediction_record.get("prediction") or {}
        gold_findings = gold_record["target"].get("findings", [])
        predicted_findings = prediction.get("findings", [])
        gold_classes = {finding["violation"] for finding in gold_findings}
        predicted_classes = {finding["violation"] for finding in predicted_findings}
        for class_name in CLASSES:
            if class_name in gold_classes and class_name in predicted_classes:
                per_class[class_name][0] += 1
            elif class_name not in gold_classes and class_name in predicted_classes:
                per_class[class_name][1] += 1
            elif class_name in gold_classes and class_name not in predicted_classes:
                per_class[class_name][2] += 1
        if gold_classes and predicted_classes:
            any_violation[0] += 1
        elif not gold_classes and predicted_classes:
            any_violation[1] += 1
        elif gold_classes and not predicted_classes:
            any_violation[2] += 1

        for finding in gold_findings:
            box = finding["person_box"]
            gold_areas.append(_box_area(box))
            candidates = [
                predicted
                for predicted in predicted_findings
                if predicted.get("violation") == finding.get("violation")
            ]
            max_ious.append(
                max(
                    (box_iou(box, predicted["person_box"]) for predicted in candidates),
                    default=0.0,
                )
            )
        prediction_areas.extend(
            _box_area(finding["person_box"]) for finding in predicted_findings
        )

    return {
        "gold_path": str(gold_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "records": len(gold),
        "image_level_presence": {
            "any_violation": _prf(any_violation),
            "per_class": {name: _prf(values) for name, values in per_class.items()},
        },
        "grounding": {
            "gold_findings": len(max_ious),
            "best_same_class_iou_mean": round(statistics.fmean(max_ious), 6),
            "best_same_class_iou_median": round(statistics.median(max_ious), 6),
            "best_same_class_iou_ge_0_1": sum(value >= 0.1 for value in max_ious),
            "best_same_class_iou_ge_0_3": sum(value >= 0.3 for value in max_ious),
            "gold_box_area_median": round(statistics.median(gold_areas), 6),
            "prediction_box_area_median": round(statistics.median(prediction_areas), 6),
        },
    }


def _parse_pair(value: str) -> tuple[str, Path, Path]:
    name, separator, paths = value.partition("=")
    gold, comma, predictions = paths.partition(",")
    if not separator or not comma or not name or not gold or not predictions:
        raise ValueError(f"invalid --pair {value!r}")
    return name, Path(gold), Path(predictions)


def main() -> int:
    args = parse_args()
    report = {
        name: diagnose(gold, predictions)
        for name, gold, predictions in (_parse_pair(value) for value in args.pair)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Diagnostic saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
