from __future__ import annotations

import csv
import json
from pathlib import Path

from visionguard.evaluation import build_metrics_report, write_metrics_exports


def test_metric_exports_include_overall_and_per_class_values(tmp_path: Path) -> None:
    per_class = [
        {
            "class_id": 0,
            "class_name": "person",
            "precision": 0.8,
            "recall": 0.7,
            "ap50": 0.75,
            "ap50_95": 0.5,
        }
    ]
    report = build_metrics_report(
        {0: "person"},
        [0.8, 0.7, 0.75, 0.5],
        per_class,
        10,
        42,
        {"model": "best.pt", "data": "data.yaml", "device": "cpu", "imgsz": 512},
    )

    exports = write_metrics_exports(report, tmp_path)

    payload = json.loads(exports["json"].read_text(encoding="utf-8"))
    assert payload["metrics"]["map50_95"] == 0.5
    assert payload["total_prediction_boxes"] == 42
    with exports["csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["class_name"] == "person"
    assert "mAP@50-95" in exports["markdown"].read_text(encoding="utf-8")
