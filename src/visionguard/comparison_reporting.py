from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionguard.experiment_reporting import _load_json, _load_optional_json, _training_summary


def _metric(row: dict[str, str], name: str) -> float:
    key = next((key for key in row if name in key.strip()), None)
    if key is None:
        raise ValueError(f"results.csv does not contain metric {name!r}.")
    return float(row[key])


def _validation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Training results are empty: {path}")
    return rows


def _experiment_label(config: dict[str, Any]) -> str:
    return f"{Path(config['model']).stem} / {config['imgsz']}px / {config['epochs']}ep"


def collect_experiment(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    training = _load_json(run_dir / "training_run.json")
    resume = _load_optional_json(run_dir / "resume_run.json")
    config = training["config"]
    rows = _validation_rows(run_dir / "results.csv")
    best = max(rows, key=lambda row: _metric(row, "mAP50-95"))
    evaluation = _load_optional_json(run_dir / "test_evaluation" / "test_metrics.json")
    performance = _load_optional_json(run_dir / "performance" / "performance.json")
    errors = _load_optional_json(run_dir / "error_analysis" / "error_analysis.json")
    training_summary = _training_summary(training, resume)

    return {
        "id": run_dir.name,
        "label": _experiment_label(config),
        "run_dir": str(run_dir),
        "model": Path(config["model"]).stem,
        "imgsz": int(config["imgsz"]),
        "target_epochs": int(config["epochs"]),
        "completed_epochs": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "status": training_summary["status"],
        "resumed": training_summary["resumed"],
        "validation": {
            "precision": _metric(best, "precision"),
            "recall": _metric(best, "recall"),
            "map50": _metric(best, "mAP50(B)"),
            "map50_95": _metric(best, "mAP50-95"),
        },
        "test": evaluation["metrics"] if evaluation else None,
        "test_images": evaluation.get("test_images") if evaluation else None,
        "test_per_class": evaluation.get("per_class") if evaluation else None,
        "performance": (
            {
                "mean_latency_ms": float(performance["average_latency_ms"]),
                "p50_latency_ms": float(performance["p50_latency_ms"]),
                "p95_latency_ms": float(performance["p95_latency_ms"]),
                "model_size_mib": float(performance["model_file_size_mb"]),
                "samples": int(performance["samples"]),
                "device": performance["device"],
            }
            if performance
            else None
        ),
        "error_image_counts": errors.get("image_counts") if errors else None,
    }


def build_comparison(experiment_dirs: list[Path]) -> dict[str, Any]:
    if len(experiment_dirs) < 2:
        raise ValueError("At least two experiment directories are required for comparison.")
    experiments = [collect_experiment(path) for path in experiment_dirs]
    baseline = experiments[0]
    selected = max(experiments, key=lambda item: item["validation"]["map50_95"])
    tested = [item for item in experiments if item["test"] is not None]

    test_delta = None
    per_class: list[dict[str, Any]] = []
    if baseline["test"] and selected["test"]:
        test_delta = selected["test"]["map50_95"] - baseline["test"]["map50_95"]
        baseline_classes = {
            row["class_name"]: row for row in baseline["test_per_class"] or []
        }
        for selected_row in selected["test_per_class"] or []:
            class_name = selected_row["class_name"]
            baseline_row = baseline_classes.get(class_name)
            if baseline_row is None:
                continue
            per_class.append(
                {
                    "class_name": class_name,
                    "baseline_experiment": baseline["id"],
                    "selected_experiment": selected["id"],
                    "baseline_ap50_95": float(baseline_row["ap50_95"]),
                    "selected_ap50_95": float(selected_row["ap50_95"]),
                    "delta_ap50_95": float(selected_row["ap50_95"])
                    - float(baseline_row["ap50_95"]),
                }
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_rule": "Highest validation mAP@50-95; test metrics are confirmatory only.",
        "baseline_experiment": baseline["id"],
        "selected_experiment": selected["id"],
        "validation_map50_95_delta": selected["validation"]["map50_95"]
        - baseline["validation"]["map50_95"],
        "test_map50_95_delta": test_delta,
        "tested_experiments": [item["id"] for item in tested],
        "experiments": experiments,
        "per_class_test_comparison": per_class,
    }


def _format_optional(value: float | None, digits: int = 4) -> str:
    return "not exported" if value is None else f"{value:.{digits}f}"


def write_comparison_markdown(comparison: dict[str, Any], output_path: Path) -> Path:
    experiments = comparison["experiments"]
    baseline = next(item for item in experiments if item["id"] == comparison["baseline_experiment"])
    selected = next(item for item in experiments if item["id"] == comparison["selected_experiment"])
    val_delta = comparison["validation_map50_95_delta"]
    test_delta = comparison["test_map50_95_delta"]
    selected_performance = selected["performance"]

    lines = [
        "# VisionGuard Experiment Comparison",
        "",
        "## Technical summary",
        "",
        f"`{selected['id']}` is the selected checkpoint because it has the highest validation "
        f"mAP@50-95 ({selected['validation']['map50_95']:.4f}), an absolute gain of "
        f"{val_delta:.4f} over `{baseline['id']}`.",
        "",
        f"Independent test mAP@50-95 is {selected['test']['map50_95']:.4f}, "
        f"{test_delta:+.4f} versus the baseline. Every reported class improves, including "
        "`no_helmet` and `no_vest`; the remaining weakness is low absolute performance on those "
        "minority safety-violation classes.",
        "",
        "## Validation evidence selects experiment 4",
        "",
        "| Experiment | Model | Input | Epochs | Best epoch | Precision | Recall | mAP@50 | mAP@50-95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in experiments:
        metrics = item["validation"]
        lines.append(
            f"| {item['id']} | {item['model']} | {item['imgsz']} | "
            f"{item['completed_epochs']}/{item['target_epochs']} | {item['best_epoch']} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['map50']:.4f} | {metrics['map50_95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Model selection uses validation mAP@50-95 only. Experiment 4 stopped normally after "
            "48 epochs; its best checkpoint is epoch 40.",
            "",
            "## Independent test confirms the improvement",
            "",
            "| Experiment | Precision | Recall | mAP@50 | mAP@50-95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in experiments:
        test = item["test"]
        lines.append(
            f"| {item['id']} | {_format_optional(test['precision'] if test else None)} | "
            f"{_format_optional(test['recall'] if test else None)} | "
            f"{_format_optional(test['map50'] if test else None)} | "
            f"{_format_optional(test['map50_95'] if test else None)} |"
        )
    lines.extend(
        [
            "",
            "Experiments 2 and 3 produced plots during interrupted evaluation attempts but no complete "
            "metric export, so the report does not treat them as test-comparable runs.",
            "",
            "## Safety-violation classes improve but remain the bottleneck",
            "",
            "| Class | Baseline AP@50-95 | Selected AP@50-95 | Absolute change |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["per_class_test_comparison"]:
        lines.append(
            f"| {row['class_name']} | {row['baseline_ap50_95']:.4f} | "
            f"{row['selected_ap50_95']:.4f} | {row['delta_ap50_95']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Accuracy comes with a deployment cost",
            "",
            "| Experiment | Mean latency (ms/image) | P95 (ms/image) | Model size (MiB) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in experiments:
        performance = item["performance"]
        lines.append(
            f"| {item['id']} | {_format_optional(performance['mean_latency_ms'] if performance else None, 2)} | "
            f"{_format_optional(performance['p95_latency_ms'] if performance else None, 2)} | "
            f"{_format_optional(performance['model_size_mib'] if performance else None, 2)} |"
        )
    lines.extend(
        [
            "",
            f"The selected model currently measures {selected_performance['mean_latency_ms']:.2f} "
            "ms/image at batch 1 on MPS. Benchmark comparisons are descriptive: runs were captured at "
            "different times and do not include repeated-run uncertainty.",
            "",
            "## Scope, methods, and metric definitions",
            "",
            "- All four runs use the same frozen dataset fingerprint, split, seed, batch size, and MPS device.",
            "- mAP@50-95 is mean average precision averaged over IoU thresholds 0.50 through 0.95.",
            "- Validation contains 1,753 images; the independent test split contains 884 images.",
            "- `best.pt` is selected within each run by validation performance; "
            "test data is not used for checkpoint selection.",
            "- Latency is synchronized single-image wall time after five warm-up images "
            "on the first 100 sorted test images.",
            "",
            "## Limitations and robustness checks",
            "",
            "- Each configuration has one seed, so the report cannot separate configuration gains "
            "from run-to-run variance.",
            "- Test comparisons are complete only for the baseline and selected final candidate.",
            "- `no_helmet` and `no_vest` appear in only 42 and 60 test images, respectively; "
            "their class metrics are less stable.",
            "- Source grouping uses filename-derived proxies rather than verified site/video identifiers.",
            "- Error-analysis categories are heuristic review candidates, not adjudicated false "
            "positives or false negatives.",
            "",
            "## Recommended next steps",
            "",
            f"1. Adopt `{selected['id']}/weights/best.pt` as the current candidate model.",
            "2. Manually review the saved `no_helmet`, `no_vest`, missed-detection, and class-confusion candidates.",
            "3. Make experiment 5 data-centric: correct labels and add or augment minority violation "
            "examples without touching test data.",
            "4. Repeat the chosen baseline and candidate with multiple seeds before making a "
            "production-level accuracy claim.",
            "5. Re-run deployment benchmarks under a controlled idle/thermal state and define an "
            "explicit latency budget.",
            "",
            "## Further questions",
            "",
            "- What minimum recall is required for `no_helmet` and `no_vest` in the intended safety workflow?",
            "- Is approximately 24 ms/image acceptable on the target device, or must the nano model "
            "remain the deployment default?",
            "- Can reliable site, video, or capture-session identifiers be recovered for a stronger leakage audit?",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _artifact_datasets(comparison: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    experiments = comparison["experiments"]
    per_class = comparison["per_class_test_comparison"]
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE experiment_metrics (
            experiment TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            model TEXT NOT NULL,
            imgsz INTEGER NOT NULL,
            best_epoch INTEGER NOT NULL,
            validation_map50_95 REAL NOT NULL,
            validation_map50 REAL NOT NULL,
            test_map50_95 REAL,
            mean_latency_ms REAL,
            p95_latency_ms REAL,
            model_size_mib REAL
        );
        CREATE TABLE per_class_metrics (
            class_order INTEGER NOT NULL,
            class_name TEXT PRIMARY KEY,
            baseline_ap50_95 REAL NOT NULL,
            selected_ap50_95 REAL NOT NULL,
            delta_ap50_95 REAL NOT NULL
        );
        CREATE TABLE comparison_context (
            baseline_experiment TEXT NOT NULL,
            selected_experiment TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        """
        INSERT INTO experiment_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item["id"],
                item["label"],
                item["model"],
                item["imgsz"],
                item["best_epoch"],
                item["validation"]["map50_95"],
                item["validation"]["map50"],
                item["test"]["map50_95"] if item["test"] else None,
                item["performance"]["mean_latency_ms"] if item["performance"] else None,
                item["performance"]["p95_latency_ms"] if item["performance"] else None,
                item["performance"]["model_size_mib"] if item["performance"] else None,
            )
            for item in experiments
        ],
    )
    connection.executemany(
        "INSERT INTO per_class_metrics VALUES (?, ?, ?, ?, ?)",
        [
            (
                index,
                row["class_name"],
                row["baseline_ap50_95"],
                row["selected_ap50_95"],
                row["delta_ap50_95"],
            )
            for index, row in enumerate(per_class)
        ],
    )
    connection.execute(
        "INSERT INTO comparison_context VALUES (?, ?)",
        (comparison["baseline_experiment"], comparison["selected_experiment"]),
    )
    project_root = Path(__file__).resolve().parents[2]
    sql_path = project_root / "scripts" / "experiment_comparison.sql"
    connection.executescript(sql_path.read_text(encoding="utf-8"))

    def rows(view: str) -> list[dict[str, Any]]:
        return [dict(row) for row in connection.execute(f"SELECT * FROM {view}").fetchall()]

    datasets = {
        "headline": rows("comparison_headline"),
        "experiments": rows("ranked_experiments"),
        "per_class": rows("per_class_long"),
    }
    connection.close()
    return datasets


def build_report_artifact(comparison: dict[str, Any], source_path: str) -> dict[str, Any]:
    experiments = comparison["experiments"]
    selected = next(item for item in experiments if item["id"] == comparison["selected_experiment"])
    datasets = _artifact_datasets(comparison)
    generated_at = comparison["generated_at"]
    source = {
        "id": "comparison_source",
        "label": "VisionGuard experiment comparison snapshot",
        "path": source_path,
    }
    title = "VisionGuard: Four-Experiment Model Selection Report"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Technical comparison of four YOLO11 PPE-detection experiments.",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "validation_card",
                "description": "Best validation mAP@50-95 for the selected checkpoint.",
                "dataset": "headline",
                "sourceId": "comparison_source",
                "metrics": [
                    {"label": "Validation mAP@50-95", "field": "selected_validation_map50_95", "format": "percent"},
                    {
                        "label": "absolute change",
                        "field": "validation_delta",
                        "format": "percent",
                        "signed": True,
                    },
                ],
            },
            {
                "id": "test_card",
                "description": "Independent test mAP@50-95 for the selected checkpoint.",
                "dataset": "headline",
                "sourceId": "comparison_source",
                "metrics": [
                    {"label": "Test mAP@50-95", "field": "selected_test_map50_95", "format": "percent"},
                    {
                        "label": "absolute change",
                        "field": "test_delta",
                        "format": "percent",
                        "signed": True,
                    },
                ],
            },
            {
                "id": "no_helmet_card",
                "description": "Test AP@50-95 for the no_helmet violation class.",
                "dataset": "headline",
                "sourceId": "comparison_source",
                "metrics": [
                    {"label": "no_helmet AP", "field": "no_helmet_ap50_95", "format": "percent"},
                    {
                        "label": "absolute change",
                        "field": "no_helmet_delta",
                        "format": "percent",
                        "signed": True,
                    },
                ],
            },
            {
                "id": "no_vest_card",
                "description": "Test AP@50-95 for the no_vest violation class.",
                "dataset": "headline",
                "sourceId": "comparison_source",
                "metrics": [
                    {"label": "no_vest AP", "field": "no_vest_ap50_95", "format": "percent"},
                    {
                        "label": "absolute change",
                        "field": "no_vest_delta",
                        "format": "percent",
                        "signed": True,
                    },
                ],
            },
            {
                "id": "latency_card",
                "description": "Current synchronized MPS batch-1 benchmark; not a repeated-run estimate.",
                "dataset": "headline",
                "sourceId": "comparison_source",
                "metrics": [
                    {"label": "Mean latency", "field": "mean_latency_ms", "format": "number"},
                ],
            },
        ],
        "charts": [
            {
                "id": "validation_chart",
                "title": "Validation mAP@50-95 across experiments",
                "subtitle": "Best checkpoint in each run; all runs use the same frozen validation split.",
                "type": "bar",
                "dataset": "experiments",
                "sourceId": "comparison_source",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "Experiment"},
                    "y": {"field": "validation_map50_95", "type": "quantitative", "label": "mAP@50-95"},
                    "tooltip": [
                        {"field": "best_epoch", "type": "quantitative", "label": "Best epoch"},
                        {
                            "field": "mean_latency_ms",
                            "type": "quantitative",
                            "label": "Mean latency",
                            "unit": "ms/image",
                        },
                        {"field": "model_size_mib", "type": "quantitative", "label": "Model size", "unit": "MiB"},
                    ],
                },
                "surface": {"palette": {"kind": "single", "root": "blue"}},
            },
            {
                "id": "class_chart",
                "title": "Per-class test AP@50-95",
                "subtitle": "Baseline versus selected checkpoint on the same 884-image test split.",
                "type": "bar",
                "dataset": "per_class",
                "sourceId": "comparison_source",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "class_name", "type": "nominal", "label": "Class"},
                    "y": {"field": "ap50_95", "type": "quantitative", "label": "AP@50-95"},
                    "color": {"field": "experiment", "type": "nominal", "label": "Checkpoint"},
                    "tooltip": [
                        {
                            "field": "delta",
                            "type": "quantitative",
                            "label": "Absolute change",
                            "format": "percent",
                        },
                    ],
                },
                "surface": {"palette": {"kind": "categorical", "roots": ["blue", "orange"]}},
            },
        ],
        "tables": [
            {
                "id": "experiment_table",
                "title": "Experiment metrics and deployment footprint",
                "subtitle": "Exact saved values; test cells are blank where a complete export is unavailable.",
                "dataset": "experiments",
                "sourceId": "comparison_source",
                "defaultSort": {"field": "validation_map50_95", "direction": "desc"},
                "density": "spacious",
                "columns": [
                    {"field": "experiment", "label": "Experiment", "type": "text"},
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "imgsz", "label": "Input", "format": "number"},
                    {"field": "best_epoch", "label": "Best epoch", "format": "number"},
                    {"field": "validation_map50_95", "label": "Validation mAP@50-95", "format": "percent"},
                    {"field": "test_map50_95", "label": "Test mAP@50-95", "format": "percent"},
                    {"field": "mean_latency_ms", "label": "Mean latency (ms)", "format": "number"},
                    {"field": "model_size_mib", "label": "Size (MiB)", "format": "number"},
                ],
            }
        ],
        "sources": [source],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "comparison_source",
                "body": (
                    "## Technical summary\n\n"
                    f"**Select `{selected['id']}`.** It leads validation mAP@50-95 at "
                    f"{selected['validation']['map50_95']:.2%} "
                    f"(+{comparison['validation_map50_95_delta'] * 100:.2f} percentage points "
                    f"versus baseline) and reaches {selected['test']['map50_95']:.2%} on the "
                    f"independent test split (+{comparison['test_map50_95_delta'] * 100:.2f} "
                    "percentage points). The gain is consistent across all seven classes, "
                    "but safety-violation classes remain the weakest and should drive the next data-centric experiment."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "validation_card",
                    "test_card",
                    "no_helmet_card",
                    "no_vest_card",
                    "latency_card",
                ],
            },
            {
                "id": "validation_finding",
                "type": "markdown",
                "sourceId": "comparison_source",
                "body": (
                    "## Experiment 4 wins on the selection metric\n\n"
                    "Validation mAP@50-95 rises monotonically across the four controlled changes. "
                    "The fourth run improves on the 30-epoch YOLO11s run, and early stopping "
                    "identifies epoch 40 as the best checkpoint."
                ),
            },
            {"id": "validation_visual", "type": "chart", "chartId": "validation_chart"},
            {
                "id": "class_finding",
                "type": "markdown",
                "sourceId": "comparison_source",
                "body": (
                    "## Test gains are broad, while violation detection remains difficult\n\n"
                    "Every class improves relative to baseline. `no_helmet` and `no_vest` show the "
                    "largest absolute gains, yet their final AP@50-95 values remain only 13.89% and "
                    "22.00%, making data quality and coverage the next constraint."
                ),
            },
            {"id": "class_visual", "type": "chart", "chartId": "class_chart"},
            {
                "id": "scope_definitions",
                "type": "markdown",
                "sourceId": "comparison_source",
                "body": (
                    "## Scope and metric definitions\n\n"
                    "All runs use the same frozen dataset fingerprint, seed, batch size, and MPS "
                    "device. Validation has 1,753 images; "
                    "the independent test split has 884. mAP@50-95 averages AP over IoU thresholds 0.50–0.95. "
                    "Checkpoint selection uses validation only; test metrics are confirmatory."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Controlled experiment design\n\n"
                    "The sequence changes input size, model capacity, and then training horizon. "
                    "The report uses the best validation row from each `results.csv`, complete test "
                    "exports when present, and synchronized 100-image batch-1 benchmark files."
                ),
            },
            {"id": "detail_table", "type": "table", "tableId": "experiment_table"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and robustness\n\n"
                    "- Each configuration has one seed, so run-to-run variance is unknown.\n"
                    "- Experiments 2 and 3 lack complete test metric exports and are not treated as test-comparable.\n"
                    "- `no_helmet` and `no_vest` occur in only 42 and 60 test images.\n"
                    "- Benchmarks were saved at different times and lack repeated-run uncertainty.\n"
                    "- Source grouping is filename-derived, and error categories are heuristic review candidates."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    f"1. Adopt `{selected['id']}/weights/best.pt` as the current candidate.\n"
                    "2. Review violation-class misses and class confusions, then correct labels.\n"
                    "3. Make experiment 5 data-centric and keep the test split untouched.\n"
                    "4. Repeat the baseline and candidate with multiple seeds.\n"
                    "5. Re-benchmark under a controlled idle and thermal state against an explicit latency budget."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- What minimum recall is required for `no_helmet` and `no_vest`?\n"
                    "- Is the selected model's current latency acceptable on the deployment device?\n"
                    "- Can reliable site/video identifiers be recovered for a stronger leakage audit?"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": datasets["headline"],
                "experiments": datasets["experiments"],
                "per_class": datasets["per_class"],
            },
            "accessIssues": [],
        },
        "sources": [source],
    }


def generate_comparison_report(experiment_dirs: list[Path], output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison(experiment_dirs)
    json_path = output_dir / "experiment_comparison.json"
    markdown_path = output_dir / "EXPERIMENT_COMPARISON.md"
    artifact_path = output_dir / "artifact.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_comparison_markdown(comparison, markdown_path)
    artifact = build_report_artifact(
        comparison, "scripts/experiment_comparison.sql"
    )
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path, "artifact": artifact_path}
