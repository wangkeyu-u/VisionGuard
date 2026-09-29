#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frozen VisionGuard Dev ablations.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=METRICS_JSON",
        help="Named metrics file. Repeat once per ablation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "outputs" / "dev_formal" / "EXPERIMENT_CONFIG.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "vlm" / "outputs" / "dev_formal",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        name, separator, path_value = value.partition("=")
        if not separator or not name or not path_value:
            raise ValueError(f"invalid --run {value!r}; expected NAME=METRICS_JSON")
        if name in names:
            raise ValueError(f"duplicate run name: {name}")
        names.add(name)
        runs.append((name, Path(path_value).expanduser().resolve()))
    return runs


def _row(name: str, path: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "metrics_path": str(path),
        "records": metrics["evaluated_records"],
        "precision": metrics["overall"]["precision"],
        "recall": metrics["overall"]["recall"],
        "f1": metrics["overall"]["f1"],
        "no_helmet_f1": metrics["per_class"]["no_helmet"]["f1"],
        "no_vest_f1": metrics["per_class"]["no_vest"]["f1"],
        "hallucination_rate": metrics["hallucination_rate_on_clear_images"],
        "parsed_json_rate": metrics["parsed_json_rate"],
        "schema_valid_rate": metrics["schema_valid_rate"],
        "repair_records": metrics["schema_repair"]["records_with_repair_attempt"],
        "runtime_errors": metrics["runtime_error_count"],
        "mean_latency_ms": metrics["latency"]["mean_ms"],
        "p95_latency_ms": metrics["latency"]["p95_ms"],
    }


def build_comparison(
    named_paths: list[tuple[str, Path]], config: dict[str, Any]
) -> dict[str, Any]:
    rows = [_row(name, path, _load_json(path)) for name, path in named_paths]
    record_counts = {row["records"] for row in rows}
    if len(record_counts) != 1:
        raise ValueError(f"runs do not evaluate the same record count: {sorted(record_counts)}")
    selected = max(
        rows,
        key=lambda row: (
            row["f1"],
            -row["hallucination_rate"],
            -row["mean_latency_ms"],
        ),
    )
    return {
        "experiment": config["experiment"],
        "selection_split": config["selection_split"],
        "test_predictions_allowed": config["test_predictions_allowed"],
        "selection_rule": "highest overall Dev F1; then lower hallucination; then lower latency",
        "selected_run": selected["name"],
        "runs": rows,
        "decision": (
            "Freeze YOLO-only for the held-out Test evaluation. The current Qwen3-VL-only and "
            "grounded variants are rejected because they reduce Dev F1 and sharply increase "
            "hallucinations on clear compliant images."
        ),
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(comparison: dict[str, Any], path: Path) -> None:
    rows = comparison["runs"]
    lines = [
        "# VisionGuard formal Dev ablation",
        "",
        f"Selected configuration: **{comparison['selected_run']}**.",
        "",
        "Test predictions were not generated. Selection used the frozen Dev split only.",
        "",
        "| Run | Precision | Recall | F1 | no_helmet F1 | no_vest F1 | "
        "Clear-image hallucination | Mean latency | Repairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['no_helmet_f1']:.3f} | "
            f"{row['no_vest_f1']:.3f} | {_percent(row['hallucination_rate'])} | "
            f"{row['mean_latency_ms']:.1f} ms | {row['repair_records']} |"
        )
    selected = next(row for row in rows if row["name"] == comparison["selected_run"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            comparison["decision"],
            "",
            "YOLO-only is the only configuration that currently satisfies the task objective: "
            f"Dev F1 {selected['f1']:.3f}, clear-image hallucination "
            f"{_percent(selected['hallucination_rate'])}, and mean latency "
            f"{selected['mean_latency_ms']:.1f} ms/image.",
            "",
            "The grounded result is a negative but useful finding: detector overlays and textual "
            "context did not help the 2B VLM reason conservatively. They reduced localization F1 "
            "and did not reduce compliant-image hallucinations.",
            "",
            "## Frozen next step",
            "",
            "Run the selected YOLO-only configuration exactly once on Test. Do not tune confidence, "
            "prompt, checkpoint, or IoU threshold after viewing Test output.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    comparison = build_comparison(_parse_runs(args.run), _load_json(args.config.resolve()))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "DEV_ABLATION_REPORT.json"
    markdown_path = output_dir / "DEV_ABLATION_REPORT.md"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_markdown(comparison, markdown_path)
    print(json.dumps(comparison, indent=2))
    print(f"Reports saved to {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
