from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionguard.fingerprint import verify_frozen_dataset
from visionguard.training import _resolved_ultralytics_data_yaml, resolve_device

SAFETY_CLASSES = ("no_helmet", "no_vest")


@dataclass(frozen=True)
class SafetySelectionConfig:
    run_dir: Path
    data: Path
    expected_fingerprint: str
    imgsz: int = 512
    batch: int = 1
    workers: int = 0
    device: str = "auto"

    def resolved(self) -> SafetySelectionConfig:
        if self.imgsz < 32 or self.batch < 1 or self.workers < 0:
            raise ValueError("imgsz must be at least 32, batch positive, and workers non-negative")
        return SafetySelectionConfig(
            run_dir=self.run_dir.expanduser().resolve(),
            data=self.data.expanduser().resolve(),
            expected_fingerprint=self.expected_fingerprint,
            imgsz=self.imgsz,
            batch=self.batch,
            workers=self.workers,
            device=resolve_device(self.device),
        )


def _harmonic_mean(values: list[float]) -> float:
    return len(values) / sum(1 / value for value in values) if values and all(values) else 0.0


def add_safety_scores(record: dict[str, Any]) -> dict[str, Any]:
    per_class = record.get("per_class", {})
    rare_values = [float(per_class.get(name, {}).get("ap50_95", 0.0)) for name in SAFETY_CLASSES]
    enriched = dict(record)
    enriched["safety_macro_map50_95"] = sum(rare_values) / len(rare_values)
    enriched["safety_hmean_map50_95"] = _harmonic_mean(rare_values)
    return enriched


def select_safety_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no checkpoint records were supplied")
    enriched = [add_safety_scores(record) for record in records]
    return max(
        enriched,
        key=lambda record: (
            record["safety_hmean_map50_95"],
            record["safety_macro_map50_95"],
            float(record.get("overall_map50_95", 0.0)),
        ),
    )


def _checkpoint_candidates(run_dir: Path) -> list[Path]:
    weights = run_dir / "weights"
    if not weights.is_dir():
        raise FileNotFoundError(f"weights directory does not exist: {weights}")
    paths = [weights / "best.pt", weights / "last.pt", *sorted(weights.glob("epoch*.pt"))]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if path.is_file() and resolved not in seen:
            candidates.append(path)
            seen.add(resolved)
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found in {weights}")
    return candidates


def _metric_record(checkpoint: Path, metrics: Any, names: dict[int, str]) -> dict[str, Any]:
    maps = metrics.box.maps.tolist() if hasattr(metrics.box.maps, "tolist") else list(metrics.box.maps)
    per_class = {
        names[index]: {"ap50_95": float(value)}
        for index, value in enumerate(maps)
        if index in names
    }
    return {
        "checkpoint": str(checkpoint),
        "overall_map50_95": float(metrics.box.map),
        "overall_map50": float(metrics.box.map50),
        "per_class": per_class,
    }


def select_safety_checkpoint(config: SafetySelectionConfig) -> dict[str, Any]:
    """Select a checkpoint on validation safety-class AP without touching test data."""
    config = config.resolved()
    verification = verify_frozen_dataset(config.data, config.expected_fingerprint)
    checkpoints = _checkpoint_candidates(config.run_dir)
    runtime_data = config.run_dir / "resolved_data.yaml"
    if not runtime_data.is_file():
        runtime_data = _resolved_ultralytics_data_yaml(
            config.data, config.run_dir / "selection_data.yaml"
        )
    from ultralytics import YOLO

    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(runtime_data),
            split="val",
            imgsz=config.imgsz,
            batch=config.batch,
            workers=config.workers,
            device=config.device,
            plots=False,
            verbose=False,
        )
        records.append(add_safety_scores(_metric_record(checkpoint, metrics, model.names)))
    selected = select_safety_record(records)
    destination = config.run_dir / "weights" / "best_safety.pt"
    shutil.copy2(selected["checkpoint"], destination)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_split": "validation",
        "test_data_used": False,
        "selection_rule": (
            "maximize harmonic mean of no_helmet and no_vest AP@50-95; "
            "tie-break by their macro AP, then overall AP"
        ),
        "data_fingerprint": verification.to_dict(),
        "imgsz": config.imgsz,
        "device": config.device,
        "checkpoints": records,
        "selected": {**selected, "copied_to": str(destination)},
    }
    output = config.run_dir / "safety_selection.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = [
        "# Safety checkpoint selection",
        "",
        "Selection uses validation only. Test data is not loaded.",
        "",
        "| Checkpoint | Overall mAP | no_helmet AP | no_vest AP | Safety H-mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        rows.append(
            f"| {Path(record['checkpoint']).name} | {record['overall_map50_95']:.4f} | "
            f"{record['per_class']['no_helmet']['ap50_95']:.4f} | "
            f"{record['per_class']['no_vest']['ap50_95']:.4f} | "
            f"{record['safety_hmean_map50_95']:.4f} |"
        )
    rows.extend(
        [
            "",
            f"Selected: `{Path(selected['checkpoint']).name}` → `weights/best_safety.pt`.",
        ]
    )
    (config.run_dir / "SAFETY_SELECTION.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return report
