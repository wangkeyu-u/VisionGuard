from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from visionguard.dataset import load_dataset_definition
from visionguard.environment import collect_environment_info
from visionguard.fingerprint import verify_frozen_dataset

FROZEN_DATASET_SHA256 = "6d40b6e5d09cc4e1ed3a6d8d9a8a88259f1e18dc9f117d664fa5694473db3bb2"


@dataclass(frozen=True)
class TrainingConfig:
    data: Path
    model: str = "yolo11n.pt"
    epochs: int = 30
    imgsz: int = 512
    batch: int = 4
    patience: int = 8
    workers: int = 0
    seed: int = 42
    device: str = "auto"
    project: Path = Path("outputs/experiments")
    name: str = "baseline_yolo11n_512"
    expected_fingerprint: str = FROZEN_DATASET_SHA256

    def resolved(self) -> TrainingConfig:
        if self.epochs < 1 or self.imgsz < 32 or self.batch < 1:
            raise ValueError("epochs and batch must be positive; imgsz must be at least 32.")
        if self.patience < 0 or self.workers < 0:
            raise ValueError("patience and workers cannot be negative.")
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty.")
        return TrainingConfig(
            data=self.data.expanduser().resolve(),
            model=self.model,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            patience=self.patience,
            workers=self.workers,
            seed=self.seed,
            device=resolve_device(self.device),
            project=self.project.expanduser().resolve(),
            name=self.name,
            expected_fingerprint=self.expected_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data"] = str(self.data)
        values["project"] = str(self.project)
        return values


def resolve_device(requested: str) -> str:
    import torch

    normalized = requested.strip().lower()
    if normalized == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was explicitly requested but is unavailable in this PyTorch runtime."
        )
    return requested


def git_commit_hash(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable (repository has no commit)"
    return result.stdout.strip()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _resolved_ultralytics_data_yaml(source: Path, destination: Path) -> Path:
    """Materialize a private dataset copy so Ultralytics cannot write into frozen data."""
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid dataset YAML: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {source}")
    definition = load_dataset_definition(source)
    runtime_root = destination.parent / "runtime_dataset"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    split_directories = {"train": "train", "val": "valid", "test": "test"}
    for split, output_split in split_directories.items():
        sources = definition.split_sources[split]
        if len(sources) != 1 or not sources[0].is_dir():
            raise ValueError(
                "Read-only Ultralytics runtime preparation requires one image directory per split."
            )
        source_images = sources[0]
        source_labels = source_images.parent / "labels"
        if not source_labels.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {source_labels}")
        shutil.copytree(source_images, runtime_root / output_split / "images")
        shutil.copytree(source_labels, runtime_root / output_split / "labels")
    payload["path"] = str(runtime_root)
    payload["train"] = "train/images"
    payload["val"] = "valid/images"
    payload["test"] = "test/images"
    _write_yaml(destination, payload)
    return destination


def _archive_incomplete_attempt(run_dir: Path) -> None:
    metadata_path = run_dir / "training_run.json"
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if metadata.get("status") == "completed":
        return
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(attempts_dir.glob("attempt_*.json"))) + 1
    status = "failed" if metadata.get("status") == "failed" else "interrupted"
    shutil.copy2(metadata_path, attempts_dir / f"attempt_{index}_{status}.json")
    log_path = run_dir / "training.log"
    if log_path.is_file():
        shutil.copy2(log_path, attempts_dir / f"attempt_{index}_{status}.log")
        log_path.unlink()


def train_yolo(config: TrainingConfig, project_root: Path) -> dict[str, Any]:
    """Verify the frozen dataset, train YOLO, and persist auditable run metadata."""
    config = config.resolved()
    run_dir = config.project / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    _archive_incomplete_attempt(run_dir)
    _write_yaml(run_dir / "training_config.yaml", config.to_dict())
    ultralytics_data = _resolved_ultralytics_data_yaml(config.data, run_dir / "resolved_data.yaml")

    verification = verify_frozen_dataset(config.data, config.expected_fingerprint)
    environment = collect_environment_info().to_dict()
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "git_commit": git_commit_hash(project_root),
        "environment": environment,
        "dataset_fingerprint": verification.to_dict(),
        "config": config.to_dict(),
        "run_directory": str(run_dir),
        "ultralytics_data_yaml": str(ultralytics_data),
    }
    metadata_path = run_dir / "training_run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _set_seeds(config.seed)

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    log_path = run_dir / "training.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
    started = time.perf_counter()
    try:
        model = YOLO(config.model)
        model.train(
            data=str(ultralytics_data),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            patience=config.patience,
            workers=config.workers,
            seed=config.seed,
            device=config.device,
            project=str(config.project),
            name=config.name,
            pretrained=True,
            deterministic=True,
            plots=True,
            exist_ok=True,
            verbose=True,
        )
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    else:
        metadata["status"] = "completed"
    finally:
        metadata["ended_at"] = datetime.now(UTC).isoformat()
        metadata["duration_seconds"] = round(time.perf_counter() - started, 6)
        results_path = run_dir / "results.csv"
        if results_path.is_file():
            metadata["completed_epochs"] = max(
                0, len(results_path.read_text(encoding="utf-8").splitlines()) - 1
            )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.removeHandler(handler)
        handler.close()
    return metadata


def resume_yolo(
    checkpoint: Path,
    data: Path,
    expected_fingerprint: str,
    device: str,
) -> dict[str, Any]:
    """Resume optimizer/scheduler state from last.pt after verifying the frozen source data."""
    checkpoint = checkpoint.expanduser().resolve()
    data = data.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    resolved_device = resolve_device(device)
    verification = verify_frozen_dataset(data, expected_fingerprint)

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    checkpoint_args = model.ckpt.get("train_args", {})
    runtime_data = Path(str(checkpoint_args.get("data", ""))).expanduser()
    if not runtime_data.is_file():
        raise FileNotFoundError(
            "The checkpoint's read-only runtime data YAML is missing: " f"{runtime_data}"
        )
    checkpoint_epoch = int(model.ckpt.get("epoch", -1))
    total_epochs = int(checkpoint_args.get("epochs", 0))
    if checkpoint_epoch + 1 >= total_epochs:
        raise RuntimeError(
            f"Checkpoint already completed {checkpoint_epoch + 1}/{total_epochs} epochs."
        )
    run_dir = checkpoint.parent.parent
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "checkpoint": str(checkpoint),
        "resume_from_epoch": checkpoint_epoch + 2,
        "target_total_epochs": total_epochs,
        "device": resolved_device,
        "dataset_fingerprint": verification.to_dict(),
    }
    metadata_path = run_dir / "resume_run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    try:
        model.train(resume=True, device=resolved_device)
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    else:
        metadata["status"] = "completed"
    finally:
        metadata["ended_at"] = datetime.now(UTC).isoformat()
        metadata["duration_seconds"] = round(time.perf_counter() - started, 6)
        results_path = run_dir / "results.csv"
        if results_path.is_file():
            metadata["completed_epochs"] = max(
                0, len(results_path.read_text(encoding="utf-8").splitlines()) - 1
            )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
