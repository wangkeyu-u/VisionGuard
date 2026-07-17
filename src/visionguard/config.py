from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
)
SPLIT_ALIASES = {"train": ("train",), "val": ("val", "validation"), "test": ("test",)}
SAFETY_TARGET_CLASSES = (
    "person",
    "helmet",
    "vest",
    "gloves",
    "boots",
    "no_helmet",
    "no_vest",
    "no_gloves",
    "no_boots",
)
SAFETY_CLASS_NAME_MAPPING = {
    "Human": "person",
    "Helmet": "helmet",
    "helmet": "helmet",
    "hat": "helmet",
    "Safety Vest": "vest",
    "vest": "vest",
    "Gloves": "gloves",
    "gloves": "gloves",
    "Safety Boot": "boots",
    "boots": "boots",
    "no hat": "no_helmet",
    "no vest": "no_vest",
    "no gloves": "no_gloves",
    "no boot": "no_boots",
    "no boots": "no_boots",
}
SAFETY_REMOVED_CLASSES = frozenset({"glasses"})
FINAL_TARGET_CLASSES = SAFETY_TARGET_CLASSES[:7]
FINAL_REMOVED_CLASSES = frozenset({"no_gloves", "no_boots"})


@dataclass(frozen=True)
class DatasetValidationConfig:
    data_path: Path
    output_dir: Path
    hash_algorithm: str = "sha256"

    def resolved(self) -> DatasetValidationConfig:
        return DatasetValidationConfig(
            data_path=self.data_path.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            hash_algorithm=self.hash_algorithm,
        )


@dataclass(frozen=True)
class DatasetVisualizationConfig:
    data_path: Path
    output_dir: Path
    samples_per_split: int = 8
    seed: int = 42

    def resolved(self) -> DatasetVisualizationConfig:
        if self.samples_per_split < 1:
            raise ValueError("samples_per_split must be at least 1.")
        return DatasetVisualizationConfig(
            data_path=self.data_path.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            samples_per_split=self.samples_per_split,
            seed=self.seed,
        )


@dataclass(frozen=True)
class DatasetRemapConfig:
    data_path: Path
    output_dir: Path

    def resolved(self) -> DatasetRemapConfig:
        return DatasetRemapConfig(
            data_path=self.data_path.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
        )


@dataclass(frozen=True)
class DatasetFreezeConfig:
    data_path: Path
    output_dir: Path
    seed: int = 42
    overwrite: bool = False
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10

    def resolved(self) -> DatasetFreezeConfig:
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("All split ratios must be greater than zero.")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")
        return DatasetFreezeConfig(
            data_path=self.data_path.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            seed=self.seed,
            overwrite=self.overwrite,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
