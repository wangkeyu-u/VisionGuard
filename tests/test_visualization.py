from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from visionguard.config import DatasetVisualizationConfig
from visionguard.reporting import write_dataset_preview_reports
from visionguard.visualization import class_color, visualize_dataset


def _build_visualization_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    split_colors = {"train": 30, "val": 80, "test": 130}
    for split, base_color in split_colors.items():
        for index in range(3):
            image_path = root / "images" / split / f"frame_{index}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (120, 90), (base_color + index, 20, 30)).save(image_path)
            label_path = root / "labels" / split / f"frame_{index}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                f"{index} 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.2 0.2\n",
                encoding="utf-8",
            )

    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "validation": "images/val",
                "test": "images/test",
                "names": ["person", "helmet", "safety_vest"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def _selected_sources(report: dict[str, object]) -> dict[str, list[str]]:
    splits = report["splits"]
    assert isinstance(splits, dict)
    return {
        split: [sample["source_image"] for sample in stats["samples"]]
        for split, stats in splits.items()
    }


def test_visualization_writes_samples_and_preview_reports(tmp_path: Path) -> None:
    data_path = _build_visualization_dataset(tmp_path)
    output_dir = tmp_path / "preview"

    report = visualize_dataset(DatasetVisualizationConfig(data_path, output_dir, 2, 17))
    json_path, markdown_path = write_dataset_preview_reports(report, output_dir)

    assert report["summary"]["images_visualized"] == 6
    assert report["summary"]["objects_visualized"] == 12
    assert report["summary"]["average_objects_per_image"] == 2.0
    assert report["warnings"] == []
    for split in ("train", "val", "test"):
        assert report["splits"][split]["images_visualized"] == 2
        assert len(list((output_dir / split).glob("sample_*.png"))) == 2

    assert json.loads(json_path.read_text(encoding="utf-8"))["configuration"]["seed"] == 17
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "VisionGuard Dataset Preview Report" in markdown
    assert "Average objects per image" in markdown

    rendered_colors: set[tuple[int, int, int]] = set()
    for output_image in output_dir.glob("*/*.png"):
        with Image.open(output_image) as image:
            rgb_image = image.convert("RGB")
            rendered_colors.update(
                rgb_image.getpixel((x, y))
                for y in range(rgb_image.height)
                for x in range(rgb_image.width)
            )
    for class_name, count in report["summary"]["class_distribution"].items():
        if count:
            class_id = {"person": 0, "helmet": 1, "safety_vest": 2}[class_name]
            assert class_color(class_id) in rendered_colors


def test_sampling_is_deterministic_and_preserves_unrelated_files(tmp_path: Path) -> None:
    data_path = _build_visualization_dataset(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    sentinel = first_output / "train" / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")

    first = visualize_dataset(DatasetVisualizationConfig(data_path, first_output, 2, 99))
    second = visualize_dataset(DatasetVisualizationConfig(data_path, second_output, 2, 99))

    assert _selected_sources(first) == _selected_sources(second)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert len({class_color(class_id) for class_id in range(3)}) == 3


def test_visualization_rejects_nonpositive_sample_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        visualize_dataset(
            DatasetVisualizationConfig(tmp_path / "data.yaml", tmp_path / "out", 0, 42)
        )
