from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visionguard.fingerprint import sha256_file, verify_frozen_dataset
from visionguard.quality_dataset import (
    LabelBox,
    QualityDatasetConfig,
    build_quality_dataset,
    sampling_decision,
)


def _make_frozen_source(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    samples = {
        "train": {
            "normal": "0 0.5 0.5 0.4 0.8\n",
            "rare": "5 0.5 0.2 0.2 0.2\n",
            "rare_small": "6 0.5 0.5 0.05 0.05\n",
        },
        "valid": {"valid_sample": "5 0.5 0.5 0.2 0.2\n"},
        "test": {"test_sample": "6 0.5 0.5 0.2 0.2\n"},
    }
    files: list[dict[str, str]] = []
    digest = hashlib.sha256()
    for split, records in samples.items():
        for stem, label_text in records.items():
            image = root / split / "images" / f"{stem}.jpg"
            label = root / split / "labels" / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{split}-{stem}".encode())
            label.write_text(label_text, encoding="utf-8")
            relative_image = image.relative_to(root)
            relative_label = label.relative_to(root)
            image_hash = sha256_file(image)
            label_hash = sha256_file(label)
            files.append(
                {
                    "image": str(relative_image),
                    "label": str(relative_label),
                    "image_sha256": image_hash,
                    "label_sha256": label_hash,
                }
            )
            digest.update(
                f"{relative_image}|{image_hash}|{relative_label}|{label_hash}\n".encode()
            )
    fingerprint = digest.hexdigest()
    (root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "names: [person, helmet, vest, gloves, boots, no_helmet, no_vest]\n",
        encoding="utf-8",
    )
    (root / "freeze_manifest.json").write_text(
        json.dumps(
            {"hash_algorithm": "sha256", "dataset_sha256": fingerprint, "files": files}
        ),
        encoding="utf-8",
    )
    return root / "data.yaml", fingerprint


def test_sampling_decision_escalates_small_rare_boxes() -> None:
    regular = [LabelBox(5, 0.5, 0.5, 0.2, 0.2)]
    hard = [LabelBox(6, 0.5, 0.5, 0.05, 0.05)]

    assert sampling_decision(regular, 3, 4, 0.01, 8, 0.1) == (3, ["rare_violation"])
    assert sampling_decision(hard, 3, 4, 0.01, 8, 0.1) == (
        4,
        ["rare_violation", "small_rare_box"],
    )


def test_quality_dataset_resamples_only_train_and_freezes_output(tmp_path: Path) -> None:
    data, fingerprint = _make_frozen_source(tmp_path)
    output = tmp_path / "quality"

    report = build_quality_dataset(
        QualityDatasetConfig(
            data=data,
            output_dir=output,
            expected_fingerprint=fingerprint,
            rare_multiplier=3,
            hard_multiplier=4,
        )
    )

    assert report["split_stats"]["train"]["original_images"] == 3
    assert report["split_stats"]["train"]["derived_images"] == 8
    assert report["split_stats"]["valid"]["derived_images"] == 1
    assert report["split_stats"]["test"]["derived_images"] == 1
    assert report["split_stats"]["train"]["derived_class_annotations"] == {
        "person": 1,
        "no_helmet": 3,
        "no_vest": 4,
    }
    assert len(list((output / "train" / "images").glob("*__vg_repeat_*"))) == 5
    assert report["audit"]["validation_content_hashes_match_source"] is True
    assert report["audit"]["test_content_hashes_match_source"] is True
    assert report["audit"]["test_was_not_used_for_sampling"] is True

    verification = verify_frozen_dataset(
        output / "data.yaml", report["freeze"]["dataset_sha256"]
    )
    assert verification.passed
