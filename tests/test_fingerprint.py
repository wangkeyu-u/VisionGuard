from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from visionguard.fingerprint import sha256_file, verify_frozen_dataset


def _frozen_dataset(tmp_path: Path) -> tuple[Path, str, Path]:
    root = tmp_path / "dataset"
    image_path = root / "train" / "images" / "sample.jpg"
    label_path = root / "train" / "labels" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image bytes")
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text("path: .\n", encoding="utf-8")
    image_relative = Path("train/images/sample.jpg")
    label_relative = Path("train/labels/sample.txt")
    image_hash = sha256_file(image_path)
    label_hash = sha256_file(label_path)
    digest = hashlib.sha256(
        f"{image_relative}|{image_hash}|{label_relative}|{label_hash}\n".encode()
    ).hexdigest()
    manifest = {
        "hash_algorithm": "sha256",
        "dataset_sha256": digest,
        "files": [
            {
                "image": str(image_relative),
                "label": str(label_relative),
                "image_sha256": image_hash,
                "label_sha256": label_hash,
            }
        ],
    }
    (root / "freeze_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "data.yaml", digest, label_path


def test_frozen_dataset_fingerprint_passes(tmp_path: Path) -> None:
    data_path, expected, _ = _frozen_dataset(tmp_path)

    verification = verify_frozen_dataset(data_path, expected)

    assert verification.passed
    assert verification.files_verified == 1


def test_frozen_dataset_fingerprint_rejects_tampering(tmp_path: Path) -> None:
    data_path, expected, label_path = _frozen_dataset(tmp_path)
    label_path.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="label hash mismatch"):
        verify_frozen_dataset(data_path, expected)
