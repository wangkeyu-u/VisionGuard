from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FingerprintVerification:
    dataset_root: Path
    expected_sha256: str
    manifest_sha256: str
    computed_sha256: str
    files_verified: int

    @property
    def passed(self) -> bool:
        return self.expected_sha256 == self.manifest_sha256 == self.computed_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "expected_sha256": self.expected_sha256,
            "manifest_sha256": self.manifest_sha256,
            "computed_sha256": self.computed_sha256,
            "files_verified": self.files_verified,
            "passed": self.passed,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_dataset(data_path: Path, expected_sha256: str) -> FingerprintVerification:
    """Verify every frozen image/label hash and the ordered dataset digest."""
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")
    dataset_root = data_path.parent
    manifest_path = dataset_root / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen dataset manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError("freeze_manifest.json must use SHA-256.")
    manifest_sha256 = str(manifest.get("dataset_sha256", ""))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("freeze_manifest.json contains no file records.")

    digest = hashlib.sha256()
    for index, record in enumerate(files, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid manifest record at index {index}.")
        image_relative = Path(str(record.get("image", "")))
        label_relative = Path(str(record.get("label", "")))
        if image_relative.is_absolute() or label_relative.is_absolute():
            raise ValueError(f"Manifest paths must be relative (record {index}).")
        image_path = dataset_root / image_relative
        label_path = dataset_root / label_relative
        if not image_path.is_file() or not label_path.is_file():
            missing = image_path if not image_path.is_file() else label_path
            raise FileNotFoundError(f"Frozen dataset file is missing: {missing}")
        image_sha256 = sha256_file(image_path)
        label_sha256 = sha256_file(label_path)
        if image_sha256 != record.get("image_sha256"):
            raise RuntimeError(f"Frozen image hash mismatch: {image_path}")
        if label_sha256 != record.get("label_sha256"):
            raise RuntimeError(f"Frozen label hash mismatch: {label_path}")
        digest.update(f"{image_relative}|{image_sha256}|{label_relative}|{label_sha256}\n".encode())

    verification = FingerprintVerification(
        dataset_root=dataset_root,
        expected_sha256=expected_sha256,
        manifest_sha256=manifest_sha256,
        computed_sha256=digest.hexdigest(),
        files_verified=len(files),
    )
    if not verification.passed:
        raise RuntimeError(
            "Frozen dataset fingerprint mismatch: "
            f"expected={expected_sha256}, manifest={manifest_sha256}, "
            f"computed={verification.computed_sha256}. Training aborted."
        )
    return verification
