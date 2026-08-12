from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVIEW_STATUSES = {"pending", "accepted", "rejected", "corrected"}
REVIEW_CLASSES = {"person", "helmet", "vest", "gloves", "boots", "no_helmet", "no_vest"}
VIOLATION_CLASSES = {"no_helmet", "no_vest"}
ANNOTATION_PROTOCOL_VERSION = "visionguard-review-v1"
DATASET_SPLITS = {"train", "dev", "test", "inference"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def validate_corrections(corrections: Any) -> list[dict[str, Any]]:
    if not isinstance(corrections, list):
        raise ValueError("corrections must be a list")
    validated = []
    for index, correction in enumerate(corrections):
        if not isinstance(correction, dict):
            raise ValueError(f"corrections[{index}] must be an object")
        action = correction.get("action")
        if action not in {"add", "update", "delete"}:
            raise ValueError(f"corrections[{index}].action must be add, update, or delete")
        original_index = correction.get("original_index")
        if action in {"update", "delete"} and (
            not isinstance(original_index, int) or original_index < 0
        ):
            raise ValueError(f"corrections[{index}].original_index is required")
        if action != "delete":
            if correction.get("class_name") not in REVIEW_CLASSES:
                raise ValueError(f"corrections[{index}].class_name is invalid")
            box = correction.get("xyxy")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(not isinstance(value, (int, float)) for value in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError(f"corrections[{index}].xyxy must be ordered coordinates")
        validated.append(correction)
    return validated


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_review_protocol(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("protocol_version") != ANNOTATION_PROTOCOL_VERSION:
        errors.append("annotation protocol version is missing or unsupported")
    if record.get("dataset_split") not in DATASET_SPLITS:
        errors.append("dataset_split is invalid")
    if record.get("status") not in {"accepted", "corrected"}:
        errors.append("review status is not accepted or corrected")
    if not str(record.get("reviewer", "")).strip():
        errors.append("reviewer is required")
    if record.get("dirty"):
        errors.append("dirty sample is excluded from evaluation")
    if record.get("excluded"):
        errors.append("explicitly excluded sample cannot enter evaluation")
    corrections = record.get("corrections", [])
    try:
        validate_corrections(corrections)
    except ValueError as exc:
        errors.append(str(exc))
    reviewed_annotations = corrections or record.get("predictions", [])
    expected_violations = {
        annotation["class_name"]
        for annotation in reviewed_annotations
        if annotation.get("action") != "delete"
        and annotation.get("class_name") in VIOLATION_CLASSES
    }
    confirmations = record.get("violation_confirmations", [])
    if not isinstance(confirmations, list):
        errors.append("violation_confirmations must be a list")
        confirmations = []
    confirmed = {
        item.get("class_name")
        for item in confirmations
        if isinstance(item, dict) and item.get("confirmed") is True
    }
    missing = expected_violations - confirmed
    if missing:
        errors.append(f"violations require explicit confirmation: {sorted(missing)}")
    if record.get("dataset_split") == "test" and not record.get("blind_review"):
        errors.append("test samples require blind_review=true")
    return errors


class ReviewStore:
    def __init__(self, database: Path) -> None:
        self.database = database.expanduser().resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    source_name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    result_url TEXT NOT NULL,
                    report_url TEXT NOT NULL,
                    predictions_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    corrections_json TEXT NOT NULL DEFAULT '[]',
                    reviewer TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_at TEXT
                )
                """
            )
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(reviews)").fetchall()
            }
            migrations = {
                "dataset_split": "TEXT NOT NULL DEFAULT 'inference'",
                "blind_review": "INTEGER NOT NULL DEFAULT 0",
                "ground_truth_json": "TEXT NOT NULL DEFAULT 'null'",
                "dirty": "INTEGER NOT NULL DEFAULT 0",
                "excluded": "INTEGER NOT NULL DEFAULT 0",
                "exclusion_reason": "TEXT NOT NULL DEFAULT ''",
                "violation_confirmations_json": "TEXT NOT NULL DEFAULT '[]'",
                "protocol_version": f"TEXT NOT NULL DEFAULT '{ANNOTATION_PROTOCOL_VERSION}'",
            }
            for column, definition in migrations.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE reviews ADD COLUMN {column} {definition}")

    @staticmethod
    def _record(row: sqlite3.Row, *, include_ground_truth: bool = False) -> dict[str, Any]:
        record = dict(row)
        record["predictions"] = json.loads(record.pop("predictions_json"))
        record["corrections"] = json.loads(record.pop("corrections_json"))
        ground_truth = json.loads(record.pop("ground_truth_json"))
        record["violation_confirmations"] = json.loads(
            record.pop("violation_confirmations_json")
        )
        record["dirty"] = bool(record["dirty"])
        record["excluded"] = bool(record["excluded"])
        record["blind_review"] = bool(record["blind_review"])
        if include_ground_truth:
            record["ground_truth"] = ground_truth
        return record

    def enqueue(self, session_id: str, report: dict[str, Any]) -> dict[str, Any]:
        timestamp = _now()
        predictions = report.get("detections", [])
        dataset_split = str(report.get("dataset_split", "inference"))
        if dataset_split not in DATASET_SPLITS:
            raise ValueError(f"Unknown dataset split: {dataset_split}")
        blind_review = dataset_split == "test"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reviews (
                    session_id, source_name, media_type, source_url, result_url, report_url,
                    predictions_json, created_at, updated_at, dataset_split, blind_review,
                    ground_truth_json, protocol_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (
                    session_id,
                    report["source_name"],
                    report["media_type"],
                    report["source_url"],
                    report["result_url"],
                    report["report_url"],
                    json.dumps(predictions),
                    timestamp,
                    timestamp,
                    dataset_split,
                    int(blind_review),
                    json.dumps(report.get("ground_truth")),
                    ANNOTATION_PROTOCOL_VERSION,
                ),
            )
        return self.get_by_session(session_id)

    def get_by_session(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown review session: {session_id}")
        return self._record(row)

    def get(self, review_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown review item: {review_id}")
        return self._record(row)

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status: {status}")
        query = "SELECT * FROM reviews"
        parameters: tuple[str, ...] = ()
        if status:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record(row) for row in rows]

    def update(self, review_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        status = payload.get("status")
        if status not in REVIEW_STATUSES - {"pending"}:
            raise ValueError("status must be accepted, rejected, or corrected")
        corrections = validate_corrections(payload.get("corrections", []))
        if status == "corrected" and not corrections:
            raise ValueError("corrected reviews require at least one correction")
        if status != "corrected" and corrections:
            raise ValueError("corrections require corrected status")
        reviewer = str(payload.get("reviewer", "")).strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        notes = str(payload.get("notes", "")).strip()
        dirty = bool(payload.get("dirty", False))
        excluded = bool(payload.get("excluded", status == "rejected"))
        exclusion_reason = str(payload.get("exclusion_reason", "")).strip()
        if (dirty or excluded) and not exclusion_reason:
            raise ValueError("dirty or excluded samples require an exclusion_reason")
        confirmations = payload.get("violation_confirmations", [])
        if not isinstance(confirmations, list):
            raise ValueError("violation_confirmations must be a list")
        for index, confirmation in enumerate(confirmations):
            if (
                not isinstance(confirmation, dict)
                or confirmation.get("class_name") not in VIOLATION_CLASSES
                or not isinstance(confirmation.get("confirmed"), bool)
            ):
                raise ValueError(f"violation_confirmations[{index}] is invalid")
        timestamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE reviews SET status = ?, corrections_json = ?, reviewer = ?, notes = ?,
                    dirty = ?, excluded = ?, exclusion_reason = ?,
                    violation_confirmations_json = ?, protocol_version = ?,
                    updated_at = ?, reviewed_at = ? WHERE id = ?
                """,
                (
                    status,
                    json.dumps(corrections),
                    reviewer,
                    notes,
                    int(dirty),
                    int(excluded),
                    exclusion_reason,
                    json.dumps(confirmations),
                    ANNOTATION_PROTOCOL_VERSION,
                    timestamp,
                    timestamp,
                    review_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown review item: {review_id}")
        return self.get(review_id)

    def evaluation_records(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM reviews ORDER BY id").fetchall()
        eligible = []
        for row in rows:
            record = self._record(row, include_ground_truth=True)
            errors = validate_review_protocol(record)
            if errors:
                continue
            eligible.append(
                {
                    "id": record["session_id"],
                    "source_name": record["source_name"],
                    "dataset_split": record["dataset_split"],
                    "blind_review": record["blind_review"],
                    "predictions": record["predictions"],
                    "reviewed_corrections": record["corrections"],
                    "violation_confirmations": record["violation_confirmations"],
                    "ground_truth": record["ground_truth"],
                    "review": {
                        "reviewer": record["reviewer"],
                        "reviewed_at": record["reviewed_at"],
                        "protocol_version": record["protocol_version"],
                    },
                }
            )
        return eligible

    def export_evaluation_jsonl(self, path: Path) -> int:
        records = self.evaluation_records()
        atomic_write_jsonl(path, records)
        return len(records)

    def export_json(self) -> bytes:
        payload = {
            "schema_version": 1,
            "exported_at": _now(),
            "reviews": self.list(),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    def export_csv(self) -> bytes:
        buffer = io.StringIO()
        fields = [
            "id", "session_id", "source_name", "media_type", "status", "reviewer",
            "notes", "prediction_count", "correction_count", "created_at", "reviewed_at",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for record in self.list():
            writer.writerow(
                {
                    **{field: record.get(field) for field in fields},
                    "prediction_count": len(record["predictions"]),
                    "correction_count": len(record["corrections"]),
                }
            )
        return buffer.getvalue().encode("utf-8")
