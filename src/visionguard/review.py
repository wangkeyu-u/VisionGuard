from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVIEW_STATUSES = {"pending", "accepted", "rejected", "corrected"}
REVIEW_CLASSES = {"person", "helmet", "vest", "gloves", "boots", "no_helmet", "no_vest"}


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

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["predictions"] = json.loads(record.pop("predictions_json"))
        record["corrections"] = json.loads(record.pop("corrections_json"))
        return record

    def enqueue(self, session_id: str, report: dict[str, Any]) -> dict[str, Any]:
        timestamp = _now()
        predictions = report.get("detections", [])
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reviews (
                    session_id, source_name, media_type, source_url, result_url, report_url,
                    predictions_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        timestamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE reviews SET status = ?, corrections_json = ?, reviewer = ?, notes = ?,
                    updated_at = ?, reviewed_at = ? WHERE id = ?
                """,
                (status, json.dumps(corrections), reviewer, notes, timestamp, timestamp, review_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown review item: {review_id}")
        return self.get(review_id)

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
