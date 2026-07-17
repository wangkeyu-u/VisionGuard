from __future__ import annotations

import json
import mimetypes
import os
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from visionguard.demo_server import _safe_child
from visionguard.vlm import normalized_xywh_to_xyxy, read_jsonl, validate_inspection

YOLO_CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "vest",
    3: "gloves",
    4: "boots",
    5: "no_helmet",
    6: "no_vest",
}
SPLITS = ("train", "dev", "test")
DEFAULT_TARGETS = {"train": 40, "dev": 15, "test": 20}
QUALITY_TAGS = {"blur", "occlusion", "low_light", "crowded", "small_person"}
ANNOTATION_PROTOCOL_VERSION = "full-visible-person-v2"


@dataclass(frozen=True)
class ReviewServerConfig:
    project_root: Path
    candidates: Path
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 7861
    reviewer: str = ""
    max_request_kb: int = 256


def parse_yolo_labels(path: Path) -> list[dict[str, Any]]:
    """Read a YOLO label file into normalized 0–1000 xyxy annotations."""
    if not path.is_file():
        return []
    annotations: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            box = normalized_xywh_to_xyxy(float(value) for value in parts[1:5])
        except (TypeError, ValueError):
            continue
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        annotations.append(
            {
                "class_id": class_id,
                "class_name": YOLO_CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "box": box,
                "source_line": line_number,
            }
        )
    return annotations


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records
    )
    if payload:
        payload += "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class ReviewStore:
    """Thread-safe access to candidates and human-reviewed JSONL manifests."""

    def __init__(self, config: ReviewServerConfig) -> None:
        self.config = config
        self.project_root = config.project_root.expanduser().resolve()
        self.candidates_path = config.candidates.expanduser().resolve()
        self.data_dir = config.data_dir.expanduser().resolve()
        self._lock = threading.RLock()
        self._candidates: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self.reload_candidates()

    def reload_candidates(self) -> None:
        candidates, issues = read_jsonl(self.candidates_path)
        if issues:
            raise ValueError("; ".join(str(issue) for issue in issues))
        by_id: dict[str, dict[str, Any]] = {}
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = candidate.get("id")
            split = candidate.get("source_split")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"{self.candidates_path}:{index}: candidate id is required")
            if candidate_id in by_id:
                raise ValueError(f"{self.candidates_path}:{index}: duplicate id {candidate_id}")
            if split not in SPLITS:
                raise ValueError(f"{self.candidates_path}:{index}: invalid source_split {split!r}")
            image = self._project_file(candidate.get("image"), "image")
            if not image.is_file():
                raise FileNotFoundError(f"candidate image does not exist: {image}")
            label_value = candidate.get("label")
            if label_value is not None:
                self._project_file(label_value, "label")
            by_id[candidate_id] = candidate
        self._candidates = candidates
        self._by_id = by_id

    def _project_file(self, value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"candidate {field} must be a non-empty relative path")
        path = (self.project_root / value).resolve()
        if path != self.project_root and self.project_root not in path.parents:
            raise ValueError(f"candidate {field} escapes the project root")
        return path

    def _read_records(self, filename: str) -> list[dict[str, Any]]:
        records, issues = read_jsonl(self.data_dir / filename)
        missing_only = issues and all(issue.line == 0 for issue in issues)
        if issues and not missing_only:
            raise ValueError("; ".join(str(issue) for issue in issues))
        return records

    def _review_state(self) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        gold_by_id: dict[str, dict[str, Any]] = {}
        statuses: dict[str, str] = {}
        for split in SPLITS:
            for record in self._read_records(f"{split}.jsonl"):
                record_id = record.get("id")
                if isinstance(record_id, str):
                    gold_by_id[record_id] = record
                    statuses[record_id] = "approved"
        image_to_id = {
            str(candidate.get("image")): str(candidate["id"]) for candidate in self._candidates
        }
        for decision in self._read_records("review_log.jsonl"):
            candidate_id = decision.get("candidate_id")
            if not isinstance(candidate_id, str):
                candidate_id = image_to_id.get(str(decision.get("image")))
            if candidate_id in self._by_id and decision.get("decision") == "exclude":
                statuses[candidate_id] = "excluded"
            elif candidate_id in self._by_id and decision.get("decision") == "approve":
                statuses[candidate_id] = "approved"
        return gold_by_id, statuses

    def _public_candidate(
        self,
        candidate: dict[str, Any],
        status: str,
        saved_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        candidate_id = str(candidate["id"])
        split = str(candidate["source_split"])
        blind = split == "test"
        label_path = candidate.get("label")
        annotations = []
        if not blind and isinstance(label_path, str):
            annotations = parse_yolo_labels(self._project_file(label_path, "label"))
        result: dict[str, Any] = {
            "id": candidate_id,
            "source_split": split,
            "source_group": candidate.get("source_group"),
            "status": status,
            "blind": blind,
            "image_url": f"/review/images/{candidate_id}",
            "category": "blind_review" if blind else candidate.get("category"),
            "source_classes": [] if blind else candidate.get("source_classes", []),
            "source_annotations": annotations,
        }
        if saved_record is not None:
            result["saved_target"] = saved_record.get("target")
            result["saved_review"] = saved_record.get("review", {})
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            gold_by_id, statuses = self._review_state()
            candidates = [
                self._public_candidate(
                    candidate,
                    statuses.get(str(candidate["id"]), "pending"),
                    gold_by_id.get(str(candidate["id"])),
                )
                for candidate in self._candidates
            ]
            gold_records = list(gold_by_id.values())
            split_counts = Counter(str(record.get("source_split")) for record in gold_records)
            composition = Counter()
            quality = Counter()
            for record in gold_records:
                findings = record.get("target", {}).get("findings", [])
                composition["violation" if findings else "compliant"] += 1
                uncertainties = record.get("target", {}).get("uncertainties", [])
                if uncertainties:
                    composition["uncertain"] += 1
                for tag in record.get("review", {}).get("quality_tags", []):
                    quality[str(tag)] += 1
            return {
                "reviewer": self.config.reviewer,
                "output_dir": self._display_path(self.data_dir),
                "targets": DEFAULT_TARGETS,
                "counts": {split: split_counts.get(split, 0) for split in SPLITS},
                "status_counts": dict(Counter(item["status"] for item in candidates)),
                "composition": dict(composition),
                "quality": dict(quality),
                "quality_tags": sorted(QUALITY_TAGS),
                "candidates": candidates,
            }

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def image_path(self, candidate_id: str) -> Path:
        candidate = self._by_id.get(candidate_id)
        if candidate is None:
            raise KeyError("Unknown candidate")
        return self._project_file(candidate["image"], "image")

    def save_decision(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        candidate_id = payload.get("candidate_id")
        decision = payload.get("decision")
        if not isinstance(candidate_id, str) or candidate_id not in self._by_id:
            raise ValueError("candidate_id is unknown")
        if decision not in {"approve", "exclude"}:
            raise ValueError("decision must be approve or exclude")
        candidate = self._by_id[candidate_id]
        reviewer = str(payload.get("reviewer") or self.config.reviewer).strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        quality_tags = payload.get("quality_tags", [])
        if not isinstance(quality_tags, list) or any(tag not in QUALITY_TAGS for tag in quality_tags):
            raise ValueError(f"quality_tags must use {sorted(QUALITY_TAGS)}")

        with self._lock:
            if decision == "approve":
                self._approve(candidate, payload, reviewer, quality_tags)
            else:
                self._exclude(candidate, payload, reviewer, quality_tags)
            return self.snapshot()

    def _approve(
        self,
        candidate: dict[str, Any],
        payload: dict[str, Any],
        reviewer: str,
        quality_tags: list[str],
    ) -> None:
        target = payload.get("target")
        errors = validate_inspection(target)
        if errors:
            raise ValueError("; ".join(errors))
        if payload.get("person_box_contract_confirmed") is not True:
            raise ValueError("confirm that every person box covers the full visible person")
        person_boxes = payload.get("person_boxes")
        box_errors = _validate_person_boxes(person_boxes, target)
        if box_errors:
            raise ValueError("; ".join(box_errors))
        split = str(candidate["source_split"])
        group = str(candidate.get("source_group") or "")
        if not group:
            raise ValueError("candidate source_group is missing")
        for other_split in SPLITS:
            if other_split == split:
                continue
            if any(record.get("source_group") == group for record in self._read_records(f"{other_split}.jsonl")):
                raise ValueError(f"source_group already exists in {other_split}; refusing split leakage")
        timestamp = datetime.now(UTC).isoformat()
        record = {
            "id": candidate["id"],
            "image": candidate["image"],
            "source_group": group,
            "source_split": split,
            "target": target,
            "review": {
                "reviewer": reviewer,
                "reviewed_at": timestamp,
                "quality_tags": sorted(set(quality_tags)),
                "annotation_protocol_version": ANNOTATION_PROTOCOL_VERSION,
                "person_box_contract_confirmed": True,
                "person_boxes": person_boxes,
            },
        }
        path = self.data_dir / f"{split}.jsonl"
        records = self._read_records(path.name)
        records = [item for item in records if item.get("id") != candidate["id"]]
        records.append(record)
        _atomic_write_jsonl(path, records)
        self._append_review_log(candidate, "approve", reviewer, payload.get("note", ""), quality_tags)

    def _exclude(
        self,
        candidate: dict[str, Any],
        payload: dict[str, Any],
        reviewer: str,
        quality_tags: list[str],
    ) -> None:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("an exclusion reason is required")
        split_path = self.data_dir / f"{candidate['source_split']}.jsonl"
        records = self._read_records(split_path.name)
        remaining = [item for item in records if item.get("id") != candidate["id"]]
        if len(remaining) != len(records):
            _atomic_write_jsonl(split_path, remaining)
        self._append_review_log(candidate, "exclude", reviewer, reason, quality_tags)

    def _append_review_log(
        self,
        candidate: dict[str, Any],
        decision: str,
        reviewer: str,
        reason: Any,
        quality_tags: list[str],
    ) -> None:
        path = self.data_dir / "review_log.jsonl"
        records = self._read_records(path.name)
        records.append(
            {
                "candidate_id": candidate["id"],
                "image": candidate["image"],
                "source_label": candidate.get("source_classes", []),
                "source_split": candidate["source_split"],
                "source_group": candidate.get("source_group"),
                "decision": decision,
                "reason": str(reason).strip(),
                "reviewer": reviewer,
                "quality_tags": sorted(set(quality_tags)),
                "reviewed_at": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_write_jsonl(path, records)


def _validate_person_boxes(person_boxes: Any, target: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(person_boxes, list) or not person_boxes:
        return ["person_boxes must contain at least one full visible person box"]
    normalized: list[list[int]] = []
    for index, box in enumerate(person_boxes):
        prefix = f"person_boxes[{index}]"
        if not isinstance(box, list) or len(box) != 4:
            errors.append(f"{prefix} must contain four integers")
            continue
        if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
            errors.append(f"{prefix} values must be integers")
            continue
        if any(value < 0 or value > 1000 for value in box):
            errors.append(f"{prefix} values must be between 0 and 1000")
            continue
        if box[0] >= box[2] or box[1] >= box[3]:
            errors.append(f"{prefix} must satisfy x1 < x2 and y1 < y2")
            continue
        normalized.append(box)
    if errors or not isinstance(target, dict):
        return errors
    if len({tuple(box) for box in normalized}) != len(normalized):
        errors.append("person_boxes must not contain duplicates")
    if normalized != sorted(normalized, key=lambda box: (box[0], box[1])):
        errors.append("person_boxes must be ordered from left to right")
    expected = {f"p{index}": box for index, box in enumerate(normalized, start=1)}
    for finding_index, finding in enumerate(target.get("findings", [])):
        if not isinstance(finding, dict):
            continue
        person_id = finding.get("person_id")
        if person_id not in expected:
            errors.append(f"findings[{finding_index}].person_id does not reference person_boxes")
        elif finding.get("person_box") != expected[person_id]:
            errors.append(f"findings[{finding_index}].person_box does not match {person_id}")
    return errors


def _handler_factory(
    store: ReviewStore,
    static_dir: Path,
    max_request_kb: int,
) -> type[BaseHTTPRequestHandler]:
    request_limit = max_request_kb * 1024

    class ReviewRequestHandler(BaseHTTPRequestHandler):
        server_version = "VisionGuardReview/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path in {"/", "/review"}:
                self._send_file(static_dir / "index.html")
                return
            if path == "/api/review/snapshot":
                self._send_json(store.snapshot())
                return
            if path.startswith("/review/static/"):
                try:
                    file_path = _safe_child(static_dir, path.removeprefix("/review/static/"))
                except ValueError as exc:
                    self._send_error(400, str(exc))
                    return
                self._send_file(file_path)
                return
            if path.startswith("/review/images/"):
                candidate_id = path.removeprefix("/review/images/")
                try:
                    self._send_file(store.image_path(candidate_id))
                except KeyError as exc:
                    self._send_error(404, str(exc))
                return
            if path == "/api/review/health":
                query = parse_qs(parsed.query)
                self._send_json({"ok": True, "echo": query.get("echo", [""])[0]})
                return
            self._send_error(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/review/decision":
                self._send_error(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("request body is empty")
                if length > request_limit:
                    self._send_error(413, f"request exceeds {max_request_kb} KB")
                    return
                content_type = self.headers.get("Content-Type", "")
                if "application/json" not in content_type.lower():
                    raise ValueError("Content-Type must be application/json")
                payload = json.loads(self.rfile.read(length))
                self._send_json(store.save_decision(payload))
            except json.JSONDecodeError as exc:
                self._send_error(400, f"invalid JSON: {exc.msg}")
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                self._send_error(400, str(exc))

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_error(404, "File not found")
                return
            payload = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[review] {self.address_string()} - {format_string % args}")

    return ReviewRequestHandler


def run_review_server(config: ReviewServerConfig) -> None:
    project_root = config.project_root.expanduser().resolve()
    static_dir = project_root / "review" / "static"
    if not (static_dir / "index.html").is_file():
        raise FileNotFoundError(f"review frontend is missing: {static_dir / 'index.html'}")
    store = ReviewStore(config)
    handler = _handler_factory(store, static_dir, config.max_request_kb)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    print(f"VisionGuard review desk ready: http://{config.host}:{config.port}")
    print(f"Candidates: {config.candidates} | reviewer={config.reviewer or 'required in UI'}")
    print("Test candidates run in blind mode. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
