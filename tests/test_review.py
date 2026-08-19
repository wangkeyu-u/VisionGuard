from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from visionguard.demo_server import DemoServerConfig, _handler_factory
from visionguard.review import ReviewStore, atomic_write_jsonl, validate_corrections


def _sample_report() -> dict[str, object]:
    return {
        "source_name": "sample.jpg",
        "media_type": "image",
        "source_url": "/results/session/input.jpg",
        "result_url": "/results/session/annotated.jpg",
        "report_url": "/results/session/report.json",
        "detections": [
            {"class_name": "no_helmet", "confidence": 0.8, "xyxy": [10, 20, 80, 100]}
        ],
    }


def test_review_store_persists_decision_corrections_and_exports(tmp_path: Path) -> None:
    database = tmp_path / "reviews.sqlite3"
    store = ReviewStore(database)
    queued = store.enqueue("session", _sample_report())

    corrected = store.update(
        queued["id"],
        {
            "status": "corrected",
            "reviewer": "qa-user",
            "notes": "Box tightened after visual inspection.",
            "violation_confirmations": [
                {"class_name": "no_helmet", "confirmed": True}
            ],
            "corrections": [
                {
                    "action": "update",
                    "original_index": 0,
                    "class_name": "no_helmet",
                    "xyxy": [12, 22, 75, 96],
                }
            ],
        },
    )

    reloaded = ReviewStore(database).get(queued["id"])
    assert corrected["status"] == reloaded["status"] == "corrected"
    assert reloaded["reviewer"] == "qa-user"
    assert reloaded["corrections"][0]["xyxy"] == [12, 22, 75, 96]
    assert json.loads(store.export_json())["reviews"][0]["status"] == "corrected"
    assert b"prediction_count,correction_count" in store.export_csv()


def test_test_split_is_blind_and_ground_truth_is_not_public(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    report = {
        **_sample_report(),
        "dataset_split": "test",
        "ground_truth": {"private": "must not leak"},
    }
    review = store.enqueue("blind-test", report)
    assert review["blind_review"] is True
    assert "ground_truth" not in review
    assert "private" not in json.dumps(store.list())


def test_eval_gate_allows_only_reviewed_clean_protocol_records(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    clean = store.enqueue("clean", {**_sample_report(), "dataset_split": "dev"})
    dirty = store.enqueue("dirty", {**_sample_report(), "dataset_split": "dev"})
    pending = store.enqueue("pending", {**_sample_report(), "dataset_split": "dev"})
    store.update(
        clean["id"],
        {
            "status": "accepted",
            "reviewer": "reviewer",
            "notes": "checked",
            "corrections": [],
            "violation_confirmations": [
                {"class_name": "no_helmet", "confirmed": True}
            ],
        },
    )
    store.update(
        dirty["id"],
        {
            "status": "accepted",
            "reviewer": "reviewer",
            "notes": "contaminated",
            "corrections": [],
            "dirty": True,
            "exclusion_reason": "label leakage discovered",
            "violation_confirmations": [
                {"class_name": "no_helmet", "confirmed": True}
            ],
        },
    )
    assert pending["status"] == "pending"
    records = store.evaluation_records()
    assert [record["id"] for record in records] == ["clean"]


def test_eval_gate_blocks_unconfirmed_violation(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    review = store.enqueue("unconfirmed", {**_sample_report(), "dataset_split": "dev"})
    store.update(
        review["id"],
        {
            "status": "accepted",
            "reviewer": "reviewer",
            "notes": "forgot confirmation",
            "corrections": [],
            "violation_confirmations": [],
        },
    )
    assert store.evaluation_records() == []


def test_atomic_jsonl_failure_preserves_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text('{"old":true}\n', encoding="utf-8")
    original_replace = Path.replace

    def fail_replace(self: Path, target: Path) -> Path:
        if self.suffix == ".tmp":
            raise OSError("simulated interrupted write")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_jsonl(path, [{"new": True}])
    assert path.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_review_validation_rejects_ambiguous_edits() -> None:
    with pytest.raises(ValueError, match="ordered coordinates"):
        validate_corrections(
            [{"action": "add", "class_name": "person", "xyxy": [20, 0, 10, 30]}]
        )


def test_review_http_queue_patch_and_export(tmp_path: Path) -> None:
    from http.server import ThreadingHTTPServer

    class FakeEngine:
        class Config:
            model = Path("fixture.pt")
            device = "fixture"
            imgsz = 512
            confidence = 0.25

        config = Config()

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("fixture", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    store = ReviewStore(tmp_path / "review.sqlite3")
    review = store.enqueue("session", _sample_report())
    config = DemoServerConfig(model=Path("fixture.pt"), output_dir=output_dir)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler_factory(FakeEngine(), config, static_dir, output_dir, store),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/api/reviews?status=pending")
        response = connection.getresponse()
        queue = json.loads(response.read())
        assert response.status == 200
        assert queue["reviews"][0]["id"] == review["id"]

        body = json.dumps(
            {
                "status": "accepted",
                "reviewer": "api-reviewer",
                "notes": "checked",
                "corrections": [],
                "violation_confirmations": [
                    {"class_name": "no_helmet", "confirmed": True}
                ],
            }
        )
        connection.request(
            "PATCH",
            f"/api/reviews/{review['id']}",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        updated = json.loads(response.read())
        assert response.status == 200
        assert updated["status"] == "accepted"

        connection.request("GET", "/api/reviews/export.csv")
        response = connection.getresponse()
        exported = response.read()
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/csv")
        assert b"api-reviewer" in exported
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
