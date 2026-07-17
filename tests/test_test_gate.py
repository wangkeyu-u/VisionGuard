from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from visionguard.vlm import write_jsonl


def load_gate_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "vlm" / "src" / "gate_test_data.py"
    spec = importlib.util.spec_from_file_location("gate_test_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gold_record(group: str = "new-group") -> dict[str, object]:
    box = [100, 100, 700, 900]
    return {
        "id": "test-item",
        "image": "images/test.jpg",
        "source_group": group,
        "source_split": "test",
        "target": {
            "findings": [
                {
                    "person_id": "p1",
                    "person_box": box,
                    "violation": "no_vest",
                    "evidence": "The visible torso has no high-visibility safety vest.",
                    "confidence": "high",
                }
            ],
            "uncertainties": [],
            "recommended_action": "human_review",
        },
        "review": {
            "reviewer": "Human Reviewer",
            "annotation_protocol_version": "full-visible-person-v2",
            "person_box_contract_confirmed": True,
            "person_boxes": [box],
        },
    }


def test_test_gate_passes_independent_protocol_v2_data(tmp_path: Path) -> None:
    gate = load_gate_module()
    gate.PROJECT_ROOT = tmp_path
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "test.jpg").write_bytes(b"image")
    test_path = tmp_path / "test.jsonl"
    history_path = tmp_path / "history.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    record = gold_record()
    write_jsonl(test_path, [record])
    write_jsonl(history_path, [])
    write_jsonl(reference_path, [record])

    report = gate.audit_test_data(test_path, [history_path], reference_path, min_records=1)

    assert report["status"] == "PASS"
    assert report["record_count"] == 1
    assert report["geometry"]["median_area_ratio"] == 1.0


def test_test_gate_blocks_prior_source_group_and_missing_contract(tmp_path: Path) -> None:
    gate = load_gate_module()
    gate.PROJECT_ROOT = tmp_path
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "test.jpg").write_bytes(b"image")
    record = gold_record()
    record["review"]["person_box_contract_confirmed"] = False  # type: ignore[index]
    test_path = tmp_path / "test.jsonl"
    history_path = tmp_path / "history.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    write_jsonl(test_path, [record])
    write_jsonl(history_path, [{"source_group": "new-group"}])
    write_jsonl(reference_path, [gold_record("reference-group")])

    report = gate.audit_test_data(test_path, [history_path], reference_path, min_records=1)

    assert report["status"] == "BLOCKED"
    assert any("prior gold data" in error for error in report["errors"])
    assert any("contract was not confirmed" in error for error in report["errors"])
