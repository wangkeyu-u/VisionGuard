from __future__ import annotations

from pathlib import Path

import pytest

from visionguard.review_server import ReviewServerConfig, ReviewStore, parse_yolo_labels
from visionguard.vlm import read_jsonl, write_jsonl


def valid_target() -> dict[str, object]:
    return {
        "findings": [
            {
                "person_id": "p1",
                "person_box": [100, 120, 600, 950],
                "violation": "no_helmet",
                "evidence": "The visible head is not protected by a safety helmet.",
                "confidence": "high",
            }
        ],
        "uncertainties": [],
        "recommended_action": "human_review",
    }


def geometry_confirmation() -> dict[str, object]:
    return {
        "person_boxes": [[100, 120, 600, 950]],
        "person_box_contract_confirmed": True,
    }


def make_store(tmp_path: Path) -> ReviewStore:
    data_dir = tmp_path / "vlm" / "data"
    image_dir = tmp_path / "dataset" / "train" / "images"
    label_dir = tmp_path / "dataset" / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "train.jpg").write_bytes(b"not-a-real-image")
    (label_dir / "train.txt").write_text("0 0.5 0.5 0.4 0.8\n5 0.5 0.2 0.1 0.1\n", encoding="utf-8")
    test_image_dir = tmp_path / "dataset" / "test" / "images"
    test_label_dir = tmp_path / "dataset" / "test" / "labels"
    test_image_dir.mkdir(parents=True)
    test_label_dir.mkdir(parents=True)
    (test_image_dir / "test.jpg").write_bytes(b"not-a-real-image")
    (test_label_dir / "test.txt").write_text("0 0.5 0.5 0.5 0.9\n", encoding="utf-8")
    candidates = [
        {
            "id": "train-candidate",
            "image": "dataset/train/images/train.jpg",
            "label": "dataset/train/labels/train.txt",
            "source_group": "train-group",
            "source_split": "train",
            "category": "no_helmet",
            "source_classes": ["person", "no_helmet"],
        },
        {
            "id": "test-candidate",
            "image": "dataset/test/images/test.jpg",
            "label": "dataset/test/labels/test.txt",
            "source_group": "test-group",
            "source_split": "test",
            "category": "no_labeled_violation",
            "source_classes": ["person"],
        },
    ]
    candidates_path = data_dir / "candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    for filename in ("train.jsonl", "dev.jsonl", "test.jsonl", "review_log.jsonl"):
        write_jsonl(data_dir / filename, [])
    return ReviewStore(
        ReviewServerConfig(
            project_root=tmp_path,
            candidates=candidates_path,
            data_dir=data_dir,
            reviewer="Test Reviewer",
        )
    )


def test_parse_yolo_labels_exposes_normalized_xyxy_boxes(tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text("0 0.5 0.5 0.4 0.8\ninvalid row\n", encoding="utf-8")

    assert parse_yolo_labels(path) == [
        {
            "class_id": 0,
            "class_name": "person",
            "box": [300, 100, 700, 900],
            "source_line": 1,
        }
    ]


def test_snapshot_hides_all_source_signals_for_test_candidates(tmp_path: Path) -> None:
    snapshot = make_store(tmp_path).snapshot()
    train_candidate, test_candidate = snapshot["candidates"]

    assert train_candidate["source_annotations"][0]["class_name"] == "person"
    assert test_candidate["blind"] is True
    assert test_candidate["category"] == "blind_review"
    assert test_candidate["source_classes"] == []
    assert test_candidate["source_annotations"] == []


def test_approve_writes_valid_gold_record_and_updates_progress(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    snapshot = store.save_decision(
        {
            "candidate_id": "train-candidate",
            "decision": "approve",
            "reviewer": "Human Reviewer",
            "quality_tags": ["occlusion", "blur"],
            "target": valid_target(),
            **geometry_confirmation(),
        }
    )

    records, issues = read_jsonl(tmp_path / "vlm" / "data" / "train.jsonl")
    assert issues == []
    assert records[0]["target"] == valid_target()
    assert records[0]["review"]["reviewer"] == "Human Reviewer"
    assert records[0]["review"]["quality_tags"] == ["blur", "occlusion"]
    assert records[0]["review"]["annotation_protocol_version"] == "full-visible-person-v2"
    assert records[0]["review"]["person_boxes"] == [[100, 120, 600, 950]]
    assert snapshot["counts"]["train"] == 1
    assert snapshot["composition"] == {"violation": 1}


def test_exclude_requires_reason_and_records_auditable_decision(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="reason"):
        store.save_decision(
            {
                "candidate_id": "train-candidate",
                "decision": "exclude",
                "reviewer": "Human Reviewer",
                "quality_tags": [],
            }
        )

    snapshot = store.save_decision(
        {
            "candidate_id": "train-candidate",
            "decision": "exclude",
            "reviewer": "Human Reviewer",
            "quality_tags": ["blur"],
            "reason": "The subject is not recognizable.",
        }
    )

    records, issues = read_jsonl(tmp_path / "vlm" / "data" / "review_log.jsonl")
    assert issues == []
    assert records[-1]["decision"] == "exclude"
    assert records[-1]["reason"] == "The subject is not recognizable."
    assert records[-1]["source_group"] == "train-group"
    assert snapshot["status_counts"]["excluded"] == 1


def test_approve_rejects_schema_invalid_target(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="person_box"):
        store.save_decision(
            {
                "candidate_id": "train-candidate",
                "decision": "approve",
                "reviewer": "Human Reviewer",
                "quality_tags": [],
                **geometry_confirmation(),
                "target": {
                    **valid_target(),
                    "findings": [{**valid_target()["findings"][0], "person_box": [8, 8, 2, 2]}],  # type: ignore[index]
                },
            }
        )


def test_approve_requires_explicit_full_person_box_confirmation(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="full visible person"):
        store.save_decision(
            {
                "candidate_id": "train-candidate",
                "decision": "approve",
                "reviewer": "Human Reviewer",
                "quality_tags": [],
                "person_boxes": [[100, 120, 600, 950]],
                "target": valid_target(),
            }
        )


def test_approve_preserves_compliant_person_boxes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    target = {"findings": [], "uncertainties": [], "recommended_action": "no_action"}

    snapshot = store.save_decision(
        {
            "candidate_id": "train-candidate",
            "decision": "approve",
            "reviewer": "Human Reviewer",
            "quality_tags": [],
            "person_boxes": [[200, 100, 800, 980]],
            "person_box_contract_confirmed": True,
            "target": target,
        }
    )

    saved = snapshot["candidates"][0]["saved_review"]
    assert saved["person_boxes"] == [[200, 100, 800, 980]]
    assert saved["person_box_contract_confirmed"] is True


def test_approve_rejects_finding_box_that_differs_from_person_registry(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="does not match p1"):
        store.save_decision(
            {
                "candidate_id": "train-candidate",
                "decision": "approve",
                "reviewer": "Human Reviewer",
                "quality_tags": [],
                "person_boxes": [[90, 100, 610, 970]],
                "person_box_contract_confirmed": True,
                "target": valid_target(),
            }
        )
