from __future__ import annotations

from pathlib import Path

import pytest

from visionguard.demo import DetectionAccumulator, detections_from_result
from visionguard.demo_server import _parse_multipart, _safe_child


class FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[object]:
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[1.234, 2.345, 30.0, 40.0], [5.0, 6.0, 20.0, 21.0]])
    cls = FakeTensor([0.0, 5.0])
    conf = FakeTensor([0.87654321, 0.612345])


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "person", 5: "no_helmet"}


def test_detections_from_result_normalizes_ultralytics_output() -> None:
    detections = detections_from_result(FakeResult())

    assert detections == [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.876543,
            "xyxy": [1.23, 2.35, 30.0, 40.0],
        },
        {
            "class_id": 5,
            "class_name": "no_helmet",
            "confidence": 0.612345,
            "xyxy": [5.0, 6.0, 20.0, 21.0],
        },
    ]


def test_accumulator_counts_detection_events_and_violation_frames() -> None:
    accumulator = DetectionAccumulator()
    accumulator.add(
        [
            {"class_name": "person"},
            {"class_name": "no_helmet"},
            {"class_name": "no_vest"},
        ]
    )
    accumulator.add([{"class_name": "helmet"}])

    assert accumulator.summary() == {
        "status": "violation_detected",
        "frames_processed": 2,
        "frames_with_violations": 1,
        "total_detection_events": 4,
        "violation_detection_events": 2,
        "max_detections_in_frame": 3,
        "max_violations_in_frame": 2,
        "class_counts": {
            "helmet": 1,
            "no_helmet": 1,
            "no_vest": 1,
            "person": 1,
        },
        "violation_counts": {"no_helmet": 1, "no_vest": 1},
    }


def test_safe_child_blocks_path_traversal(tmp_path: Path) -> None:
    assert _safe_child(tmp_path, "session/report.json") == (
        tmp_path / "session" / "report.json"
    ).resolve()
    with pytest.raises(ValueError, match="escapes"):
        _safe_child(tmp_path, "../secret.txt")


def test_parse_multipart_extracts_uploaded_file() -> None:
    boundary = "visionguard-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="sample.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + b"image-bytes\r\n" + f"--{boundary}--\r\n".encode()

    fields = _parse_multipart(
        f"multipart/form-data; boundary={boundary}",
        body,
    )

    assert fields["file"] == {"filename": "sample.jpg", "content": b"image-bytes"}
