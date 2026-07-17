from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionguard.training import resolve_device

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VIOLATION_CLASSES = {"no_helmet", "no_vest"}


@dataclass(frozen=True)
class DemoConfig:
    model: Path
    output_dir: Path
    imgsz: int = 512
    confidence: float = 0.25
    device: str = "auto"
    max_video_seconds: float = 120.0

    def resolved(self) -> DemoConfig:
        model = self.model.expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Demo checkpoint does not exist: {model}")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if self.imgsz < 32:
            raise ValueError("Input size must be at least 32 pixels.")
        return DemoConfig(
            model=model,
            output_dir=self.output_dir.expanduser().resolve(),
            imgsz=self.imgsz,
            confidence=self.confidence,
            device=resolve_device(self.device),
            max_video_seconds=self.max_video_seconds,
        )


@dataclass
class DetectionAccumulator:
    class_counts: Counter[str] = field(default_factory=Counter)
    violation_counts: Counter[str] = field(default_factory=Counter)
    frames_processed: int = 0
    frames_with_violations: int = 0
    max_detections_in_frame: int = 0
    max_violations_in_frame: int = 0

    def add(self, detections: list[dict[str, Any]]) -> None:
        self.frames_processed += 1
        frame_violations = 0
        for detection in detections:
            class_name = detection["class_name"]
            self.class_counts[class_name] += 1
            if class_name in VIOLATION_CLASSES:
                self.violation_counts[class_name] += 1
                frame_violations += 1
        if frame_violations:
            self.frames_with_violations += 1
        self.max_detections_in_frame = max(self.max_detections_in_frame, len(detections))
        self.max_violations_in_frame = max(self.max_violations_in_frame, frame_violations)

    def summary(self) -> dict[str, Any]:
        violation_events = sum(self.violation_counts.values())
        return {
            "status": "violation_detected" if violation_events else "no_violation_detected",
            "frames_processed": self.frames_processed,
            "frames_with_violations": self.frames_with_violations,
            "total_detection_events": sum(self.class_counts.values()),
            "violation_detection_events": violation_events,
            "max_detections_in_frame": self.max_detections_in_frame,
            "max_violations_in_frame": self.max_violations_in_frame,
            "class_counts": dict(sorted(self.class_counts.items())),
            "violation_counts": {
                class_name: self.violation_counts[class_name]
                for class_name in sorted(VIOLATION_CLASSES)
            },
        }


def detections_from_result(result: Any) -> list[dict[str, Any]]:
    coordinates = result.boxes.xyxy.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    names = result.names
    detections = []
    for xyxy, class_id, confidence in zip(
        coordinates, class_ids, confidences, strict=True
    ):
        integer_class_id = int(class_id)
        detections.append(
            {
                "class_id": integer_class_id,
                "class_name": str(names[integer_class_id]),
                "confidence": round(float(confidence), 6),
                "xyxy": [round(float(value), 2) for value in xyxy],
            }
        )
    return detections


def _base_report(config: DemoConfig, source: Path, media_type: str) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "media_type": media_type,
        "source_name": source.name,
        "model": str(config.model),
        "device": config.device,
        "imgsz": config.imgsz,
        "confidence_threshold": config.confidence,
        "interpretation": (
            "Counts are detection events, not unique tracked people. A clear result does not prove "
            "that the scene is safe; it means the model found no violation above the threshold."
        ),
    }


class VisionGuardDemoEngine:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config.resolved()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO

        self.model = YOLO(str(self.config.model))

    def infer(self, source: Path, session_dir: Path) -> dict[str, Any]:
        suffix = source.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return self._infer_image(source, session_dir)
        if suffix in VIDEO_SUFFIXES:
            return self._infer_video(source, session_dir)
        raise ValueError(f"Unsupported media type: {suffix or 'no extension'}")

    def _predict(self, source: Any) -> Any:
        return self.model.predict(
            source=source,
            imgsz=self.config.imgsz,
            conf=self.config.confidence,
            device=self.config.device,
            verbose=False,
        )[0]

    def _infer_image(self, source: Path, session_dir: Path) -> dict[str, Any]:
        import cv2

        result = self._predict(str(source))
        detections = detections_from_result(result)
        accumulator = DetectionAccumulator()
        accumulator.add(detections)
        output_path = session_dir / "annotated.jpg"
        if not cv2.imwrite(str(output_path), result.plot()):
            raise OSError(f"Could not write annotated image: {output_path}")
        report = {
            **_base_report(self.config, source, "image"),
            "summary": accumulator.summary(),
            "detections": detections,
            "output_file": output_path.name,
        }
        self._write_report(report, session_dir)
        return report

    def _infer_video(self, source: Path, session_dir: Path) -> dict[str, Any]:
        import cv2

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps if fps > 0 else 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_total / fps if frame_total > 0 else 0.0
        if duration_seconds > self.config.max_video_seconds:
            capture.release()
            raise ValueError(
                f"Video is {duration_seconds:.1f}s; the demo limit is "
                f"{self.config.max_video_seconds:.1f}s."
            )
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("Video dimensions could not be read.")

        output_path = session_dir / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise OSError("OpenCV could not initialize the MP4 writer.")

        accumulator = DetectionAccumulator()
        violation_frames: list[dict[str, Any]] = []
        frame_index = 0
        max_frames = max(1, int(self.config.max_video_seconds * fps))
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index >= max_frames:
                    raise ValueError(
                        "Video exceeds the demo limit when decoded "
                        f"({self.config.max_video_seconds:.1f}s)."
                    )
                result = self._predict(frame)
                detections = detections_from_result(result)
                accumulator.add(detections)
                violations = [
                    detection
                    for detection in detections
                    if detection["class_name"] in VIOLATION_CLASSES
                ]
                if violations and len(violation_frames) < 100:
                    violation_frames.append(
                        {
                            "frame": frame_index,
                            "time_seconds": round(frame_index / fps, 3),
                            "violations": violations,
                        }
                    )
                writer.write(result.plot())
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        if frame_index == 0:
            raise ValueError("The uploaded video did not contain readable frames.")
        report = {
            **_base_report(self.config, source, "video"),
            "video": {
                "fps": round(fps, 3),
                "frames_processed": frame_index,
                "duration_seconds": round(frame_index / fps, 3),
            },
            "summary": accumulator.summary(),
            "violation_frame_samples": violation_frames,
            "output_file": output_path.name,
        }
        self._write_report(report, session_dir)
        return report

    @staticmethod
    def _write_report(report: dict[str, Any], session_dir: Path) -> None:
        (session_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
