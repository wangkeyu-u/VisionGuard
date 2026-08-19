from __future__ import annotations

import json
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_VIOLATIONS = {"no_helmet", "no_vest"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def validate_finding_document(document: dict[str, Any]) -> list[str]:
    """Return schema errors for the stable VLM finding contract."""
    errors: list[str] = []
    if not isinstance(document.get("scene_summary"), str):
        errors.append("scene_summary must be a string")
    findings = document.get("findings")
    if not isinstance(findings, list):
        errors.append("findings must be a list")
        findings = []
    for index, finding in enumerate(findings):
        prefix = f"findings[{index}]"
        if not isinstance(finding, dict):
            errors.append(f"{prefix} must be an object")
            continue
        box = finding.get("person_box")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
            or any(not 0 <= float(value) <= 1 for value in box)
            or (len(box) == 4 and (box[0] >= box[2] or box[1] >= box[3]))
        ):
            errors.append(f"{prefix}.person_box must be normalized xyxy")
        if finding.get("violation") not in ALLOWED_VIOLATIONS:
            errors.append(f"{prefix}.violation must be one of {sorted(ALLOWED_VIOLATIONS)}")
        if not isinstance(finding.get("evidence"), str) or not finding.get("evidence", "").strip():
            errors.append(f"{prefix}.evidence must be a non-empty string")
        if finding.get("confidence") not in ALLOWED_CONFIDENCE:
            errors.append(f"{prefix}.confidence must be one of {sorted(ALLOWED_CONFIDENCE)}")
    uncertainties = document.get("uncertainties")
    if not isinstance(uncertainties, list) or any(not isinstance(item, str) for item in uncertainties):
        errors.append("uncertainties must be a list of strings")
    if not isinstance(document.get("recommended_action"), str):
        errors.append("recommended_action must be a string")
    return errors


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model output did not contain a JSON object.")
        candidate = candidate[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON must be an object.")
    return parsed


@dataclass(frozen=True)
class VLMExample:
    id: str
    image: Path
    target: dict[str, Any]


@dataclass(frozen=True)
class AdapterResult:
    document: dict[str, Any]
    latency_ms: float
    raw_output: str | None = None
    cache_hit: bool = False


class VisionAdapter(ABC):
    name: str
    execution_mode: str

    @abstractmethod
    def predict(self, example: VLMExample) -> AdapterResult:
        raise NotImplementedError


class CachedAdapter(VisionAdapter):
    """Atomic per-example prediction cache for resumable expensive inference."""

    def __init__(self, adapter: VisionAdapter, cache_path: Path, signature: str) -> None:
        self.adapter = adapter
        self.name = adapter.name
        self.execution_mode = adapter.execution_mode
        self.runtime_metadata = getattr(adapter, "runtime_metadata", {})
        self.cache_path = cache_path
        self.signature = signature
        self._records: dict[str, dict[str, Any]] = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("signature") == signature:
                        self._records[str(record["id"])] = record

    def _write(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.cache_path.name}.", suffix=".tmp", dir=self.cache_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for record in self._records.values():
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.cache_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def predict(self, example: VLMExample) -> AdapterResult:
        cached = self._records.get(example.id)
        if cached is not None:
            return AdapterResult(
                cached["document"],
                float(cached["latency_ms"]),
                cached.get("raw_output"),
                cache_hit=True,
            )
        result = self.adapter.predict(example)
        self._records[example.id] = {
            "signature": self.signature,
            "id": example.id,
            "document": result.document,
            "latency_ms": result.latency_ms,
            "raw_output": result.raw_output,
        }
        self._write()
        return result


class FixtureAdapter(VisionAdapter):
    """Deterministic pipeline fixture. It is never evidence of model quality."""

    execution_mode = "fixture"

    def __init__(self, name: str, predictions_path: Path) -> None:
        self.name = name
        self.predictions_path = predictions_path
        self._predictions: dict[str, dict[str, Any]] = {}
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self._predictions[str(row["id"])] = row["prediction"]

    def predict(self, example: VLMExample) -> AdapterResult:
        if example.id not in self._predictions:
            raise KeyError(f"Fixture has no prediction for example {example.id!r}.")
        return AdapterResult(self._predictions[example.id], latency_ms=0.0)


class YOLO11Adapter(VisionAdapter):
    execution_mode = "real"

    def __init__(self, name: str, model: str | Path, imgsz: int, confidence: float, device: str) -> None:
        from ultralytics import YOLO

        self.name = name
        self.model = YOLO(str(model))
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = device
        self.runtime_metadata = {
            "adapter": "yolo11",
            "model": str(model),
            "device_requested": device,
            "image_size": imgsz,
        }

    def detect(self, image: Path) -> list[dict[str, Any]]:
        result = self.model.predict(
            source=str(image), imgsz=self.imgsz, conf=self.confidence, device=self.device, verbose=False
        )[0]
        width, height = result.orig_shape[1], result.orig_shape[0]
        detections = []
        for xyxy, class_id, confidence in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            strict=True,
        ):
            detections.append(
                {
                    "class_name": str(result.names[int(class_id)]),
                    "confidence": float(confidence),
                    "xyxy": [xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height],
                }
            )
        return detections

    def predict(self, example: VLMExample) -> AdapterResult:
        start = time.perf_counter()
        detections = self.detect(example.image)
        persons = [detection for detection in detections if detection["class_name"] == "person"]

        def associated_person_box(violation: dict[str, Any]) -> list[float]:
            box = violation["xyxy"]
            center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            containing = [
                person
                for person in persons
                if person["xyxy"][0] <= center_x <= person["xyxy"][2]
                and person["xyxy"][1] <= center_y <= person["xyxy"][3]
            ]
            if not containing:
                return box
            return min(
                containing,
                key=lambda person: (
                    (person["xyxy"][2] - person["xyxy"][0])
                    * (person["xyxy"][3] - person["xyxy"][1])
                ),
            )["xyxy"]

        findings = [
            {
                "person_box": associated_person_box(detection),
                "violation": detection["class_name"],
                "evidence": "YOLO11 violation detection associated to the smallest containing person box.",
                "confidence": "high" if detection["confidence"] >= 0.75 else "medium",
            }
            for detection in detections
            if detection["class_name"] in ALLOWED_VIOLATIONS
        ]
        document = {
            "scene_summary": f"YOLO11 produced {len(detections)} detections.",
            "findings": findings,
            "uncertainties": [
                "Person association uses center-in-person; unmatched violation boxes remain grounding proxies."
            ],
            "recommended_action": "Human review required." if findings else "Continue routine human review.",
        }
        return AdapterResult(document, (time.perf_counter() - start) * 1000)


class Qwen3VLAdapter(VisionAdapter):
    execution_mode = "real"

    def __init__(self, name: str, model_id: str, revision: str | None, device_map: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.name = name
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, revision=revision, device_map=device_map
        )
        self.runtime_metadata = {
            "adapter": "qwen3_vl",
            "model_id": model_id,
            "revision": revision,
            "device_requested": device_map,
            "model_device": str(self.model.device),
            "max_new_tokens": max_new_tokens,
        }

    def _prompt(self, grounding: list[dict[str, Any]] | None = None) -> str:
        grounding_text = ""
        if grounding is not None:
            grounding_text = f"\nDetector proposals (normalized xyxy): {json.dumps(grounding)}"
        return (
            "Inspect visible workers for no_helmet and no_vest only. Return exactly one JSON object "
            "with scene_summary, findings, uncertainties, recommended_action. Each finding requires "
            "person_box normalized xyxy, violation, evidence, confidence (low/medium/high). Abstain when "
            f"evidence is unclear.{grounding_text}"
        )

    def generate(self, image: Path, grounding: list[dict[str, Any]] | None = None) -> AdapterResult:
        from PIL import Image

        start = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(image)},
                    {"type": "text", "text": self._prompt(grounding)},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return AdapterResult(_extract_json(raw), (time.perf_counter() - start) * 1000, raw)

    def predict(self, example: VLMExample) -> AdapterResult:
        return self.generate(example.image)


class GroundedQwen3VLAdapter(VisionAdapter):
    execution_mode = "real"

    def __init__(self, name: str, detector: YOLO11Adapter, vlm: Qwen3VLAdapter) -> None:
        self.name = name
        self.detector = detector
        self.vlm = vlm

    def predict(self, example: VLMExample) -> AdapterResult:
        start = time.perf_counter()
        detections = self.detector.detect(example.image)
        result = self.vlm.generate(example.image, detections)
        return AdapterResult(result.document, (time.perf_counter() - start) * 1000, result.raw_output)


def load_examples(path: Path) -> list[VLMExample]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = Path(row["image"])
        if not image.is_absolute():
            image = (path.parent / image).resolve()
        examples.append(VLMExample(str(row["id"]), image, row["target"]))
    if not examples:
        raise ValueError("Evaluation manifest is empty.")
    return examples


def build_adapter(run: dict[str, Any], config_dir: Path) -> VisionAdapter:
    execution = run.get("execution", "real")
    if execution == "fixture":
        fixture = (config_dir / run["fixture_predictions"]).resolve()
        return FixtureAdapter(run["name"], fixture)
    kind = run["adapter"]
    if kind == "yolo11":
        return YOLO11Adapter(run["name"], run["model"], run["imgsz"], run["confidence"], run["device"])
    if kind == "qwen3_vl":
        return Qwen3VLAdapter(
            run["name"],
            run["model_id"],
            run.get("revision"),
            run.get("device_map", "auto"),
            run.get("max_new_tokens", 512),
        )
    if kind == "grounded_qwen3_vl":
        detector = YOLO11Adapter(
            f"{run['name']}-detector", run["detector_model"], run["imgsz"], run["confidence"], run["device"]
        )
        vlm = Qwen3VLAdapter(
            f"{run['name']}-vlm",
            run["model_id"],
            run.get("revision"),
            run.get("device_map", "auto"),
            run.get("max_new_tokens", 512),
        )
        return GroundedQwen3VLAdapter(run["name"], detector, vlm)
    raise ValueError(f"Unknown adapter kind: {kind}")


def load_evaluation_config(path: Path) -> tuple[dict[str, Any], list[VLMExample]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("runs"), list):
        raise ValueError("Expected VLM evaluation config schema_version 1 with a runs list.")
    manifest = Path(config["manifest"])
    if not manifest.is_absolute():
        manifest = (path.parent / manifest).resolve()
    return config, load_examples(manifest)
