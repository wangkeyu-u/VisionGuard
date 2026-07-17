#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionguard.training import resolve_device  # noqa: E402
from visionguard.vlm import (  # noqa: E402
    VIOLATION_CLASSES,
    extract_json_object,
    pixel_xyxy_to_normalized,
    read_jsonl,
    validate_inspection,
    write_jsonl,
)

POSITIVE_GROUNDING_CLASSES = {"person", "helmet", "vest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VisionGuard VLM ablation baselines.")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--mode", choices=("yolo", "vlm", "grounded"), default="vlm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto", help="VLM device: auto, mps, cpu, or cuda.")
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--schema-retries",
        type=int,
        default=1,
        help="Bounded text-only correction attempts after invalid VLM JSON.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", type=Path, help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments" / "exp4_yolo11s_512_e50" / "weights" / "best.pt",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve_vlm_device(requested: str) -> str:
    import torch

    normalized = requested.lower().strip()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return resolve_device(normalized)


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def _expanded_person_box(box: list[int], violation: str) -> list[int]:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if violation == "no_helmet":
        return [
            max(0, round(x1 - width * 0.8)),
            max(0, round(y1 - height * 0.4)),
            min(1000, round(x2 + width * 0.8)),
            min(1000, round(y2 + height * 6.0)),
        ]
    return [
        max(0, round(x1 - width * 0.35)),
        max(0, round(y1 - height * 0.9)),
        min(1000, round(x2 + width * 0.35)),
        min(1000, round(y2 + height * 1.8)),
    ]


def _contains_center(person_box: list[int], object_box: list[int]) -> bool:
    center_x = (object_box[0] + object_box[2]) / 2
    center_y = (object_box[1] + object_box[3]) / 2
    return person_box[0] <= center_x <= person_box[2] and person_box[1] <= center_y <= person_box[3]


class YoloGrounder:
    def __init__(self, model_path: Path, device: str, imgsz: int, confidence: float) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.confidence = confidence

    def detect(self, image: Path) -> tuple[Any, list[dict[str, Any]]]:
        result = self.model.predict(
            source=str(image),
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        detections: list[dict[str, Any]] = []
        for xyxy, class_id, confidence in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            strict=True,
        ):
            detections.append(
                {
                    "class_name": str(result.names[int(class_id)]),
                    "confidence": round(float(confidence), 6),
                    "box": pixel_xyxy_to_normalized(xyxy, width, height),
                    "pixel_box": [round(float(value)) for value in xyxy],
                }
            )
        return result, detections

    @staticmethod
    def yolo_prediction(detections: list[dict[str, Any]]) -> dict[str, Any]:
        people = sorted(
            (detection for detection in detections if detection["class_name"] == "person"),
            key=lambda detection: detection["box"][0],
        )
        people_with_ids = [(f"p{index}", person) for index, person in enumerate(people, start=1)]
        findings_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        uncertainties: list[str] = []
        unmatched_index = len(people_with_ids) + 1

        violations = sorted(
            (detection for detection in detections if detection["class_name"] in VIOLATION_CLASSES),
            key=lambda detection: detection["box"][0],
        )
        for detection in violations:
            containing = [
                (person_id, person)
                for person_id, person in people_with_ids
                if _contains_center(person["box"], detection["box"])
            ]
            if containing:
                person_id, person = min(
                    containing,
                    key=lambda item: (item[1]["box"][2] - item[1]["box"][0]) * (item[1]["box"][3] - item[1]["box"][1]),
                )
                person_box = person["box"]
            else:
                person_id = f"p{unmatched_index}"
                unmatched_index += 1
                person_box = _expanded_person_box(detection["box"], detection["class_name"])
                uncertainties.append(f"{person_id}: violation detection was not contained by a YOLO person box.")
            finding = {
                "person_id": person_id,
                "person_box": person_box,
                "violation": detection["class_name"],
                "evidence": (f"YOLO detected {detection['class_name']} with confidence {detection['confidence']:.3f}."),
                "confidence": _confidence_label(detection["confidence"]),
            }
            key = (person_id, detection["class_name"])
            previous = findings_by_key.get(key)
            if previous is None or detection["confidence"] > previous["_score"]:
                findings_by_key[key] = {**finding, "_score": detection["confidence"]}

        findings = []
        for finding in findings_by_key.values():
            finding.pop("_score")
            findings.append(finding)
        findings.sort(key=lambda finding: (finding["person_box"][0], finding["violation"]))
        return {
            "findings": findings,
            "uncertainties": uncertainties,
            "recommended_action": "human_review" if findings or uncertainties else "no_action",
        }

    @staticmethod
    def grounded_image_and_context(
        result: Any,
        detections: list[dict[str, Any]],
        output_path: Path,
    ) -> str:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(result.orig_img[:, :, ::-1])
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        colors = {"person": "#ffd43b", "helmet": "#2dd4bf", "vest": "#60a5fa"}
        context_rows: list[str] = []
        allowed = [detection for detection in detections if detection["class_name"] in POSITIVE_GROUNDING_CLASSES]
        for index, detection in enumerate(allowed, start=1):
            box = detection["pixel_box"]
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            color = colors[detection["class_name"]]
            draw.rectangle(box, outline=color, width=3)
            text_box = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((box[0], box[1]), label, fill="black", font=font)
            context_rows.append(
                f"d{index}: class={detection['class_name']}, "
                f"confidence={detection['confidence']:.3f}, box={detection['box']}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        if not context_rows:
            return "The detector produced no person, helmet, or vest grounding above threshold."
        return "Detector grounding (fallible; verify against the image):\n" + "\n".join(context_rows)


class QwenVlm:
    def __init__(
        self,
        model_id: str,
        device: str,
        max_new_tokens: int,
        adapter: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = _resolve_vlm_device(device)
        self.max_new_tokens = max_new_tokens
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        if adapter is not None:
            if not adapter.is_dir():
                raise FileNotFoundError(f"LoRA adapter directory not found: {adapter}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter.resolve()))
        self.model.eval()

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        trimmed = generated[0][inputs["input_ids"].shape[-1] :]
        return self.processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate(self, image: Path, prompt: str) -> str:
        return self._generate(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(image.resolve())},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    def correct_json(self, raw_output: str, schema_errors: list[str]) -> str:
        correction_prompt = (
            "Correct the JSON object below so it passes validation. Return exactly one JSON "
            "object and nothing else. Preserve the visible findings and coordinates unless a "
            "validation error requires changing them.\n\nValidation errors:\n- "
            + "\n- ".join(schema_errors)
            + "\n\nJSON to correct:\n"
            + raw_output
        )
        return self._generate([{"role": "user", "content": [{"type": "text", "text": correction_prompt}]}])


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.schema_retries < 0:
        raise SystemExit("--schema-retries must be non-negative")
    data_path = PROJECT_ROOT / "vlm" / "data" / f"{args.split}.jsonl"
    records, issues = read_jsonl(data_path)
    if issues:
        raise SystemExit("\n".join(str(issue) for issue in issues))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No gold records in {data_path}")

    output_path = args.output or (PROJECT_ROOT / "vlm" / "outputs" / f"{args.split}_{args.mode}.jsonl")
    prompt = (PROJECT_ROOT / "vlm" / "prompts" / "safety_inspection.txt").read_text(encoding="utf-8").strip()
    grounder = None
    if args.mode in {"yolo", "grounded"}:
        grounder = YoloGrounder(args.yolo_model.resolve(), args.yolo_device, args.imgsz, args.confidence)
    vlm = None
    if args.mode in {"vlm", "grounded"}:
        vlm = QwenVlm(args.model_id, args.device, args.max_new_tokens, args.adapter)

    outputs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        image = (PROJECT_ROOT / record["image"]).resolve()
        started = time.perf_counter()
        prediction = None
        raw_output = None
        raw_attempts: list[str] = []
        repair_attempts = 0
        error = None
        detections = None
        try:
            if args.mode == "yolo":
                assert grounder is not None
                _, detections = grounder.detect(image)
                prediction = grounder.yolo_prediction(detections)
            elif args.mode == "vlm":
                assert vlm is not None
                raw_output = vlm.generate(image, prompt)
                raw_attempts.append(raw_output)
                prediction = extract_json_object(raw_output)
            else:
                assert grounder is not None and vlm is not None
                result, detections = grounder.detect(image)
                grounded_image = output_path.parent / "grounded_images" / f"{record['id']}.jpg"
                context = grounder.grounded_image_and_context(result, detections, grounded_image)
                grounded_prompt = (
                    f"{prompt}\n\n{context}\nDo not treat detector output as ground truth. Use only visible evidence."
                )
                raw_output = vlm.generate(grounded_image, grounded_prompt)
                raw_attempts.append(raw_output)
                prediction = extract_json_object(raw_output)
            schema_errors = validate_inspection(prediction)
            while schema_errors and vlm is not None and repair_attempts < args.schema_retries:
                repair_attempts += 1
                raw_output = vlm.correct_json(raw_output or "", schema_errors)
                raw_attempts.append(raw_output)
                prediction = extract_json_object(raw_output)
                schema_errors = validate_inspection(prediction)
            if schema_errors:
                error = "Schema validation: " + "; ".join(schema_errors)
        except Exception as exc:  # Record per-sample model/runtime failures for auditability.
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        outputs.append(
            {
                "id": record["id"],
                "mode": args.mode,
                "model": (args.yolo_model.name if args.mode == "yolo" else str(args.adapter or args.model_id)),
                "prediction": prediction,
                "raw_output": raw_output,
                "raw_attempts": raw_attempts,
                "repair_attempts": repair_attempts,
                "error": error,
                "latency_ms": round(elapsed_ms, 3),
                "detections": detections,
            }
        )
        print(f"[{index}/{len(records)}] {record['id']}: {'ok' if error is None else error}")
        write_jsonl(output_path, outputs)

    print(f"Wrote {len(outputs)} predictions to {output_path}")
    return 0 if all(record["error"] is None for record in outputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
