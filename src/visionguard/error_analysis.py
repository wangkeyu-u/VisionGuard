from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from visionguard.dataset import (
    collect_split_images,
    label_path_for_image,
    load_dataset_definition,
)
from visionguard.inference import iter_batched_predictions
from visionguard.training import resolve_device


@dataclass(frozen=True)
class Box:
    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    model: Path
    data: Path
    output_dir: Path
    imgsz: int = 512
    device: str = "auto"
    batch: int = 1
    confidence: float = 0.25
    low_confidence: float = 0.10
    iou_threshold: float = 0.50
    max_examples_per_category: int = 20


CATEGORIES = (
    "low_confidence",
    "suspected_missed_detection",
    "suspected_false_positive",
    "class_confusion",
    "small_object_failure",
    "occlusion_scene_failure",
    "dense_scene_failure",
)


def box_iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _ground_truth_boxes(image_path: Path, dataset_root: Path) -> list[Box]:
    with Image.open(image_path) as image:
        width, height = image.size
    label_path = label_path_for_image(image_path, dataset_root)
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, x_text, y_text, width_text, height_text = line.split()
        x, y, box_width, box_height = map(float, (x_text, y_text, width_text, height_text))
        boxes.append(
            Box(
                int(class_id_text),
                (
                    (x - box_width / 2) * width,
                    (y - box_height / 2) * height,
                    (x + box_width / 2) * width,
                    (y + box_height / 2) * height,
                ),
            )
        )
    return boxes


def _prediction_boxes(result: Any) -> list[Box]:
    xyxy = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    return [
        Box(int(class_id), tuple(float(value) for value in coordinates), float(confidence))
        for coordinates, class_id, confidence in zip(xyxy, classes, confidences, strict=True)
    ]


def _match_same_class(
    ground_truth: list[Box], predictions: list[Box], iou_threshold: float
) -> tuple[set[int], set[int]]:
    candidates = sorted(
        (
            (box_iou(gt, pred), gt_index, pred_index)
            for gt_index, gt in enumerate(ground_truth)
            for pred_index, pred in enumerate(predictions)
            if gt.class_id == pred.class_id and box_iou(gt, pred) >= iou_threshold
        ),
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    for _, gt_index, pred_index in candidates:
        if gt_index not in matched_gt and pred_index not in matched_predictions:
            matched_gt.add(gt_index)
            matched_predictions.add(pred_index)
    return matched_gt, matched_predictions


def _box_area_ratio(box: Box, width: int, height: int) -> float:
    x1, y1, x2, y2 = box.xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)


def _draw_example(
    image_path: Path,
    ground_truth: list[Box],
    predictions: list[Box],
    names: dict[int, str],
    title: str,
    destination: Path,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((5, 5), title, fill=(255, 255, 255))
    for box in ground_truth:
        draw.rectangle(box.xyxy, outline=(40, 220, 80), width=3)
        draw.text(
            (box.xyxy[0], max(25, box.xyxy[1])), f"GT {names[box.class_id]}", fill=(40, 220, 80)
        )
    for box in predictions:
        draw.rectangle(box.xyxy, outline=(230, 60, 60), width=2)
        label = f"P {names[box.class_id]} {box.confidence:.2f}"
        draw.text((box.xyxy[0], max(25, box.xyxy[1] + 12)), label, fill=(230, 60, 60))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def analyze_yolo_errors(config: ErrorAnalysisConfig) -> dict[str, Any]:
    model_path = config.model.expanduser().resolve()
    data_path = config.data.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    device = resolve_device(config.device)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    definition = load_dataset_definition(data_path)
    test_images = collect_split_images(definition, "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    predictions = iter_batched_predictions(
        model,
        test_images,
        config.batch,
        imgsz=config.imgsz,
        conf=config.low_confidence,
        device=device,
        verbose=False,
    )
    event_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    saved_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {category: [] for category in CATEGORIES}

    for image_index, (image_path, result) in enumerate(
        zip(test_images, predictions, strict=True), 1
    ):
        ground_truth = _ground_truth_boxes(image_path, definition.root)
        all_predictions = _prediction_boxes(result)
        active_predictions = [
            prediction
            for prediction in all_predictions
            if prediction.confidence is not None and prediction.confidence >= config.confidence
        ]
        low_predictions = [
            prediction for prediction in all_predictions if prediction not in active_predictions
        ]
        matched_gt, matched_predictions = _match_same_class(
            ground_truth, active_predictions, config.iou_threshold
        )
        missed_indices = set(range(len(ground_truth))) - matched_gt
        extra_indices = set(range(len(active_predictions))) - matched_predictions
        with Image.open(image_path) as image:
            width, height = image.size

        category_events: dict[str, int] = {}
        if low_predictions:
            category_events["low_confidence"] = len(low_predictions)
        if missed_indices:
            category_events["suspected_missed_detection"] = len(missed_indices)
        if extra_indices:
            category_events["suspected_false_positive"] = len(extra_indices)
        confusions = sum(
            1
            for gt_index in missed_indices
            for pred_index in extra_indices
            if ground_truth[gt_index].class_id != active_predictions[pred_index].class_id
            and box_iou(ground_truth[gt_index], active_predictions[pred_index])
            >= config.iou_threshold
        )
        if confusions:
            category_events["class_confusion"] = confusions
        small_misses = sum(
            _box_area_ratio(ground_truth[index], width, height) < 0.01 for index in missed_indices
        )
        if small_misses:
            category_events["small_object_failure"] = small_misses
        overlapping_gt = {
            index
            for index, box in enumerate(ground_truth)
            if any(
                index != other_index and box_iou(box, other_box) >= 0.10
                for other_index, other_box in enumerate(ground_truth)
            )
        }
        occluded_misses = len(missed_indices & overlapping_gt)
        if occluded_misses:
            category_events["occlusion_scene_failure"] = occluded_misses
        if len(ground_truth) >= 8 and (missed_indices or extra_indices):
            category_events["dense_scene_failure"] = len(missed_indices) + len(extra_indices)

        for category, count in category_events.items():
            event_counts[category] += count
            image_counts[category] += 1
            if saved_counts[category] < config.max_examples_per_category:
                destination = output_dir / category / f"{image_index:04d}_{image_path.stem}.jpg"
                _draw_example(
                    image_path,
                    ground_truth,
                    all_predictions,
                    definition.class_names,
                    category.replace("_", " "),
                    destination,
                )
                saved_counts[category] += 1
                examples[category].append(str(destination))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": str(model_path),
        "data": str(data_path),
        "device": device,
        "imgsz": config.imgsz,
        "test_images": len(test_images),
        "heuristics": {
            "active_prediction_confidence": config.confidence,
            "low_confidence_range": [config.low_confidence, config.confidence],
            "same_class_match_iou": config.iou_threshold,
            "class_confusion_iou": config.iou_threshold,
            "small_object_area_ratio_below": 0.01,
            "occlusion_proxy_ground_truth_iou_at_least": 0.10,
            "dense_scene_ground_truth_objects_at_least": 8,
        },
        "disclaimer": (
            "These categories are heuristic candidates for human review, not absolute false-positive "
            "or false-negative determinations. Occlusion and density are proxy rules."
        ),
        "event_counts": {category: event_counts[category] for category in CATEGORIES},
        "image_counts": {category: image_counts[category] for category in CATEGORIES},
        "saved_examples": examples,
    }
    (output_dir / "error_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Error analysis",
        "",
        f"Analyzed `{len(test_images)}` test images on `{device}` at `{config.imgsz}px`.",
        "",
        "All findings below are heuristic review candidates, not absolute false positives or false negatives.",
        "Ground-truth matching uses same-class IoU >= 0.50 at confidence >= 0.25.",
        "Small objects occupy <1% of image area; the occlusion proxy uses GT overlap IoU >= 0.10;",
        "and dense scenes contain at least 8 ground-truth objects.",
        "",
        "| Category | Candidate events | Images | Saved examples |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category in CATEGORIES:
        lines.append(
            f"| {category.replace('_', ' ')} | {event_counts[category]} | "
            f"{image_counts[category]} | {saved_counts[category]} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Label omissions or annotation noise can make a correct prediction look like a suspected false positive.",
            "- A missed match may reflect localization IoU below the threshold rather than total non-detection.",
            "- Occlusion and density labels are inferred from box geometry/object count, not human scene tags.",
        ]
    )
    (output_dir / "ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
