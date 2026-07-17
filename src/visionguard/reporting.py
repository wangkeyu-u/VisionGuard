from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataset_report_markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["valid"] else "FAIL"
    summary = report["summary"]
    lines = [
        "# VisionGuard Dataset Validation Report",
        "",
        f"- **Status:** {status}",
        f"- **Generated:** {report['generated_at']}",
        f"- **Data YAML:** `{report['data_yaml']}`",
        f"- **Dataset root:** `{report['dataset_root']}`",
        f"- **Images:** {summary['images']}",
        f"- **Label files:** {summary['label_files']}",
        f"- **Valid annotations:** {summary['valid_annotations']} / {summary['annotation_lines']}",
        f"- **Errors / warnings:** {summary['errors']} / {summary['warnings']}",
        f"- **Duplicate groups:** {summary['duplicate_groups']}",
        "",
        "## Split summary",
        "",
        "| Split | Images | Label files | Annotation lines | Valid annotations |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, stats in report["splits"].items():
        lines.append(
            f"| {split} | {stats['images']} | {stats['label_files']} | "
            f"{stats['annotation_lines']} | {stats['valid_annotations']} |"
        )

    lines.extend(
        [
            "",
            "## Image counts by class",
            "",
            "An image containing multiple objects of one class is counted once for that class.",
            "",
            "| Class | Train | Validation | Test | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name, total in summary["images_by_class"].items():
        lines.append(
            f"| {_escape(class_name)} | {report['splits']['train']['images_by_class'][class_name]} | "
            f"{report['splits']['val']['images_by_class'][class_name]} | "
            f"{report['splits']['test']['images_by_class'][class_name]} | {total} |"
        )

    lines.extend(
        [
            "",
            "## Annotation counts by class",
            "",
            "| Class | Train | Validation | Test | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name, total in summary["annotations_by_class"].items():
        lines.append(
            f"| {_escape(class_name)} | {report['splits']['train']['annotations_by_class'][class_name]} | "
            f"{report['splits']['val']['annotations_by_class'][class_name]} | "
            f"{report['splits']['test']['annotations_by_class'][class_name]} | {total} |"
        )

    lines.extend(["", "## Duplicate image groups", ""])
    if report["duplicates"]:
        for index, group in enumerate(report["duplicates"], 1):
            lines.append(f"### Group {index}: `{group['sha256']}`")
            lines.append("")
            for occurrence in group["occurrences"]:
                lines.append(f"- `{occurrence['split']}` — `{occurrence['path']}`")
            lines.append("")
    else:
        lines.extend(["No duplicate image content was detected.", ""])

    lines.extend(
        [
            "## Issues",
            "",
            "| Severity | Code | Split | Image | Label | Line | Message |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(
                "| {severity} | {code} | {split} | {image} | {label} | {line} | {message} |".format(
                    severity=_escape(issue.get("severity", "")),
                    code=_escape(issue.get("code", "")),
                    split=_escape(issue.get("split", "")),
                    image=_escape(issue.get("image_path", "")),
                    label=_escape(issue.get("label_path", "")),
                    line=_escape(issue.get("line", "")),
                    message=_escape(issue.get("message", "")),
                )
            )
    else:
        lines.append("| - | - | - | - | - | - | No issues detected. |")
    lines.append("")
    return "\n".join(lines)


def write_dataset_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_report.json"
    markdown_path = output_dir / "dataset_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(dataset_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def dataset_preview_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VisionGuard Dataset Preview Report",
        "",
        f"- **Generated:** {report['generated_at']}",
        f"- **Data YAML:** `{report['data_yaml']}`",
        f"- **Output directory:** `{report['output_dir']}`",
        f"- **Seed:** {report['configuration']['seed']}",
        f"- **Requested samples per split:** {report['configuration']['samples_per_split']}",
        f"- **Images visualized:** {summary['images_visualized']}",
        f"- **Objects visualized:** {summary['objects_visualized']}",
        f"- **Average objects per image:** {summary['average_objects_per_image']:.4f}",
        "",
        "## Split summary",
        "",
        "| Split | Available annotated | Visualized | Objects | Average objects/image | Skipped images |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, stats in report["splits"].items():
        lines.append(
            f"| {split} | {stats['available_annotated_images']} | {stats['images_visualized']} | "
            f"{stats['objects_visualized']} | {stats['average_objects_per_image']:.4f} | "
            f"{stats['skipped_images']} |"
        )

    lines.extend(
        [
            "",
            "## Sampled-object class distribution",
            "",
            "| Class | Color | Train | Validation | Test | Total |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for class_name, total in summary["class_distribution"].items():
        lines.append(
            f"| {_escape(class_name)} | `{report['class_colors'][class_name]}` | "
            f"{report['splits']['train']['class_distribution'][class_name]} | "
            f"{report['splits']['val']['class_distribution'][class_name]} | "
            f"{report['splits']['test']['class_distribution'][class_name]} | {total} |"
        )

    lines.extend(["", "## Sampled images", ""])
    for split, stats in report["splits"].items():
        lines.extend([f"### {split}", ""])
        if stats["samples"]:
            for sample in stats["samples"]:
                lines.append(
                    f"- `{sample['output_image']}` — {sample['objects']} objects "
                    f"(source: `{sample['source_image']}`)"
                )
        else:
            lines.append("No annotated images were available for visualization.")
        lines.append("")

    lines.extend(["## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {_escape(warning)}" for warning in report["warnings"])
    else:
        lines.append("No warnings.")
    lines.append("")
    return "\n".join(lines)


def write_dataset_preview_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_preview.json"
    markdown_path = output_dir / "dataset_preview.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(dataset_preview_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def class_remap_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VisionGuard Dataset Class Remapping Report",
        "",
        f"- **Generated:** {report['generated_at']}",
        f"- **Source:** `{report['source_dataset_root']}`",
        f"- **Output:** `{report['output_dataset_root']}`",
        f"- **Source modified:** {report['source_dataset_modified']}",
        f"- **Removed `glasses` annotations:** {report['removed_glasses_annotations']}",
        f"- **Images with empty clean labels:** {report['empty_label_images_count']}",
        "",
        "## Class mapping and source annotation counts",
        "",
        "| Old class | New class/action | Source annotations |",
        "|---|---|---:|",
    ]
    for old_name, count in report["old_class_counts"].items():
        action = report["mapping"].get(old_name, "REMOVED")
        lines.append(f"| {_escape(old_name)} | {_escape(action)} | {count} |")

    lines.extend(
        [
            "",
            "## Clean annotation counts",
            "",
            "| New ID | New class | Annotations |",
            "|---:|---|---:|",
        ]
    )
    for class_id, class_name in report["target_classes"].items():
        lines.append(
            f"| {class_id} | {_escape(class_name)} | {report['new_class_counts'][class_name]} |"
        )

    lines.extend(
        [
            "",
            "## Split statistics",
            "",
            "| Split | Images | Source annotations | Written annotations | Removed | Empty labels |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, stats in report["splits"].items():
        lines.append(
            f"| {split} | {stats['images']} | {stats['source_annotations']} | "
            f"{stats['written_annotations']} | {stats['removed_annotations']} | "
            f"{stats['empty_label_images']} |"
        )

    lines.extend(["", "## Empty-label images", ""])
    if report["empty_label_images"]:
        for item in report["empty_label_images"]:
            lines.append(
                f"- `{item['split']}` — `{item['clean_image']}` (source: `{item['source_image']}`)"
            )
    else:
        lines.append("No images became empty after class remapping.")

    lines.extend(
        [
            "",
            "## Safety guarantees",
            "",
            "- The source dataset was read only; all images and rewritten labels were placed in a new directory.",
            "- `glasses` annotations were removed without deleting their images.",
            "- Output was assembled in a temporary directory and published only after successful completion.",
            "- Existing output directories are never overwritten automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def write_class_remap_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "class_remap_report.json"
    markdown_path = output_dir / "class_remap_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(class_remap_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def finalization_report_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    audit = report["audit"]
    lines = [
        "# VisionGuard Final Dataset Freeze Report",
        "",
        f"- **Generated:** {report['generated_at']}",
        f"- **Source:** `{report['source_dataset_root']}`",
        f"- **Final dataset:** `{report['output_dataset_root']}`",
        f"- **Source modified:** {report['source_dataset_modified']}",
        f"- **Seed:** {report['configuration']['seed']}",
        f"- **Input / final images:** {counts['input_images']} / {counts['final_images']}",
        f"- **Duplicate groups / removed images:** {counts['duplicate_groups']} / "
        f"{counts['duplicate_images_removed']}",
        f"- **Duplicate groups with inconsistent labels:** "
        f"{counts['duplicate_groups_with_inconsistent_labels']}",
        f"- **Empty-label images removed:** {counts['empty_label_images_removed']}",
        f"- **Final annotations:** {counts['final_annotations']}",
        f"- **Dataset SHA-256:** `{report['freeze']['dataset_sha256']}`",
        "",
        "## Processing strategy",
        "",
        f"- Deduplication: {report['strategy']['deduplication']}.",
        f"- Source grouping: {report['strategy']['source_grouping']}.",
        f"- Split method: {report['strategy']['split']}.",
        f"- Limitation: {report['strategy']['limitation']}",
        "",
        "## Removed class annotations",
        "",
        "| Class | Input annotations removed |",
        "|---|---:|",
    ]
    for class_name, count in counts["input_removed_class_annotations"].items():
        lines.append(f"| {_escape(class_name)} | {count} |")

    lines.extend(
        [
            "",
            "## Final split sizes",
            "",
            "| Split | Images | Labels | Target ratio | Actual ratio | Deviation | Annotations |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, stats in report["splits"].items():
        lines.append(
            f"| {split} | {stats['images']} | {stats['labels']} | "
            f"{stats['target_ratio']:.2%} | {stats['actual_ratio']:.2%} | "
            f"{stats['ratio_deviation']:+.2%} | {stats['annotations']} |"
        )

    lines.extend(
        [
            "",
            "## Annotation counts by class",
            "",
            "| ID | Class | Train | Validation | Test | Total | Global share |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    total_annotations = counts["final_annotations"]
    for class_id, class_name in report["target_classes"].items():
        total = report["total_class_annotations"][class_name]
        lines.append(
            f"| {class_id} | {_escape(class_name)} | "
            f"{report['splits']['train']['class_annotations'][class_name]} | "
            f"{report['splits']['valid']['class_annotations'][class_name]} | "
            f"{report['splits']['test']['class_annotations'][class_name]} | {total} | "
            f"{total / total_annotations:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Image coverage by class",
            "",
            "| ID | Class | Train | Validation | Test | Total |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for class_id, class_name in report["target_classes"].items():
        lines.append(
            f"| {class_id} | {_escape(class_name)} | "
            f"{report['splits']['train']['class_image_coverage'][class_name]} | "
            f"{report['splits']['valid']['class_image_coverage'][class_name]} | "
            f"{report['splits']['test']['class_image_coverage'][class_name]} | "
            f"{report['total_class_image_coverage'][class_name]} |"
        )

    lines.extend(
        [
            "",
            "## Freeze audit",
            "",
            f"- Audit passed: **{audit['passed']}**",
            f"- Image/label one-to-one: **{audit['image_label_pairs_valid']}**",
            f"- Valid class ID range: `{audit['class_id_range'][0]}..{audit['class_id_range'][1]}`",
            f"- Remaining duplicate SHA-256 groups: **{audit['duplicate_sha256_groups']}**",
            f"- Cross-split SHA-256 groups: **{audit['cross_split_sha256_groups']}**",
            f"- Cross-split source groups: **{audit['cross_split_source_groups']}**",
            "",
            "## Duplicate groups removed",
            "",
        ]
    )
    if report["duplicate_groups"]:
        for group in report["duplicate_groups"]:
            lines.append(f"### `{group['sha256']}`")
            lines.append("")
            lines.append(f"- Kept: `{group['kept_image']}`")
            for removed in group["removed_images"]:
                lines.append(f"- Removed: `{removed}`")
            lines.append(
                f"- Label annotations exactly consistent: {group['label_annotations_consistent']}"
            )
            lines.append("")
    else:
        lines.extend(["No input duplicate groups were found.", ""])

    lines.extend(["## All removed files", ""])
    if report["removed_files"]:
        for item in report["removed_files"]:
            kept = f"; kept `{item['kept_image']}`" if "kept_image" in item else ""
            lines.append(
                f"- `{item['reason']}` — image `{item['image']}`; label `{item['label']}`{kept}"
            )
    else:
        lines.append("No files were removed.")

    lines.extend(
        [
            "",
            "## Final data.yaml",
            "",
            "```yaml",
            "path: .",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "nc: 7",
            "names:",
        ]
    )
    for class_id, class_name in report["target_classes"].items():
        lines.append(f"  {class_id}: {class_name}")
    lines.extend(["```", "", "Training was not started.", ""])
    return "\n".join(lines)


def write_finalization_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "finalization_report.json"
    markdown_path = output_dir / "finalization_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(finalization_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path
