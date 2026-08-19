#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESUME_CLAIMS = [
    (
        "dataset",
        "构建数据清洗、去重、同源分组切分、训练评估与可视化流水线，处理 "
        "8,762 张图片、43,727 个标注、7 类 PPE 目标，使用数据指纹和 "
        "source-group 隔离控制数据泄漏。",
    ),
    (
        "yolo",
        "完成 4 组 YOLO11 控制实验，最终模型独立测试集 mAP@50–95 0.4602、"
        "mAP@50 0.6615，较基线提升 0.0289，MPS 单图平均延迟 23.61 ms。",
    ),
    (
        "review",
        "自研人工审核 Web 平台，支持人物框编辑、违规确认、Test 盲审、"
        "脏数据排除和 JSONL 原子写入，并实现标注协议校验和评估数据门禁。",
    ),
    (
        "vlm",
        "完成 YOLO/Qwen3-VL/grounded-Qwen3-VL 消融实验；通过统一 "
        "IoU-grounded 指标定位 VLM 幻觉与 grounding 问题，Dev 最优方案 F1 0.600。",
    ),
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    source = load(PROJECT_ROOT / "docs/source-verification.json")
    vlm = load(PROJECT_ROOT / "docs/evidence/vlm_asset_audit.json")
    generated_from_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dataset_verified = source["dataset"]["status"] == "verified"
    claims = [
        {
            "id": "dataset",
            "resume_text": RESUME_CLAIMS[0][1],
            "status": "verified" if dataset_verified else "implemented_unverified",
            "code_locations": [
                "src/visionguard/remapping.py",
                "src/visionguard/finalization.py",
                "src/visionguard/evidence.py",
            ],
            "verification_commands": [
                ".venv/bin/python scripts/remap_dataset_classes.py --data "
                "training/source_downloads.yaml --output-dir training/datasets/safety_clean",
                ".venv/bin/python scripts/finalize_dataset.py --data "
                "training/datasets/safety_clean/data.yaml --output-dir "
                "training/datasets/safety_final --seed 42",
                ".venv/bin/python scripts/verify_resume_evidence.py --data "
                "training/datasets/safety_final/data.yaml --output docs/source-verification.json",
            ],
            "data_model_versions_hashes": {
                "upstream": "Roboflow Construction PPE v3, CC BY 4.0",
                "final_dataset_sha256": source["dataset"].get("dataset_sha256"),
                "data_yaml_sha256": source["dataset"].get("source_sha256"),
            },
            "recomputed_values": {
                key: source["dataset"].get(key)
                for key in (
                    "images",
                    "annotations",
                    "class_count",
                    "classes",
                    "duplicate_sha256_groups",
                    "cross_split_sha256_groups",
                    "cross_split_source_groups",
                )
            },
            "generated_artifacts": [
                "docs/source-verification.json",
                "docs/evidence/class_remap_report.json",
                "docs/evidence/finalization_report.json",
                "docs/evidence/freeze_manifest.json",
            ],
            "gap": None if dataset_verified else source["dataset"].get("missing"),
        },
        {
            "id": "yolo",
            "resume_text": RESUME_CLAIMS[1][1],
            "status": "verified" if source["yolo"]["status"] == "verified" else "implemented_unverified",
            "code_locations": [
                "src/visionguard/training.py",
                "src/visionguard/evaluation.py",
                "src/visionguard/benchmarking.py",
                "src/visionguard/evidence.py",
            ],
            "verification_commands": [
                ".venv/bin/python scripts/verify_resume_evidence.py --require-verified",
                "make comparison",
            ],
            "data_model_versions_hashes": {
                "dataset_sha256": source["dataset"].get("dataset_sha256"),
                "checkpoint_sha256": None,
            },
            "recomputed_values": source["yolo"].get("derived", {}),
            "generated_artifacts": ["docs/source-verification.json"],
            "gap": source["yolo"].get("missing", []),
        },
        {
            "id": "review",
            "resume_text": RESUME_CLAIMS[2][1],
            "status": "verified",
            "code_locations": [
                "src/visionguard/review.py",
                "src/visionguard/demo_server.py",
                "demo/static/app.js",
                "schemas/review_protocol.schema.json",
            ],
            "verification_commands": [
                ".venv/bin/python -m pytest -q tests/test_review.py",
                "node --check demo/static/app.js",
            ],
            "data_model_versions_hashes": {"protocol_version": "visionguard-review-v1"},
            "generated_artifacts": [
                "schemas/review_protocol.schema.json",
                "docs/evidence/review_visual_qa.json",
            ],
            "negative_tests": [
                "Test ground truth omitted from public queue API",
                "dirty/pending/unconfirmed samples blocked from eval",
                "interrupted temp-file replace preserves prior JSONL",
            ],
            "gap": "Local prototype; no authentication, multi-user locking study, or operator study.",
        },
        {
            "id": "vlm",
            "resume_text": RESUME_CLAIMS[3][1],
            "status": "verified" if vlm["resume_f1_0_600_verified"] else "unsupported",
            "code_locations": [
                "src/visionguard/vlm.py",
                "src/visionguard/vlm_evaluation.py",
                "scripts/evaluate_vlm.py",
            ],
            "verification_commands": [
                ".venv/bin/python scripts/prepare_historical_vlm_dev.py",
                ".venv/bin/python scripts/evaluate_vlm.py --config "
                "configs/vlm_qwen_real_historical.yaml --cache-dir outputs/vlm/cache "
                "--output outputs/vlm/qwen_real_historical.json",
                "make vlm-fixture",
                ".venv/bin/python scripts/promote_vlm_diagnostic.py",
            ],
            "data_model_versions_hashes": {
                "qwen_model_id": vlm["qwen"]["model_id"],
                "qwen_revision": vlm["qwen"]["revision"],
                "qwen_weights_sha256": vlm["qwen"]["weights"]["sha256"],
                "qwen_diagnostic_recomputed": vlm["qwen"].get("diagnostic_recomputed"),
                "dev_source_audit": vlm["dev_source_audit"],
                "yolo_checkpoint_sha256": vlm["yolo"]["selected_checkpoint"]["sha256"],
            },
            "generated_artifacts": [
                "docs/evidence/vlm_asset_audit.json",
                "docs/evidence/vlm_dev_source_audit.json",
                "docs/evidence/qwen_real_historical_result.json",
                "docs/evidence/QWEN_DIAGNOSTIC_FAILURES.md",
                "outputs/vlm/fixture_evaluation.json (fixture only)",
            ],
            "gap": vlm["blocking_reasons"],
            "fixture_is_evidence_for_real_claim": False,
        },
    ]
    document = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_from_head": generated_from_head,
        "allowed_statuses": ["verified", "implemented_unverified", "unsupported"],
        "claims": claims,
    }
    json_path = PROJECT_ROOT / "docs/resume-evidence.json"
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Resume evidence matrix",
        "",
        "Generated from machine-readable source audits. README prose is not accepted as source evidence.",
        "",
        "| Claim | Status | Evidence boundary |",
        "| --- | --- | --- |",
    ]
    for claim in claims:
        gap = claim.get("gap")
        boundary = "Source artifacts and gates pass." if not gap else json.dumps(gap, ensure_ascii=False)
        lines.append(f"| `{claim['id']}` | **{claim['status']}** | {boundary} |")
    lines.extend(["", "## Claims", ""])
    for claim in claims:
        lines.extend(
            [
                f"### {claim['id']} - {claim['status']}",
                "",
                f"> {claim['resume_text']}",
                "",
                "Code: " + ", ".join(f"`{path}`" for path in claim["code_locations"]),
                "",
                "Verification:",
                "",
            ]
        )
        lines.extend(f"- `{command}`" for command in claim["verification_commands"])
        lines.extend(["", "Artifacts: " + ", ".join(f"`{path}`" for path in claim["generated_artifacts"]), ""])
        if claim.get("gap"):
            lines.extend(["Gap: " + json.dumps(claim["gap"], ensure_ascii=False), ""])
    (PROJECT_ROOT / "docs/RESUME_EVIDENCE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({claim["id"]: claim["status"] for claim in claims}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
