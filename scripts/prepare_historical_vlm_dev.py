#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_record(record: dict[str, Any], project_root: Path) -> dict[str, Any]:
    image = project_root / record["image"]
    target = record["target"]
    return {
        "id": record["id"],
        "image": str(image.resolve()),
        "target": {
            "findings": [
                {
                    "person_box": [float(value) / 1000 for value in finding["person_box"]],
                    "violation": finding["violation"],
                }
                for finding in target.get("findings", [])
            ]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and normalize the historical feature-branch Dev labels.")
    parser.add_argument("--ref", default="origin/feature/vlm-grounding")
    parser.add_argument("--source", default="vlm/data/dev.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/vlm/real_dev_manifest.jsonl")
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "docs/evidence/vlm_dev_source_audit.json")
    args = parser.parse_args()
    try:
        raw = subprocess.run(
            ["git", "show", f"{args.ref}:{args.source}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        normalized = [normalize_record(record, PROJECT_ROOT) for record in records]
        missing = [row["image"] for row in normalized if not Path(row["image"]).is_file()]
        reviewers = Counter(
            str(record.get("review", {}).get("reviewer") or "missing") for record in records
        )
        groups = [str(record["source_group"]) for record in records]
        source_commit = subprocess.run(
            ["git", "rev-parse", args.ref], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        audit = {
            "schema_version": 1,
            "source_ref": args.ref,
            "source_commit": source_commit,
            "source_path": args.source,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "records": len(records),
            "unique_source_groups": len(set(groups)),
            "duplicate_source_groups": len(groups) - len(set(groups)),
            "target_findings": sum(len(row["target"]["findings"]) for row in normalized),
            "reviewer_provenance": dict(sorted(reviewers.items())),
            "missing_images": missing,
            "evidence_grade": "historical_mixed_provenance",
            "limitation": (
                "This recovered Dev set mixes named human, Codex-assisted, and missing reviewer "
                "provenance. Real-model runs are diagnostic and cannot verify the resume F1 claim."
            ),
        }
        if missing or audit["duplicate_source_groups"]:
            raise ValueError(f"Dev source audit failed: {audit}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in normalized),
            encoding="utf-8",
        )
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Historical Dev preparation failed: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
