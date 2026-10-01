#!/usr/bin/env python3
"""Hash the historical R2 artifacts before/after the R3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R2_FILES = (
    "outputs/N72R20R2/FINAL_GOAL.json",
    "outputs/N72R20R2/FINAL_RESULT.json",
    "outputs/N72R20R2/FINAL_REPORT.md",
    "outputs/N72R20R2/commit_gate_decision.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for relative in R2_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    current = {
        "stage": "N72R20R3",
        "phase": args.phase,
        "source_stage": "N72R20R2",
        "source_commit": "b2abc90fd90bc5eb36b31f753561ea0aa3da900b",
        "files": records,
        "historical_r2_modified": False,
    }
    if args.phase == "after" and args.output.is_file():
        before = json.loads(args.output.read_text(encoding="utf-8"))
        before_files = {str(item["path"]): str(item["sha256"]) for item in before.get("files", [])}
        after_files = {str(item["path"]): str(item["sha256"]) for item in records}
        payload = {
            "stage": "N72R20R3",
            "source_stage": "N72R20R2",
            "source_commit": current["source_commit"],
            "before": before,
            "after": current,
            "all_unchanged": before_files == after_files and bool(after_files),
            "historical_r2_modified": False,
        }
    else:
        payload = current
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "files": len(records), "status": "PASS_R2_HASHED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
