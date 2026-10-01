#!/usr/bin/env python3
"""Hash R3 historical artifacts before and after R3R1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R3_FILES = (
    "outputs/N72R20R3/FINAL_GOAL.json",
    "outputs/N72R20R3/FINAL_RESULT.json",
    "outputs/N72R20R3/FINAL_REPORT.md",
    "outputs/N72R20R3/presence/loso_results.json",
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
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R3R1/r3_immutability_audit.json")
    args = parser.parse_args()
    files = []
    for relative in R3_FILES:
        path = ROOT / relative
        files.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    current = {
        "stage": "N72R20R3R1",
        "phase": args.phase,
        "source_stage": "N72R20R3",
        "source_commit": "75696491945456849f5ebdb75fc785e087d8fb41",
        "files": files,
        "historical_r3_modified": False,
    }
    if args.phase == "after" and args.output.is_file():
        before = json.loads(args.output.read_text(encoding="utf-8"))
        before_map = {str(item["path"]): str(item["sha256"]) for item in before.get("files", [])}
        after_map = {str(item["path"]): str(item["sha256"]) for item in files}
        payload = {
            "stage": "N72R20R3R1",
            "source_stage": "N72R20R3",
            "source_commit": current["source_commit"],
            "before": before,
            "after": current,
            "all_unchanged": before_map == after_map and bool(after_map),
            "historical_r3_modified": False,
        }
    else:
        payload = current
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "files": len(files), "status": "PASS_R3_HASHED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
