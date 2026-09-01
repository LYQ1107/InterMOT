#!/usr/bin/env python3
"""Create an isolated N42 baseline/final protection snapshot.

The snapshot hashes project code/configuration and selected frozen artifacts,
hashes the shared checkpoints, and records metadata digests for the large
output trees so N42 cannot silently alter MOT/OVMOT or N39--N41 evidence.
It writes only under outputs/n42/snapshot/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "outputs" / "n42" / "snapshot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_record(path: Path, include_hash: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "size": int(path.stat().st_size),
        "mtime_ns": int(path.stat().st_mtime_ns),
    }
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def tree_inventory(path: Path) -> dict[str, Any]:
    entries = []
    if path.is_dir():
        for item in sorted(path.rglob("*")):
            if item.is_file():
                entries.append(
                    {
                        "path": str(item.relative_to(ROOT)) if item.is_relative_to(ROOT) else str(item),
                        "size": int(item.stat().st_size),
                        "mtime_ns": int(item.stat().st_mtime_ns),
                    }
                )
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def run(label: str) -> dict[str, Any]:
    code_files = []
    for directory in (ROOT / "sam3_intermot", ROOT / "scripts", ROOT / "tests"):
        if directory.is_dir():
            code_files.extend(
                item for item in directory.rglob("*.py") if "__pycache__" not in item.parts
            )
    config_files = [item for item in (ROOT / "configs").rglob("*") if item.is_file()]
    protected_text = [
        ROOT / "AGENTS.md",
        ROOT / "docs" / "N39_FINAL_REPORT.md",
        ROOT / "docs" / "N41_FINAL_REPORT.md",
        ROOT / "outputs" / "n39" / "n39_final_gate.json",
        ROOT / "outputs" / "n41" / "n41_final_gate.json",
        ROOT / "outputs" / "n41" / "diagnostic" / "diagnostic_interpretation.json",
        ROOT / "outputs" / "n41" / "diagnostic" / "candidate_pair_summary.json",
        ROOT / "outputs" / "n41" / "source_replay" / "posthoc_source_results.json",
        ROOT / "outputs" / "n41" / "stage_03_status.json",
        ROOT / "outputs" / "n41" / "stage_04_status.json",
        ROOT / "outputs" / "n40" / "stage_02_pause_status.json",
    ]
    checkpoints = [
        ROOT / "checkpoints" / "sam3.1_mirror" / "sam3.1_multiplex.pt",
        ROOT / "outputs" / "n9" / "checkpoints" / "osnet_x1_0_market1501.pth",
    ]
    protected_text = [path for path in protected_text if path.is_file()]
    checkpoints = [path for path in checkpoints if path.is_file()]
    sibling = ROOT.parent / "InterMOT"
    ovmot_candidates = sorted(
        str(path)
        for path in ROOT.parent.iterdir()
        if path.is_dir() and "ovmot" in path.name.lower()
    )
    payload = {
        "protocol": "N42_ISOLATED_PROTECTION_SNAPSHOT_V1",
        "label": str(label),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"]),
        },
        "project_code_hashes": [file_record(path) for path in sorted(code_files)],
        "project_config_hashes": [file_record(path) for path in sorted(config_files)],
        "protected_text_hashes": [file_record(path) for path in sorted(protected_text)],
        "shared_checkpoint_hashes": [file_record(path) for path in sorted(checkpoints)],
        "protected_output_tree_inventories": [
            tree_inventory(ROOT / "outputs" / name) for name in ("n39", "n40", "n41")
        ],
        "mot_ovmot_boundary": {
            "sibling_intermot_root": str(sibling),
            "sibling_exists": sibling.is_dir(),
            "sibling_metadata_inventory": tree_inventory(sibling) if sibling.is_dir() else None,
            "ovmot_directories_under_interactive": ovmot_candidates,
            "n42_write_root": str(ROOT / "outputs" / "n42"),
            "production_files_modified_by_snapshot": False,
        },
    }
    output = SNAPSHOT_DIR / f"{label}.json"
    atomic_json(output, payload)
    payload["output"] = str(output.relative_to(ROOT))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result = run(args.label)
    print(json.dumps({"status": "PASS", "output": result["output"], "code_files": len(result["project_code_hashes"]), "config_files": len(result["project_config_hashes"])}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
