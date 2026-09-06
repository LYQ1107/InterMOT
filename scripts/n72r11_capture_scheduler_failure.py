#!/usr/bin/env python3
"""Seal the first N72R11 batch scheduler failure without altering child logs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OLD_MANIFEST = ROOT / "outputs/N72R11/secondary_interactions_attempt_01/batch_manifest_attempt_01.json"
OUTPUT = ROOT / "outputs/N72R11/attempts/secondary_batch_gpu_slot_failure.json"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not OLD_MANIFEST.is_file():
        raise FileNotFoundError(OLD_MANIFEST)
    atomic_json(
        OUTPUT,
        {
            "schema_version": "N72R11_BATCH_SCHEDULER_FAILURE_V1",
            "status": "FAIL_RESOURCE_SCHEDULER_SLOT_REUSE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "old_batch_manifest": str(OLD_MANIFEST),
            "old_batch_manifest_sha256": sha256(OLD_MANIFEST),
            "supervisor_pid": 31600,
            "first_observed_active_records": [
                "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:045",
                "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:050",
                "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:055",
                "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:060",
            ],
            "first_observed_active_gpu_ids": [4, 4, 4, 4],
            "nvidia_smi_observation": {
                "physical_gpu_id": 4,
                "memory_used_mib": 30034,
                "memory_total_mib": 40960,
                "child_pids": [33309, 33415, 33562, 33710],
            },
            "child_oom_failures": [
                {
                    "event_id": "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:020",
                    "returncode": 1,
                    "failure": "OutOfMemoryError",
                },
                {
                    "event_id": "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:030",
                    "returncode": 1,
                    "failure": "OutOfMemoryError",
                },
                {
                    "event_id": "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:035",
                    "returncode": 1,
                    "failure": "OutOfMemoryError",
                },
                {
                    "event_id": "n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:060",
                    "returncode": 1,
                    "failure": "OutOfMemoryError",
                },
            ],
            "root_cause": "supervisor chose gpu_ids[len(active)] rather than a currently free physical GPU slot after child completion",
            "containment": "supervisor paused before additional dispatch; all already-started children were allowed to finish; old manifest and logs retained",
            "repair": "track physical GPU IDs in active slots and allocate only from the free-slot set",
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
        },
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
