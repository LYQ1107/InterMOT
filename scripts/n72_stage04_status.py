#!/usr/bin/env python3
"""Write the N72 Stage-04 implementation/readiness status atomically."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    module = ROOT / "sam3_intermot" / "interaction" / "n72_real_human.py"
    mapping = ROOT / "sam3_intermot" / "provenance" / "mapping.py"
    cli = ROOT / "scripts" / "n72_real_human_event_cli.py"
    collection_doc = ROOT / "docs" / "N72_REAL_HUMAN_COLLECTION.md"
    stage03 = OUT / "stage_03_status.json"
    payload = {
        "schema": "N72_STAGE_STATUS_V1",
        "stage": "N72-04_REAL_HUMAN_EVENT_RECORDER_VALIDATOR",
        "status": "IMPLEMENTATION_READY_WAITING_EXTERNAL_COLLECTION",
        "overall_allowed_status": "PARTIAL_MAPPING_PASS_REAL_HUMAN_RECORDER_READY",
        "real_human_tape": {
            "verified_event_count": 0,
            "input_present": False,
            "source": "No external UI/annotator JSONL or raw payload was supplied",
            "simulated_from_gt_imported": False,
            "test_fixture_imported": False,
        },
        "implementation": {
            "module": str(module),
            "mapping_module": str(mapping),
            "cli": str(cli),
            "collection_document": str(collection_doc),
            "atomic_rejection_artifacts": str(OUT / "human_tape" / "attempts"),
            "append_only_recorder": True,
            "runtime_future_gt_read": False,
            "gt_or_simulator_import": False,
        },
        "focused_tests": {
            "command": "pytest -q tests/test_n72_real_human.py tests/test_n72_raw_provenance.py tests/test_n72_mapping.py",
            "status": "PASS",
            "passed": 25,
            "failed": 0,
            "first_fixture_failure_preserved": str(OUT / "attempts" / "n72_stage04_test_failure_attempt1.json"),
            "environment_warning_preserved": "osr_lib-1.1.0-nspkg.pth site.addpackage AttributeError; pytest still completed",
        },
        "input_hashes": {
            "stage_03_status": sha256(stage03),
            "module": sha256(module),
            "mapping_module": sha256(mapping),
            "cli": sha256(cli),
            "collection_document": sha256(collection_doc),
        },
        "downstream": {
            "full_loop_started": False,
            "replay_started": False,
            "training_started": False,
            "calibration_started": False,
            "lora_started": False,
            "reason": "Real tape is external input and currently absent; schema readiness does not authorize scientific execution.",
        },
        "next_step": "Collect genuine external UI records and raw payloads, then run the documented validator before any full-loop/replay.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "stage_04_status.json", payload)
    print(json.dumps({"status": payload["status"], "real_human_event_count": 0, "path": str(OUT / "stage_04_status.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
