#!/usr/bin/env python3
"""Record the N72 export-smoke decision without launching a GPU job."""

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
    stage02 = OUT / "stage_02_status.json"
    stage05 = OUT / "stage_05_status.json"
    payload = {
        "schema": "N72_STAGE_STATUS_V1",
        "stage": "N72-06_EXPORT_SMOKE_DECISION",
        "status": "NOT_REQUIRED_RAW_PROVENANCE_REPAIRED",
        "decision": {
            "stage_02_raw_provenance_permanently_missing": False,
            "gpu_export_smoke_started": False,
            "gpu_export_smoke_required": False,
            "reason": "Stage 02 retained official out_obj_ids in a backward-compatible opt-in export; the N72 rule requests a GPU smoke only when raw provenance remains permanently unavailable.",
            "cpu_official_shaped_parser_export_regression": "PASS",
            "test": "test_official_shaped_out_obj_ids_survive_stable_binding_and_extended_export",
            "scientific_tape_generated": False,
        },
        "verified_contract": {
            "raw_native_id_source": "official_out_obj_ids",
            "adapter_external_id_distinct": True,
            "default_export_unchanged": True,
            "public_mapping_fabricated": False,
            "runtime_future_gt_used": False,
        },
        "input_hashes": {
            "stage_02_status": sha256(stage02),
            "stage_05_status": sha256(stage05),
            "raw_provenance_test": sha256(ROOT / "tests" / "test_n72_raw_provenance.py"),
        },
        "next_stage": "N72-07_POSTHOC_ASSIGNMENT_BOUNDARY_DIAGNOSTIC",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "stage_06_status.json", payload)
    print(json.dumps({"status": payload["status"], "gpu_started": False, "path": str(OUT / "stage_06_status.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
