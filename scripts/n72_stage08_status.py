#!/usr/bin/env python3
"""Write the N72 final-stage status after the report has been materialized."""

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
    report = ROOT / "docs" / "N72_FINAL_REPORT.md"
    final_status = OUT / "n72_final_status.json"
    preservation = OUT / "preservation_audit.json"
    payload = {
        "schema": "N72_STAGE_STATUS_V1",
        "stage": "N72-08_FINAL_GATE_AND_REPORT",
        "status": "CANDIDATE_PROVENANCE_PASS_PUBLIC_MAPPING_BLOCKED",
        "research_gate": "NOT_RUN_NO_REAL_HUMAN_TAPE",
        "report": str(report),
        "report_sha256": sha256(report),
        "final_status": str(final_status),
        "final_status_sha256_at_stage_write": sha256(final_status),
        "preservation_audit": str(preservation),
        "preservation_audit_sha256": sha256(preservation),
        "real_human_event_count": 0,
        "n72_exact_mapping_rows": 0,
        "n71_public_mapping_exact_rows": 0,
        "full_project_pytest_passed": 143,
        "production_authorized": False,
        "pass_efficacy": False,
        "next_minimum_action": "External provenance-complete real-human UI event tape and exact public mapping export.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "stage_08_status.json", payload)
    print(json.dumps({"status": payload["status"], "report_sha256": payload["report_sha256"], "path": str(OUT / "stage_08_status.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
