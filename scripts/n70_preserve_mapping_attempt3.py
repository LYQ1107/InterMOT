"""Preserve the third N70 mapping repair failure before the null-safe fix."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N70"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    names = (
        "diagnosis/mapping_audit.jsonl",
        "diagnosis/mapping_summary.json",
        "cache/candidate_cache_manifest.json",
        "cache/candidate_cache_audit.json",
        "stage_01_status.json",
        "stage_02_status.json",
    )
    atomic_json(OUT / "attempts/n70_stage01_mapping_attempt3_null_public_uniqueness_failure.json", {
        "schema": "N70_MAPPING_ATTEMPT3_FAILURE_V1",
        "status": "FAIL_PRESERVED_BEFORE_MINIMAL_REPAIR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "N70_STAGE_01_MAPPING_PROVENANCE",
        "root_cause": "The public-assignment absence contract was added, but candidate-chain uniqueness still attempted int(None) for explicit null public IDs.",
        "command_exit": 1,
        "traceback": "Traceback (most recent call last):\n  File scripts/n70_prepare_cache.py, line 1047, in <module>\n    main()\n  File scripts/n70_prepare_cache.py, line 853, in main\n    rows, summary, _ = process_event(event, selected)\n  File scripts/n70_prepare_cache.py, line 662, in process_event\n    row = candidate_mapping_row(...)\n  File scripts/n70_prepare_cache.py, line 435, in candidate_mapping_row\n    len(set(int(item[\\\"mapping\\\"][\\\"public_id\\\"]) for item in candidate_records))\nTypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'",
        "first_actionable_input": "dancetrack0008 / n37-dancetrack0008-0060-authoritative_reassign-002",
        "progress_before_failure": {
            "events_written": [
                "n37-dancetrack0001-0296-authoritative_reassign-001",
                "n37-dancetrack0002-0167-authoritative_reassign-001",
                "n37-dancetrack0006-0005-authoritative_reassign-003",
                "n37-dancetrack0006-0599-atomic_id_swap-001"
            ],
            "cache_outputs_are_not_a_complete_attempt": True,
            "mapping_audit_and_summary_not_committed_by_failed_run": True
        },
        "preserved_current_files": [
            {"path": str(OUT / relative), "sha256": sha256(OUT / relative)}
            for relative in names
        ],
        "repair_scope": "Filter null public IDs from uniqueness arithmetic while requiring explicit public_id_status for every such absence; do not synthesize IDs.",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
    })
    print(json.dumps({"status": "PRESERVED", "artifact": str(OUT / "attempts/n70_stage01_mapping_attempt3_null_public_uniqueness_failure.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
