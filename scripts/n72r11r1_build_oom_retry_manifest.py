#!/usr/bin/env python3
"""Derive the exact N72R11 OOM retry set without hand-copying event IDs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/N72R11/secondary_retry_manifest_attempt_03.json"
MERGED = ROOT / "outputs/N72R11/secondary_batch_merged_attempt_03.json"
OUTPUT = ROOT / "outputs/N72R11R1/oom_retry_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    merged = json.loads(MERGED.read_text(encoding="utf-8"))
    source_records = [
        dict(item)
        for item in source.get("records", [])
        if item.get("failure_type") == "OutOfMemoryError"
    ]
    source_records.sort(key=lambda item: str(item["event_id"]))
    event_ids = [str(item["event_id"]) for item in source_records]
    if len(event_ids) != 37 or len(event_ids) != len(set(event_ids)):
        raise RuntimeError(f"expected 37 unique OOM records, found {len(event_ids)}")
    merged_by_id = {str(item["event_id"]): item for item in merged.get("records", [])}
    if set(event_ids) - set(merged_by_id):
        raise RuntimeError("OOM retry manifest contains an event absent from merged evidence")
    if any(str(merged_by_id[event_id].get("status")) != "FAIL_CHILD" for event_id in event_ids):
        raise RuntimeError("OOM retry manifest contains a non-failing merged record")
    records = []
    for item in source_records:
        records.append(
            {
                "event_id": str(item["event_id"]),
                "sequence": str(item["sequence"]),
                "secondary_frame": int(item["secondary_frame"]),
                "action_type": str(item["action_type"]),
                "failure_type": "OutOfMemoryError",
                "primary_failure_artifact": str(item["primary_failure_artifact"]),
                "primary_failure_artifact_sha256": str(item["primary_failure_artifact_sha256"]),
                "primary_returncode": int(item["primary_returncode"]),
                "primary_gpu_id": item.get("primary_gpu_id"),
                "retry_reason": "preserved_N72R11_official_SAM3_future_propagation_OOM",
                "runtime_future_gt_used": False,
            }
        )
    output = {
        "schema_version": "N72R11R1_OOM_RETRY_MANIFEST_V1",
        "status": "OOM_RETRY_LIST_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_retry_manifest": str(SOURCE),
        "source_retry_manifest_sha256": sha256_file(SOURCE),
        "source_merged_manifest": str(MERGED),
        "source_merged_manifest_sha256": sha256_file(MERGED),
        "retry_count": len(records),
        "event_ids": event_ids,
        "failure_type_counts": {"OutOfMemoryError": len(records)},
        "records": records,
        "runtime_future_gt_used": False,
        "preserves_original_failure_artifacts": True,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps({"status": output["status"], "output": str(OUTPUT), "retry_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
