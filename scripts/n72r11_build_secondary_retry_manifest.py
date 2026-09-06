#!/usr/bin/env python3
"""Freeze an explicit retry list from one completed N72R11 batch manifest.

Only records that are not PASS are selected.  The primary manifest and its
failure artifacts remain immutable; this file is a selector for a new output
root, not a replacement for the original evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = ROOT / "outputs/N72R11/secondary_interactions_attempt_02/batch_manifest_attempt_02.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11/secondary_retry_manifest_attempt_03.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-manifest", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    primary = args.primary_manifest if args.primary_manifest.is_absolute() else ROOT / args.primary_manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = read_json(primary)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("primary batch manifest has no records")
    event_ids = [str(record.get("event_id")) for record in records]
    if any(value in {"None", ""} for value in event_ids) or len(event_ids) != len(set(event_ids)):
        raise RuntimeError("primary batch manifest has missing or duplicate event IDs")
    retry_records: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["event_id"])):
        if str(record.get("status")) == "PASS":
            continue
        event_id = str(record["event_id"])
        failure_path = None
        failure: dict[str, Any] | None = None
        log_value = record.get("log")
        if log_value:
            output_root = Path(str(log_value)).parent.parent
            candidate = output_root / "attempts" / f"{event_id}.failure.json"
            if candidate.is_file():
                failure_path = str(candidate)
                failure = read_json(candidate)
        retry_records.append(
            {
                "event_id": event_id,
                "sequence": str(record.get("sequence")),
                "secondary_frame": int(record.get("secondary_frame", -1)),
                "action_type": str(record.get("action_type")),
                "retry_required": True,
                "primary_status": str(record.get("status")),
                "primary_returncode": record.get("returncode"),
                "primary_gpu_id": record.get("gpu_id"),
                "primary_log": log_value,
                "primary_failure_artifact": failure_path,
                "primary_failure_artifact_sha256": None if failure_path is None else sha256_file(Path(failure_path)),
                "failure_type": None if failure is None else failure.get("failure_type"),
                "failure_message": None if failure is None else failure.get("message"),
                "runtime_future_gt_used": False,
            }
        )
    if not retry_records:
        raise RuntimeError("primary manifest is already complete; no retry items")
    result = {
        "schema_version": "N72R11_SECONDARY_RETRY_MANIFEST_V1",
        "status": "RETRY_LIST_FROZEN",
        "created_at_utc": now_utc(),
        "primary_manifest": str(primary),
        "primary_manifest_sha256": sha256_file(primary),
        "primary_status": payload.get("status"),
        "primary_record_count": len(records),
        "retry_count": len(retry_records),
        "pass_preserved_count": len(records) - len(retry_records),
        "failure_type_counts": {
            failure_type: sum(1 for item in retry_records if item.get("failure_type") == failure_type)
            for failure_type in sorted({str(item.get("failure_type")) for item in retry_records})
        },
        "event_ids": [str(item["event_id"]) for item in retry_records],
        "records": retry_records,
        "runtime_future_gt_used": False,
        "preserves_primary_failures": True,
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "output": str(output), "retry_count": len(retry_records), "failure_type_counts": result["failure_type_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
