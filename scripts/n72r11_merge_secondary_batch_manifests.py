#!/usr/bin/env python3
"""Merge a complete primary batch with an explicit N72R11 retry batch.

The primary and retry manifests are never edited.  For every frozen schedule
key, the primary PASS record is retained; a non-PASS primary record can be
replaced only by the matching retry record.  A partial result remains partial
and is intentionally rejected by the corpus builder.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "outputs/N72R11/secondary_event_manifest.json"
PRIMARY = ROOT / "outputs/N72R11/secondary_interactions_attempt_02/batch_manifest_attempt_02.json"
RETRY = ROOT / "outputs/N72R11/secondary_interactions_retry_attempt_03/batch_manifest_attempt_03.json"
OUTPUT = ROOT / "outputs/N72R11/secondary_batch_merged_attempt_03.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


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


def record_map(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"{label} has no records list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        event_id = str(record.get("event_id"))
        if event_id in {"", "None"} or event_id in result:
            raise RuntimeError(f"{label} has missing/duplicate event_id: {event_id}")
        result[event_id] = dict(record)
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--primary", type=Path, default=PRIMARY)
    parser.add_argument("--retry", type=Path, default=RETRY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    schedule_path = args.schedule if args.schedule.is_absolute() else ROOT / args.schedule
    primary_path = args.primary if args.primary.is_absolute() else ROOT / args.primary
    retry_path = args.retry if args.retry.is_absolute() else ROOT / args.retry
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    schedule = read_json(schedule_path)
    eligible = [dict(item) for item in schedule.get("events", []) if item.get("status") == "ELIGIBLE"]
    eligible_ids = [str(item["event_id"]) for item in eligible]
    if len(eligible_ids) != 527 or len(eligible_ids) != len(set(eligible_ids)):
        raise RuntimeError(f"frozen schedule is not 527 unique eligible events: {len(eligible_ids)}")
    primary = read_json(primary_path)
    retry = read_json(retry_path)
    primary_records = record_map(primary, "primary")
    retry_records = record_map(retry, "retry")
    if len(primary_records) != 527:
        raise RuntimeError(f"primary record count is not 527: {len(primary_records)}")
    unknown_retry = sorted(set(retry_records) - set(eligible_ids))
    if unknown_retry:
        raise RuntimeError(f"retry contains keys outside frozen schedule: {unknown_retry[:5]}")
    final_records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    missing_retry: list[str] = []
    for event_id in eligible_ids:
        primary_record = primary_records.get(event_id)
        if primary_record is None:
            raise RuntimeError(f"primary missing frozen event: {event_id}")
        primary_status = str(primary_record.get("status"))
        if primary_status == "PASS":
            chosen = dict(primary_record)
            source = "primary"
        else:
            retry_record = retry_records.get(event_id)
            if retry_record is None:
                missing_retry.append(event_id)
                continue
            chosen = dict(retry_record)
            source = "retry"
        chosen["merge_source"] = source
        chosen["primary_status"] = primary_status
        chosen["retry_status"] = None if event_id not in retry_records else str(retry_records[event_id].get("status"))
        chosen["runtime_future_gt_used"] = False
        final_records.append(chosen)
        provenance.append({
            "event_id": event_id,
            "source": source,
            "primary_status": primary_status,
            "retry_status": chosen["retry_status"],
        })
    if missing_retry:
        # Keep an explicit partial artifact rather than omitting missing keys.
        for event_id in missing_retry:
            primary_record = primary_records[event_id]
            chosen = dict(primary_record)
            chosen["merge_source"] = "primary_unresolved_failure"
            chosen["primary_status"] = str(primary_record.get("status"))
            chosen["retry_status"] = None
            chosen["runtime_future_gt_used"] = False
            final_records.append(chosen)
            provenance.append({"event_id": event_id, "source": chosen["merge_source"], "primary_status": chosen["primary_status"], "retry_status": None})
    final_records.sort(key=lambda item: eligible_ids.index(str(item["event_id"])))
    final_ids = [str(item["event_id"]) for item in final_records]
    duplicate_count = len(final_ids) - len(set(final_ids))
    missing_count = len(set(eligible_ids) - set(final_ids))
    if duplicate_count or missing_count:
        raise RuntimeError(f"merge key integrity failed: duplicate={duplicate_count} missing={missing_count}")
    counts: dict[str, int] = {}
    for record in final_records:
        key = str(record.get("status"))
        counts[key] = counts.get(key, 0) + 1
    final_status = "PASS_ALL_SELECTED_AFTER_RETRY" if counts == {"PASS": 527} else "PARTIAL_WITH_FAILURES"
    result = {
        "schema_version": "N72R11_SECONDARY_BATCH_MERGED_MANIFEST_V1",
        "status": final_status,
        "created_at_utc": now_utc(),
        "schedule": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "primary_manifest": str(primary_path),
        "primary_manifest_sha256": sha256_file(primary_path),
        "retry_manifest": str(retry_path),
        "retry_manifest_sha256": sha256_file(retry_path),
        "record_count": len(final_records),
        "expected_record_count": 527,
        "duplicate_count": duplicate_count,
        "missing_count": missing_count,
        "counts": counts,
        "primary_status": primary.get("status"),
        "retry_status": retry.get("status"),
        "retry_failure_evidence_preserved": True,
        "runtime_future_gt_used": False,
        "records": final_records,
        "provenance": provenance,
    }
    atomic_json(output_path, result)
    print(json.dumps({"status": final_status, "output": str(output_path), "record_count": len(final_records), "counts": counts, "duplicate_count": duplicate_count, "missing_count": missing_count}, sort_keys=True))
    return 0 if final_status == "PASS_ALL_SELECTED_AFTER_RETRY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
