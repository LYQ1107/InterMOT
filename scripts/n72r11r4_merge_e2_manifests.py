#!/usr/bin/env python3
"""Losslessly merge N72R11R4 E2 event attempts under a strict 32-event gate.

Initial PASS records are retained.  For an initial failure, the newest
attempt is selected only when it is a sealed PASS for the same event ID.  An
unresolved event remains in the output as a non-PASS record, so the resulting
manifest cannot be consumed as a complete Oracle input.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"


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
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
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
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def frozen_events() -> dict[str, dict[str, Any]]:
    payload = read_json(PROTOCOL)
    events = payload.get("source_event_selection", {}).get("events", [])
    result = {str(item["event_id"]): dict(item) for item in events}
    if len(result) != 32 or len(events) != 32:
        raise RuntimeError(f"expected exactly 32 unique frozen events, found {len(events)} records/{len(result)} keys")
    return result


def validate_pass(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    if record.get("status") != "PASS":
        return False, f"manifest_status={record.get('status')}"
    done_value = record.get("done")
    if not done_value:
        return False, "missing_done_path"
    done_path = resolve(done_value)
    if not done_path.is_file():
        return False, f"done_missing={done_path}"
    done = read_json(done_path)
    if done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT":
        return False, f"done_status={done.get('status')}"
    if done.get("runtime_future_gt_used") is not False or done.get("posthoc_gt_used") is not True:
        return False, "done_provenance_failed"
    seal_value = done.get("runtime_event_sealed")
    if not seal_value or not resolve(seal_value).is_file():
        return False, "runtime_seal_missing"
    seal = read_json(resolve(seal_value))
    if seal.get("runtime_future_gt_used") is not False or seal.get("runtime_gt_read") is not False:
        return False, "runtime_seal_gt_provenance_failed"
    posthoc_value = done.get("posthoc")
    if not posthoc_value or not resolve(posthoc_value).is_file():
        return False, "posthoc_missing"
    posthoc = read_json(resolve(posthoc_value))
    if posthoc.get("status") != "PASS_N72R11_POSTHOC_EVENT" or posthoc.get("posthoc_gt_used") is not True:
        return False, "posthoc_provenance_failed"
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--retry", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, default=None)
    args = parser.parse_args()
    expected = frozen_events()
    paths = [args.initial, *args.retry]
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        payloads.append((path, read_json(path)))

    by_event: dict[str, list[dict[str, Any]]] = {}
    for path, payload in payloads:
        for raw_record in payload.get("records", []):
            if not isinstance(raw_record, Mapping):
                continue
            event_id = str(raw_record.get("event_id"))
            if event_id not in expected:
                raise RuntimeError(f"record is not one of the 32 frozen event IDs: {event_id}")
            record = dict(raw_record)
            record["source_manifest"] = str(path)
            record["source_manifest_sha256"] = sha256_file(path)
            by_event.setdefault(event_id, []).append(record)

    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for event_id in sorted(expected):
        attempts = by_event.get(event_id, [])
        # Later attempts are considered only for this same event and only if
        # they have a complete sealed PASS artifact.
        selected: dict[str, Any] | None = None
        validation_errors: list[str] = []
        for candidate in reversed(attempts):
            valid, reason = validate_pass(candidate)
            if valid:
                selected = candidate
                break
            if reason:
                validation_errors.append(reason)
        if selected is None:
            if not attempts:
                record = {"event_id": event_id, "sequence": expected[event_id].get("sequence"), "action_type": expected[event_id].get("action_type"), "status": "MISSING_EVENT", "attempt": None, "returncode": None}
            else:
                record = dict(attempts[-1])
                record["status"] = "BLOCKED_UNRESOLVED_EVENT"
            record["source_attempts"] = [
                {"attempt": item.get("attempt"), "status": item.get("status"), "returncode": item.get("returncode"), "source_manifest": item.get("source_manifest"), "done": item.get("done")}
                for item in attempts
            ]
            record["validation_errors"] = validation_errors
            unresolved.append({"event_id": event_id, "reason": validation_errors or ["no_attempt_record"]})
        else:
            record = dict(selected)
            record["source_attempts"] = [
                {"attempt": item.get("attempt"), "status": item.get("status"), "returncode": item.get("returncode"), "source_manifest": item.get("source_manifest"), "done": item.get("done")}
                for item in attempts
            ]
            record["selected_attempt"] = selected.get("attempt")
        records.append(record)

    ids = [str(record.get("event_id")) for record in records]
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("status"))
        counts[key] = counts.get(key, 0) + 1
    completeness = {
        "required_events": 32,
        "manifest_records": len(records),
        "unique_event_ids": len(set(ids)),
        "duplicate_event_ids": len(ids) - len(set(ids)),
        "missing_event_ids": sorted(set(expected) - set(ids)),
        "unexpected_event_ids": sorted(set(ids) - set(expected)),
        "pass_events": counts.get("PASS", 0),
        "unresolved_events": unresolved,
        "unavailable_or_nonpass_events": sum(status != "PASS" for status in (str(record.get("status")) for record in records)),
        "strict_complete": len(records) == 32 and len(set(ids)) == 32 and not unresolved and counts.get("PASS", 0) == 32,
    }
    status = "PASS_N72R11R4_E2_MERGED_COMPLETE" if completeness["strict_complete"] else "BLOCKED_N72R11R4_E2_INCOMPLETE"
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = {
        "schema_version": "N72R11R4_E2_MERGED_MANIFEST_V1",
        "status": status,
        "created_at_utc": now_utc(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "event_count": 32,
        "horizon": 100,
        "model_kind": "pctis",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "attempt_manifest_sources": [{"path": str(path), "sha256": sha256_file(path), "status": payload.get("status"), "counts": payload.get("counts")} for path, payload in payloads],
        "counts": counts,
        "completeness": completeness,
        "records": records,
        "oracle_authorized": bool(completeness["strict_complete"]),
    }
    atomic_json(output, result)
    if args.gate_output is not None:
        gate_output = args.gate_output if args.gate_output.is_absolute() else ROOT / args.gate_output
        atomic_json(
            gate_output,
            {
                "schema_version": "N72R11R4_E2_MERGE_GATE_V1",
                "created_at_utc": result["created_at_utc"],
                "status": status,
                "source_merged_manifest": str(output),
                "source_merged_manifest_sha256": sha256_file(output),
                "completeness": completeness,
                "oracle_authorized": bool(completeness["strict_complete"]),
                "preserves_unresolved_failure": bool(unresolved),
                "runtime_future_gt_used": False,
                "not_real_human_evidence": True,
            },
        )
    print(json.dumps({"status": status, "output": str(output), "counts": counts, "completeness": completeness}, sort_keys=True))
    return 0 if completeness["strict_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
