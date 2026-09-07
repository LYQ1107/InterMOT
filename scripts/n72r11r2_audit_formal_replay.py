#!/usr/bin/env python3
"""Lossless audit/merge for the N72R11R2 formal development replay.

The original batch attempts are immutable.  This utility never edits either
source manifest or an event artifact.  It selects the first sealed PASS for a
frozen event (attempt 1 before any resume attempt), validates the event's
runtime seal and posthoc artifact, and writes a new combined audit.  RUNNING,
NOT_RUN, and child failures remain explicit outcomes and can never be promoted
to PASS by the merge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
DEFAULT_ATTEMPT_01 = ROOT / "outputs/N72R11R2/formal_replay_resource_censored_attempt_01/formal_batch_manifest_attempt_01.json"
DEFAULT_ATTEMPT_03 = ROOT / "outputs/N72R11R2/formal_replay_resource_censored_attempt_03/formal_batch_manifest_attempt_03.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json"


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


def resolve_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def frozen_events(protocol_path: Path) -> list[dict[str, Any]]:
    payload = read_json(protocol_path)
    values = payload.get("source_event_selection", {}).get("events", [])
    if not isinstance(values, list) or len(values) != 32:
        raise RuntimeError(f"expected 32 frozen N72R9 events, found {len(values) if isinstance(values, list) else 'non-list'}")
    events = [dict(value) for value in values if isinstance(value, Mapping)]
    if len(events) != 32:
        raise RuntimeError("frozen event list contains non-object records")
    ids = [str(value.get("event_id")) for value in events]
    if any(value in {"None", ""} for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("frozen event IDs are missing or duplicated")
    for event in events:
        if event.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"frozen event violates runtime GT contract: {event['event_id']}")
        if event.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"unexpected frozen interaction source: {event['event_id']}")
    return sorted(events, key=lambda value: str(value["event_id"]))


def load_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = read_json(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"manifest has no records list: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not record.get("event_id"):
            raise RuntimeError(f"malformed record in {path}")
        event_id = str(record["event_id"])
        if event_id in by_id:
            raise RuntimeError(f"duplicate event in manifest {path}: {event_id}")
        by_id[event_id] = dict(record)
    return payload, by_id


def validate_sealed_record(record: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    if str(record.get("status")) != "PASS":
        reasons.append(f"manifest_status={record.get('status')}")
    if record.get("returncode") not in (0, "0"):
        reasons.append(f"returncode={record.get('returncode')}")
    done = resolve_path(record.get("done"))
    if done is None or not done.is_file():
        reasons.append("done_missing")
        return False, reasons, details
    details["done"] = str(done)
    actual_done_hash = sha256_file(done)
    details["done_sha256"] = actual_done_hash
    expected_done_hash = record.get("done_sha256")
    if expected_done_hash and str(expected_done_hash) != actual_done_hash:
        reasons.append("done_hash_mismatch")
    try:
        done_payload = read_json(done)
    except Exception as exc:
        reasons.append(f"done_unreadable={type(exc).__name__}")
        return False, reasons, details
    details["done_status"] = done_payload.get("status")
    if done_payload.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT":
        reasons.append(f"done_status={done_payload.get('status')}")
    if done_payload.get("runtime_future_gt_used") is not False:
        reasons.append("done_runtime_future_gt_used_not_false")
    if done_payload.get("interaction_source") != "simulated_from_gt":
        reasons.append("done_interaction_source_unexpected")
    runtime_seal = resolve_path(done_payload.get("runtime_event_sealed"))
    if runtime_seal is None or not runtime_seal.is_file():
        reasons.append("runtime_event_seal_missing")
    else:
        seal_hash = sha256_file(runtime_seal)
        details["runtime_event_sealed"] = str(runtime_seal)
        details["runtime_event_sealed_sha256"] = seal_hash
        expected = done_payload.get("runtime_event_sealed_sha256")
        if expected and str(expected) != seal_hash:
            reasons.append("runtime_event_seal_hash_mismatch")
        try:
            seal_payload = read_json(runtime_seal)
            if seal_payload.get("status") != "PASS_N72R11_ALL_RUNTIME_SEALED":
                reasons.append(f"runtime_event_seal_status={seal_payload.get('status')}")
            if seal_payload.get("runtime_future_gt_used") is not False:
                reasons.append("runtime_event_seal_future_gt_not_false")
            if seal_payload.get("gt_loaded") is not False:
                reasons.append("runtime_event_seal_gt_loaded_not_false")
        except Exception as exc:
            reasons.append(f"runtime_event_seal_unreadable={type(exc).__name__}")
    posthoc = resolve_path(done_payload.get("posthoc"))
    if posthoc is None or not posthoc.is_file():
        reasons.append("posthoc_missing")
    else:
        posthoc_hash = sha256_file(posthoc)
        details["posthoc"] = str(posthoc)
        details["posthoc_sha256"] = posthoc_hash
        expected = done_payload.get("posthoc_sha256")
        if expected and str(expected) != posthoc_hash:
            reasons.append("posthoc_hash_mismatch")
        try:
            posthoc_payload = read_json(posthoc)
            if posthoc_payload.get("status") != "PASS_N72R11_POSTHOC_EVENT":
                reasons.append(f"posthoc_status={posthoc_payload.get('status')}")
            if posthoc_payload.get("runtime_future_gt_used") is not False:
                reasons.append("posthoc_runtime_future_gt_not_false")
            if posthoc_payload.get("posthoc_gt_used") is not True:
                reasons.append("posthoc_gt_used_not_true")
        except Exception as exc:
            reasons.append(f"posthoc_unreadable={type(exc).__name__}")
    return not reasons, reasons, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--attempt-01", type=Path, default=DEFAULT_ATTEMPT_01)
    parser.add_argument("--attempt-03", type=Path, default=DEFAULT_ATTEMPT_03)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    attempt_01 = args.attempt_01 if args.attempt_01.is_absolute() else ROOT / args.attempt_01
    attempt_03 = args.attempt_03 if args.attempt_03.is_absolute() else ROOT / args.attempt_03
    output = args.output if args.output.is_absolute() else ROOT / args.output
    events = frozen_events(protocol)
    manifest_01, records_01 = load_records(attempt_01)
    manifest_03, records_03 = load_records(attempt_03)
    frozen_ids = {str(event["event_id"]) for event in events}
    source_out_of_scope = sorted((set(records_01) | set(records_03)) - frozen_ids)
    if source_out_of_scope:
        raise RuntimeError(f"source manifests contain out-of-scope events: {source_out_of_scope[:5]}")
    selected: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event["event_id"])
        candidates: list[tuple[str, dict[str, Any]]] = []
        for label, record_map in (("attempt_01", records_01), ("attempt_03", records_03)):
            record = record_map.get(event_id)
            if record is None:
                continue
            valid, reasons, details = validate_sealed_record(record)
            candidates.append((label, {"record": record, "valid": valid, "reasons": reasons, "details": details}))
            if valid:
                break
        chosen = next((value for value in candidates if value[1]["valid"]), None)
        source_summary = []
        for label, value in candidates:
            record = value["record"]
            source_summary.append({
                "source": label,
                "status": record.get("status"),
                "returncode": record.get("returncode"),
                "valid_sealed_pass": bool(value["valid"]),
                "reasons": value["reasons"],
                "log": record.get("log"),
            })
        base = {
            "event_id": event_id,
            "sequence": event.get("sequence"),
            "action_type": event.get("action_type"),
            "source_attempts": source_summary,
        }
        if chosen is None:
            outcomes.append({**base, "status": "UNSEALED_OR_FAILED", "selected_attempt": None})
            continue
        label, value = chosen
        record = value["record"]
        detail = value["details"]
        selected.append({
            **base,
            "status": "PASS",
            "selected_attempt": label,
            "attempt": record.get("attempt"),
            "gpu_id": record.get("gpu_id"),
            "done": detail.get("done"),
            "done_sha256": detail.get("done_sha256"),
            "runtime_event_sealed": detail.get("runtime_event_sealed"),
            "runtime_event_sealed_sha256": detail.get("runtime_event_sealed_sha256"),
            "posthoc": detail.get("posthoc"),
            "posthoc_sha256": detail.get("posthoc_sha256"),
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
        })
        outcomes.append({**base, "status": "PASS", "selected_attempt": label})
    selected_ids = [str(value["event_id"]) for value in selected]
    duplicate_ids = sorted({value for value in selected_ids if selected_ids.count(value) > 1})
    missing_ids = sorted(frozen_ids - set(selected_ids))
    counts: dict[str, int] = {}
    for value in outcomes:
        status = str(value["status"])
        counts[status] = counts.get(status, 0) + 1
    strict_complete = (
        len(events) == 32
        and len(selected) == 32
        and len(set(selected_ids)) == 32
        and not duplicate_ids
        and not missing_ids
        and all(value["status"] == "PASS" for value in outcomes)
    )
    result = {
        "schema_version": "N72R11R2_FORMAL_REPLAY_COMBINED_AUDIT_V1",
        "status": "PASS_FORMAL_DEVELOPMENT_REPLAY_COMPLETE" if strict_complete else "INCOMPLETE_FORMAL_DEVELOPMENT_REPLAY",
        "created_at_utc": now_utc(),
        "protocol": str(protocol.resolve()),
        "protocol_sha256": sha256_file(protocol),
        "source_manifests": [
            {"attempt": 1, "path": str(attempt_01.resolve()), "sha256": sha256_file(attempt_01), "status": manifest_01.get("status")},
            {"attempt": 3, "path": str(attempt_03.resolve()), "sha256": sha256_file(attempt_03), "status": manifest_03.get("status")},
        ],
        "selection_rule": "first_valid_sealed_PASS_by_attempt_order_01_then_03; never promote RUNNING/NOT_RUN/FAIL_CHILD",
        "event_count_required": 32,
        "event_count_selected": len(selected),
        "unique_event_count_selected": len(set(selected_ids)),
        "duplicate_event_ids": duplicate_ids,
        "missing_event_ids": missing_ids,
        "counts": counts,
        "strict_complete": strict_complete,
        "resource_censored_development": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "selected_records": sorted(selected, key=lambda value: str(value["event_id"])),
        "outcomes": sorted(outcomes, key=lambda value: str(value["event_id"])),
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "output": str(output), "counts": counts, "missing": missing_ids, "duplicates": duplicate_ids}, sort_keys=True))
    return 0 if strict_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
