#!/usr/bin/env python3
"""Lossless CPU-only audit for N72R5 Stage 07 official branch artifacts.

The worker JSONL schema is intentionally frame-record-only: the frame key is
``frame`` (not ``frame_id``), and the first row is the event-frame snapshot.
This auditor therefore validates the actual schema rather than assuming the
older sidecar layout.  It never rewrites worker artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


BRANCHES = (
    "B0_NO_INTERVENTION",
    "B1_SPATIAL_CORRECTION_ONLY",
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
)
RECOVERY_BRANCHES = {
    "B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
}
TVC_BRANCHES = {
    "B3_SPATIAL_CORRECTION_PLUS_TVC",
    "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC",
}
HORIZON = 100


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_worker(record: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    event_id = str(record.get("event_id"))
    branch = str(record.get("branch"))
    errors: list[str] = []
    if branch not in BRANCHES:
        errors.append(f"unknown branch: {branch}")
    if int(record.get("return_code", 1)) != 0:
        errors.append(f"worker return_code={record.get('return_code')}")
    if record.get("status") != "PASS_N72R5_OFFICIAL_FULL_LOOP_BRANCH":
        errors.append(f"worker status={record.get('status')}")
    output = Path(str(record.get("output", "")))
    done_path = Path(str(record.get("done", ""))) if record.get("done") else None
    if not output.is_file():
        errors.append(f"missing JSONL: {output}")
        return {"event_id": event_id, "branch": branch, "status": "FAIL", "errors": errors}
    done_payload: dict[str, Any] = {}
    if done_path is None or not done_path.is_file():
        errors.append(f"missing done artifact: {done_path}")
    else:
        try:
            done_payload = read_json(done_path)
        except Exception as exc:
            errors.append(f"done artifact parse: {type(exc).__name__}: {exc}")
    try:
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        errors.append(f"JSONL parse: {type(exc).__name__}: {exc}")
        return {"event_id": event_id, "branch": branch, "status": "FAIL", "errors": errors}
    # The orchestrator deliberately removes the nested worker payload from
    # ``worker_records``; its scalar fields are copied to the record itself.
    event_frame = int(done_payload.get("event_frame", record.get("event_frame", -1)))
    expected = list(range(event_frame, event_frame + HORIZON + 1))
    frames = [row.get("frame") for row in rows if isinstance(row, dict)]
    if len(rows) != HORIZON + 1:
        errors.append(f"row_count={len(rows)} expected={HORIZON + 1}")
    if frames != expected:
        errors.append(f"frame_coverage={frames[:3]}...{frames[-3:] if frames else []} expected={expected[:3]}...{expected[-3:]}")
    if len(frames) != len(set(frames)):
        errors.append("duplicate frame rows")
    event_row = rows[0] if rows else {}
    future_row = rows[1] if len(rows) > 1 else {}
    if event_row.get("phase") != "Y_PRE_FROZEN":
        errors.append(f"event phase={event_row.get('phase')}")
    if future_row.get("phase") != "FUTURE_PROPAGATION":
        errors.append(f"event+1 phase={future_row.get('phase')}")
    if event_row.get("event_frame_memory_read") is not False:
        errors.append("event-frame memory read is not false")
    if event_row.get("first_memory_visible_frame") != event_frame + 1:
        errors.append(f"first memory frame={event_row.get('first_memory_visible_frame')}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index} is not an object")
            continue
        if row.get("event_id") != event_id or row.get("branch") != branch:
            errors.append(f"row {index} event/branch mismatch")
        if row.get("record_type") != "official_candidate_frame":
            errors.append(f"row {index} record_type={row.get('record_type')}")
        for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference", "event_frame_memory_read"):
            if row.get(key) is not False:
                errors.append(f"row {index} {key}={row.get(key)!r}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or row.get("candidate_count") != len(candidates):
            errors.append(f"row {index} candidate count/list mismatch")
            continue
        if row.get("candidate_set_complete") is not True:
            errors.append(f"row {index} candidate_set_complete={row.get('candidate_set_complete')}")
        indices = [item.get("candidate_index") for item in candidates if isinstance(item, dict)]
        if len(indices) != len(candidates) or len(indices) != len(set(indices)) or row.get("candidate_order") != indices:
            errors.append(f"row {index} candidate order/uniqueness invalid")
        native_keys: list[tuple[str, Any]] = []
        for cindex, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                errors.append(f"row {index} candidate {cindex} not object")
                continue
            if candidate.get("adapter_visible_id") is None or candidate.get("adapter_external_id") is None:
                errors.append(f"row {index} candidate {cindex} adapter mapping missing")
            raw = candidate.get("raw_native_id")
            official = candidate.get("official_raw_sam_id")
            native_keys.append(("raw", raw if raw is not None else official))
            if raw is None and official is None:
                errors.append(f"row {index} candidate {cindex} native mapping missing")
            if candidate.get("public_id") is not None:
                errors.append(f"row {index} candidate {cindex} unexpectedly has public_id")
        if len(native_keys) != len(set(native_keys)):
            errors.append(f"row {index} duplicate native mapping")
    correction = event_row.get("correction")
    recovery = event_row.get("recovery")
    if branch == "B0_NO_INTERVENTION":
        if correction is not None:
            errors.append("B0 has correction")
    else:
        if not isinstance(correction, dict) or correction.get("status") != "PASS_OFFICIAL_CURRENT_FRAME_CORRECTION":
            errors.append("corrected branch missing official current-frame PASS")
    if isinstance(recovery, dict):
        if bool(recovery.get("enabled")) != (branch in RECOVERY_BRANCHES):
            errors.append("recovery enabled flag inconsistent with branch")
    else:
        errors.append("missing recovery audit")
    if bool(event_row.get("tvc_enabled_for_later_association")) != (branch in TVC_BRANCHES):
        errors.append("TVC branch marker inconsistent")
    return {
        "event_id": event_id,
        "sequence": str(record.get("sequence")),
        "branch": branch,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "row_count": len(rows),
        "frame_start": frames[0] if frames else None,
        "frame_end": frames[-1] if frames else None,
        "event_frame": event_frame,
        "event_candidate_count": event_row.get("candidate_count"),
        "event_plus_one_candidate_count": future_row.get("candidate_count"),
        "candidate_artifact_sha256": sha256_file(output),
        "y_pre_semantic_hash": done_payload.get("y_pre_semantic_hash", record.get("y_pre_semantic_hash")),
        "recovery_status": recovery.get("status") if isinstance(recovery, dict) else None,
        "correction_status": correction.get("status") if isinstance(correction, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "official_full_loop_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    workers = manifest.get("worker_records")
    errors: list[str] = []
    if not isinstance(workers, list):
        errors.append("worker_records is not a list")
        workers = []
    keys = [(str(item.get("event_id")), str(item.get("branch"))) for item in workers if isinstance(item, dict)]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        errors.append(f"duplicate worker keys: {duplicate_keys}")
    audits = [audit_worker(item, manifest) for item in workers if isinstance(item, dict)]
    event_hashes: dict[str, dict[str, str | None]] = {}
    for item in audits:
        event_hashes.setdefault(str(item["event_id"]), {})[str(item["branch"])] = item.get("y_pre_semantic_hash")
    event_summaries = []
    for event_id, branches in sorted(event_hashes.items()):
        missing = sorted(set(BRANCHES) - set(branches))
        unique_hashes = {value for value in branches.values() if value}
        event_errors = []
        if missing:
            event_errors.append(f"missing branches: {missing}")
        if len(unique_hashes) != 1 or len(branches) != len(BRANCHES):
            event_errors.append("shared Y_pre hash gate failed")
        event_summaries.append({"event_id": event_id, "branch_count": len(branches), "missing_branches": missing, "y_pre_hash_equal": not event_errors, "errors": event_errors})
    if len(workers) != int(manifest.get("event_count_expected", -1)) * len(BRANCHES):
        errors.append(f"worker record count={len(workers)} does not equal event_count_expected*5")
    errors.extend(f"{item['event_id']}/{item['branch']}: {error}" for item in audits for error in item.get("errors", []))
    errors.extend(f"{item['event_id']}: {error}" for item in event_summaries for error in item.get("errors", []))
    payload = {
        "schema_version": "N72R5_STAGE07_CPU_AUDIT_V1",
        "status": "PASS_N72R5_STAGE07_CPU_AUDIT" if not errors else "FAIL_N72R5_STAGE07_CPU_AUDIT",
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "event_count_expected": int(manifest.get("event_count_expected", 0)),
        "event_count_observed": len(event_summaries),
        "branch_count_expected": len(BRANCHES),
        "worker_record_count": len(workers),
        "duplicate_worker_keys": [list(key) for key in duplicate_keys],
        "worker_audits": audits,
        "event_summaries": event_summaries,
        "runtime_future_gt_used": False,
        "errors": errors,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve()), "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
