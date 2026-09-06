#!/usr/bin/env python3
"""Seal the N72R11R1 retry evidence for the frozen 37-key OOM set."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OOM_MANIFEST = ROOT / "outputs/N72R11R1/oom_retry_manifest.json"
SMOKE_ROOT = ROOT / "outputs/N72R11R1/secondary_oom_smoke_attempt_04"
RETRY_ROOT = ROOT / "outputs/N72R11R1/secondary_oom_retry_attempt_05"
RETRY_BATCH = RETRY_ROOT / "batch_manifest_attempt_05.json"
OUTPUT = ROOT / "outputs/N72R11R1/secondary_retry_manifest_attempt_05.json"


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


def final_failure_payload(log_path: Path) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and str(value.get("status", "")).startswith("FAIL_N72R11_SECONDARY_INTERACTION"):
                payloads.append(value)
    if not payloads:
        raise RuntimeError(f"no machine-readable child failure payload in {log_path}")
    return payloads[-1]


def read_done(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not an object")
    if value.get("status") != "PASS_N72R11_SECONDARY_INTERACTION":
        raise RuntimeError(f"{label} is not a PASS artifact")
    if value.get("runtime_future_gt_used") is not False:
        raise RuntimeError(f"{label} violates runtime GT boundary")
    return value


def main() -> int:
    oom = json.loads(OOM_MANIFEST.read_text(encoding="utf-8"))
    frozen_records = {str(item["event_id"]): dict(item) for item in oom.get("records", [])}
    frozen_ids = [str(item) for item in oom.get("event_ids", [])]
    if len(frozen_ids) != 37 or len(frozen_ids) != len(set(frozen_ids)):
        raise RuntimeError(f"expected 37 unique frozen OOM IDs, found {len(frozen_ids)}")
    if set(frozen_ids) != set(frozen_records):
        raise RuntimeError("OOM manifest event_ids and records disagree")

    smoke_paths = sorted(SMOKE_ROOT.glob("*/done.json"))
    if len(smoke_paths) != 1:
        raise RuntimeError(f"expected one required smoke done artifact, found {len(smoke_paths)}")
    smoke_path = smoke_paths[0]
    smoke = read_done(smoke_path, "required smoke")
    smoke_event_id = str(smoke["event_id"])
    if smoke_event_id not in frozen_records:
        raise RuntimeError("required smoke key is outside the frozen OOM set")

    retry = json.loads(RETRY_BATCH.read_text(encoding="utf-8"))
    retry_records = {str(item["event_id"]): dict(item) for item in retry.get("records", [])}
    if len(retry_records) != 36 or set(retry_records) != (set(frozen_ids) - {smoke_event_id}):
        raise RuntimeError("attempt-05 batch does not cover exactly the 36 non-smoke OOM keys")
    if any(str(item.get("status")) not in {"PASS", "FAIL_CHILD"} for item in retry_records.values()):
        raise RuntimeError("attempt-05 batch contains a non-terminal record")

    records: list[dict[str, Any]] = []
    failure_type_counts: dict[str, int] = {}
    pass_count = 0
    failure_count = 0

    smoke_record = frozen_records[smoke_event_id]
    records.append(
        {
            "event_id": smoke_event_id,
            "sequence": smoke["sequence"],
            "secondary_frame": int(smoke["secondary_frame"]),
            "action_type": smoke["action_type"],
            "status": "PASS",
            "attempt": int(smoke["attempt"]),
            "r1_source": "required_full_window_smoke",
            "done_artifact": str(smoke_path),
            "done_sha256": sha256_file(smoke_path),
            "engineering_smoke_only": True,
            "runtime_future_gt_used": False,
            "original_failure_artifact": smoke_record.get("primary_failure_artifact"),
            "original_failure_artifact_sha256": smoke_record.get("primary_failure_artifact_sha256"),
        }
    )
    pass_count += 1

    for event_id in sorted(retry_records):
        item = retry_records[event_id]
        output_dir = RETRY_ROOT / event_id
        row: dict[str, Any] = {
            "event_id": event_id,
            "sequence": item["sequence"],
            "secondary_frame": int(item["secondary_frame"]),
            "action_type": item["action_type"],
            "status": str(item["status"]),
            "attempt": int(item["attempt"]),
            "r1_source": "attempt_05_dynamic_scheduler",
            "gpu_id": item.get("gpu_id"),
            "gpu_snapshot_at_launch": item.get("gpu_snapshot_at_launch"),
            "gpu_selection_reason": item.get("gpu_selection_reason"),
            "external_compute_process_count": item.get("external_compute_process_count"),
            "external_compute_used_mib": item.get("external_compute_used_mib"),
            "free_mib_at_launch": item.get("free_mib_at_launch"),
            "used_mib_at_launch": item.get("used_mib_at_launch"),
            "utilization_at_launch": item.get("utilization_at_launch"),
            "returncode": item.get("returncode"),
            "log": item.get("log"),
            "runtime_future_gt_used": False,
            "original_failure_artifact": frozen_records[event_id].get("primary_failure_artifact"),
            "original_failure_artifact_sha256": frozen_records[event_id].get("primary_failure_artifact_sha256"),
        }
        if row["status"] == "PASS":
            done_path = output_dir / "done.json"
            done = read_done(done_path, f"retry done {event_id}")
            row.update({
                "done_artifact": str(done_path),
                "done_sha256": sha256_file(done_path),
                "engineering_smoke_only": False,
            })
            pass_count += 1
        else:
            log_path = Path(str(item.get("log")))
            if not log_path.is_file():
                raise RuntimeError(f"missing failure log for {event_id}: {log_path}")
            failure = final_failure_payload(log_path)
            failure_type = str(failure.get("error_type"))
            if failure_type != "OutOfMemoryError":
                raise RuntimeError(f"non-OOM failure in frozen retry set {event_id}: {failure_type}")
            row.update({
                "failure_type": failure_type,
                "failure_status": failure.get("status"),
                "failure": failure.get("error"),
                "failure_log_sha256": sha256_file(log_path),
                "engineering_smoke_only": False,
            })
            failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
            failure_count += 1
        records.append(row)

    records.sort(key=lambda item: frozen_ids.index(str(item["event_id"])))
    result = {
        "schema_version": "N72R11R1_SECONDARY_RETRY_MANIFEST_V1",
        "status": "PASS_AND_OOM_FAILURES_SEALED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_oom_manifest": str(OOM_MANIFEST),
        "source_oom_manifest_sha256": sha256_file(OOM_MANIFEST),
        "source_retry_batch": str(RETRY_BATCH),
        "source_retry_batch_sha256": sha256_file(RETRY_BATCH),
        "required_smoke_artifact": str(smoke_path),
        "required_smoke_artifact_sha256": sha256_file(smoke_path),
        "record_count": len(records),
        "expected_record_count": 37,
        "pass_count": pass_count,
        "failure_count": failure_count,
        "failure_type_counts": failure_type_counts,
        "duplicate_count": len(records) - len({str(item["event_id"]) for item in records}),
        "missing_count": len(set(frozen_ids) - {str(item["event_id"]) for item in records}),
        "runtime_future_gt_used": False,
        "original_failure_evidence_preserved": True,
        "records": records,
    }
    if result["record_count"] != 37 or result["duplicate_count"] or result["missing_count"]:
        raise RuntimeError("sealed retry manifest key integrity failed")
    if failure_count + pass_count != 37:
        raise RuntimeError("sealed retry manifest terminal counts do not sum to 37")
    atomic_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "output": str(OUTPUT),
        "record_count": result["record_count"],
        "pass_count": pass_count,
        "failure_count": failure_count,
        "failure_type_counts": failure_type_counts,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
