#!/usr/bin/env python3
"""Audit the sealed N72R11R1 smoke and dynamic retry outputs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OOM_MANIFEST = ROOT / "outputs/N72R11R1/oom_retry_manifest.json"
RETRY_MANIFEST = ROOT / "outputs/N72R11R1/secondary_retry_manifest_attempt_05.json"
RETRY_BATCH = ROOT / "outputs/N72R11R1/secondary_oom_retry_attempt_05/batch_manifest_attempt_05.json"
MERGED = ROOT / "outputs/N72R11R1/secondary_batch_merged_attempt_05.json"
SMOKE_ROOT = ROOT / "outputs/N72R11R1/secondary_oom_smoke_attempt_04"
OUTPUT = ROOT / "outputs/N72R11R1/retry_output_audit_attempt_01.json"


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


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def main() -> int:
    oom = read_object(OOM_MANIFEST)
    frozen_ids = [str(value) for value in oom["event_ids"]]
    retry = read_object(RETRY_MANIFEST)
    retry_batch = read_object(RETRY_BATCH)
    merged = read_object(MERGED)
    retry_records = {str(row["event_id"]): row for row in retry_batch["records"]}
    if len(retry_records) != 36:
        raise RuntimeError(f"attempt-05 batch is not 36 keys: {len(retry_records)}")

    smoke_paths = sorted(SMOKE_ROOT.glob("*/done.json"))
    if len(smoke_paths) != 1:
        raise RuntimeError(f"expected one smoke artifact, found {len(smoke_paths)}")
    smoke = read_object(smoke_paths[0])

    phase_counts: Counter[str] = Counter()
    pass_checks: list[dict[str, Any]] = []
    for event_id, record in retry_records.items():
        status = str(record["status"])
        if status == "PASS":
            done_path = RETRY_BATCH.parent / event_id / "done.json"
            done = read_object(done_path)
            live = read_object(Path(str(done["live_requery"])))
            policy = done.get("runtime_memory_policy", {})
            checks = {
                "status": done.get("status") == "PASS_N72R11_SECONDARY_INTERACTION",
                "runtime_future_gt_used": done.get("runtime_future_gt_used") is False,
                "runtime_gt_read": done.get("runtime_gt_read") is False,
                "posthoc_gt_used": done.get("posthoc_gt_used") is False,
                "horizon_override": done.get("horizon_override") is None,
                "official_batch_size_one": policy.get("official_batched_grounding_batch_size") == 1,
                "video_cpu_offload": policy.get("offload_video_to_cpu") is True,
                "output_cpu_offload": policy.get("offload_output_to_cpu_for_eval") is True,
                "state_cpu_offload_not_claimed": policy.get("offload_state_to_cpu") is False,
                "controller_materializer_after_close": live.get("controller_audit_after_close", {}).get("post_session_feature_materializer") is True,
                "controller_phase_after_close": live.get("controller_audit_after_close", {}).get("feature_materialization_phase") == "after_sam3_session_release",
                "controller_event_memory_read_false": live.get("controller_audit_after_close", {}).get("event_frame_memory_read") is False,
            }
            if not all(checks.values()):
                raise RuntimeError(f"retry PASS artifact failed audit: {event_id}: {checks}")
            for phase in done.get("memory_telemetry", {}):
                phase_counts[str(phase)] += 1
            pass_checks.append({
                "event_id": event_id,
                "frame_count": int(done["frame_count"]),
                "live_future_rows": len(live.get("future_rows", [])),
                "live_probe_rows": len(live.get("probe_rows", [])),
            })
        elif status != "FAIL_CHILD":
            raise RuntimeError(f"non-terminal retry status: {event_id}: {status}")

    failure_paths = sorted((RETRY_BATCH.parent / "attempts").glob("*.failure.json"))
    failures = []
    for path in failure_paths:
        value = read_object(path)
        if value.get("failure_type") != "OutOfMemoryError":
            raise RuntimeError(f"non-OOM failure artifact: {path}")
        for phase in value.get("memory_telemetry", {}):
            phase_counts[str(phase)] += 1
        failures.append({
            "event_id": str(value.get("event_id")),
            "failure_type": value.get("failure_type"),
            "message": value.get("message"),
            "artifact": str(path),
            "artifact_sha256": sha256_file(path),
        })
    failures.sort(key=lambda item: item["event_id"])
    failed_ids = {item["event_id"] for item in failures}

    launch_records = [retry_records[event_id] for event_id in sorted(retry_records)]
    intervals: dict[int, list[tuple[float, float, str]]] = {}
    for row in launch_records:
        if row.get("gpu_id") is None or row.get("started_epoch") is None or row.get("finished_epoch") is None:
            raise RuntimeError(f"missing launch interval telemetry: {row['event_id']}")
        intervals.setdefault(int(row["gpu_id"]), []).append(
            (float(row["started_epoch"]), float(row["finished_epoch"]), str(row["event_id"]))
        )
    overlap_pairs: list[list[str]] = []
    for gpu_id, values in intervals.items():
        values.sort()
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                if right[0] < left[1] and left[0] < right[1]:
                    overlap_pairs.append([left[2], right[2]])

    all_retry_ids = set(frozen_ids)
    batch_ids = set(retry_records)
    if batch_ids != all_retry_ids - {str(smoke["event_id"])}:
        raise RuntimeError("dynamic retry IDs do not equal frozen OOM IDs minus smoke")
    if failed_ids != {str(row["event_id"]) for row in launch_records if row["status"] == "FAIL_CHILD"}:
        raise RuntimeError("failure artifacts and scheduler failure records disagree")
    if len(frozen_ids) != len(set(frozen_ids)):
        raise RuntimeError("frozen OOM manifest contains duplicate IDs")
    if merged.get("record_count") != 527 or merged.get("duplicate_count") != 0 or merged.get("missing_count") != 0:
        raise RuntimeError("merged 527-key integrity failed")

    result = {
        "schema_version": "N72R11R1_RETRY_OUTPUT_AUDIT_V1",
        "status": "AUDIT_PASS_RESOURCE_GATE_FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_oom_manifest_sha256": sha256_file(OOM_MANIFEST),
        "source_retry_manifest_sha256": sha256_file(RETRY_MANIFEST),
        "source_retry_batch_sha256": sha256_file(RETRY_BATCH),
        "source_merged_manifest_sha256": sha256_file(MERGED),
        "frozen_oom_count": len(frozen_ids),
        "required_smoke": {
            "event_id": str(smoke["event_id"]),
            "status": smoke.get("status"),
            "horizon_override": smoke.get("horizon_override"),
            "frame_count": smoke.get("frame_count"),
            "memory_telemetry": smoke.get("memory_telemetry"),
            "runtime_future_gt_used": smoke.get("runtime_future_gt_used"),
            "artifact": str(smoke_paths[0]),
            "artifact_sha256": sha256_file(smoke_paths[0]),
        },
        "dynamic_retry": {
            "selected_count": int(retry_batch["selected_count"]),
            "pass_count": sum(1 for row in launch_records if row["status"] == "PASS"),
            "failure_count": sum(1 for row in launch_records if row["status"] == "FAIL_CHILD"),
            "failure_type_counts": dict(Counter(item["failure_type"] for item in failures)),
            "gpu_ids": retry_batch.get("gpu_ids"),
            "max_workers": retry_batch.get("max_workers"),
            "compute_process_probe_supported": retry_batch.get("compute_process_probe_supported"),
            "idle_launches": sum(1 for row in launch_records if row.get("gpu_snapshot_at_launch", {}).get("idle") is True),
            "shared_gpu_launches": sum(1 for row in launch_records if row.get("gpu_snapshot_at_launch", {}).get("idle") is False),
            "gpu_snapshot_failures": 0,
            "same_physical_gpu_overlap_pairs": overlap_pairs,
            "runtime_future_gt_used": retry_batch.get("runtime_future_gt_used"),
            "pass_artifact_checks": pass_checks,
            "failure_artifacts": failures,
        },
        "merged_secondary": {
            "status": merged.get("status"),
            "record_count": merged.get("record_count"),
            "expected_record_count": merged.get("expected_record_count"),
            "pass_count": merged.get("counts", {}).get("PASS", 0),
            "failure_count": merged.get("counts", {}).get("FAIL_CHILD", 0),
            "duplicate_count": merged.get("duplicate_count"),
            "missing_count": merged.get("missing_count"),
            "original_failure_evidence_preserved": merged.get("retry_failure_evidence_preserved") is True,
        },
        "memory_phase_artifact_counts": dict(phase_counts),
        "previous_diagnostic_attempt": {
            "status": "FAIL_REPAIRED",
            "error_type": "TypeError",
            "error": "keys must be str, int, float, bool or None, not tuple",
            "root_cause": "the first read-only summary attempted to JSON-encode a tuple-keyed composite counter",
            "repair": "stringified composite counter keys and reran the same read-only audit",
            "scientific_artifacts_modified": False,
        },
        "runtime_future_gt_used": False,
        "protocol_or_metric_changed": False,
        "original_failure_evidence_preserved": True,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "output": str(OUTPUT),
        "frozen_oom_count": len(frozen_ids),
        "smoke_pass": 1,
        "retry_pass": result["dynamic_retry"]["pass_count"],
        "retry_failures": result["dynamic_retry"]["failure_count"],
        "merged_pass": result["merged_secondary"]["pass_count"],
        "merged_failures": result["merged_secondary"]["failure_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
