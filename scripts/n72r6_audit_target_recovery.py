#!/usr/bin/env python3
"""CPU-only audit for one N72R6 target-session recovery artifact.

This validator checks only stream/provenance invariants.  It deliberately does
not read dataset GT or score identity effect; those are separate posthoc
operations.  A recovery smoke is not promoted to the frozen C1 stream by this
script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
from typing import Any


HORIZON = 100


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def audit(done_path: Path) -> dict[str, Any]:
    done = read_json(done_path)
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    event_id = str(done.get("event_id", ""))
    event_frame = int(done.get("event_frame", -1))
    end_frame = int(done.get("end_frame", -1))
    check(bool(event_id), "event_id_missing")
    check(
        done.get("status") in {
            "PASS_TARGET_STREAM_COMPLETE",
            "PASS_TARGET_STREAM_COMPLETE_WITH_RECOVERY_MISS",
        },
        "stream_not_complete",
    )
    check(done.get("target_session_recovery_mode") is True, "recovery_mode_not_enabled")
    check(end_frame - event_frame == HORIZON, "horizon_mismatch")
    check(int(done.get("frame_count", -1)) == HORIZON + 1, "frame_count_mismatch")
    for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
        check(done.get(flag) is False, f"done_{flag}")

    frames_path = Path(str(done.get("frames", "")))
    if not frames_path.is_absolute():
        frames_path = Path.cwd() / frames_path
    check(frames_path.is_file(), "frames_missing")
    if frames_path.is_file():
        rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = list(range(event_frame, end_frame + 1))
        actual = [int(row.get("frame", -1)) for row in rows]
        check(actual == expected, "frame_axis_not_contiguous")
        check(len(rows) == HORIZON + 1, "frame_rows_mismatch")
        for row in rows:
            frame = int(row.get("frame", -1))
            prefix = f"{event_id}:{frame}"
            check(row.get("event_id") == event_id, f"{prefix}:event_mismatch")
            check(row.get("candidate_set_complete") is True, f"{prefix}:candidate_set_incomplete")
            check(row.get("event_frame_memory_read") is False, f"{prefix}:event_memory_read")
            check(row.get("memory_read") is False, f"{prefix}:memory_read")
            for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                check(row.get(flag) is False, f"{prefix}:{flag}")
            candidates = row.get("candidate_rows")
            check(isinstance(candidates, list) and len(candidates) <= 1, f"{prefix}:candidate_cardinality")
            if not isinstance(candidates, list):
                continue
            check(int(row.get("candidate_count", -1)) == len(candidates), f"{prefix}:candidate_count")
            for candidate in candidates:
                check(candidate.get("candidate_kind") == "TARGET_CORRECTION_SESSION_CANDIDATE", f"{prefix}:kind")
                check(candidate.get("public_id") is None, f"{prefix}:candidate_public_id")
                check(candidate.get("public_id_inference") is False, f"{prefix}:candidate_public_inference")
                check(candidate.get("runtime_future_gt_used") is False, f"{prefix}:candidate_future_gt")
                check(candidate.get("runtime_gt_read") is False, f"{prefix}:candidate_gt")
                feature = candidate.get("feature")
                check(isinstance(feature, list) and len(feature) == 512, f"{prefix}:feature_shape")
                if isinstance(feature, list):
                    values = [float(value) for value in feature]
                    check(all(math.isfinite(value) for value in values), f"{prefix}:feature_nonfinite")
                    check(math.sqrt(sum(value * value for value in values)) > 1.0e-6, f"{prefix}:feature_zero_norm")

    session_audit = done.get("target_session_audit")
    check(isinstance(session_audit, dict), "session_audit_missing")
    attempts = session_audit.get("recovery_attempts", []) if isinstance(session_audit, dict) else []
    check(isinstance(attempts, list), "recovery_attempts_not_list")
    if isinstance(attempts, list):
        check(len(attempts) == int(done.get("target_session_recovery_attempt_count", -1)), "recovery_attempt_count_mismatch")
        for index, item in enumerate(attempts):
            prefix = f"recovery:{index}"
            frame = int(item.get("global_frame", -1))
            source = int(item.get("source_frame", -1))
            check(event_frame < frame <= end_frame, f"{prefix}:frame_out_of_range")
            check(source < frame, f"{prefix}:noncausal_source")
            for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"):
                check(item.get(flag) is False, f"{prefix}:{flag}")
            response = item.get("official_response", {})
            check(isinstance(response, dict), f"{prefix}:response_missing")
            if isinstance(response, dict):
                if str(item.get("status", "")).startswith("FAIL_TARGET_RECOVERY"):
                    check(
                        response.get("status") == "FAIL_TARGET_RECOVERY_NO_OFFICIAL_OBSERVATION",
                        f"{prefix}:failure_status",
                    )
                    check(response.get("retained_official_count") == 0, f"{prefix}:failure_retained_count")
                    check(response.get("retained_raw_sam_id") is None, f"{prefix}:failure_raw_id")
                    prompt_attempts = response.get("recovery_prompt_attempts")
                    check(
                        isinstance(prompt_attempts, list) and len(prompt_attempts) >= 1,
                        f"{prefix}:failure_prompt_audit",
                    )
                    continue
                check(response.get("runtime_future_gt_used") is False, f"{prefix}:response_future_gt")
                check(response.get("retained_official_count") == 1, f"{prefix}:retained_count")
                isolation = response.get("isolation", {})
                check(isinstance(isolation, dict), f"{prefix}:isolation_missing")
                if isinstance(isolation, dict):
                    check(isolation.get("official_target_only_propagation_action") == "refine", f"{prefix}:not_target_only")
                    check(isolation.get("persistent_public_identity_touched") is False, f"{prefix}:identity_touched")
                    check(isolation.get("runtime_future_gt_used") is False, f"{prefix}:isolation_future_gt")
                    check(isolation.get("metadata_ids_after") == [int(response.get("retained_raw_sam_id", -1))], f"{prefix}:raw_isolation")

    if errors:
        raise ValueError(";".join(errors))
    recovery_failure_count = sum(
        isinstance(item, dict) and str(item.get("status", "")).startswith("FAIL_TARGET_RECOVERY")
        for item in attempts
    )
    # Streams written before the recovery-miss extension do not contain the
    # optional field.  Absence means that the legacy stream recorded no
    # recovery miss; an explicit value is still checked strictly.  Treating
    # the absent field as -1 would turn every previously valid stream into a
    # false audit failure.
    declared_failure_count = done.get("target_session_recovery_failure_count", 0)
    if declared_failure_count is None:
        declared_failure_count = 0
    check(int(declared_failure_count) == recovery_failure_count, "recovery_failure_count_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {
        "schema_version": "N72R6_TARGET_RECOVERY_AUDIT_V1",
        "status": (
            "PASS_TARGET_SESSION_RECOVERY_STREAM_AUDIT_WITH_LEGITIMATE_LOSS"
            if recovery_failure_count
            else "PASS_TARGET_SESSION_RECOVERY_STREAM_AUDIT"
        ),
        "event_id": event_id,
        "sequence": str(done.get("sequence")),
        "event_frame": event_frame,
        "end_frame": end_frame,
        "frame_count": int(done.get("frame_count")),
        "candidate_row_count": int(done.get("candidate_row_count")),
        "recovery_attempt_count": len(attempts),
        "recovery_failure_count": int(recovery_failure_count),
        "all_recovery_sources_strictly_past": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "scientific_effect_scored": False,
        "done": str(done_path),
        "created_at_utc": now_utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--done", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    args = parser.parse_args()
    done = args.done if args.done.is_absolute() else Path.cwd() / args.done
    output = args.status_output if args.status_output.is_absolute() else Path.cwd() / args.status_output
    result = audit(done)
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "event_id": result["event_id"], "recovery_attempt_count": result["recovery_attempt_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
