#!/usr/bin/env python3
"""Audit the sealed R5 paired replay without loading GT."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _clean_runtime(value: Any, location: str, errors: list[dict[str, Any]]) -> None:
    forbidden = {"dataset_gt_id", "gt_box", "future_gt", "future_identity_error", "h20", "h50", "h100"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in forbidden:
                errors.append({"location": location, "reason": "forbidden_runtime_key", "key": str(key)})
            _clean_runtime(nested, f"{location}/{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _clean_runtime(nested, f"{location}/{index}", errors)


def _audit_manifest(path: Path, *, expected_variant: str, event_id: str, require_requery: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    manifest = read_json(path)
    if manifest.get("status") != "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY" or manifest.get("variant") != expected_variant:
        errors.append({"event_id": event_id, "reason": "manifest_status_or_variant"})
    frames_path = Path(str(manifest.get("frames", "")))
    if not frames_path.is_absolute():
        frames_path = ROOT / frames_path
    if not frames_path.is_file():
        errors.append({"event_id": event_id, "reason": "missing_frames"})
        return [], errors
    rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 101:
        errors.append({"event_id": event_id, "reason": "frame_count", "value": len(rows)})
    if rows:
        event_frame = int(rows[0]["event_frame"])
        expected = list(range(event_frame, event_frame + 101))
        if [int(row["frame"]) for row in rows] != expected:
            errors.append({"event_id": event_id, "reason": "frame_axis"})
    for index, row in enumerate(rows):
        location = f"{expected_variant}/{event_id}/{row.get('frame')}"
        _clean_runtime(row, location, errors)
        for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
            if row.get(flag) is not False:
                errors.append({"location": location, "reason": f"flag_{flag}"})
        candidates = list(row.get("candidate_rows", []))
        if int(row.get("candidate_count", -1)) != len(candidates):
            errors.append({"location": location, "reason": "candidate_count"})
        uids = [str(item.get("candidate_uid")) for item in candidates]
        if len(uids) != len(set(uids)):
            errors.append({"location": location, "reason": "duplicate_candidate_uid"})
        if index == 0:
            if candidates or row.get("candidate_count") != 0 or row.get("memory_read") is not False:
                errors.append({"location": location, "reason": "event_frame_boundary"})
            continue
        pool = row.get("candidate_pool")
        if not isinstance(pool, Mapping):
            errors.append({"location": location, "reason": "missing_pool_audit"})
            continue
        pool_rows = list(pool.get("candidate_rows", []))
        if [str(item.get("candidate_uid")) for item in pool_rows] != uids:
            errors.append({"location": location, "reason": "pool_output_uid_mismatch"})
        if pool.get("runtime_future_gt_used") is not False or pool.get("public_id_inference") is not False:
            errors.append({"location": location, "reason": "pool_causal_flags"})
        if require_requery:
            if pool.get("schema_version") != "N72R7_TARGET_CANDIDATE_POOL_WITH_REQUERY_V1":
                errors.append({"location": location, "reason": "requery_pool_schema_missing"})
            requery_count = int(pool.get("target_session_requery_candidate_count", 0))
            actual_requery_count = sum(
                int(item.get("candidate_source") == "TARGET_SESSION_REQUERY")
                for item in pool_rows
            )
            if requery_count != actual_requery_count:
                errors.append({
                    "location": location,
                    "reason": "requery_count_mismatch",
                    "declared": requery_count,
                    "actual": actual_requery_count,
                })
        score = row.get("score_audit")
        if not isinstance(score, Mapping):
            errors.append({"location": location, "reason": "score_audit_missing"})
        else:
            fused = np.asarray(score.get("fused_score_matrix", []), dtype=np.float64)
            base = np.asarray(score.get("base_score_matrix", []), dtype=np.float64)
            states = list(score.get("association_state_axis", []))
            publics = list(score.get("public_id_axis", []))
            if fused.ndim != 2 or base.shape != fused.shape or fused.shape != (len(candidates), len(states)) or len(states) != len(publics) or not np.all(np.isfinite(fused)) or not np.all(np.isfinite(base)):
                errors.append({"location": location, "reason": "score_matrix_shape_or_finite"})
        assignment = row.get("assignment")
        if not isinstance(assignment, Mapping) or not isinstance(assignment.get("solver"), Mapping):
            errors.append({"location": location, "reason": "solver_audit_missing"})
        else:
            if assignment["solver"].get("runtime_future_gt_used") is not False:
                errors.append({"location": location, "reason": "solver_future_gt_flag"})
    return rows, errors


def audit(root: Path, *, attempt: int) -> dict[str, Any]:
    batch = read_json(root / f"batch_attempt{int(attempt)}.json")
    errors: list[dict[str, Any]] = []
    event_summaries: list[dict[str, Any]] = []
    for item in batch.get("results", []):
        event_id = str(item["event_id"])
        worker = read_json(root / event_id / "worker_status.json")
        if worker.get("status") != "PASS_N72R7_R5_REQUERY_REPLAY":
            errors.append({"event_id": event_id, "reason": "worker_not_pass"})
        current_rows, current_errors = _audit_manifest(
            root / "current" / event_id / "event_manifest.json",
            expected_variant="D2",
            event_id=event_id,
            require_requery=False,
        )
        treatment_rows, treatment_errors = _audit_manifest(
            root / "requery" / event_id / "event_manifest.json",
            expected_variant="R5_REQUERY",
            event_id=event_id,
            require_requery=True,
        )
        errors.extend(current_errors)
        errors.extend(treatment_errors)
        if current_rows and treatment_rows:
            if [int(row["frame"]) for row in current_rows] != [int(row["frame"]) for row in treatment_rows]:
                errors.append({"event_id": event_id, "reason": "pair_frame_axis"})
            event_summaries.append({
                "event_id": event_id,
                "sequence": current_rows[0].get("sequence"),
                "action_type": current_rows[0].get("action_type"),
                "current_future_candidate_rows": sum(int(row.get("candidate_count", 0)) for row in current_rows[1:]),
                "requery_future_candidate_rows": sum(int(row.get("candidate_count", 0)) for row in treatment_rows[1:]),
                "current_frames": len(current_rows),
                "requery_frames": len(treatment_rows),
            })
    passed = not errors and not batch.get("failures") and len(event_summaries) == int(batch.get("requested_event_count", -1))
    return {
        "schema_version": "N72R7_R5_REQUERY_RUNTIME_AUDIT_V1",
        "status": "PASS_N72R7_R5_REQUERY_RUNTIME_AUDIT" if passed else "FAIL_N72R7_R5_REQUERY_RUNTIME_AUDIT",
        "input_batch": str(root / f"batch_attempt{int(attempt)}.json"),
        "attempt": int(attempt),
        "event_count": len(event_summaries),
        "errors": errors,
        "batch_failures": batch.get("failures", []),
        "events": event_summaries,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = audit(root, attempt=int(args.attempt))
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "event_count": result["event_count"], "errors": len(result["errors"])}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
