#!/usr/bin/env python3
"""CPU-only audit for the sealed N72R7 confirmation D1/D2 replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    forbidden = {
        "dataset_gt_id", "other_dataset_gt_id", "gt_box", "future_gt", "future_identity_error",
        "h20", "h50", "h100", "iou", "identity_error", "id_switch",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in forbidden:
                errors.append({"location": location, "reason": "forbidden_runtime_key", "key": str(key)})
            _clean_runtime(nested, f"{location}/{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _clean_runtime(nested, f"{location}/{index}", errors)


def _audit_variant(
    manifest_path: Path,
    *,
    event_id: str,
    expected_variant: str,
    target_public_id: int,
    event_frame: int,
    expected_main_uids: list[list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        return [], [{"event_id": event_id, "variant": expected_variant, "reason": "missing_manifest"}], {}
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY" or manifest.get("variant") != expected_variant:
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "manifest_status_or_variant"})
    if int(manifest.get("event_frame", -1)) != event_frame or int(manifest.get("target_public_id", -1)) != target_public_id:
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "manifest_authority"})
    frames_path = resolve(str(manifest.get("frames", "")))
    if not frames_path.is_file():
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "missing_frames"})
        return [], errors, manifest
    declared_hash = str(manifest.get("frames_sha256", ""))
    if declared_hash != sha256(frames_path):
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "frames_hash"})
    rows = read_jsonl(frames_path)
    expected_frames = list(range(event_frame, event_frame + 101))
    if len(rows) != 101 or [int(row.get("frame", -1)) for row in rows] != expected_frames:
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "frame_axis", "count": len(rows)})
    if manifest.get("event_frame_memory_read") is not False or manifest.get("first_memory_visible_frame") != event_frame + 1:
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "manifest_memory_boundary"})
    if manifest.get("runtime_future_gt_used") is not False or manifest.get("runtime_gt_read") is not False or manifest.get("posthoc_gt_used") is not False or manifest.get("public_id_inference") is not False:
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "manifest_causal_flags"})
    if not bool(manifest.get("raw_switch_preserves_public_id")):
        errors.append({"event_id": event_id, "variant": expected_variant, "reason": "raw_switch_changed_public_id"})
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
            if candidates or row.get("candidate_count") != 0 or row.get("memory_read") is not False or row.get("memory_write") is not True:
                errors.append({"location": location, "reason": "event_frame_boundary"})
            continue
        if row.get("record_kind") != "future_association_frame" or row.get("memory_read") is not True:
            errors.append({"location": location, "reason": "future_memory_marker"})
        if int(row.get("frame_horizon", -1)) != index or int(row.get("first_memory_visible_frame", -1)) != event_frame + 1:
            errors.append({"location": location, "reason": "future_horizon_boundary"})
        pool = row.get("candidate_pool")
        if not isinstance(pool, Mapping):
            errors.append({"location": location, "reason": "missing_pool"})
            continue
        pool_rows = list(pool.get("candidate_rows", []))
        pool_uids = [str(item.get("candidate_uid")) for item in pool_rows]
        if pool_uids != uids or int(pool.get("candidate_count", -1)) != len(pool_rows):
            errors.append({"location": location, "reason": "pool_output_mismatch"})
        if pool.get("runtime_future_gt_used") is not False or pool.get("public_id_inference") is not False:
            errors.append({"location": location, "reason": "pool_causal_flags"})
        if any(item.get("public_id") is not None or item.get("public_id_authority") is not None for item in pool_rows):
            errors.append({"location": location, "reason": "source_candidate_has_public_authority"})
        if expected_variant == "D1":
            if any(item.get("candidate_source") != "MAIN_B0_CANDIDATE" for item in pool_rows):
                errors.append({"location": location, "reason": "D1_non_main_source"})
            if uids != expected_main_uids[index]:
                errors.append({"location": location, "reason": "D1_did_not_preserve_B0", "expected": expected_main_uids[index], "actual": uids})
        else:
            if uids[: len(expected_main_uids[index])] != expected_main_uids[index]:
                errors.append({"location": location, "reason": "D2_B0_prefix_changed"})
            extra = pool_rows[len(expected_main_uids[index]):]
            if len(extra) != 1 or extra[0].get("candidate_source") != "TARGET_SESSION_CURRENT_RAW":
                errors.append({"location": location, "reason": "D2_target_source_count", "extra_count": len(extra)})
        assignment = row.get("assignment")
        solver = assignment.get("solver") if isinstance(assignment, Mapping) else None
        if not isinstance(assignment, Mapping) or not isinstance(solver, Mapping):
            errors.append({"location": location, "reason": "solver_audit_missing"})
        else:
            if solver.get("runtime_future_gt_used") is not False or assignment.get("solver_public_id_immutable") is not True:
                errors.append({"location": location, "reason": "solver_causal_or_immutability"})
            assigned = [item.get("public_id") for item in candidates if item.get("public_id") is not None]
            if len(assigned) != len(set(assigned)):
                errors.append({"location": location, "reason": "duplicate_public_assignment"})
        score = row.get("score_audit")
        if not isinstance(score, Mapping):
            errors.append({"location": location, "reason": "score_audit_missing"})
        else:
            matrix = np.asarray(score.get("fused_score_matrix", []), dtype=np.float64)
            base = np.asarray(score.get("base_score_matrix", []), dtype=np.float64)
            states = list(score.get("association_state_axis", []))
            publics = list(score.get("public_id_axis", []))
            if matrix.ndim != 2 or base.shape != matrix.shape or matrix.shape != (len(candidates), len(states)) or len(states) != len(publics) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(base)):
                errors.append({"location": location, "reason": "score_shape_or_finite"})
            if target_public_id not in [int(value) for value in publics]:
                errors.append({"location": location, "reason": "target_public_missing_from_authority_axis"})
    summary = {
        "event_id": event_id,
        "variant": expected_variant,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "frame_count": len(rows),
        "future_frame_count": max(0, len(rows) - 1),
        "candidate_rows_total": sum(int(row.get("candidate_count", 0)) for row in rows[1:]),
        "raw_binding_switch_count": int(manifest.get("raw_binding_switch_count", -1)),
        "raw_switch_preserves_public_id": manifest.get("raw_switch_preserves_public_id"),
    }
    return rows, errors, summary


def audit(root: Path, *, attempt: int) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    batch = read_json(root / f"batch_attempt{int(attempt)}.json")
    protocol = read_json(ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json")
    specs = {str(item["event_id"]): item for item in protocol.get("events", [])}
    records = {str(item["event_id"]): item for item in batch.get("results", [])}
    if batch.get("status") != "PASS_N72R7_CONFIRMATION_REPLAY_BATCH" or int(batch.get("requested_event_count", -1)) != len(specs) or int(batch.get("completed_event_count", -1)) != len(specs) or int(batch.get("failed_event_count", -1)) != 0 or batch.get("failures"):
        errors.append({"reason": "batch_incomplete_or_not_pass", "status": batch.get("status")})
    if set(records) != set(specs):
        errors.append({"reason": "batch_event_keys", "expected": sorted(specs), "actual": sorted(records)})
    event_summaries: list[dict[str, Any]] = []
    for event_id in sorted(specs):
        spec = specs[event_id]
        record = records.get(event_id)
        if record is None:
            continue
        worker_path = root / event_id / "worker_status.json"
        if not worker_path.is_file():
            errors.append({"event_id": event_id, "reason": "missing_worker_status"})
            continue
        worker = read_json(worker_path)
        if worker.get("status") != "PASS_N72R7_CONFIRMATION_REPLAY" or worker.get("runtime_future_gt_used") is not False or worker.get("posthoc_gt_used") is not False:
            errors.append({"event_id": event_id, "reason": "worker_status_or_causal_flags"})
        frozen = worker.get("frozen_manifest")
        if not isinstance(frozen, Mapping):
            errors.append({"event_id": event_id, "reason": "missing_frozen_manifest"})
            continue
        c0_rows = read_jsonl(resolve(str(frozen["c0"]["path"])))
        expected_main_uids = [[str(item["candidate_uid"]) for item in row.get("candidate_rows", [])] for row in c0_rows]
        event_frame = int(spec["event_frame"])
        target_public_id = int(spec["target_public_id"])
        d1_rows, d1_errors, d1_summary = _audit_variant(
            resolve(str(worker["D1_manifest"])),
            event_id=event_id, expected_variant="D1", target_public_id=target_public_id,
            event_frame=event_frame, expected_main_uids=expected_main_uids,
        )
        d2_rows, d2_errors, d2_summary = _audit_variant(
            resolve(str(worker["D2_manifest"])),
            event_id=event_id, expected_variant="D2", target_public_id=target_public_id,
            event_frame=event_frame, expected_main_uids=expected_main_uids,
        )
        errors.extend(d1_errors)
        errors.extend(d2_errors)
        if d1_rows and d2_rows and [int(row["frame"]) for row in d1_rows] != [int(row["frame"]) for row in d2_rows]:
            errors.append({"event_id": event_id, "reason": "D1_D2_frame_axis_mismatch"})
        event_summaries.append({
            "event_id": event_id,
            "sequence": str(spec["sequence"]),
            "action_type": str(spec["action_type"]),
            "event_frame": event_frame,
            "target_public_id": target_public_id,
            "D1": d1_summary,
            "D2": d2_summary,
        })
    passed = not errors and len(event_summaries) == len(specs)
    return {
        "schema_version": "N72R7_CONFIRMATION_RUNTIME_AUDIT_V1",
        "status": "PASS_N72R7_CONFIRMATION_RUNTIME_AUDIT" if passed else "FAIL_N72R7_CONFIRMATION_RUNTIME_AUDIT",
        "input_batch": str(root / f"batch_attempt{int(attempt)}.json"),
        "input_batch_sha256": sha256(root / f"batch_attempt{int(attempt)}.json"),
        "protocol": str(ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json"),
        "protocol_sha256": sha256(ROOT / "outputs/N72R7/confirmation/confirmation_protocol.json"),
        "attempt": int(attempt),
        "event_count": len(event_summaries),
        "errors": errors,
        "events": event_summaries,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "posthoc_scoring_authorized": bool(passed),
        "created_at_utc": now_utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = resolve(args.root)
    output = resolve(args.output)
    try:
        result = audit(root, attempt=int(args.attempt))
    except Exception as exc:
        failure = {
            "schema_version": "N72R7_CONFIRMATION_RUNTIME_AUDIT_FAILURE_V1",
            "status": "FAIL_N72R7_CONFIRMATION_RUNTIME_AUDIT_INPUT",
            "root": str(root),
            "attempt": int(args.attempt),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "created_at_utc": now_utc(),
        }
        atomic_json(output.parent / "attempts" / f"runtime_audit_failure_attempt{int(args.attempt)}.json", failure)
        print(json.dumps({"status": failure["status"], "error": str(exc)}, ensure_ascii=False))
        return 1
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "event_count": result["event_count"], "errors": len(result["errors"])}, ensure_ascii=False))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
