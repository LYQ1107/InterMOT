#!/usr/bin/env python3
"""Independently audit the sealed runtime rows of the N72R11R2 replay.

This is a read-only audit.  It verifies hashes, frame axes, causal boundaries,
candidate uniqueness/mapping, score-matrix dimensions and runtime GT flags for
the sealed records selected by the combined formal-replay audit.  The failed
event is intentionally outside the selected set and remains in that audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED = ROOT / "outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11R2/formal_replay_resource_censored_runtime_audit.json"
VARIANTS = ("E0_BASELINE_B0", "E1_V3_BRIDGE", "E2_V3_BRIDGE_LIVE_ON_DEMAND")
ROW_VARIANT_ALIASES = {
    "E0_BASELINE_B0": {"E0_BASELINE_B0", "BASELINE_B0"},
    "E1_V3_BRIDGE": {"E1_V3_BRIDGE", "V3_BRIDGE"},
    "E2_V3_BRIDGE_LIVE_ON_DEMAND": {
        "E2_V3_BRIDGE_LIVE_ON_DEMAND",
        "V3_BRIDGE_LIVE_ON_DEMAND",
    },
}
HORIZON = 100


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


def resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def finite_matrix(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for row in value:
        if not isinstance(row, list):
            return False
        for number in row:
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                return False
    return True


def audit_event(record: Mapping[str, Any], counters: Counter[str], errors: list[str]) -> None:
    event_id = str(record.get("event_id"))
    sealed_path = resolve_path(record.get("runtime_event_sealed"))
    if not sealed_path.is_file():
        errors.append(f"{event_id}:runtime_event_sealed_missing")
        return
    sealed = read_json(sealed_path)
    if sealed.get("status") != "PASS_N72R11_ALL_RUNTIME_SEALED":
        errors.append(f"{event_id}:runtime_seal_status={sealed.get('status')}")
    for flag in ("gt_loaded", "posthoc_gt_used", "runtime_future_gt_used", "runtime_gt_read"):
        if sealed.get(flag) is not False:
            errors.append(f"{event_id}:seal_{flag}_not_false")
    manifests = sealed.get("runtime_manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != set(VARIANTS):
        errors.append(f"{event_id}:variant_manifest_axis")
        return
    for variant in VARIANTS:
        manifest = manifests[variant]
        if not isinstance(manifest, Mapping):
            errors.append(f"{event_id}/{variant}:manifest_not_object")
            continue
        frames_path = resolve_path(manifest.get("frames"))
        if not frames_path.is_file():
            errors.append(f"{event_id}/{variant}:frames_missing")
            continue
        expected_hash = manifest.get("frames_sha256")
        actual_hash = sha256_file(frames_path)
        if expected_hash != actual_hash:
            errors.append(f"{event_id}/{variant}:frames_hash_mismatch")
        if manifest.get("status") != "PASS_N72R11_RUNTIME_ARTIFACT_SEALED":
            errors.append(f"{event_id}/{variant}:manifest_status={manifest.get('status')}")
        for flag in ("posthoc_gt_used", "runtime_future_gt_used", "runtime_gt_read"):
            if manifest.get(flag) is not False:
                errors.append(f"{event_id}/{variant}:manifest_{flag}_not_false")
        event_frame = int(record.get("event_frame", manifest.get("event_frame")))
        expected_frames = list(range(event_frame, event_frame + HORIZON + 1))
        observed_frames: list[int] = []
        line_count = 0
        with frames_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line_count += 1
                try:
                    row = json.loads(line)
                except Exception as exc:
                    errors.append(f"{event_id}/{variant}:json_line_{line_number}={type(exc).__name__}")
                    continue
                if not isinstance(row, Mapping):
                    errors.append(f"{event_id}/{variant}:row_{line_number}_not_object")
                    continue
                frame = row.get("frame")
                if isinstance(frame, bool) or not isinstance(frame, int):
                    errors.append(f"{event_id}/{variant}:row_{line_number}_frame_invalid")
                    continue
                observed_frames.append(int(frame))
                counters["runtime_rows"] += 1
                if str(row.get("event_id")) != event_id or str(row.get("variant")) not in ROW_VARIANT_ALIASES[variant]:
                    errors.append(f"{event_id}/{variant}:{frame}:authority_axis")
                if str(row.get("sequence")) != str(record.get("sequence")):
                    errors.append(f"{event_id}/{variant}:{frame}:sequence_axis")
                if int(row.get("event_frame", -1)) != event_frame:
                    errors.append(f"{event_id}/{variant}:{frame}:event_frame_axis")
                for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                    if row.get(flag) is not False:
                        errors.append(f"{event_id}/{variant}:{frame}:{flag}_not_false")
                if row.get("public_id_immutable") is not True:
                    errors.append(f"{event_id}/{variant}:{frame}:public_id_not_immutable")
                if frame == event_frame:
                    counters["event_frame_rows"] += 1
                    if row.get("candidate_rows") != [] or row.get("candidate_count") != 0:
                        errors.append(f"{event_id}/{variant}:{frame}:event_candidate_not_empty")
                    if row.get("candidate_pool") is not None or row.get("assignment") is not None or row.get("score_audit") is not None:
                        errors.append(f"{event_id}/{variant}:{frame}:event_audit_not_empty")
                    if row.get("memory_read") is not False or row.get("event_frame_memory_read") is not False:
                        errors.append(f"{event_id}/{variant}:{frame}:event_memory_read_leak")
                    if row.get("memory_write") is not True:
                        errors.append(f"{event_id}/{variant}:{frame}:event_memory_write_missing")
                    if row.get("first_memory_visible_frame") != event_frame + 1:
                        errors.append(f"{event_id}/{variant}:{frame}:first_visible_boundary")
                    continue
                if frame not in expected_frames[1:]:
                    errors.append(f"{event_id}/{variant}:{frame}:frame_out_of_range")
                candidates = row.get("candidate_rows")
                pool = row.get("candidate_pool")
                score = row.get("score_audit")
                assignment = row.get("assignment")
                if not isinstance(candidates, list) or not isinstance(pool, Mapping) or not isinstance(score, Mapping) or not isinstance(assignment, Mapping):
                    errors.append(f"{event_id}/{variant}:{frame}:future_audit_missing")
                    continue
                try:
                    candidate_count = int(row.get("candidate_count"))
                except (TypeError, ValueError):
                    candidate_count = -1
                uids = [str(item.get("candidate_uid")) for item in candidates if isinstance(item, Mapping)]
                if candidate_count != len(candidates) or len(uids) != len(set(uids)):
                    errors.append(f"{event_id}/{variant}:{frame}:candidate_axis")
                counters["candidate_rows"] += len(candidates)
                pool_rows = pool.get("candidate_rows")
                pool_uids = pool.get("candidate_uids")
                if not isinstance(pool_rows, list) or [str(item.get("candidate_uid")) for item in pool_rows] != uids or list(pool_uids or []) != uids:
                    errors.append(f"{event_id}/{variant}:{frame}:pool_mapping_axis")
                for item in candidates:
                    if not isinstance(item, Mapping):
                        errors.append(f"{event_id}/{variant}:{frame}:candidate_not_object")
                        continue
                    # Current/persisted solver rows may carry the exact global
                    # mapping after the solver.  The pool remains the
                    # pre-solver, all-null authority axis checked above.
                    mapped = item.get("public_id")
                    if item.get("public_id_inference") is not False:
                        errors.append(f"{event_id}/{variant}:{frame}:candidate_public_authority")
                    if mapped is not None and item.get("public_id_authority") != "exact_global_solver_output":
                        errors.append(f"{event_id}/{variant}:{frame}:candidate_public_authority")
                    if mapped is not None and item.get("assigned_public_id") != mapped:
                        errors.append(f"{event_id}/{variant}:{frame}:candidate_assignment_mapping")
                if pool.get("runtime_future_gt_used") is not False or pool.get("runtime_gt_read") is not False or pool.get("posthoc_gt_used") is not False:
                    errors.append(f"{event_id}/{variant}:{frame}:pool_gt_flags")
                public_axis = score.get("public_id_axis")
                if not isinstance(public_axis, list):
                    errors.append(f"{event_id}/{variant}:{frame}:public_axis_missing")
                    continue
                matrix_keys = ["fused_score_matrix"]
                if "base_score_matrix" in score:
                    matrix_keys.insert(0, "base_score_matrix")
                elif variant != "E0_BASELINE_B0":
                    errors.append(f"{event_id}/{variant}:{frame}:base_score_matrix_missing")
                else:
                    counters["legacy_e0_score_schema_rows"] += 1
                for matrix_key in matrix_keys:
                    matrix = score.get(matrix_key)
                    if not finite_matrix(matrix) or len(matrix) != len(candidates) or any(len(r) != len(public_axis) for r in matrix):
                        errors.append(f"{event_id}/{variant}:{frame}:{matrix_key}_shape_or_finite")
                for vector_key in ("base_target_scores", "fused_target_scores"):
                    vector = score.get(vector_key)
                    if not finite_matrix([vector]) or not isinstance(vector, list) or len(vector) != len(candidates):
                        errors.append(f"{event_id}/{variant}:{frame}:{vector_key}_shape_or_finite")
                solver = assignment.get("solver")
                if not isinstance(solver, Mapping) or solver.get("runtime_future_gt_used") is not False:
                    errors.append(f"{event_id}/{variant}:{frame}:solver_gt_flags")
        if line_count != HORIZON + 1 or observed_frames != expected_frames:
            errors.append(f"{event_id}/{variant}:frame_axis_expected_{len(expected_frames)}_got_{line_count}")
        counters["variant_artifacts"] += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    combined = args.combined if args.combined.is_absolute() else ROOT / args.combined
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = read_json(combined)
    selected = manifest.get("selected_records")
    errors: list[str] = []
    counters: Counter[str] = Counter()
    if not isinstance(selected, list):
        errors.append("combined_selected_records_missing")
        selected = []
    for record in selected:
        if isinstance(record, Mapping) and record.get("status") == "PASS":
            audit_event(record, counters, errors)
        else:
            errors.append("selected_record_not_PASS")
    result = {
        "schema_version": "N72R11R2_FORMAL_RUNTIME_ROWS_AUDIT_V1",
        "status": "PASS_FORMAL_RUNTIME_ROWS_31_EVENTS" if not errors else "FAIL_FORMAL_RUNTIME_ROWS_AUDIT",
        "created_at_utc": now_utc(),
        "source_combined_manifest": str(combined),
        "source_combined_manifest_sha256": sha256_file(combined),
        "source_status": manifest.get("status"),
        "selected_event_count": len(selected),
        "required_event_count": manifest.get("event_count_required"),
        "variants": list(VARIANTS),
        "expected_frames_per_variant": HORIZON + 1,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "counters": dict(sorted(counters.items())),
        "errors": sorted(set(errors)),
        "failed_event_excluded_by_combined_audit": manifest.get("missing_event_ids", []),
        "note": "This audit covers sealed PASS records only; the failed event remains preserved in the combined audit and its failure artifact.",
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "output": str(output), "errors": len(errors), "counters": result["counters"]}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
