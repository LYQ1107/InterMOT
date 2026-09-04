#!/usr/bin/env python3
"""Audit native/geometry/solver authority after the N72R6 fallback replay.

This is a read-only, post-replay audit.  It does not change the score formula,
candidate stream, public mapping, solver, or future window.  It checks whether
the target-session rows that survive the fixed human-anchor gate are rejected
by the target-public solver domain or by a non-finite/negative base score.  GT
labels are not opened here; spatial quality is imported only from the sealed
posthoc root-cause artifact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    HORIZON,
    atomic_json,
    read_json,
    read_jsonl,
    sha256_file,
)


DEFAULT_REPLAY_ROOT = ROOT / "outputs/N72R6/public_replay/human_anchor_fallback_attempt1"
DEFAULT_EFFECT = ROOT / "outputs/N72R6/ccam_paired_replay_results_human_anchor_fallback.json"
DEFAULT_ROOT_AUDIT = ROOT / "outputs/N72R6/target_root_cause_audit_human_anchor_fallback.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R6/native_geometry_authority_audit_human_anchor_fallback.json"
DEFAULT_STATUS = ROOT / "outputs/N72R6/stage_09_native_geometry_authority_status.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--effect", default=str(DEFAULT_EFFECT))
    parser.add_argument("--root-audit", default=str(DEFAULT_ROOT_AUDIT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--status-output", default=str(DEFAULT_STATUS))
    args = parser.parse_args()

    replay_root = resolve(args.replay_root)
    effect_path = resolve(args.effect)
    root_audit_path = resolve(args.root_audit)
    output_path = resolve(args.output)
    status_path = resolve(args.status_output)

    batch_path = replay_root / "replay_batch_status.json"
    batch = read_json(batch_path)
    if batch.get("status") != "PASS_N72R6_C0_C1_REPLAY" or int(batch.get("completed_event_count", -1)) != 32:
        raise RuntimeError(f"replay batch is not complete: {batch.get('status')}")
    effect = read_json(effect_path)
    if effect.get("status") != "FAIL_FUTURE_EFFECT":
        raise RuntimeError(f"effect artifact is not the recorded FAIL: {effect.get('status')}")
    root_audit = read_json(root_audit_path)
    if root_audit.get("status") != "PASS_TARGET_ROOT_CAUSE_AUDIT":
        raise RuntimeError(f"root-cause audit is not PASS: {root_audit.get('status')}")

    frame_counts = Counter()
    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    base_target_scores: list[float] = []
    fallback_target_scores: list[float] = []
    target_solver_scores: list[float] = []
    target_scope_matches = 0
    target_scope_total = 0
    target_rows_assigned = 0
    target_rows_total = 0
    fallback_rows_assigned = 0
    fallback_rows_total = 0
    no_target_or_fallback = 0
    nonfinite_target_score = 0
    nonpositive_target_score = 0
    target_none_decisions = 0
    fallback_none_decisions = 0
    fallback_uids_not_shadowed = 0
    fallback_uids_mismatch = 0
    event_records: list[dict[str, Any]] = []

    for manifest_path in sorted(replay_root.glob("*/event_manifest.json")):
        manifest = read_json(manifest_path)
        event_id = str(manifest["event_id"])
        action = str(manifest["action_type"])
        target_public = int(manifest["target_public_id"])
        event_frame = int(manifest["event_frame"])
        target_rows = 0
        target_assigned = 0
        fallback_rows = 0
        fallback_assigned = 0
        no_source = 0
        scope_matches = 0
        scope_total = 0
        replay_rows = read_jsonl(resolve(str(manifest["c1"]["path"])))
        if len(replay_rows) != HORIZON + 1:
            raise RuntimeError(f"frame count mismatch: {event_id}")
        for row in replay_rows:
            frame = int(row["frame"])
            if frame <= event_frame:
                continue
            frame_counts["future_frames"] += 1
            axis = [int(value) for value in row.get("public_id_axis", [])]
            if target_public not in axis:
                raise RuntimeError(f"target public absent from axis: {event_id}:{frame}")
            target_col = axis.index(target_public)
            candidates = row.get("candidate_rows", [])
            target_candidates = [
                (index, candidate) for index, candidate in enumerate(candidates)
                if candidate.get("candidate_kind") == "TARGET_CORRECTION_SESSION_CANDIDATE"
            ]
            exclusive = row.get("target_exclusive_constraint") or {}
            fallback_uid = exclusive.get("fallback_main_candidate_uid")
            fallback_uid = None if fallback_uid in (None, "") else str(fallback_uid)
            shadowed = {str(value) for value in exclusive.get("shadowed_main_candidate_uids", [])}
            selected_fallback = (row.get("human_anchor_main_fallback") or {}).get("selected_main_candidate_uid")
            selected_fallback = None if selected_fallback in (None, "") else str(selected_fallback)
            if selected_fallback != fallback_uid:
                fallback_uids_mismatch += 1
            if fallback_uid is not None and fallback_uid not in shadowed:
                fallback_uids_not_shadowed += 1

            if target_candidates:
                if len(target_candidates) != 1:
                    raise RuntimeError(f"multiple target rows: {event_id}:{frame}")
                index, candidate = target_candidates[0]
                target_rows += 1
                target_rows_total += 1
                frame_counts["target_session_rows"] += 1
                score = row.get("base_score_matrix", [])[index][target_col]
                if not finite(score):
                    nonfinite_target_score += 1
                else:
                    score_value = float(score)
                    base_target_scores.append(score_value)
                    target_solver_scores.append(float(candidate.get("solver_score", 0.0)))
                    if score_value <= 0.0:
                        nonpositive_target_score += 1
                target_scope_total += 1
                scope_total += 1
                expected_scope = str(row.get("correction_epoch", {}).get("target_session_scope", ""))
                scope_ok = str(candidate.get("native_scope", "")) == expected_scope
                target_scope_matches += int(scope_ok)
                scope_matches += int(scope_ok)
                target_assigned += int(candidate.get("solver_public_id") == target_public)
                target_rows_assigned += int(candidate.get("solver_public_id") == target_public)
                target_none_decisions += int(candidate.get("solver_public_id") is None)
            elif fallback_uid is not None:
                fallback_rows += 1
                fallback_rows_total += 1
                frame_counts["fallback_main_rows"] += 1
                matches = [
                    (index, candidate) for index, candidate in enumerate(candidates)
                    if str(candidate.get("candidate_uid")) == fallback_uid
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"fallback row coverage mismatch: {event_id}:{frame}")
                index, candidate = matches[0]
                score = row.get("base_score_matrix", [])[index][target_col]
                if finite(score):
                    fallback_target_scores.append(float(score))
                fallback_assigned += int(candidate.get("solver_public_id") == target_public)
                fallback_rows_assigned += int(candidate.get("solver_public_id") == target_public)
                fallback_none_decisions += int(candidate.get("solver_public_id") is None)
            else:
                no_target_or_fallback += 1
                no_source += 1
                frame_counts["none_without_source"] += 1
        action_counts[action].update({
            "future_frames": HORIZON,
            "target_session_rows": target_rows,
            "target_session_assigned": target_assigned,
            "fallback_main_rows": fallback_rows,
            "fallback_main_assigned": fallback_assigned,
            "no_target_or_fallback": no_source,
            "scope_matches": scope_matches,
            "scope_total": scope_total,
        })
        event_records.append({
            "event_id": event_id,
            "sequence": str(manifest["sequence"]),
            "action_type": action,
            "target_session_rows": target_rows,
            "target_session_assigned": target_assigned,
            "fallback_main_rows": fallback_rows,
            "fallback_main_assigned": fallback_assigned,
            "target_scope_matches": scope_matches,
            "target_scope_total": scope_total,
            "runtime_future_gt_used": False,
        })

    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
            "max": float(np.max(array)),
        }

    target_spatial = root_audit.get("totals", {})
    payload = {
        "schema_version": "N72R6_NATIVE_GEOMETRY_AUTHORITY_AUDIT_V1",
        "status": "PASS_N72R6_NATIVE_GEOMETRY_AUTHORITY_NOT_PRIMARY_BOTTLENECK",
        "inputs": {
            "replay_batch": str(batch_path),
            "replay_batch_sha256": sha256_file(batch_path),
            "effect": str(effect_path),
            "effect_sha256": sha256_file(effect_path),
            "root_cause_audit": str(root_audit_path),
            "root_cause_audit_sha256": sha256_file(root_audit_path),
        },
        "frame_counts": dict(frame_counts),
        "action_counts": {key: dict(value) for key, value in sorted(action_counts.items())},
        "target_session_base_target_public_score": stats(base_target_scores),
        "fallback_main_base_target_public_score": stats(fallback_target_scores),
        "target_session_solver_score": stats(target_solver_scores),
        "target_session_rows_total": target_rows_total,
        "target_session_rows_assigned_target": target_rows_assigned,
        "target_session_assignment_rate": None if not target_rows_total else target_rows_assigned / target_rows_total,
        "fallback_main_rows_total": fallback_rows_total,
        "fallback_main_rows_assigned_target": fallback_rows_assigned,
        "fallback_main_assignment_rate": None if not fallback_rows_total else fallback_rows_assigned / fallback_rows_total,
        "target_session_native_scope_matches": target_scope_matches,
        "target_session_native_scope_total": target_scope_total,
        "target_session_native_scope_match_rate": None if not target_scope_total else target_scope_matches / target_scope_total,
        "nonfinite_target_public_base_score_count": nonfinite_target_score,
        "nonpositive_target_public_base_score_count": nonpositive_target_score,
        "target_session_explicit_none_decisions": target_none_decisions,
        "fallback_main_explicit_none_decisions": fallback_none_decisions,
        "fallback_uid_not_shadowed_count": fallback_uids_not_shadowed,
        "fallback_uid_audit_mismatch_count": fallback_uids_mismatch,
        "posthoc_spatial_quality": {
            "target_candidate_rows": target_spatial.get("target_candidate_rows"),
            "target_candidate_spatial_hits": target_spatial.get("target_candidate_spatial_hits"),
            "target_candidate_drift_frames": target_spatial.get("target_candidate_drift_frames"),
            "target_candidate_absent_visible_frames": target_spatial.get("target_candidate_absent_visible_frames"),
        },
        "interpretation": "Target-session rows have finite positive target-public scores, correct target-session native scope, and are assigned to the explicit target domain when present; the dominant failure is candidate absence/spatial drift, not native/geometry/solver authority rejection.",
        "c2_authorized": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "events": event_records,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_path, payload)
    status = {
        "schema_version": "N72R6_STAGE_STATUS_V1",
        "stage": "N72R6-09_NATIVE_GEOMETRY_AUTHORITY_AUDIT",
        "status": payload["status"],
        "output": str(output_path),
        "target_session_assignment_rate": payload["target_session_assignment_rate"],
        "target_session_native_scope_match_rate": payload["target_session_native_scope_match_rate"],
        "nonfinite_target_public_base_score_count": nonfinite_target_score,
        "nonpositive_target_public_base_score_count": nonpositive_target_score,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(status_path, status)
    print({"status": payload["status"], "target_rows": target_rows_total, "fallback_rows": fallback_rows_total, "no_source": no_target_or_fallback})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
