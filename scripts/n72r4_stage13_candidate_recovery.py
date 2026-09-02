#!/usr/bin/env python3
"""CPU-only track-centric candidate-recovery probe for N72R4.

This is a bounded diagnostic branch, not a production detector.  It consumes
the already validated official B1 corrected future stream and proposes one
candidate from the persistent target state only when the explicit-NONE
association leaves the target public identity unassigned.  The proposal has
no public-ID authority; only the exact public-ID+NONE solver can associate it.
The official candidate rows are retained verbatim and are evaluated separately
from recovery proposals.

The rule is frozen before posthoc GT is opened.  GT is used only after all
runtime artifacts have passed the causal and mapping validator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.evaluation.interaction_effect_metrics import (  # noqa: E402
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    metric_record,
    sequence_cluster_bootstrap,
)
from scripts.n72r4_stage10_cpu_analysis import (  # noqa: E402
    M0_ROOT,
    OUT,
    PAIR_MANIFEST,
    State,
    association_score,
    atomic_json,
    atomic_jsonl,
    box_iou,
    current_post_anchor,
    feature_digest,
    finite_feature,
    initialise_b1,
    load_events,
    load_stage16,
    load_stage18_public_map,
    load_branch_rows,
    now_utc,
    predicted_box,
    read_json,
    read_jsonl,
    row_view,
    sha256_file,
    state_copy,
    update_state,
)
from scripts.n72r4_stage11_memory_replay import (  # noqa: E402
    SolverState,
    _candidate_output_rows,
    _mapping_from_solver,
    _solver_rows,
    _state_axis,
)


BRANCHES = ("R0_M0_NO_RECOVERY", "R1_TRACK_CENTRIC_RECOVERY")
HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.5
NONE_SCORE = 0.0
STAGE_NAME = "13_TRACK_CENTRIC_CANDIDATE_RECOVERY"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _prestate_axis(event_id: str) -> tuple[dict[int, int], Path]:
    path = OUT / "event_prestate" / "attempt5" / event_id / "persistent_runtime_snapshot.json"
    snapshot = read_json(path)
    payload = snapshot.get("payload")
    if snapshot.get("runtime_future_gt_used") is not False or not isinstance(payload, dict):
        raise RuntimeError(f"persistent prestate violates causal contract: {event_id}")
    values = payload.get("public_to_state")
    if not isinstance(values, dict) or not values:
        raise RuntimeError(f"persistent prestate public/state authority missing: {event_id}")
    mapping = {int(public): int(state) for public, state in values.items()}
    if len(mapping) != len(set(mapping)) or len(mapping) != len(set(mapping.values())):
        raise RuntimeError(f"persistent public/state axis duplicated: {event_id}")
    return mapping, path


def _make_recovery_candidate(
    *,
    event_id: str,
    frame: int,
    target_public: int,
    state: State,
    official_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if state.last_box is None or state.last_feature is None:
        return None
    box = predicted_box(state, frame)
    if box is None or box.size != 4 or not np.all(np.isfinite(box)):
        return None
    if float(box[2] - box[0]) <= 0.0 or float(box[3] - box[1]) <= 0.0:
        return None
    existing_indices = [int(row["candidate_index"]) for row in official_rows]
    next_index = max(existing_indices, default=-1) + 1
    uid = f"recovery:{event_id}:frame:{int(frame)}:target_public:{int(target_public)}"
    feature = finite_feature(state.last_feature)
    return {
        "candidate_key": uid,
        "candidate_index": next_index,
        "official_raw_sam_id": -1,
        "raw_native_id": -1,
        "native_tid": -1,
        "adapter_external_id": -1,
        "adapter_visible_id": -1,
        "box_xyxy": [float(value) for value in box.tolist()],
        "feature": feature,
        "feature_sha256": feature_digest(feature),
        "confidence": 0.0,
        "source": "track_centric_recovery_motion_feature",
        "candidate_kind": "RECOVERY_PROPOSAL_NO_PUBLIC_AUTHORITY",
        "recovery_target_public_id_hint": int(target_public),
    }


def _solve_without_mutating(
    *,
    rows: list[dict[str, Any]],
    states: dict[int, State],
    public_to_state: dict[int, int],
    event_id: str,
    branch: str,
    frame: int,
) -> tuple[list[int | None], list[str], dict[str, Any], list[int], list[SolverState], np.ndarray]:
    publics, solver_states = _state_axis(public_to_state, states)
    scores = np.asarray(
        [[association_score(states[public], row, frame) for row in rows] for public in publics],
        dtype=np.float64,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(f"recovery base score nonfinite: {event_id}/{branch}/{frame}")
    solver = __import__("scripts.n72r4_stage11_memory_replay", fromlist=["solve_effect_assignment"]).solve_effect_assignment(
        candidate_rows=_solver_rows(rows),
        persistent_states=solver_states,
        fused_state_candidate_scores=scores,
        source_run_id=f"n72r4-stage13:{event_id}:{branch}:{frame}",
        session_id=f"n72r4-stage13:{event_id}:{branch}",
        none_score=NONE_SCORE,
    )
    public_ids, statuses = _mapping_from_solver(solver, rows)
    return public_ids, statuses, solver, publics, solver_states, scores


def _apply_assignment(
    *,
    rows: list[dict[str, Any]],
    public_ids: list[int | None],
    states: dict[int, State],
    frame: int,
) -> None:
    assigned = {int(value) for value in public_ids if value is not None}
    for public, state in states.items():
        if public not in assigned:
            state.status = "LOST"
            continue
        index = public_ids.index(public)
        update_state(state, rows[index], frame)


def _runtime_row(
    *,
    event: dict[str, Any],
    branch: str,
    variant: str,
    source_row: dict[str, Any],
    rows: list[dict[str, Any]],
    public_ids: list[int | None],
    statuses: list[str],
    solver: dict[str, Any],
    public_to_state: dict[int, int],
    states: dict[int, State],
    public_order: list[int],
    state_order: list[SolverState],
    scores: np.ndarray,
    recovery_triggered: bool,
    recovery_accepted: bool,
    recovery_reason: str,
    official_count: int,
) -> dict[str, Any]:
    frame = int(source_row["frame"])
    event_frame = int(event["event_frame"])
    output_rows = _candidate_output_rows(rows, public_ids, statuses)
    for output, row in zip(output_rows, rows):
        output["candidate_kind"] = str(row.get("candidate_kind", "OFFICIAL_SAM3_CANDIDATE"))
        output["is_recovery_proposal"] = bool(row.get("candidate_kind") == "RECOVERY_PROPOSAL_NO_PUBLIC_AUTHORITY")
        output["recovery_public_authority"] = False
    return {
        "schema_version": "N72R4_STAGE13_RECOVERY_FRAME_V1",
        "event_id": str(event["event_id"]),
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "branch": branch,
        "variant": variant,
        "candidate_stream_kind": "OFFICIAL_SAM3_FUTURE_PROPAGATION_PLUS_EXPLICIT_RECOVERY_PROPOSAL",
        "official_candidate_count": int(official_count),
        "recovery_candidate_count": int(len(rows) - official_count),
        "candidate_stream_sha256": str(source_row["y_pre_semantic_hash"] if frame == event_frame else source_row["frame_hash_sha256"]),
        "event_frame": event_frame,
        "frame": frame,
        "frame_horizon": frame - event_frame,
        "phase": "FUTURE_ASSOCIATION",
        "candidate_order": [int(row["candidate_index"]) for row in rows],
        "candidate_rows": output_rows,
        "association_state_axis": [int(item.association_state_id) for item in state_order],
        "public_id_order": public_order,
        "target_public_id": int(event["_target_public_id"]),
        "target_state_index": public_order.index(int(event["_target_public_id"])) if int(event["_target_public_id"]) in public_order else None,
        "base_score_matrix": scores.astype(float).tolist(),
        "appearance_score_matrix": np.zeros_like(scores, dtype=np.float64).tolist(),
        "appearance_score_deltas": np.zeros_like(scores, dtype=np.float64).tolist(),
        "fused_score_matrix": scores.astype(float).tolist(),
        "solver_executed": True,
        "solver": solver,
        "assignment_public_ids": [None if value is None else int(value) for value in public_ids],
        "assignment_status": statuses,
        "assignment_map": {
            str(row["candidate_key"]): None if public is None else int(public)
            for row, public in zip(rows, public_ids)
        },
        "recovery_triggered": bool(recovery_triggered),
        "recovery_accepted": bool(recovery_accepted),
        "recovery_reason": str(recovery_reason),
        "recovery_has_no_public_authority_before_solver": True,
        "public_id_created": False,
        "public_id_authority": "PERSISTENT_PRESTATE_ONLY_AFTER_EXACT_SOLVER",
        "causal_boundary": {
            "event_frame_memory_read": False,
            "first_future_frame": event_frame + 1,
            "runtime_future_gt_used": False,
        },
        "public_state_axis_after_frame": [
            {
                "public_id": int(public),
                "association_state_id": int(public_to_state[public]),
                "last_frame": int(states[public].last_frame),
                "last_native": None if states[public].last_native is None else int(states[public].last_native),
                "status": str(states[public].status),
            }
            for public in sorted(states)
        ],
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _event_frame(
    *,
    event: dict[str, Any],
    source_row: dict[str, Any],
    public_by_raw: dict[int, int],
    public_to_state: dict[int, int],
    states: dict[int, State],
    stage16: dict[str, Any],
    target_post: dict[str, Any],
    public_to_post: dict[int, int],
    branch: str,
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    frame = int(event["event_frame"])
    rows = row_view("B1_CURRENT_FRAME_CORRECTION", event_id, source_row)
    mapping = [public_by_raw.get(int(row["official_raw_sam_id"])) for row in rows]
    if len([value for value in mapping if value is not None]) != len({value for value in mapping if value is not None}):
        raise RuntimeError(f"event pre-mapping duplicate: {event_id}")
    publics = sorted(states)
    zeros = np.zeros((len(publics), len(rows)), dtype=np.float64)
    base = np.asarray(
        [[association_score(states[public], row, frame) for row in rows] for public in publics],
        dtype=np.float64,
    )
    return {
        "schema_version": "N72R4_STAGE13_RECOVERY_FRAME_V1",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "branch": branch,
        "variant": "R0_M0_NO_RECOVERY" if branch == "R0_M0_NO_RECOVERY" else "R1_TRACK_CENTRIC_RECOVERY",
        "candidate_stream_kind": "OFFICIAL_SAM3_EVENT_FRAME_PRE_CORRECTION_STREAM",
        "official_candidate_count": len(rows),
        "recovery_candidate_count": 0,
        "candidate_stream_sha256": str(source_row["y_pre_semantic_hash"]),
        "event_frame": frame,
        "frame": frame,
        "frame_horizon": 0,
        "phase": "CURRENT_FRAME_CORRECTION",
        "candidate_order": [int(row["candidate_index"]) for row in rows],
        "candidate_rows": [
            {
                **item,
                "public_id": None if public is None else int(public),
                "assignment_status": "PRESTATE_PERSISTENT_MAPPING" if public is not None else "EXPLICIT_NONE_UNMAPPED_PRESTATE_CANDIDATE",
                "candidate_kind": "OFFICIAL_SAM3_CANDIDATE",
                "is_recovery_proposal": False,
                "recovery_public_authority": False,
            }
            for item, public in zip(
                [
                    {
                        "candidate_uid": str(row["candidate_key"]),
                        "candidate_index": int(row["candidate_index"]),
                        "official_raw_sam_id": int(row["official_raw_sam_id"]),
                        "raw_native_id": int(row["raw_native_id"]),
                        "native_tid": int(row["native_tid"]),
                        "adapter_external_id": int(row["adapter_external_id"]),
                        "adapter_visible_id": int(row["adapter_visible_id"]),
                        "box_xyxy": [float(value) for value in row["box_xyxy"]],
                        "feature_sha256": str(row["feature_sha256"]),
                        "confidence": float(row["confidence"]),
                        "source": str(row["source"]),
                    }
                    for row in rows
                ],
                mapping,
            )
        ],
        "association_state_axis": [int(public_to_state[public]) for public in publics],
        "public_id_order": publics,
        "target_public_id": int(event["_target_public_id"]),
        "target_state_index": publics.index(int(event["_target_public_id"])),
        "base_score_matrix": base.tolist(),
        "appearance_score_matrix": zeros.tolist(),
        "appearance_score_deltas": zeros.tolist(),
        "fused_score_matrix": base.tolist(),
        "solver_executed": False,
        "assignment_public_ids": [None if value is None else int(value) for value in mapping],
        "assignment_status": ["PRESTATE_PERSISTENT_MAPPING" if value is not None else "EXPLICIT_NONE_UNMAPPED_PRESTATE_CANDIDATE" for value in mapping],
        "assignment_map": {
            str(row["candidate_key"]): None if public is None else int(public)
            for row, public in zip(rows, mapping)
        },
        "correction": {
            "target_post_raw_id": int(target_post["official_raw_sam_id"]),
            "public_to_post_raw": {str(public): int(raw) for public, raw in sorted(public_to_post.items())},
            "post_observation_source": str(stage16["official_current_correction"]["post_observation"]["source"]),
            "spatial_correction_before_any_recovery": True,
        },
        "recovery_triggered": False,
        "recovery_accepted": False,
        "recovery_reason": "EVENT_FRAME_RECOVERY_FORBIDDEN",
        "recovery_has_no_public_authority_before_solver": True,
        "public_id_created": False,
        "public_id_authority": "N72R3_STAGE18_PERSISTENT_RUNTIME_PRESTATE",
        "causal_boundary": {
            "event_frame_memory_read": False,
            "first_future_frame": frame + 1,
            "runtime_future_gt_used": False,
        },
        "public_state_axis_after_frame": [
            {
                "public_id": int(public),
                "association_state_id": int(public_to_state[public]),
                "last_frame": int(states[public].last_frame),
                "last_native": None if states[public].last_native is None else int(states[public].last_native),
                "status": str(states[public].status),
            }
            for public in publics
        ],
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _run_event(event: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_id = str(event["event_id"])
    stage16 = load_stage16(event_id)
    event["_target_public_id"] = int(stage16["persistent_identity"]["public_id"])
    public_to_state, prestate_path = _prestate_axis(event_id)
    public_by_raw = load_stage18_public_map(str(event["sequence"]), int(event["event_frame"]))
    branch_rows = load_branch_rows(M0_ROOT, event_id)
    event_row = branch_rows[int(event["event_frame"])]
    target_post, post_candidates = current_post_anchor(stage16, event_row)
    initial_states, public_to_post = initialise_b1(
        event_row,
        post_candidates,
        target_post,
        public_by_raw,
        int(event["_target_public_id"]),
        event_id,
    )
    if set(initial_states) != set(public_to_state):
        raise RuntimeError(f"recovery initial public axis changed: {event_id}")
    rows_out: list[dict[str, Any]] = []
    for branch in BRANCHES:
        states = {public: state_copy(value) for public, value in initial_states.items()}
        rows_out.append(
            _event_frame(
                event=event,
                source_row=event_row,
                public_by_raw=public_by_raw,
                public_to_state=public_to_state,
                states=states,
                stage16=stage16,
                target_post=target_post,
                public_to_post=public_to_post,
                branch=branch,
            )
        )
        for frame in range(int(event["event_frame"]) + 1, int(event["event_frame"]) + 101):
            source_row = branch_rows[frame]
            official_rows = row_view("B1_CURRENT_FRAME_CORRECTION", event_id, source_row)
            before = {public: state_copy(value) for public, value in states.items()}
            public_ids, statuses, solver, publics, solver_states, scores = _solve_without_mutating(
                rows=official_rows,
                states=before,
                public_to_state=public_to_state,
                event_id=event_id,
                branch=branch,
                frame=frame,
            )
            target_public = int(event["_target_public_id"])
            recovery_triggered = False
            recovery_accepted = False
            recovery_reason = "TARGET_ALREADY_ASSIGNED_OR_BRANCH_HAS_NO_PROPOSAL"
            rows = official_rows
            official_count = len(official_rows)
            if branch == "R1_TRACK_CENTRIC_RECOVERY" and target_public not in {value for value in public_ids if value is not None}:
                recovery_triggered = True
                proposal = _make_recovery_candidate(
                    event_id=event_id,
                    frame=frame,
                    target_public=target_public,
                    state=before[target_public],
                    official_rows=official_rows,
                )
                if proposal is None:
                    recovery_reason = "TARGET_UNASSIGNED_BUT_PERSISTENT_STATE_NOT_PROPOSABLE"
                else:
                    rows = official_rows + [proposal]
                    public_ids, statuses, solver, publics, solver_states, scores = _solve_without_mutating(
                        rows=rows,
                        states=before,
                        public_to_state=public_to_state,
                        event_id=event_id,
                        branch=branch,
                        frame=frame,
                    )
                    recovery_accepted = bool(public_ids[-1] == target_public)
                    recovery_reason = "RECOVERY_PROPOSAL_ASSOCIATED_TO_TARGET_PUBLIC_ID" if recovery_accepted else "RECOVERY_PROPOSAL_REMAINED_EXPLICIT_NONE_OR_OTHER_ID"
            elif branch == "R1_TRACK_CENTRIC_RECOVERY":
                recovery_reason = "TARGET_ALREADY_ASSIGNED_NO_RECOVERY_PROPOSAL_NEEDED"
            _apply_assignment(rows=rows, public_ids=public_ids, states=states, frame=frame)
            rows_out.append(
                _runtime_row(
                    event=event,
                    branch=branch,
                    variant=branch,
                    source_row=source_row,
                    rows=rows,
                    public_ids=public_ids,
                    statuses=statuses,
                    solver=solver,
                    public_to_state=public_to_state,
                    states=states,
                    public_order=publics,
                    state_order=solver_states,
                    scores=scores,
                    recovery_triggered=recovery_triggered,
                    recovery_accepted=recovery_accepted,
                    recovery_reason=recovery_reason,
                    official_count=official_count,
                )
            )
    rows_out.sort(key=lambda row: (int(row["frame"]), BRANCHES.index(str(row["branch"]))))
    return rows_out, {
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": int(event["event_frame"]),
        "future_frame_count": 100,
        "branch_count": len(BRANCHES),
        "branch_frame_count": len(rows_out),
        "prestate_snapshot": str(prestate_path),
        "prestate_snapshot_sha256": sha256_file(prestate_path),
        "target_public_id": int(event["_target_public_id"]),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def _candidate_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("candidate_uid")),
        int(item.get("candidate_index", -1)),
        tuple(float(value) for value in item.get("box_xyxy", [])),
        str(item.get("feature_sha256")),
        bool(item.get("is_recovery_proposal", False)),
    )


def _validate_runtime(events: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    files = []
    checked = 0
    recovery_proposals = 0
    accepted_recovery = 0
    for event in events:
        event_id = str(event["event_id"])
        rows = read_jsonl(root / f"{event_id}.jsonl")
        if len(rows) != len(BRANCHES) * 101:
            raise RuntimeError(f"recovery artifact row count mismatch: {event_id}/{len(rows)}")
        keyed = {(str(row.get("branch")), int(row.get("frame", -1))): row for row in rows}
        if len(keyed) != len(rows):
            raise RuntimeError(f"recovery artifact duplicate branch/frame: {event_id}")
        expected = {(branch, frame) for branch in BRANCHES for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101)}
        if set(keyed) != expected:
            raise RuntimeError(f"recovery artifact missing branch/frame: {event_id}")
        for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101):
            official_signatures = None
            for branch in BRANCHES:
                row = keyed[(branch, frame)]
                if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                    raise RuntimeError(f"recovery runtime GT boundary failed: {event_id}/{branch}/{frame}")
                if row.get("public_id_created") is not False or row.get("recovery_has_no_public_authority_before_solver") is not True:
                    raise RuntimeError(f"recovery public authority invariant failed: {event_id}/{branch}/{frame}")
                candidate_rows = row.get("candidate_rows")
                if not isinstance(candidate_rows, list):
                    raise RuntimeError(f"recovery candidate rows missing: {event_id}/{branch}/{frame}")
                for item in candidate_rows:
                    if item.get("is_recovery_proposal"):
                        recovery_proposals += 1
                        if item.get("recovery_public_authority") is not False:
                            raise RuntimeError(f"recovery proposal has public authority: {event_id}/{branch}/{frame}")
                accepted_recovery += int(bool(row.get("recovery_accepted")))
                signature = tuple(_candidate_signature(item) for item in candidate_rows if not item.get("is_recovery_proposal", False))
                if official_signatures is None:
                    official_signatures = signature
                elif official_signatures != signature:
                    raise RuntimeError(f"official candidate stream changed between recovery branches: {event_id}/{frame}")
                base = np.asarray(row.get("base_score_matrix"), dtype=np.float64)
                fused = np.asarray(row.get("fused_score_matrix"), dtype=np.float64)
                appearance = np.asarray(row.get("appearance_score_matrix"), dtype=np.float64)
                shape = (len(row.get("public_id_order", [])), len(candidate_rows))
                if base.shape != shape or fused.shape != shape or appearance.shape != shape:
                    raise RuntimeError(f"recovery score matrix shape mismatch: {event_id}/{branch}/{frame}")
                if not (np.isfinite(base).all() and np.isfinite(fused).all() and np.array_equal(base, fused) and np.array_equal(appearance, np.zeros_like(appearance))):
                    raise RuntimeError(f"recovery score matrix invalid: {event_id}/{branch}/{frame}")
                public_values = [item.get("public_id") for item in candidate_rows if item.get("public_id") is not None]
                if len(public_values) != len(set(public_values)):
                    raise RuntimeError(f"recovery public assignment duplicated: {event_id}/{branch}/{frame}")
                checked += 1
        files.append({"event_id": event_id, "path": str(root / f"{event_id}.jsonl"), "sha256": sha256_file(root / f"{event_id}.jsonl"), "row_count": len(rows)})
    return {
        "status": "PASS_STAGE13_RECOVERY_RUNTIME_VALIDATION",
        "event_count": len(events),
        "branch_count": len(BRANCHES),
        "checked_branch_frame_rows": checked,
        "official_candidate_stream_unchanged": True,
        "explicit_none_retained": True,
        "recovery_proposals_have_no_public_authority": True,
        "recovery_proposal_rows": recovery_proposals,
        "accepted_recovery_assignments": accepted_recovery,
        "runtime_future_gt_used": False,
        "gt_loaded_in_worker": False,
        "files": files,
    }


def _load_gt(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
    path = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack") / "train" / sequence / "gt/gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            frame, gid = int(parts[0]), int(parts[1])
            x, y, width, height = [float(item) for item in parts[2:6]]
            result[frame - 1][gid] = {"box": [x, y, x + width, y + height]}
    return result


def _candidate_best(row: dict[str, Any], box: list[float], *, official_only: bool = False) -> tuple[float, dict[str, Any] | None]:
    values = [
        (box_iou(item["box_xyxy"], box), item)
        for item in row.get("candidate_rows", [])
        if not official_only or not item.get("is_recovery_proposal", False)
    ]
    values.sort(key=lambda item: (-item[0], str(item[1].get("candidate_uid"))))
    return (float(values[0][0]), values[0][1]) if values else (0.0, None)


def _public_iou(row: dict[str, Any], public_id: int, box: list[float]) -> float:
    return max(
        (
            box_iou(item["box_xyxy"], box)
            for item in row.get("candidate_rows", [])
            if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)
        ),
        default=0.0,
    )


def _assignment_map(row: dict[str, Any]) -> dict[str, int | None]:
    return {
        str(item["candidate_uid"]): None if item.get("public_id") is None else int(item["public_id"])
        for item in row.get("candidate_rows", [])
    }


def _metric_template() -> dict[str, float | int]:
    return {
        "evaluated_frames": 0,
        "recovery_target_iou_sum": 0.0,
        "m0_target_iou_sum": 0.0,
        "recovery_correct_frames": 0,
        "m0_correct_frames": 0,
        "recovery_missing_frames": 0,
        "m0_missing_frames": 0,
        "recovery_candidate_present_frames": 0,
        "m0_candidate_present_frames": 0,
        "recovery_identity_error_frames": 0,
        "m0_identity_error_frames": 0,
        "recovery_id_switch_count": 0,
        "m0_id_switch_count": 0,
        "recovery_recorrect_count": 0,
        "m0_recorrect_count": 0,
        "assignment_change_count": 0,
        "true_correct_crossings": 0,
        "true_incorrect_crossings": 0,
        "directional_improvements": 0,
        "directional_regressions": 0,
        "neutral_changes": 0,
        "recovery_proposal_count": 0,
        "recovery_proposal_accepted_count": 0,
        "identity_error_reduction_sum": 0.0,
        "delta_iou_sum": 0.0,
    }


def _finalize(metric: dict[str, Any]) -> dict[str, Any]:
    denom = max(1, int(metric["evaluated_frames"]))
    metric["recovery_mean_iou"] = float(metric["recovery_target_iou_sum"] / denom)
    metric["m0_mean_iou"] = float(metric["m0_target_iou_sum"] / denom)
    metric["delta_iou_mean"] = float(metric["delta_iou_sum"] / denom)
    metric["identity_error_reduction"] = float(metric["identity_error_reduction_sum"] / denom)
    metric["recovery_identity_error"] = float(metric["recovery_identity_error_frames"] / denom)
    metric["m0_identity_error"] = float(metric["m0_identity_error_frames"] / denom)
    metric["recovery_missing_rate"] = float(metric["recovery_missing_frames"] / denom)
    metric["m0_missing_rate"] = float(metric["m0_missing_frames"] / denom)
    metric["recovery_candidate_recall"] = float(metric["recovery_candidate_present_frames"] / denom)
    metric["m0_candidate_recall"] = float(metric["m0_candidate_present_frames"] / denom)
    metric["recovery_id_switch_rate"] = float(metric["recovery_id_switch_count"] / denom)
    metric["m0_id_switch_rate"] = float(metric["m0_id_switch_count"] / denom)
    metric["recovery_recorrection_rate"] = float(metric["recovery_recorrect_count"] / denom)
    metric["m0_recorrection_rate"] = float(metric["m0_recorrect_count"] / denom)
    metric["assignment_change_rate"] = float(metric["assignment_change_count"] / denom)
    return metric


def _score_event(
    *,
    event: dict[str, Any],
    rows: dict[tuple[str, int], dict[str, Any]],
    gt_frames: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_frame = int(event["event_frame"])
    target_public = int(event["_target_public_id"])
    target_gid = int(event["dataset_gt_id"])
    horizons = {}
    for horizon in HORIZONS:
        metric = _metric_template()
        details = []
        previous_r_pid = previous_m_pid = None
        previous_r_error = previous_m_error = False
        for frame in range(event_frame + 1, event_frame + horizon + 1):
            gt = gt_frames.get(frame, {}).get(target_gid)
            if gt is None:
                continue
            r = rows[("R1_TRACK_CENTRIC_RECOVERY", frame)]
            m0 = rows[("R0_M0_NO_RECOVERY", frame)]
            box = gt["box"]
            r_iou = _public_iou(r, target_public, box)
            m_iou = _public_iou(m0, target_public, box)
            r_best_iou, r_best = _candidate_best(r, box)
            m_best_iou, m_best = _candidate_best(m0, box)
            r_correct, m_correct = r_iou >= IOU_THRESHOLD, m_iou >= IOU_THRESHOLD
            r_missing = not any(item.get("public_id") is not None and int(item["public_id"]) == target_public for item in r.get("candidate_rows", []))
            m_missing = not any(item.get("public_id") is not None and int(item["public_id"]) == target_public for item in m0.get("candidate_rows", []))
            r_obs = int(r_best["public_id"]) if r_best is not None and r_best_iou >= IOU_THRESHOLD and r_best.get("public_id") is not None else None
            m_obs = int(m_best["public_id"]) if m_best is not None and m_best_iou >= IOU_THRESHOLD and m_best.get("public_id") is not None else None
            r_switch = previous_r_pid is not None and r_obs is not None and r_obs != previous_r_pid
            m_switch = previous_m_pid is not None and m_obs is not None and m_obs != previous_m_pid
            if r_obs is not None:
                previous_r_pid = r_obs
            if m_obs is not None:
                previous_m_pid = m_obs
            r_recorrect = (not r_correct) and not previous_r_error
            m_recorrect = (not m_correct) and not previous_m_error
            previous_r_error, previous_m_error = not r_correct, not m_correct
            rmap, mmap = _assignment_map(r), _assignment_map(m0)
            changed = rmap != mmap
            record = metric_record(
                baseline_iou=m_iou,
                treatment_iou=r_iou,
                baseline_correct=m_correct,
                treatment_correct=r_correct,
                assignment_changed=changed,
            )
            proposal_rows = [item for item in r.get("candidate_rows", []) if item.get("is_recovery_proposal")]
            metric["evaluated_frames"] += 1
            metric["recovery_target_iou_sum"] += r_iou
            metric["m0_target_iou_sum"] += m_iou
            metric["recovery_correct_frames"] += int(r_correct)
            metric["m0_correct_frames"] += int(m_correct)
            metric["recovery_missing_frames"] += int(r_missing)
            metric["m0_missing_frames"] += int(m_missing)
            metric["recovery_candidate_present_frames"] += int(r_best_iou >= IOU_THRESHOLD)
            metric["m0_candidate_present_frames"] += int(m_best_iou >= IOU_THRESHOLD)
            metric["recovery_identity_error_frames"] += int(not r_correct)
            metric["m0_identity_error_frames"] += int(not m_correct)
            metric["recovery_id_switch_count"] += int(r_switch)
            metric["m0_id_switch_count"] += int(m_switch)
            metric["recovery_recorrect_count"] += int(r_recorrect)
            metric["m0_recorrect_count"] += int(m_recorrect)
            metric["assignment_change_count"] += int(changed)
            metric["true_correct_crossings"] += int(record["true_correct_crossing"])
            metric["true_incorrect_crossings"] += int(record["true_incorrect_crossing"])
            metric["directional_improvements"] += int(record["directional_improvement"])
            metric["directional_regressions"] += int(record["directional_regression"])
            metric["neutral_changes"] += int(record["assignment_change_type"] == "NEUTRAL_CHANGE")
            metric["recovery_proposal_count"] += len(proposal_rows)
            metric["recovery_proposal_accepted_count"] += int(any(item.get("public_id") == target_public for item in proposal_rows))
            metric["identity_error_reduction_sum"] += float(record["identity_error_reduction"])
            metric["delta_iou_sum"] += float(record["delta_iou"])
            details.append(
                {
                    "frame": frame,
                    "recovery_target_iou": float(r_iou),
                    "m0_target_iou": float(m_iou),
                    "recovery_candidate_best_iou": float(r_best_iou),
                    "m0_candidate_best_iou": float(m_best_iou),
                    "recovery_correct": r_correct,
                    "m0_correct": m_correct,
                    "recovery_missing": r_missing,
                    "m0_missing": m_missing,
                    "recovery_proposal_count": len(proposal_rows),
                    "recovery_proposal_accepted_to_target": bool(any(item.get("public_id") == target_public for item in proposal_rows)),
                    "assignment_changed": changed,
                    "assignment_change_type": record["assignment_change_type"],
                    "identity_error_reduction": record["identity_error_reduction"],
                    "delta_iou": record["delta_iou"],
                    "runtime_future_gt_used": False,
                    "posthoc_gt_used": True,
                }
            )
        horizons[str(horizon)] = {**_finalize(metric), "frame_details": details}
    return {
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "action_type": str(event["action_type"]),
        "event_frame": event_frame,
        "target_public_id": target_public,
        "target_dataset_gt_id": target_gid,
        "horizons": horizons,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_recovery_runtime_validation",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }


def _posthoc_score(events: list[dict[str, Any]], root: Path, validation: dict[str, Any]) -> dict[str, Any]:
    gt_by_sequence = {str(event["sequence"]): _load_gt(str(event["sequence"])) for event in events}
    event_results = []
    runtime_by_event: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for event in events:
        event_id = str(event["event_id"])
        event["_target_public_id"] = int(event.get("_target_public_id") or load_stage16(event_id)["persistent_identity"]["public_id"])
        keyed = {}
        for row in read_jsonl(root / f"{event_id}.jsonl"):
            keyed[(str(row["branch"]), int(row["frame"]))] = row
        runtime_by_event[event_id] = keyed
        event_results.append(_score_event(event=event, rows=keyed, gt_frames=gt_by_sequence[str(event["sequence"])]))
    aggregate = {branch: {} for branch in BRANCHES}
    for horizon in HORIZONS:
        selected = [item for item in event_results]
        for branch in BRANCHES:
            total = _metric_template()
            for event_result in selected:
                value = event_result["horizons"][str(horizon)]
                _add = {key: value[key] for key in _metric_template()}
                for key, item in _add.items():
                    total[key] += item
            # Both branches are represented as a pairwise record; expose a
            # common aggregate in the result and retain R1-vs-R0 deltas below.
            if branch == "R1_TRACK_CENTRIC_RECOVERY":
                aggregate[branch][str(horizon)] = _finalize(total)
            else:
                # Derive the R0 standalone fields from the pairwise frame
                # details so the branch remains auditable.
                base = _metric_template()
                for event_result in selected:
                    for detail in event_result["horizons"][str(horizon)]["frame_details"]:
                        base["evaluated_frames"] += 1
                        base["m0_target_iou_sum"] += detail["m0_target_iou"]
                        base["m0_correct_frames"] += int(detail["m0_correct"])
                        base["m0_missing_frames"] += int(detail["m0_missing"])
                        base["m0_candidate_present_frames"] += int(detail["m0_candidate_best_iou"] >= IOU_THRESHOLD)
                base["m0_identity_error_frames"] = base["evaluated_frames"] - base["m0_correct_frames"]
                aggregate[branch][str(horizon)] = _finalize(base)
    sequence_values = {
        str(event["sequence"]): [float(event_result["horizons"]["20"]["identity_error_reduction"])]
        for event, event_result in zip(events, event_results)
    }
    for horizon in HORIZONS:
        aggregate["R1_TRACK_CENTRIC_RECOVERY"][str(horizon)]["sequence_cluster_bootstrap_95ci"] = sequence_cluster_bootstrap(
            {
                str(event["sequence"]): [float(result["horizons"][str(horizon)]["identity_error_reduction"])]
                for event, result in zip(events, event_results)
            },
            seed=BOOTSTRAP_SEED,
            repetitions=BOOTSTRAP_REPETITIONS,
        )
    recovery_rows = [detail for result in event_results for detail in result["horizons"]["20"]["frame_details"]]
    return {
        "schema_version": "N72R4_STAGE13_RECOVERY_RESULTS_V1",
        "status": "PASS_STAGE13_RECOVERY_POSTHOC_SCORING",
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events}),
        "branches": list(BRANCHES),
        "horizons": list(HORIZONS),
        "event_results": event_results,
        "aggregate": aggregate,
        "recovery_proposal_summary": {
            "h20_proposal_rows": sum(1 for item in recovery_rows if item["recovery_proposal_count"] > 0),
            "h20_proposal_accepted_to_target": sum(1 for item in recovery_rows if item["recovery_proposal_accepted_to_target"]),
            "h20_official_best_iou_gain_available": sum(1 for item in recovery_rows if item["recovery_candidate_best_iou"] > item["m0_candidate_best_iou"] + 1.0e-12),
            "runtime_trigger": "target persistent public ID unassigned by exact official-candidate solve",
            "proposal_geometry": "persistent_state_last_box_plus_velocity_and_last_feature",
            "public_id_authority": "none_until_exact_solver_assignment",
        },
        "runtime_validation": validation,
        "runtime_future_gt_used": False,
        "gt_usage": "posthoc_only_after_runtime_validation",
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "scientific_result": "CANDIDATE_RECOVERY_AVAILABILITY_AND_IDENTITY_DIAGNOSTIC_NOT_PRODUCTION_GATE",
    }


def _add_metric(destination: dict[str, Any], source: dict[str, Any]) -> None:
    for key in _metric_template():
        destination[key] = destination.get(key, 0) + source.get(key, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="attempt1")
    parser.add_argument("--artifact-root", type=Path, default=OUT / "recovery" / "candidate_recovery_probe_attempt1")
    parser.add_argument("--manifest-path", type=Path, default=OUT / "recovery" / "candidate_recovery_manifest_attempt1.json")
    parser.add_argument("--results-path", type=Path, default=OUT / "recovery" / "candidate_recovery_results_attempt1.json")
    parser.add_argument("--status-path", type=Path, default=OUT / "stage_status" / "stage_13_status.json")
    args = parser.parse_args()
    artifact_root = args.artifact_root if args.artifact_root.is_absolute() else ROOT / args.artifact_root
    manifest_path = args.manifest_path if args.manifest_path.is_absolute() else ROOT / args.manifest_path
    results_path = args.results_path if args.results_path.is_absolute() else ROOT / args.results_path
    status_path = args.status_path if args.status_path.is_absolute() else ROOT / args.status_path
    started = now_utc()
    try:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise RuntimeError(f"recovery artifact root is not empty: {artifact_root}")
        if manifest_path.exists() or results_path.exists() or status_path.exists():
            raise RuntimeError("refusing to overwrite Stage13 output path")
        events = load_events()
        artifact_root.mkdir(parents=True, exist_ok=True)
        event_manifests = []
        for event in events:
            rows, summary = _run_event(event)
            atomic_jsonl(artifact_root / f"{event['event_id']}.jsonl", rows)
            event_manifests.append(summary)
        validation = _validate_runtime(events, artifact_root)
        manifest = {
            "schema_version": "N72R4_STAGE13_RECOVERY_MANIFEST_V1",
            "status": validation["status"],
            "stage": STAGE_NAME,
            "attempt": str(args.attempt),
            "events": event_manifests,
            "branch_definitions": {
                "R0_M0_NO_RECOVERY": "official corrected B1 stream, exact persistent-public+NONE association, no proposal",
                "R1_TRACK_CENTRIC_RECOVERY": "same stream plus at most one persistent-state proposal when target public is unassigned",
            },
            "recovery_rule": {
                "trigger": "target persistent public ID absent from exact official-candidate assignment",
                "box": "persistent target last_box extrapolated with persistent velocity",
                "feature": "persistent target last_feature",
                "proposal_count_per_frame": 1,
                "proposal_has_public_authority": False,
                "new_public_ids_created": False,
                "gt_used_for_trigger": False,
            },
            "pair_manifest": str(PAIR_MANIFEST),
            "pair_manifest_sha256": sha256_file(PAIR_MANIFEST),
            "official_corrected_root": str(M0_ROOT),
            "runtime_future_gt_used": False,
            "gt_loaded_in_worker": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
        }
        atomic_json(manifest_path, manifest)
        for event in events:
            event["_target_public_id"] = int(load_stage16(str(event["event_id"]))["persistent_identity"]["public_id"])
        result = _posthoc_score(events, artifact_root, validation)
        result["inputs"] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "pair_manifest_sha256": sha256_file(PAIR_MANIFEST),
            "official_corrected_root": str(M0_ROOT),
        }
        atomic_json(results_path, result)
        stage_status = {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": STAGE_NAME,
            "status": "PASS_STAGE13_CANDIDATE_RECOVERY_DIAGNOSTIC",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "attempt": str(args.attempt),
            "event_count": len(events),
            "independent_sequence_count": len({str(event["sequence"]) for event in events}),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "results": str(results_path),
            "results_sha256": sha256_file(results_path),
            "runtime_validation": validation,
            "recovery_rule": manifest["recovery_rule"],
            "recovery_summary": result["recovery_proposal_summary"],
            "decision": "RECOVERY_DIAGNOSTIC_COMPLETE",
            "runtime_future_gt_used": False,
            "gt_loaded_in_worker": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "production_authorized": False,
        }
        atomic_json(status_path, stage_status)
        print(json.dumps({"status": stage_status["status"], "events": len(events), "results": str(results_path)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure_root = OUT / "attempts" / "stage13"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = failure_root / f"{str(args.attempt)}_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_json(
            failure_path,
            {
                "schema_version": "N72R4_FAILURE_V1",
                "stage": STAGE_NAME,
                "attempt": str(args.attempt),
                "status": "FAIL",
                "started_at_utc": started,
                "finished_at_utc": now_utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "artifact_root": str(artifact_root),
                "runtime_future_gt_used": False,
            },
        )
        print(json.dumps({"status": "FAIL", "failure_artifact": str(failure_path), "error": str(exc)}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
