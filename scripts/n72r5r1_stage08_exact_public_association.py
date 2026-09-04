#!/usr/bin/env python3
"""N72R5R1 Stage 08: attach the frozen SAM3 streams to public identity.

This is a CPU-side continuation of the sealed N72R5 official run.  It never
changes or re-runs the 200 Stage07 workers.  A sequence-persistent runtime is
rebuilt from the frozen N36 prefix, then each Stage07 event/future stream is
solved through the explicit-NONE public-assignment wrapper.

The runner deliberately keeps three things separate:

* the official, session-local candidate stream;
* association state IDs used as solver columns;
* immutable public IDs allocated by the outer runtime.

The simulated oracle is a post-freeze current-frame side channel.  Its private
GT-to-public map is written only under ``simulation_private`` and is never
passed to scoring, state, candidate, or solver code.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.simulation.human_oracle import SimulatedHumanOracle  # noqa: E402
from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    BRANCHES,
    HORIZON,
    IOU_THRESHOLD,
    TVC_BRANCHES,
    apply_exact_frame,
    assignment_by_uid,
    atomic_json,
    atomic_jsonl,
    best_candidate_for_box,
    candidate_map,
    choose_tvc_competitors,
    current_gt_input,
    exact_solve,
    learned_tvc_residual,
    json_hash,
    load_gt,
    load_prefix_rows,
    load_stage07_event_rows,
    new_runtime,
    new_state_manager,
    now_utc,
    public_axis,
    read_json,
    robust_tvc_scale,
    runtime_invariants,
    score_existing,
    sha256_file,
    tvc_residual,
)

EVENT_MANIFEST = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
STAGE07_ROOT = ROOT / "outputs/N72R5/mechanism_rounds/round_07_official_full_loop_attempt5"
OUT = Path(os.environ.get("N72R5R1_RUN_ROOT", str(ROOT / "outputs/N72R5R1")))
PUBLIC_ROOT = OUT / "public_assignment"
PRESTATE_ROOT = OUT / "event_prestate"
PRIVATE_ROOT = OUT / "simulation_private"
STAGE08_STATUS = OUT / "stage_08_status.json"
SMOKE_STATUS = OUT / "stage_08_smoke_status.json"
RUNTIME_MANIFEST = OUT / "stage08_runtime_manifest.json"

TVC_MODE = str(os.environ.get("N72R5R1_TVC_MODE", "V0")).upper()
if TVC_MODE not in {"V0", "V1"}:
    raise ValueError(f"unsupported N72R5R1_TVC_MODE: {TVC_MODE}")
PERSISTENCE_MODE = str(os.environ.get("N72R5R1_PERSISTENCE_MODE", "OFF")).upper()
if PERSISTENCE_MODE not in {"OFF", "FREEZE_MACHINE_PROTOTYPE_AFTER_EVENT"}:
    raise ValueError(f"unsupported N72R5R1_PERSISTENCE_MODE: {PERSISTENCE_MODE}")
TVC_MODEL_PATH = Path(os.environ.get("N72R5R1_TVC_MODEL", ""))
TVC_MODEL = None
TVC_MODEL_SHA256 = None
if TVC_MODE == "V1":
    if not TVC_MODEL_PATH.is_file():
        raise FileNotFoundError(f"TVC_V1 model is missing: {TVC_MODEL_PATH}")
    TVC_MODEL = read_json(TVC_MODEL_PATH)
    TVC_MODEL_SHA256 = sha256_file(TVC_MODEL_PATH)

def _read_events() -> list[dict[str, Any]]:
    payload = read_json(EVENT_MANIFEST)
    if payload.get("status") != "PASS_N72R5_EVENT_POLICY_FROZEN":
        raise RuntimeError(f"Stage06 event policy is not frozen PASS: {payload.get('status')}")
    events = [dict(item) for item in payload.get("events", [])]
    if len(events) != 40 or len({str(item.get("event_id")) for item in events}) != 40:
        raise RuntimeError(f"Stage06 event set is not exactly 40 unique events: {len(events)}")
    for event in events:
        if str(event.get("interaction_source")) != "simulated_from_gt":
            raise RuntimeError(f"unexpected interaction source: {event.get('event_id')}")
        if event.get("runtime_future_gt_used") is not False or event.get("runtime_gt_read") is not False:
            raise RuntimeError(f"Stage06 event has a runtime GT flag: {event.get('event_id')}")
        frame = int(event["event_frame"])
        frame_count = int(event["sequence_frame_count"])
        if frame < 1 or frame + HORIZON >= frame_count:
            raise RuntimeError(f"event lacks prefix/H100 coverage: {event.get('event_id')}")
        tape = Path(str(event["candidate_tape_ref"]))
        if not tape.is_file():
            raise FileNotFoundError(f"frozen N36 tape is missing: {tape}")
    return sorted(events, key=lambda item: str(item["event_id"]))


def _event_by_id(events: Sequence[Mapping[str, Any]], event_id: str) -> dict[str, Any]:
    for event in events:
        if str(event["event_id"]) == str(event_id):
            return dict(event)
    raise KeyError(f"unknown frozen event: {event_id}")


def _content_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(candidate["candidate_index"]),
        int(candidate["official_raw_sam_id"]),
        int(candidate["adapter_external_id"]),
        tuple(round(float(value), 7) for value in candidate["box_xyxy"]),
        str(candidate["feature_sha256"]),
    )


def _semantic_assignment(applied: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in applied["candidate_decisions"]:
        result.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "official_raw_sam_id": int(row["official_raw_sam_id"]),
                "adapter_external_id": int(row["adapter_external_id"]),
                "box_xyxy": [float(value) for value in row["box_xyxy"]],
                "feature_sha256": str(row["feature_sha256"]),
                "public_id": None if row.get("public_id") is None else int(row["public_id"]),
                "assignment_status": str(row["assignment_status"]),
            }
        )
    return sorted(result, key=lambda item: (item["candidate_index"], item["official_raw_sam_id"]))


def _stream_content(rows: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[int, list[tuple[Any, ...]]]:
    return {int(frame): sorted((_content_key(candidate) for candidate in candidates)) for frame, candidates in rows.items()}


def _assert_frozen_branch_contract(
    event: Mapping[str, Any],
    rows_by_branch: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    raw_by_branch: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> str:
    event_id = str(event["event_id"])
    hashes = {
        str(raw_by_branch[branch][int(event["event_frame"])].get("y_pre_semantic_hash"))
        for branch in BRANCHES
    }
    if len(hashes) != 1 or "None" in hashes:
        raise RuntimeError(f"Y_pre semantic hash mismatch: {event_id}: {sorted(hashes)}")
    expected_pairs = (
        ("B1_SPATIAL_CORRECTION_ONLY", "B3_SPATIAL_CORRECTION_PLUS_TVC"),
        ("B2_SPATIAL_CORRECTION_PLUS_IMAGE_RECOVERY", "B4_SPATIAL_CORRECTION_PLUS_RECOVERY_PLUS_TVC"),
    )
    for left, right in expected_pairs:
        if _stream_content(rows_by_branch[left]) != _stream_content(rows_by_branch[right]):
            raise RuntimeError(f"frozen candidate stream mismatch for {event_id}: {left} != {right}")
    return next(iter(hashes))


def _predictions(applied: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_uid": str(row["candidate_uid"]),
            "public_id": int(row["public_id"]),
            "box": [float(value) for value in row["box_xyxy"]],
        }
        for row in applied["candidate_decisions"]
        if row.get("public_id") is not None
    ]


def _commit_prefix_oracle(
    oracle: SimulatedHumanOracle,
    frame: int,
    gt_frames: Mapping[int, Mapping[int, Mapping[str, Any]]],
    applied: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read only current GT after the current runtime result is frozen."""

    decisions = oracle.choose_actions(
        int(frame),
        current_gt_input(gt_frames, int(frame)),
        _predictions(applied),
        localization_iou_threshold=IOU_THRESHOLD,
    )
    committed: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.dataset_gt_id is None or decision.matched_runtime_public_id is None:
            continue
        gt_id = int(decision.dataset_gt_id)
        public = int(decision.matched_runtime_public_id)
        existing = oracle.gt_to_public.get(gt_id)
        if existing is None:
            try:
                oracle.commit_mapping(gt_id, public, reason="current_frame_runtime_confirmation")
                committed.append({"dataset_gt_id": gt_id, "public_id": public})
            except ValueError as exc:
                # A frozen prefix can contain a public-ID collision caused by
                # an earlier association error.  Keep the private oracle
                # immutable and continue the public-runtime replay; the
                # conflict is surfaced in the event diagnostics and makes the
                # affected posthoc event unavailable rather than aborting all
                # five branches.
                committed.append(
                    {
                        "dataset_gt_id": gt_id,
                        "public_id": public,
                        "conflict": True,
                        "reason": "PUBLIC_MAPPING_CONFLICT",
                        "error": str(exc),
                    }
                )
        elif int(existing) != public:
            # The human-facing oracle has already established a different
            # public identity.  This is a protocol-visible association error,
            # not an opportunity to rewrite the private map.
            committed.append(
                {"dataset_gt_id": gt_id, "public_id": public, "existing_public_id": int(existing), "conflict": True}
            )
    return committed


def _run_frame(
    runtime: Any,
    manager: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    frame: int,
    event_id: str,
    branch: str,
    session_id: str,
    event_frame: int,
    source_path: str,
    candidate_role: str | None = None,
    tvc: Mapping[str, Any] | None = None,
    memory_read: bool = False,
    memory_read_source: str | None = None,
    score_override: np.ndarray | None = None,
    freeze_public_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    runtime.clear_current_session_bindings(int(frame), reason="stage08_frame_binding_refresh")
    states, base, score_audit = score_existing(manager, candidates, int(frame))
    fused = np.asarray(base if score_override is None else score_override, dtype=np.float64)
    solver = exact_solve(
        runtime,
        states,
        candidates,
        fused,
        event_id=event_id,
        branch=branch,
        frame=int(frame),
        session_id=session_id,
    )
    applied = apply_exact_frame(
        runtime,
        manager,
        states,
        candidates,
        np.asarray(base, dtype=np.float64),
        fused,
        solver,
        frame=int(frame),
        event_id=event_id,
        branch=branch,
        session_id=session_id,
        event_frame=int(event_frame),
        source_path=source_path,
        candidate_role=candidate_role,
        tvc=tvc,
        birth_none=True,
        memory_read=memory_read,
        memory_read_source=memory_read_source,
        freeze_public_ids=freeze_public_ids,
        persistence_mode=PERSISTENCE_MODE if freeze_public_ids else None,
    )
    applied["score_audit"] = deepcopy(score_audit)
    return applied


def _create_outer_birth(
    runtime: Any,
    manager: Any,
    candidate: Mapping[str, Any],
    *,
    frame: int,
    session_id: str,
) -> Any:
    record = runtime.create_identity(
        int(frame),
        # The helper accepts the canonical observation object; import here to
        # keep the common module free of transaction policy.
        __import__("sam3_intermot.association.branch_public_replay", fromlist=["candidate_obs"]).candidate_obs(
            candidate, int(frame), source="simulated_human_correction"
        ),
        session_id=session_id,
        adapter_external_id=int(candidate["adapter_external_id"]),
        raw_sam_id=int(candidate["official_raw_sam_id"]),
        candidate_uid=str(candidate["candidate_uid"]),
        appearance_state={
            "last_machine_feature": list(candidate["feature"]),
            "last_machine_feature_sha256": str(candidate["feature_sha256"]),
            "last_machine_feature_frame": int(frame),
        },
        motion_state_ref={"last_box": list(candidate["box_xyxy"]), "last_frame": int(frame)},
    )
    manager.register_from_persistent_identity(
        record,
        {
            "feat": np.asarray(candidate["feature"], dtype=np.float32),
            "box": np.asarray(candidate["box_xyxy"], dtype=float),
            "native_tid": int(candidate["native_tid"]),
        },
        int(frame),
    )
    return record


def _forced_matrix(
    base: np.ndarray,
    states: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    runtime: Any,
    forced: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    matrix = np.asarray(base, dtype=np.float64).copy()
    state_by_public = {}
    for index, state in enumerate(states):
        record = runtime.get_identity_by_state_id(int(state.pid))
        if record is None:
            raise RuntimeError(f"state {state.pid} has no persistent public authority")
        state_by_public[int(record.public_id)] = index
    candidate_by_uid = {str(candidate["candidate_uid"]): index for index, candidate in enumerate(candidates)}
    if len(candidate_by_uid) != len(candidates):
        raise RuntimeError("duplicate candidate UID before authority override")
    for item in forced:
        uid = str(item["candidate_uid"])
        public = int(item["public_id"])
        if uid not in candidate_by_uid or public not in state_by_public:
            raise RuntimeError(f"forced authority axis is unresolved: {uid}/{public}")
        row = candidate_by_uid[uid]
        col = state_by_public[public]
        matrix[:, col] = -1.0e6
        matrix[row, :] = -1.0e6
        matrix[row, col] = 1.0e6
    return matrix


def _protected_transaction_matrix(
    base: np.ndarray,
    states: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    runtime: Any,
    forced: Sequence[Mapping[str, Any]],
    *,
    event_id: str,
    branch: str,
    frame: int,
    session_id: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply an authority transaction while preserving untouched assignments.

    The current correction is allowed to displace only the public IDs and
    candidate rows explicitly named by the interaction.  All other
    state/candidate pairs from the same pre-treatment exact solve are pinned
    before the final global solve.  This is a causal association-interface
    probe: it uses only the current candidate stream and the current-frame
    authority supplied by the simulated human, never future GT.
    """

    baseline = exact_solve(
        runtime,
        states,
        candidates,
        np.asarray(base, dtype=np.float64),
        event_id=event_id,
        branch=f"{branch}:PRE_TREATMENT",
        frame=int(frame),
        session_id=session_id,
    )
    assignments = assignment_by_uid(baseline)
    touched_public = {int(item["public_id"]) for item in forced}
    touched_uids = {str(item["candidate_uid"]) for item in forced}
    locks: list[dict[str, Any]] = []
    for uid, assignment in assignments.items():
        public = assignment.get("public_id")
        if public is None or int(public) in touched_public or str(uid) in touched_uids:
            continue
        locks.append(
            {
                "candidate_uid": str(uid),
                "public_id": int(public),
                "reason": "preserve_untouched_pre_treatment_assignment",
            }
        )
    combined = locks + [dict(item) for item in forced]
    return (
        _forced_matrix(np.asarray(base, dtype=np.float64), states, candidates, runtime, combined),
        locks,
    )


def _event_target_candidate(
    event: Mapping[str, Any],
    raw_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[float, Mapping[str, Any] | None, str | None]:
    correction = raw_row.get("correction") if isinstance(raw_row.get("correction"), Mapping) else {}
    box = correction.get("human_box") or event.get("current_gt_box")
    iou, candidate = best_candidate_for_box(candidates, box or [])
    if candidate is None or iou < IOU_THRESHOLD:
        return (
            float(iou),
            None,
            f"TARGET_CANDIDATE_UNAVAILABLE_IOU_{float(iou):.6f}_THRESHOLD_{IOU_THRESHOLD}",
        )
    return float(iou), candidate, None


def _event_other_candidate(
    event: Mapping[str, Any],
    gt_frames: Mapping[int, Mapping[int, Mapping[str, Any]]],
    candidates: Sequence[Mapping[str, Any]],
    excluded: Iterable[str],
) -> tuple[float, Mapping[str, Any] | None, str | None]:
    other_gt_id = event.get("other_dataset_gt_id")
    if other_gt_id is None:
        return 0.0, None, "ATOMIC_OTHER_IDENTITY_MISSING_FROM_EVENT_CONTRACT"
    item = gt_frames.get(int(event["event_frame"]), {}).get(int(other_gt_id))
    if item is None:
        return 0.0, None, "ATOMIC_OTHER_IDENTITY_NOT_VISIBLE_AT_EVENT"
    iou, candidate = best_candidate_for_box(candidates, item["box"], excluded_uids=excluded)
    if candidate is None or iou < IOU_THRESHOLD:
        return float(iou), None, f"ATOMIC_OTHER_CANDIDATE_UNAVAILABLE_IOU_{float(iou):.6f}_THRESHOLD_{IOU_THRESHOLD}"
    return float(iou), candidate, None


def _public_state_axis(runtime: Any, manager: Any) -> list[dict[str, Any]]:
    result = []
    for state in sorted(manager.states.values(), key=lambda item: int(item.pid)):
        record = runtime.get_identity_by_state_id(int(state.pid))
        if record is None:
            continue
        result.append(
            {
                "association_state_id": int(record.association_state_id),
                "public_id": int(record.public_id),
                "state": str(state.state),
                "last_seen_frame": int(state.last_seen_frame),
                "last_native_tid": int(state.last_native_tid),
            }
        )
    return result


def _write_prestate(
    event: Mapping[str, Any],
    runtime: Any,
    manager: Any,
    oracle: SimulatedHumanOracle,
    prefix_snapshot: Mapping[str, Any],
    y_pre: Mapping[str, Any],
) -> None:
    event_id = str(event["event_id"])
    root = PRESTATE_ROOT / event_id
    atomic_json(root / "persistent_runtime_snapshot.json", prefix_snapshot)
    mapping = oracle.gt_to_public
    atomic_json(
        root / "oracle_private_mapping_digest.json",
        {
            "schema_version": "N72R5R1_ORACLE_MAPPING_DIGEST_V1",
            "mapping_sha256": json_hash(mapping),
            "mapping_count": len(mapping),
            "oracle_audit": oracle.audit.as_dict(),
            "contains_mapping": False,
            "runtime_future_gt_used": False,
        },
    )
    axis = public_axis(runtime)
    atomic_json(root / "public_axis.json", {"event_id": event_id, "public_axis": axis, "runtime_future_gt_used": False})
    atomic_json(
        root / "state_axis.json",
        {"event_id": event_id, "state_axis": _public_state_axis(runtime, manager), "runtime_future_gt_used": False},
    )
    atomic_json(
        root / "event_y_pre_assignment.json",
        {
            "event_id": event_id,
            "sequence": event["sequence"],
            "event_frame": int(event["event_frame"]),
            "candidate_role": "PRE_INTERVENTION_Y_PRE",
            "y_pre_semantic_hash": str(y_pre["y_pre_semantic_hash"]),
            "assignment": _semantic_assignment(y_pre["applied"]),
            "public_axis": axis,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
        },
    )


def _run_event(event: Mapping[str, Any], gt_frames: Mapping[int, Mapping[int, Mapping[str, Any]]]) -> dict[str, Any]:
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = int(event["event_frame"])
    tape_path = Path(str(event["candidate_tape_ref"]))
    prefix = load_prefix_rows(tape_path, event_frame, sequence)
    rows_by_branch, raw_by_branch, source_paths = load_stage07_event_rows(
        STAGE07_ROOT, event_id, event_frame, sequence
    )
    y_pre_hash = _assert_frozen_branch_contract(event, rows_by_branch, raw_by_branch)

    runtime = new_runtime(sequence, event_id)
    manager = new_state_manager(runtime)
    oracle = SimulatedHumanOracle(sequence)
    runtime.begin_new_sam_session(f"{event_id}:prefix", boundary_frame=event_frame - 1)
    prefix_commits: list[dict[str, Any]] = []
    for frame in range(event_frame):
        applied = _run_frame(
            runtime,
            manager,
            prefix[frame],
            frame=frame,
            event_id=event_id,
            branch="PREFIX_N36_TAPE",
            session_id=f"{event_id}:prefix",
            event_frame=event_frame,
            source_path=str(tape_path),
        )
        prefix_commits.extend(_commit_prefix_oracle(oracle, frame, gt_frames, applied))
    prefix_snapshot = runtime.snapshot()
    # Every B0--B4 branch is a counterfactual from the same Y_PRE state.
    # Keep the prefix oracle map immutable while a branch commits its own
    # current-frame action.  Reusing the mutable oracle here would let an
    # earlier branch (notably ADD_NEW_IDENTITY) change the precondition seen
    # by later branches and invalidate the paired protocol.
    prefix_oracle_mapping = oracle.gt_to_public
    prefix_prototypes = {
        int(item["public_id"]): np.asarray(item.get("appearance_state", {}).get("last_machine_feature"), dtype=np.float32)
        for item in prefix_snapshot.get("identities", [])
        if item.get("public_id") is not None
        and item.get("appearance_state", {}).get("last_machine_feature") is not None
    }

    # Build one reference Y_pre from the common B0 event row.  Treatment
    # branches run this same transaction internally, but expose only Y_post at
    # the event-frame output row.
    ypre_runtime = new_runtime(sequence, event_id)
    ypre_runtime.restore(deepcopy(prefix_snapshot))
    ypre_manager = new_state_manager(ypre_runtime)
    ypre_runtime.begin_new_sam_session(f"{event_id}:Y_PRE_REFERENCE", boundary_frame=event_frame - 1)
    ypre_applied = _run_frame(
        ypre_runtime,
        ypre_manager,
        rows_by_branch["B0_NO_INTERVENTION"][event_frame],
        frame=event_frame,
        event_id=event_id,
        branch="Y_PRE_REFERENCE",
        session_id=f"{event_id}:Y_PRE_REFERENCE",
        event_frame=event_frame,
        source_path=source_paths["B0_NO_INTERVENTION"],
        candidate_role="PRE_INTERVENTION_Y_PRE",
    )
    ypre = {"applied": ypre_applied, "y_pre_semantic_hash": y_pre_hash}
    _write_prestate(event, runtime, manager, oracle, prefix_snapshot, ypre)

    branch_results: list[dict[str, Any]] = []
    branch_oracle_public_maps: list[dict[int, int]] = []
    for branch in BRANCHES:
        branch_oracle = SimulatedHumanOracle(
            sequence,
            known_gt_to_public=prefix_oracle_mapping,
        )
        branch_runtime = new_runtime(sequence, event_id)
        branch_runtime.restore(deepcopy(prefix_snapshot))
        branch_manager = new_state_manager(branch_runtime)
        session_id = f"{event_id}:{branch}"
        branch_runtime.begin_new_sam_session(session_id, boundary_frame=event_frame - 1)
        pre_event = _run_frame(
            branch_runtime,
            branch_manager,
            rows_by_branch["B0_NO_INTERVENTION"][event_frame],
            frame=event_frame,
            event_id=event_id,
            branch=branch,
            session_id=session_id,
            event_frame=event_frame,
            source_path=source_paths["B0_NO_INTERVENTION"],
            candidate_role="PRE_INTERVENTION_Y_PRE",
        )
        if _semantic_assignment(pre_event) != _semantic_assignment(ypre_applied):
            raise RuntimeError(f"shared PUBLIC_Y_PRE mismatch: {event_id}/{branch}")

        output_rows: list[dict[str, Any]] = []
        forced: list[dict[str, Any]] = []
        protected_locks: list[dict[str, Any]] = []
        target_public: int | None = None
        target_candidate_uid: str | None = None
        treatment_applied: dict[str, Any] | None = None
        action_diagnostic: dict[str, Any] = {
            "status": "NOT_APPLICABLE" if branch == "B0_NO_INTERVENTION" else "NOT_ATTEMPTED",
            "human_intervention_applied": False,
            "target_candidate_available": False,
            "target_candidate_iou": None,
            "other_candidate_available": None,
            "other_candidate_iou": None,
            "authority_override_count": 0,
            "runtime_future_gt_used": False,
        }
        event_candidates = rows_by_branch[branch][event_frame]
        if branch == "B0_NO_INTERVENTION":
            pre_event["frame_record"]["shared_y_pre_semantic_hash"] = y_pre_hash
            pre_event["frame_record"]["human_intervention_applied"] = False
            pre_event["frame_record"]["treatment_diagnostic"] = deepcopy(action_diagnostic)
            output_rows.append(pre_event["frame_record"])
        else:
            branch_runtime.clear_current_session_bindings(event_frame, reason="stage08_after_y_pre_before_treatment")
            action = str(event["action_type"])
            raw_event = raw_by_branch[branch][event_frame]
            target_iou, target_candidate, target_reason = _event_target_candidate(event, raw_event, event_candidates)
            action_diagnostic.update(
                {
                    "target_candidate_available": target_candidate is not None,
                    "target_candidate_iou": float(target_iou),
                    "target_candidate_uid": None if target_candidate is None else str(target_candidate["candidate_uid"]),
                }
            )
            if target_candidate is not None:
                target_candidate_uid = str(target_candidate["candidate_uid"])

            known = branch_oracle.gt_to_public.get(int(event["dataset_gt_id"]))
            if action == "ADD_NEW_IDENTITY" and known is not None:
                action_diagnostic.update(
                    {
                        "status": "ADD_TARGET_ALREADY_HAS_PREFIX_PUBLIC",
                        "reason": "frozen_prefix_oracle_already_contains_target",
                    }
                )
            elif action != "ADD_NEW_IDENTITY" and known is None:
                action_diagnostic.update(
                    {
                        "status": "TARGET_PUBLIC_MAPPING_UNAVAILABLE",
                        "reason": "frozen_prefix_oracle_has_no_target_public",
                    }
                )
            elif target_candidate is None:
                action_diagnostic.update(
                    {
                        "status": "TARGET_CANDIDATE_UNAVAILABLE",
                        "reason": target_reason,
                    }
                )
            elif action == "ATOMIC_ID_SWAP":
                other_gt = event.get("other_dataset_gt_id")
                other_known = None if other_gt is None else branch_oracle.gt_to_public.get(int(other_gt))
                other_iou, other_candidate, other_reason = _event_other_candidate(
                    event, gt_frames, event_candidates, excluded=(target_candidate_uid,)
                )
                action_diagnostic.update(
                    {
                        "other_candidate_available": other_candidate is not None,
                        "other_candidate_iou": float(other_iou),
                        "other_candidate_uid": None if other_candidate is None else str(other_candidate["candidate_uid"]),
                    }
                )
                if other_known is None:
                    action_diagnostic.update(
                        {
                            "status": "ATOMIC_OTHER_PUBLIC_MAPPING_UNAVAILABLE",
                            "reason": "frozen_prefix_oracle_has_no_other_public",
                        }
                    )
                elif other_candidate is None:
                    action_diagnostic.update(
                        {
                            "status": "ATOMIC_OTHER_CANDIDATE_UNAVAILABLE",
                            "reason": other_reason,
                        }
                    )
                elif int(other_known) == int(known) or str(other_candidate["candidate_uid"]) == target_candidate_uid:
                    action_diagnostic.update(
                        {
                            "status": "ATOMIC_PUBLIC_AXIS_NOT_DISTINCT",
                            "reason": "target_and_other_authority_axis_collides",
                        }
                    )
                else:
                    target_public = int(known)
                    forced = [
                        {
                            "candidate_uid": target_candidate_uid,
                            "public_id": target_public,
                            "reason": "current_frame_authoritative_target",
                            "target_box_iou": float(target_iou),
                        },
                        {
                            "candidate_uid": str(other_candidate["candidate_uid"]),
                            "public_id": int(other_known),
                            "reason": "atomic_other_authoritative_target",
                            "target_box_iou": float(other_iou),
                        },
                    ]
                    action_diagnostic.update({"status": "APPLIED", "authority_override_count": 2})
            else:
                if action == "ADD_NEW_IDENTITY":
                    # ADD is intentionally a new allocator decision.  The
                    # branch starts from the same prefix snapshot as every
                    # other treatment branch, so this public ID is stable
                    # within the event while remaining runtime allocated.
                    new_record = _create_outer_birth(
                        branch_runtime,
                        branch_manager,
                        target_candidate,
                        frame=event_frame,
                        session_id=session_id,
                    )
                    target_public = int(new_record.public_id)
                    try:
                        branch_oracle.commit_mapping(
                            int(event["dataset_gt_id"]),
                            target_public,
                            reason="outer_allocator_birth",
                        )
                    except ValueError as exc:
                        action_diagnostic.update(
                            {
                                "status": "ORACLE_PUBLIC_MAPPING_CONFLICT",
                                "reason": str(exc),
                            }
                        )
                        target_public = None
                    else:
                        action_diagnostic.update({"status": "APPLIED", "authority_override_count": 1})
                else:
                    target_public = int(known)
                    action_diagnostic.update({"status": "APPLIED", "authority_override_count": 1})
                if action_diagnostic["status"] == "APPLIED":
                    forced = [
                        {
                            "candidate_uid": target_candidate_uid,
                            "public_id": target_public,
                            "reason": "current_frame_authoritative_target",
                            "target_box_iou": float(target_iou),
                        }
                    ]

            if forced:
                states, base, _ = score_existing(branch_manager, event_candidates, event_frame)
                # ADD creates the state before the score audit; re-read the
                # state axis so the allocator-owned public identity is
                # explicit in the exact solver input.
                states = branch_manager.candidates(event_frame)
                if np.asarray(base).shape != (len(event_candidates), len(states)):
                    _, base, _ = score_existing(branch_manager, event_candidates, event_frame, states=states)
                fused, protected_locks = _protected_transaction_matrix(
                    np.asarray(base, dtype=np.float64),
                    states,
                    event_candidates,
                    branch_runtime,
                    forced,
                    event_id=event_id,
                    branch=branch,
                    frame=event_frame,
                    session_id=session_id,
                )
            else:
                fused = None
            treatment_applied = _run_frame(
                branch_runtime,
                branch_manager,
                event_candidates,
                frame=event_frame,
                event_id=event_id,
                branch=branch,
                session_id=session_id,
                event_frame=event_frame,
                source_path=source_paths[branch],
                candidate_role="POST_INTERVENTION_Y_POST",
                score_override=fused,
            )
            treatment_applied["frame_record"]["shared_y_pre_semantic_hash"] = y_pre_hash
            action_diagnostic["human_intervention_applied"] = bool(forced)
            treatment_applied["frame_record"]["human_intervention_applied"] = bool(forced)
            treatment_applied["frame_record"]["authority_overrides"] = deepcopy(forced)
            treatment_applied["frame_record"]["protected_pre_treatment_locks"] = deepcopy(protected_locks)
            action_diagnostic["protected_lock_count"] = len(protected_locks)
            treatment_applied["frame_record"]["target_box_iou"] = float(target_iou)
            treatment_applied["frame_record"]["treatment_action"] = action
            treatment_applied["frame_record"]["memory_write"] = False
            treatment_applied["frame_record"]["memory_write_public_ids"] = []
            treatment_applied["frame_record"]["treatment_diagnostic"] = deepcopy(action_diagnostic)
            output_rows.append(treatment_applied["frame_record"])

        # Future association starts at event+1.  Only B3/B4 read the frozen
        # TVC anchor; no future GT or future public map is available here.
        event_anchor = (
            None
            if not action_diagnostic["human_intervention_applied"] or target_candidate_uid is None
            else candidate_map(event_candidates)[target_candidate_uid]
        )
        for frame in range(event_frame + 1, event_frame + HORIZON + 1):
            candidates = rows_by_branch[branch][frame]
            tvc_meta: dict[str, Any] | None = None
            override = None
            use_memory = False
            freeze_public_ids = (
                {int(target_public)}
                if (
                    PERSISTENCE_MODE == "FREEZE_MACHINE_PROTOTYPE_AFTER_EVENT"
                    and branch != "B0_NO_INTERVENTION"
                    and action_diagnostic["human_intervention_applied"]
                    and target_public is not None
                )
                else None
            )
            if branch in TVC_BRANCHES and target_public is not None and event_anchor is not None:
                # The target row can legitimately leave the state axis after
                # the configured lost gap.  Record that as an unapplied TVC
                # observation rather than manufacturing a state.
                states_now = branch_manager.candidates(frame)
                target_on_axis = any(
                    (branch_runtime.get_identity_by_state_id(int(state.pid)) is not None
                     and int(branch_runtime.get_identity_by_state_id(int(state.pid)).public_id) == int(target_public))
                    for state in states_now
                )
                if target_on_axis and candidates:
                    _, base_now, _ = score_existing(branch_manager, candidates, frame, states=states_now)
                    if TVC_MODE == "V1":
                        residual, residual_details = learned_tvc_residual(
                            states_now,
                            branch_runtime,
                            candidates,
                            target_public=int(target_public),
                            human_anchor=np.asarray(event_anchor["feature"], dtype=np.float32),
                            persistent_target=prefix_prototypes.get(int(target_public)),
                            model=TVC_MODEL,
                        )
                        tvc_meta = {
                            "name": "TVC_V1_LEARNED_CANDIDATE_VERIFIER",
                            "enabled": True,
                            "applied": True,
                            "model_sha256": TVC_MODEL_SHA256,
                            "model_path": str(TVC_MODEL_PATH),
                            "max_abs_residual": float(TVC_MODEL.get("max_abs_residual", 8.0)),
                            "target_public_id": int(target_public),
                            "target_row_only": True,
                            "residual_details": residual_details,
                            "runtime_future_gt_used": False,
                        }
                    else:
                        competitors = choose_tvc_competitors(states_now, branch_runtime, np.asarray(base_now).T, target_public)
                        target_record = branch_runtime.get_identity_by_public_id(int(target_public))
                        persistent = np.asarray(
                            (target_record.appearance_state.get("last_machine_feature") if target_record else None)
                            or next(state.prototype for state in states_now if int(branch_runtime.get_identity_by_state_id(int(state.pid)).public_id) == int(target_public)),
                            dtype=np.float32,
                        )
                        scale_info = robust_tvc_scale(np.asarray(base_now).T)
                        residual, residual_details = tvc_residual(
                            states_now,
                            branch_runtime,
                            candidates,
                            target_public=int(target_public),
                            competitor_publics=competitors["selected_public_ids"],
                            human_anchor=np.asarray(event_anchor["feature"], dtype=np.float32),
                            persistent_target=persistent,
                            scale=float(scale_info["scale"]),
                        )
                        tvc_meta = {
                            "name": "TVC_V0_TARGET_VS_COMPETITOR",
                            "enabled": True,
                            "applied": True,
                            "trust_radius": 1.0,
                            "top_k": 3,
                            "scale": scale_info,
                            "competitors": competitors,
                            "target_public_id": int(target_public),
                            "target_row_only": True,
                            "residual_details": residual_details,
                            "runtime_future_gt_used": False,
                        }
                    override = np.asarray(base_now, dtype=np.float64) + np.asarray(residual, dtype=np.float64).T
                    use_memory = True
                else:
                    tvc_meta = {
                        "name": "TVC_V1_LEARNED_CANDIDATE_VERIFIER" if TVC_MODE == "V1" else "TVC_V0_TARGET_VS_COMPETITOR",
                        "enabled": True,
                        "applied": False,
                        "reason": "NO_CANDIDATES_OR_TARGET_PUBLIC_NOT_ON_STATE_AXIS",
                        "target_public_id": int(target_public),
                        "runtime_future_gt_used": False,
                    }
            applied = _run_frame(
                branch_runtime,
                branch_manager,
                candidates,
                frame=frame,
                event_id=event_id,
                branch=branch,
                session_id=session_id,
                event_frame=event_frame,
                source_path=source_paths[branch],
                tvc=tvc_meta,
                memory_read=use_memory,
                memory_read_source=f"TVC_{TVC_MODE}" if use_memory else None,
                score_override=override,
                freeze_public_ids=freeze_public_ids,
            )
            output_rows.append(applied["frame_record"])

        if len(output_rows) != HORIZON + 1:
            raise RuntimeError(f"Stage08 output frame count mismatch: {event_id}/{branch}/{len(output_rows)}")
        output_path = PUBLIC_ROOT / event_id / f"{branch}.jsonl"
        atomic_jsonl(output_path, output_rows)
        done = {
            "schema_version": "N72R5R1_PUBLIC_ASSIGNMENT_DONE_V1",
            "event_id": event_id,
            "sequence": sequence,
            "branch": branch,
            "status": "PASS_N72R5R1_PUBLIC_ASSOCIATION_BRANCH",
            "frame_count": len(output_rows),
            "event_frame": event_frame,
            "event_candidate_role": str(output_rows[0]["candidate_role"]),
            "y_pre_semantic_hash": y_pre_hash,
            "shared_y_pre_assignment_sha256": json_hash(_semantic_assignment(ypre_applied)),
            "output_sha256": sha256_file(output_path),
            "target_public_id": target_public,
            "target_candidate_uid": target_candidate_uid,
            "public_axis": public_axis(branch_runtime),
            "runtime_invariants": runtime_invariants(branch_runtime),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "action_precondition_status": str(action_diagnostic["status"]),
            "human_intervention_applied": bool(action_diagnostic["human_intervention_applied"]),
            "treatment_diagnostic": deepcopy(action_diagnostic),
            "protected_pre_treatment_lock_count": len(protected_locks),
        }
        done_path = PUBLIC_ROOT / event_id / f"{branch}.done.json"
        done["output"] = str(output_path)
        done["done"] = str(done_path)
        # The path fields are written in the done manifest itself so later
        # validators never have to infer a branch artifact location.
        atomic_json(done_path, done)
        branch_results.append(done)
        branch_oracle_public_maps.append(branch_oracle.gt_to_public)

    # Build one deterministic posthoc map from the immutable prefix map plus
    # the branch-local current-event map.  Branches must agree on every public
    # mapping used for pairing, but their mutable oracle histories must never
    # leak into one another.
    posthoc_mapping = dict(prefix_oracle_mapping)
    mapping_conflicts: list[dict[str, Any]] = []
    for branch_map, branch_result in zip(branch_oracle_public_maps, branch_results):
        for gt_id, public_id in branch_map.items():
            previous = posthoc_mapping.get(int(gt_id))
            if previous is None:
                posthoc_mapping[int(gt_id)] = int(public_id)
            elif int(previous) != int(public_id):
                mapping_conflicts.append(
                    {
                        "branch": str(branch_result["branch"]),
                        "dataset_gt_id": int(gt_id),
                        "prefix_or_posthoc_public_id": int(previous),
                        "branch_public_id": int(public_id),
                    }
                )
    if mapping_conflicts:
        raise RuntimeError(f"branch-local public mapping conflict: {event_id}: {mapping_conflicts}")

    # The full oracle map is intentionally kept away from all runtime/frame
    # rows.  It is written only after all current-event allocator decisions,
    # including ADD_NEW_IDENTITY, have completed; Stage10 uses this file as a
    # posthoc simulation side channel.
    atomic_json(
        PRIVATE_ROOT / event_id / "oracle_private_mapping.json",
        {
            "schema_version": "N72R5R1_PRIVATE_SIMULATION_MAP_V1",
            "event_id": event_id,
            "sequence": sequence,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "dataset_gt_to_public": dict(sorted(posthoc_mapping.items())),
            "prefix_dataset_gt_to_public": dict(sorted(prefix_oracle_mapping.items())),
            "branch_local_dataset_gt_to_public": [
                {
                    "branch": str(branch_result["branch"]),
                    "mapping": dict(sorted(branch_map.items())),
                }
                for branch_map, branch_result in zip(branch_oracle_public_maps, branch_results)
            ],
            "branch_mapping_conflicts": mapping_conflicts,
            "runtime_future_gt_used": False,
            "posthoc_only": True,
        },
    )

    return {
        "event_id": event_id,
        "sequence": sequence,
        "event_frame": event_frame,
        "action_type": str(event["action_type"]),
        "branch_count": len(branch_results),
        "branches": branch_results,
        "prefix_frame_count": len(prefix),
        "prefix_oracle_commit_count": len(prefix_commits),
        "prefix_oracle_conflict_count": sum(1 for item in prefix_commits if item.get("conflict")),
        "action_precondition_statuses": sorted({str(item.get("action_precondition_status")) for item in branch_results}),
        "action_transaction_complete": all(
            str(item.get("action_precondition_status")) in {"NOT_APPLICABLE", "APPLIED"}
            for item in branch_results
        ),
        "oracle_mapping_digest": json_hash(posthoc_mapping),
        "oracle_mapping_count": len(posthoc_mapping),
        "branch_oracle_isolated": True,
        "branch_mapping_conflict_count": len(mapping_conflicts),
        "y_pre_semantic_hash": y_pre_hash,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "status": "PASS_N72R5R1_PUBLIC_ASSOCIATION_EVENT",
    }


def _failure_path(event_id: str) -> Path:
    root = OUT / "attempts"
    candidate = root / f"{event_id}.failure.json"
    if not candidate.exists():
        return candidate
    for index in range(2, 10):
        candidate = root / f"{event_id}.failure.attempt{index}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many preserved failure artifacts for {event_id}")


def _write_failure(event: Mapping[str, Any], exc: BaseException) -> str:
    path = _failure_path(str(event["event_id"]))
    atomic_json(
        path,
        {
            "schema_version": "N72R5R1_STAGE08_FAILURE_V1",
            "event_id": str(event["event_id"]),
            "sequence": str(event["sequence"]),
            "status": "FAIL_STAGE08_EVENT",
            "failure_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "created_at_utc": now_utc(),
            "runtime_future_gt_used": False,
            "original_event_manifest_sha256": sha256_file(EVENT_MANIFEST),
        },
    )
    return str(path)


def _status(
    *,
    selected: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    scope: str,
) -> dict[str, Any]:
    full = len(selected) == 40 and not failures and len(results) == 40
    action_incomplete = [
        str(item["event_id"])
        for item in results
        if item.get("action_transaction_complete") is not True
    ]
    return {
        "schema_version": "N72R5R1_STAGE_STATUS_V1",
        "stage": "08_EXACT_PERSISTENT_PUBLIC_ASSOCIATION",
        "status": "PASS_N72R5R1_EXACT_PUBLIC_ASSOCIATION" if full else ("PARTIAL_N72R5R1_EXACT_PUBLIC_ASSOCIATION" if results else "BLOCKED_N72R5R1_EXACT_PUBLIC_ASSOCIATION"),
        "scope": scope,
        "selected_event_count": len(selected),
        "completed_event_count": len(results),
        "failure_count": len(failures),
        "expected_event_count": 40,
        "expected_branch_count": 200 if full else len(results) * len(BRANCHES),
        "public_assignment_complete": bool(full),
        "action_transaction_complete_event_count": int(len(results) - len(action_incomplete)),
        "action_diagnostic_event_count": int(len(action_incomplete)),
        "action_incomplete_events": action_incomplete,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "stage06_event_manifest": str(EVENT_MANIFEST),
        "stage06_event_manifest_sha256": sha256_file(EVENT_MANIFEST),
        "stage07_root": str(STAGE07_ROOT),
        "stage07_manifest_sha256": sha256_file(STAGE07_ROOT / "official_full_loop_manifest.json"),
        "failures": [dict(item) for item in failures],
        "minimal_next_action": None if full else "repair preserved Stage08 event failures and rerun only unfinished event units",
        "created_at_utc": now_utc(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    events = _read_events()
    if args.event_id:
        selected = [_event_by_id(events, value) for value in args.event_id]
    elif args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        selected = events[: int(args.limit)]
    else:
        selected = events
    scope = "SMOKE" if args.smoke or len(selected) < len(events) else "FULL_40_EVENT_SET"
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event in selected:
        try:
            results.append(_run_event(event, load_gt(str(event["sequence"]))))
        except Exception as exc:
            failure_path = _write_failure(event, exc)
            failures.append(
                {
                    "event_id": str(event["event_id"]),
                    "sequence": str(event["sequence"]),
                    "failure_path": failure_path,
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    manifest = {
        "schema_version": "N72R5R1_STAGE08_RUNTIME_MANIFEST_V1",
        "status": "PASS_N72R5R1_EXACT_PUBLIC_ASSOCIATION" if len(selected) == 40 and not failures else "PARTIAL_N72R5R1_EXACT_PUBLIC_ASSOCIATION",
        "scope": scope,
        "event_count_expected": 40,
        "event_count_selected": len(selected),
        "event_count_completed": len(results),
        "branch_count_completed": sum(int(item["branch_count"]) for item in results),
        "public_assignment_complete": bool(len(selected) == 40 and not failures and len(results) == 40),
        "action_transaction_complete_event_count": int(
            sum(1 for item in results if item.get("action_transaction_complete") is True)
        ),
        "action_diagnostic_event_count": int(
            sum(1 for item in results if item.get("action_transaction_complete") is not True)
        ),
        "action_incomplete_events": [
            str(item["event_id"])
            for item in results
            if item.get("action_transaction_complete") is not True
        ],
        "events": results,
        "failures": failures,
        "branches": list(BRANCHES),
        "horizon": HORIZON,
        "assignment_solver": "solve_effect_assignment",
        "outer_birth_policy": "explicit_none_then_sequence_runtime_allocator",
        "tvc": (
            {
                "name": "TVC_V1_LEARNED_CANDIDATE_VERIFIER",
                "model_sha256": TVC_MODEL_SHA256,
                "model_path": str(TVC_MODEL_PATH),
                "max_abs_residual": float(TVC_MODEL.get("max_abs_residual", 8.0)),
                "target_row_only": True,
            }
            if TVC_MODE == "V1"
            else {
                "name": "TVC_V0_TARGET_VS_COMPETITOR",
                "trust_radius": 1.0,
                "top_k": 3,
                "mad_normalization": True,
                "target_row_only": True,
            }
        ),
        "tvc_mode": TVC_MODE,
        "persistence_mode": PERSISTENCE_MODE,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "stage06_event_manifest_sha256": sha256_file(EVENT_MANIFEST),
        "stage07_manifest_sha256": sha256_file(STAGE07_ROOT / "official_full_loop_manifest.json"),
        "created_at_utc": now_utc(),
    }
    atomic_json(RUNTIME_MANIFEST, manifest)
    status = _status(selected=selected, results=results, failures=failures, scope=scope)
    atomic_json(SMOKE_STATUS if scope == "SMOKE" else STAGE08_STATUS, status)
    print(json.dumps({"status": status["status"], "completed": len(results), "failures": len(failures), "scope": scope}, ensure_ascii=False))
    return 0 if not failures and (scope == "FULL_40_EVENT_SET" or scope == "SMOKE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
