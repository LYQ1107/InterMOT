"""Causal paired rollouts for N72R13 Temporal Intervention Value.

This module deliberately contains no dataset-GT or post-hoc metric code.  It
uses the already frozen N72R11/N72R12 candidate adapter, scorer and exact
public-ID solver, and adds only the counterfactual state fork needed to measure
the future consequence of one accepted PCTIS assignment.

The contract is intentionally narrow:

* a branch commits exactly one assignment at ``frame``;
* all later frames use the BASE score matrix and exact solver;
* later frames never inject another proposal and never run live re-query;
* each branch owns an independent ``TemporalIdentityState``;
* runtime artifacts carry explicit false future-GT flags.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from sam3_intermot.association.target_edge_interface import EDGE_MODE_BASE, EDGE_MODE_LEGACY_INJECTION
from sam3_intermot.reacquisition.temporal_state_policy import (
    TemporalIdentityState,
    state_audit,
    update_temporal_state,
)


PRIMARY_VALUE_HORIZON = 20
VALUE_HORIZONS = (5, 20, 50)


def clone_temporal_state(state: TemporalIdentityState) -> TemporalIdentityState:
    """Deep-copy a temporal state, including every feature array.

    ``deepcopy`` is not used as the only operation here because the explicit
    copies make the no-shared-memory contract auditable and remain stable if
    the dataclass gains a non-array helper field later.
    """

    return TemporalIdentityState(
        predicted_box=[float(value) for value in state.predicted_box],
        previous_raw_sam_id=None if state.previous_raw_sam_id is None else int(state.previous_raw_sam_id),
        previous_native_scope=None if state.previous_native_scope is None else str(state.previous_native_scope),
        previous_score=float(state.previous_score),
        previous_uncertainty=float(state.previous_uncertainty),
        trusted_age=int(state.trusted_age),
        recent_trusted=[np.asarray(value, dtype=np.float32).copy() for value in state.recent_trusted],
        long_term_trusted=[np.asarray(value, dtype=np.float32).copy() for value in state.long_term_trusted],
        distractors=[np.asarray(value, dtype=np.float32).copy() for value in state.distractors],
    )


def temporal_state_digest(state: TemporalIdentityState) -> str:
    """Hash all causal state values, not just the compact audit summary."""

    payload = {
        "audit": state_audit(state),
        "recent_trusted": [np.asarray(value, dtype="<f4").tolist() for value in state.recent_trusted],
        "long_term_trusted": [np.asarray(value, dtype="<f4").tolist() for value in state.long_term_trusted],
        "distractors": [np.asarray(value, dtype="<f4").tolist() for value in state.distractors],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _replay_module():
    """Import the frozen replay helpers lazily to keep this module lightweight."""

    from scripts import n72r11_on_demand_replay as replay

    return replay


def _candidate_rows_for_frame(inputs: Mapping[str, Any], frame: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay = _replay_module()
    event_id = str(inputs["event_id"])
    sequence = str(inputs["sequence"])
    c0_row = inputs["rows"]["c0_source"][int(frame)]
    target_row = inputs["rows"]["target_stream_source"][int(frame)]
    main_candidates = list(c0_row.get("candidate_rows", []))
    current_candidates = [replay._strip_target_authority(item) for item in target_row.get("candidate_rows", [])]
    if len(current_candidates) > 1:
        raise RuntimeError(f"target stream is not singleton at {event_id}:{frame}")
    pool, audit = replay.build_candidate_pool(
        main_candidates,
        current_candidates,
        sequence=sequence,
        frame=int(frame),
        include_target_session=True,
        require_positive_geometry=bool(inputs.get("require_positive_geometry", False)),
    )
    if not pool:
        raise RuntimeError(f"empty frozen BASE candidate pool at {event_id}:{frame}")
    return pool, audit


def build_base_frame(
    *,
    inputs: Mapping[str, Any],
    frame: int,
    state: TemporalIdentityState,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Build one causal BASE frame without reading GT or applying PCTIS."""

    replay = _replay_module()
    pool, pool_audit = _candidate_rows_for_frame(inputs, int(frame))
    scored = replay._score_pool(
        inputs,
        int(frame),
        pool,
        None,
        None,
        torch.device(device),
        state,
        EDGE_MODE_BASE,
    )
    return {
        "pool": pool,
        "pool_audit": pool_audit,
        "scored": scored,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def build_pctis_proposal(
    *,
    inputs: Mapping[str, Any],
    frame: int,
    state: TemporalIdentityState,
    pctis_model: Any,
    device: torch.device | str,
) -> dict[str, Any]:
    """Build the PCTIS proposal on one frozen pool and audit its opportunity."""

    replay = _replay_module()
    base = build_base_frame(inputs=inputs, frame=int(frame), state=state, device="cpu")
    proposal = replay._score_pool(
        inputs,
        int(frame),
        base["pool"],
        pctis_model,
        None,
        torch.device(device),
        state,
        EDGE_MODE_LEGACY_INJECTION,
    )
    if proposal["state_axis"] != base["scored"]["state_axis"] or proposal["public_axis"] != base["scored"]["public_axis"]:
        raise RuntimeError(f"PCTIS proposal changed authority axes at {inputs['event_id']}:{frame}")
    if not np.array_equal(np.asarray(proposal["base_matrix"]), np.asarray(base["scored"]["base_matrix"])):
        raise RuntimeError(f"PCTIS proposal did not reuse BASE matrix at {inputs['event_id']}:{frame}")
    selected_uid = proposal["selection"].get("selected_candidate_uid")
    base_target_uid = base["scored"].get("base_target_uid")
    proposal_target_uid = proposal.get("target_uid")
    accepted = selected_uid is not None
    changed = str(proposal_target_uid) != str(base_target_uid)
    effective = bool(accepted and changed)
    return {
        "pool": base["pool"],
        "pool_audit": base["pool_audit"],
        "base": base["scored"],
        "proposal": proposal,
        "selection_accepted": accepted,
        "assignment_changed": changed,
        "effective_opportunity": effective,
        "no_effect_reason": None if effective else "NO_EFFECTIVE_INTERVENTION_OPPORTUNITY",
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _base_target_margin(scored: Mapping[str, Any], target_public: int) -> tuple[int, float, float, float]:
    public_axis = [int(value) for value in scored["public_axis"]]
    if int(target_public) not in public_axis:
        raise RuntimeError(f"target public ID missing from score axis: {target_public}")
    target_column = public_axis.index(int(target_public))
    values = np.asarray(scored["base_matrix"], dtype=np.float64)[:, target_column]
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), str(scored["pool"][index]["candidate_uid"]) if "pool" in scored else index))
    top = float(values[order[0]]) if order else 0.0
    second = float(values[order[1]]) if len(order) > 1 else 0.0
    return target_column, top, second, float(top - second)


def _commit_frame(
    *,
    inputs: Mapping[str, Any],
    frame: int,
    state: TemporalIdentityState,
    pool: Sequence[Mapping[str, Any]],
    pool_audit: Mapping[str, Any],
    scored: Mapping[str, Any],
    solver: Mapping[str, Any],
    matrix: np.ndarray,
    target_uid: str | None,
    selected_uid: str | None,
    selection: Mapping[str, Any] | None,
    branch: str,
    base_scored: Mapping[str, Any],
) -> dict[str, Any]:
    replay = _replay_module()
    target_public = int(inputs["target_public_id"])
    public_axis = [int(value) for value in scored["public_axis"]]
    target_column = public_axis.index(target_public)
    base_matrix = np.asarray(base_scored["base_matrix"], dtype=np.float64)
    base_values = base_matrix[:, target_column]
    base_order = sorted(range(len(base_values)), key=lambda index: (-float(base_values[index]), str(pool[index]["candidate_uid"])))
    base_top = float(base_values[base_order[0]]) if base_order else 0.0
    base_second = float(base_values[base_order[1]]) if len(base_order) > 1 else 0.0
    selected_score = None if selection is None else selection.get("selected_score")
    selected_margin = None if selection is None else selection.get("best_minus_second_margin")
    assigned = next((item for item in pool if str(item["candidate_uid"]) == str(target_uid)), None)
    before_digest = temporal_state_digest(state)
    state_update = update_temporal_state(
        state,
        candidates=pool,
        target_uid=None if target_uid is None else str(target_uid),
        selected_uid=None if selected_uid is None else str(selected_uid),
        selected_score=selected_score,
        selected_margin=selected_margin,
        fused_target_scores=np.asarray(matrix, dtype=np.float64)[:, target_column],
        frame_horizon=int(frame) - int(inputs["event_frame"]),
        assigned_candidate=assigned,
        base_top_score=base_top,
        base_second_score=base_second,
    )
    source_rows = [replay.serializable_candidate(candidate, include_feature=False) for candidate in pool]
    for source_row in source_rows:
        source_row["public_id"] = None
        source_row["public_id_authority"] = None
    committed_rows = replay._solver_rows(pool, solver)
    row = {
        "schema_version": "N72R13_TIV_RUNTIME_FRAME_V1",
        "record_kind": "tiv_current_commit_frame" if branch in {"KEEP", "APPLY"} else "tiv_passive_future_frame",
        "branch": str(branch),
        "event_id": str(inputs["event_id"]),
        "sequence": str(inputs["sequence"]),
        "action_type": str(inputs.get("action_type", "UNKNOWN")),
        "event_frame": int(inputs["event_frame"]),
        "frame": int(frame),
        "frame_horizon": int(frame) - int(inputs["event_frame"]),
        "target_public_id": target_public,
        "candidate_rows": committed_rows,
        "candidate_count": len(committed_rows),
        "candidate_pool": {**dict(pool_audit), "candidate_rows": source_rows},
        "assignment": {
            "target_public_id": target_public,
            "target_assigned_candidate_uid": None if target_uid is None else str(target_uid),
            "target_base_assigned_candidate_uid": None if base_scored.get("base_target_uid") is None else str(base_scored.get("base_target_uid")),
            "target_selected_candidate_uid": None if selected_uid is None else str(selected_uid),
            "target_selector_and_solver_agree": bool(selected_uid is not None and str(selected_uid) == str(target_uid)),
            "solver": deepcopy(dict(solver)),
            "solver_public_id_immutable": True,
            "runtime_future_gt_used": False,
        },
        "score_audit": {
            "association_state_axis": [int(value) for value in scored["state_axis"]],
            "public_id_axis": public_axis,
            "base_score_matrix": base_matrix.astype(float).tolist(),
            "committed_score_matrix": np.asarray(matrix, dtype=np.float64).astype(float).tolist(),
            "base_target_scores": base_values.astype(float).tolist(),
            "committed_target_scores": np.asarray(matrix, dtype=np.float64)[:, target_column].astype(float).tolist(),
            "base_top1_score": base_top,
            "base_top2_score": base_second,
            "base_assignment_margin": float(base_top - base_second),
            "proposal_score_matrix": None,
            "proposal_target_scores": None,
            "model_logit_by_candidate": None,
            "model_score_by_candidate": None,
            "none_logit": None,
            "legacy_injection_delta": 0.0,
            "runtime_future_gt_used": False,
        },
        "memory_read": True,
        "memory_write": bool(state_update["trusted_admitted"]),
        "event_frame_memory_read": False,
        "first_memory_visible_frame": int(inputs["event_frame"]) + 1,
        "temporal_state_before_digest": before_digest,
        "temporal_state_update": state_update,
        "temporal_state_after": state_audit(state),
        "temporal_state_after_digest": temporal_state_digest(state),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
        "public_id_immutable": True,
    }
    return row


def _commit_base_frame(
    *, inputs: Mapping[str, Any], frame: int, state: TemporalIdentityState, base: Mapping[str, Any], branch: str
) -> dict[str, Any]:
    scored = base["scored"]
    return _commit_frame(
        inputs=inputs,
        frame=int(frame),
        state=state,
        pool=base["pool"],
        pool_audit=base["pool_audit"],
        scored=scored,
        solver=scored["solver"],
        matrix=np.asarray(scored["base_matrix"], dtype=np.float64),
        target_uid=scored.get("base_target_uid"),
        selected_uid=None,
        selection=None,
        branch=branch,
        base_scored=scored,
    )


def _commit_proposal_frame(
    *, inputs: Mapping[str, Any], frame: int, state: TemporalIdentityState, proposal: Mapping[str, Any]
) -> dict[str, Any]:
    base = proposal["base"]
    scored = proposal["proposal"]
    selection = scored["selection"]
    selected_uid = selection.get("selected_candidate_uid")
    target_uid = scored.get("target_uid")
    trusted_uid = selected_uid if selected_uid is not None and str(selected_uid) == str(target_uid) else None
    return _commit_frame(
        inputs=inputs,
        frame=int(frame),
        state=state,
        pool=proposal["pool"],
        pool_audit=proposal["pool_audit"],
        scored=scored,
        solver=scored["solver"],
        matrix=np.asarray(scored["fused_matrix"], dtype=np.float64),
        target_uid=target_uid,
        selected_uid=trusted_uid,
        selection=selection,
        branch="APPLY",
        base_scored=base,
    )


@dataclass
class BranchRolloutResult:
    runtime_rows: list[dict[str, Any]]
    final_state: TemporalIdentityState
    horizon: int


def rollout_passive_future(
    *,
    inputs: Mapping[str, Any],
    start_frame: int,
    initial_state: TemporalIdentityState,
    horizon: int,
    require_positive_geometry: bool,
) -> BranchRolloutResult:
    """Roll BASE-only continuation for ``start_frame+1 .. +horizon``.

    No GT, PCTIS injection, live re-query or post-hoc code is reachable from
    this function.  The caller owns the state passed in; it is copied before
    the first update so a result can never mutate its caller's branch.
    """

    if int(horizon) < 1 or int(horizon) > 50:
        raise ValueError("passive horizon must be in 1..50")
    event_end = int(inputs["event_frame"]) + int(inputs.get("horizon", 100))
    if int(start_frame) + int(horizon) > event_end:
        raise RuntimeError(
            f"passive continuation exceeds frozen input window: {inputs['event_id']}:{start_frame}+{horizon}>{event_end}"
        )
    state = clone_temporal_state(initial_state)
    rows: list[dict[str, Any]] = []
    local_inputs = dict(inputs)
    local_inputs["require_positive_geometry"] = bool(require_positive_geometry)
    for frame in range(int(start_frame) + 1, int(start_frame) + int(horizon) + 1):
        base = build_base_frame(inputs=local_inputs, frame=frame, state=state, device="cpu")
        rows.append(_commit_base_frame(inputs=local_inputs, frame=frame, state=state, base=base, branch="PASSIVE_BASE"))
    return BranchRolloutResult(runtime_rows=rows, final_state=state, horizon=int(horizon))


def rollout_intervention_pair(
    *,
    inputs: Mapping[str, Any],
    frame: int,
    state_before: TemporalIdentityState,
    pctis_proposal: Mapping[str, Any],
    max_horizon: int = 50,
    require_positive_geometry: bool = True,
) -> dict[str, Any]:
    """Fork one accepted proposal and passively continue both branches."""

    if not bool(pctis_proposal.get("effective_opportunity")):
        raise ValueError("paired rollout requires an effective intervention opportunity")
    if int(max_horizon) not in VALUE_HORIZONS and int(max_horizon) != 50:
        raise ValueError("max_horizon must be one of the frozen value horizons")
    before = clone_temporal_state(state_before)
    keep_state = clone_temporal_state(before)
    apply_state = clone_temporal_state(before)
    before_digest = temporal_state_digest(before)
    keep_initial_digest = temporal_state_digest(keep_state)
    apply_initial_digest = temporal_state_digest(apply_state)
    base = pctis_proposal["base"]
    keep_row = _commit_frame(
        inputs=inputs,
        frame=int(frame),
        state=keep_state,
        pool=pctis_proposal["pool"],
        pool_audit=pctis_proposal["pool_audit"],
        scored=base,
        solver=base["solver"],
        matrix=np.asarray(base["base_matrix"], dtype=np.float64),
        target_uid=base.get("base_target_uid"),
        selected_uid=None,
        selection=None,
        branch="KEEP",
        base_scored=base,
    )
    apply_row = _commit_proposal_frame(inputs=inputs, frame=int(frame), state=apply_state, proposal=pctis_proposal)
    keep = rollout_passive_future(
        inputs=inputs,
        start_frame=int(frame),
        initial_state=keep_state,
        horizon=int(max_horizon),
        require_positive_geometry=bool(require_positive_geometry),
    )
    applied = rollout_passive_future(
        inputs=inputs,
        start_frame=int(frame),
        initial_state=apply_state,
        horizon=int(max_horizon),
        require_positive_geometry=bool(require_positive_geometry),
    )
    return {
        "schema_version": "N72R13_TIV_INTERVENTION_PAIR_V1",
        "event_id": str(inputs["event_id"]),
        "sequence": str(inputs["sequence"]),
        "action_type": str(inputs.get("action_type", "UNKNOWN")),
        "event_frame": int(inputs["event_frame"]),
        "intervention_frame": int(frame),
        "frame_range": [int(frame), int(frame) + int(max_horizon)],
        "state_before": state_audit(before),
        "state_before_sha256": before_digest,
        "keep_initial_state_sha256": keep_initial_digest,
        "apply_initial_state_sha256": apply_initial_digest,
        "keep": {
            "current_commit_row": keep_row,
            "state_after_current": state_audit(keep_state),
            "state_after_current_sha256": temporal_state_digest(keep_state),
            "_state_object": keep_state,
            "passive": keep,
        },
        "apply": {
            "current_commit_row": apply_row,
            "state_after_current": state_audit(apply_state),
            "state_after_current_sha256": temporal_state_digest(apply_state),
            "_state_object": apply_state,
            "passive": applied,
        },
        "proposal_audit": {
            "selection_accepted": bool(pctis_proposal.get("selection_accepted")),
            "assignment_changed": bool(pctis_proposal.get("assignment_changed")),
            "effective_opportunity": True,
            "base_target_uid": None if base.get("base_target_uid") is None else str(base.get("base_target_uid")),
            "proposal_target_uid": None if pctis_proposal["proposal"].get("target_uid") is None else str(pctis_proposal["proposal"].get("target_uid")),
            "selected_candidate_uid": pctis_proposal["proposal"]["selection"].get("selected_candidate_uid"),
        },
        "initial_state_equal_for_branches": bool(keep_initial_digest == apply_initial_digest == before_digest),
        "current_assignment_differs": bool(
            str(keep_row["assignment"]["target_assigned_candidate_uid"])
            != str(apply_row["assignment"]["target_assigned_candidate_uid"])
        ),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "runtime_eligible": True,
        "oracle_upper_bound_only": False,
    }


def serialize_branch_result(result: BranchRolloutResult) -> dict[str, Any]:
    """Convert a branch result to an atomic-artifact-safe JSON object."""

    return {
        "runtime_rows": deepcopy(result.runtime_rows),
        "final_state": state_audit(result.final_state),
        "final_state_sha256": temporal_state_digest(result.final_state),
        "horizon": int(result.horizon),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def serialize_intervention_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Drop live NumPy/model objects while preserving the runtime audit."""

    output = {
        key: value
        for key, value in pair.items()
        if key not in {"keep", "apply"}
    }
    for name in ("keep", "apply"):
        branch = pair[name]
        output[name] = {
            "current_commit_row": deepcopy(branch["current_commit_row"]),
            "state_after_current": deepcopy(branch["state_after_current"]),
            "state_after_current_sha256": str(branch["state_after_current_sha256"]),
            "passive": serialize_branch_result(branch["passive"]),
        }
    output["runtime_future_gt_used"] = False
    output["runtime_gt_read"] = False
    output["posthoc_gt_used"] = False
    return output


__all__ = [
    "PRIMARY_VALUE_HORIZON",
    "VALUE_HORIZONS",
    "BranchRolloutResult",
    "build_base_frame",
    "build_pctis_proposal",
    "clone_temporal_state",
    "rollout_intervention_pair",
    "rollout_passive_future",
    "serialize_branch_result",
    "serialize_intervention_pair",
    "state_audit",
    "temporal_state_digest",
]
