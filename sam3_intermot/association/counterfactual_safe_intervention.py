"""Counterfactual-safe authority for the N72R12 intervention probe.

The module deliberately sits *after* the frozen PCTIS proposal solver and
*before* the runtime state update.  It does not score candidates, infer
public IDs, or implement another assignment solver.  Its only authority is
the exact base/proposal assignment audit supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


KEEP_BASELINE = "KEEP_BASELINE"
APPLY_PCTIS = "APPLY_PCTIS"
COUNTERFACTUAL_SAFE_V1 = "COUNTERFACTUAL_SAFE_V1"
EPSILON = 1.0e-9

_REJECTION_ORDER = (
    "NO_ACCEPTED_SELECTION",
    "NO_TARGET_CHANGE",
    "PROPOSAL_TARGET_NOT_SELECTED",
    "INVALID_GEOMETRY",
    "SELECTED_OWNED_BY_OTHER_PUBLIC",
    "NON_TARGET_PUBLIC_COLLATERAL_CHANGE",
    "TARGET_EDGE_NOT_DOMINANT",
    "NO_POSITIVE_TARGET_EDGE_GAIN",
)


def _solver_rows(solver: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = solver.get("assignment_rows")
    if not isinstance(rows, list):
        raise ValueError("solver assignment_rows must be a list")
    return [row for row in rows if isinstance(row, Mapping)]


def solver_public_assignment_map(
    solver: Mapping[str, Any],
) -> dict[int, str | None]:
    """Return the solver's explicit public-ID -> candidate UID map.

    The public assignment list is the authority.  This function never
    reconstructs it from candidate rows or from numeric state IDs.
    """

    assignments = solver.get("public_assignments")
    if not isinstance(assignments, list):
        raise ValueError("solver public_assignments must be a list")
    candidate_uids = {str(row.get("candidate_uid")) for row in _solver_rows(solver)}
    result: dict[int, str | None] = {}
    for item in assignments:
        if not isinstance(item, Mapping) or item.get("public_id") is None:
            raise ValueError("every public assignment must contain a public_id")
        public_id = int(item["public_id"])
        if public_id in result:
            raise ValueError(f"duplicate public_id in solver map: {public_id}")
        candidate_uid = item.get("candidate_uid")
        if candidate_uid in (None, "", "None"):
            result[public_id] = None
        else:
            uid = str(candidate_uid)
            if uid not in candidate_uids:
                raise ValueError(f"public assignment points outside candidate axis: {uid}")
            result[public_id] = uid
    return result


def solver_candidate_public_map(
    solver: Mapping[str, Any],
) -> dict[str, int | None]:
    """Return candidate UID -> public ID using explicit assignment rows."""

    rows = _solver_rows(solver)
    result: dict[str, int | None] = {}
    for row in rows:
        candidate_uid = row.get("candidate_uid")
        if candidate_uid in (None, "", "None"):
            raise ValueError("solver assignment row lacks candidate_uid")
        uid = str(candidate_uid)
        if uid in result:
            raise ValueError(f"duplicate candidate UID in solver rows: {uid}")
        public_id = row.get("public_id")
        result[uid] = None if public_id is None else int(public_id)
    return result


def compare_global_assignments(
    *,
    base_solver: Mapping[str, Any],
    proposal_solver: Mapping[str, Any],
    target_public_id: int,
) -> dict[str, Any]:
    """Compare the two explicit global assignments on the public axis."""

    base_map = solver_public_assignment_map(base_solver)
    proposal_map = solver_public_assignment_map(proposal_solver)
    public_axis = sorted(set(base_map) | set(proposal_map))
    changed_public_ids = [
        public_id
        for public_id in public_axis
        if base_map.get(public_id) != proposal_map.get(public_id)
    ]
    non_target = [public_id for public_id in changed_public_ids if public_id != int(target_public_id)]
    base_candidate_map = solver_candidate_public_map(base_solver)
    proposal_candidate_map = solver_candidate_public_map(proposal_solver)
    base_target_uid = base_map.get(int(target_public_id))
    proposal_target_uid = proposal_map.get(int(target_public_id))
    return {
        "base_public_map": base_map,
        "proposal_public_map": proposal_map,
        "changed_public_ids": changed_public_ids,
        "non_target_changed_public_ids": non_target,
        "changed_public_count": len(changed_public_ids),
        "non_target_changed_public_count": len(non_target),
        "base_target_candidate_uid": base_target_uid,
        "proposal_target_candidate_uid": proposal_target_uid,
        "base_candidate_public_map": base_candidate_map,
        "proposal_candidate_public_map": proposal_candidate_map,
    }


def _selected_index_and_uid(
    selection: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
) -> tuple[bool, int | None, str | None]:
    accepted = bool(selection.get("accepted") is True)
    selected_uid = selection.get("selected_candidate_uid")
    if not accepted or selected_uid in (None, "", "None"):
        return accepted, None, None
    selected_index = selection.get("selected_index")
    if isinstance(selected_index, bool) or not isinstance(selected_index, (int, np.integer)):
        raise ValueError("accepted selection must provide an integer selected_index")
    index = int(selected_index)
    if index < 0 or index >= len(pool):
        raise ValueError("accepted selected_index is outside the pool")
    uid = str(selected_uid)
    if str(pool[index].get("candidate_uid")) != uid:
        raise ValueError("selected_index does not identify selected_candidate_uid")
    return True, index, uid


def build_counterfactual_audit(
    *,
    target_public_id: int,
    pool: Sequence[Mapping[str, Any]],
    base_matrix: np.ndarray,
    proposed_matrix: np.ndarray,
    public_axis: Sequence[int],
    base_solver: Mapping[str, Any],
    proposal_solver: Mapping[str, Any],
    selection: Mapping[str, Any],
    model_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete runtime-causal CSI audit for one frame."""

    base = np.asarray(base_matrix, dtype=np.float64)
    proposal = np.asarray(proposed_matrix, dtype=np.float64)
    axis = [int(value) for value in public_axis]
    if base.ndim != 2 or proposal.shape != base.shape or base.shape[1] != len(axis):
        raise ValueError("counterfactual matrices and public axis are misaligned")
    if not np.isfinite(base).all() or not np.isfinite(proposal).all():
        raise ValueError("counterfactual matrices must be finite")
    if len(pool) != base.shape[0]:
        raise ValueError("counterfactual pool and matrix are misaligned")
    if len(axis) != len(set(axis)):
        raise ValueError("counterfactual public axis contains duplicates")

    comparison = compare_global_assignments(
        base_solver=base_solver,
        proposal_solver=proposal_solver,
        target_public_id=int(target_public_id),
    )
    accepted, selected_index, selected_uid = _selected_index_and_uid(selection, pool)
    target_column = axis.index(int(target_public_id)) if int(target_public_id) in axis else None
    base_target_uid = comparison["base_target_candidate_uid"]
    proposal_target_uid = comparison["proposal_target_candidate_uid"]
    target_changed = base_target_uid != proposal_target_uid
    causal_order = list(model_values.get("causal_order", []))
    causal_top_uid = None
    if causal_order:
        top_index = int(causal_order[0])
        if 0 <= top_index < len(pool):
            causal_top_uid = str(pool[top_index].get("candidate_uid"))

    audit: dict[str, Any] = {
        "schema_version": "N72R12_COUNTERFACTUAL_AUDIT_V1",
        "policy": COUNTERFACTUAL_SAFE_V1,
        "target_public_id": int(target_public_id),
        "public_axis": axis,
        "selected_candidate_uid": selected_uid,
        "selected_candidate_index": selected_index,
        "selected_candidate_source": None if selected_index is None else str(pool[selected_index].get("candidate_source")),
        "selection_accepted": accepted,
        "selected_score": None if selected_index is None else _optional_float(selection.get("selected_score")),
        "selected_margin": None if selected_index is None else _optional_float(selection.get("best_minus_second_margin")),
        "base_assignment_margin": _optional_float(model_values.get("base_assignment_margin")),
        "base_target_uid": base_target_uid,
        "proposal_target_uid": proposal_target_uid,
        "target_assignment_changed": bool(target_changed),
        "proposal_target_is_selected": bool(selected_uid is not None and proposal_target_uid == selected_uid),
        "selected_base_public_id": None,
        "selected_base_target_edge": None,
        "selected_proposed_target_edge": None,
        "selected_best_other_edge": None,
        "selected_target_minus_best_other": None,
        "target_edge_gain": None,
        "selected_geometry_valid": None,
        "selected_confidence": None,
        "selected_presence": None,
        "selected_motion_iou": None,
        "causal_top_candidate_uid": causal_top_uid,
        "causal_top_agrees_with_selected": bool(causal_top_uid is not None and causal_top_uid == selected_uid),
        "changed_public_ids": comparison["changed_public_ids"],
        "non_target_changed_public_ids": comparison["non_target_changed_public_ids"],
        "changed_public_count": int(comparison["changed_public_count"]),
        "non_target_changed_public_count": int(comparison["non_target_changed_public_count"]),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }
    if selected_index is not None:
        if target_column is None:
            raise ValueError("target public ID is absent from public axis")
        selected = pool[selected_index]
        selected_base_public = comparison["base_candidate_public_map"].get(selected_uid)
        best_other_values = np.asarray(model_values.get("legacy_best_other_scores", []), dtype=np.float64).reshape(-1)
        motion_values = np.asarray(model_values.get("motion_iou", []), dtype=np.float64).reshape(-1)
        if best_other_values.shape != (len(pool),) or motion_values.shape != (len(pool),):
            raise ValueError("selected-edge audit vectors are not aligned with pool")
        base_edge = float(base[selected_index, target_column])
        proposed_edge = float(proposal[selected_index, target_column])
        best_other = float(best_other_values[selected_index])
        audit.update(
            {
                "selected_base_public_id": None if selected_base_public is None else int(selected_base_public),
                "selected_base_target_edge": base_edge,
                "selected_proposed_target_edge": proposed_edge,
                "selected_best_other_edge": best_other,
                "selected_target_minus_best_other": proposed_edge - best_other,
                "target_edge_gain": proposed_edge - base_edge,
                "selected_geometry_valid": bool(selected.get("geometry_valid") is True),
                "selected_confidence": _optional_float(selected.get("confidence")),
                "selected_presence": _optional_float(selected.get("presence_score")),
                "selected_motion_iou": float(motion_values[selected_index]),
            }
        )
    return audit


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("counterfactual audit scalar is non-finite")
    return result


def decide_counterfactual_safe_v1(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered no-new-threshold CSI decision."""

    selected_uid = audit.get("selected_candidate_uid")
    selected_geometry = audit.get("selected_geometry_valid")
    selected_base_public = audit.get("selected_base_public_id")
    target_public_id = int(audit.get("target_public_id"))
    reasons: list[str] = []
    if audit.get("selection_accepted") is not True or selected_uid in (None, "", "None"):
        reasons.append("NO_ACCEPTED_SELECTION")
    if audit.get("target_assignment_changed") is not True:
        reasons.append("NO_TARGET_CHANGE")
    if audit.get("proposal_target_is_selected") is not True:
        reasons.append("PROPOSAL_TARGET_NOT_SELECTED")
    if selected_uid not in (None, "", "None") and selected_geometry is not True:
        reasons.append("INVALID_GEOMETRY")
    if selected_uid not in (None, "", "None") and selected_base_public not in (None, target_public_id):
        reasons.append("SELECTED_OWNED_BY_OTHER_PUBLIC")
    if int(audit.get("non_target_changed_public_count", 0)) != 0:
        reasons.append("NON_TARGET_PUBLIC_COLLATERAL_CHANGE")
    proposed_edge = audit.get("selected_proposed_target_edge")
    best_other_edge = audit.get("selected_best_other_edge")
    if proposed_edge is None or best_other_edge is None or not float(proposed_edge) > max(float(best_other_edge), 0.0) + EPSILON:
        reasons.append("TARGET_EDGE_NOT_DOMINANT")
    target_edge_gain = audit.get("target_edge_gain")
    if target_edge_gain is None or not float(target_edge_gain) > EPSILON:
        reasons.append("NO_POSITIVE_TARGET_EDGE_GAIN")
    unique_reasons = [reason for reason in _REJECTION_ORDER if reason in set(reasons)]
    apply = not unique_reasons
    return {
        "schema_version": "N72R12_COUNTERFACTUAL_DECISION_V1",
        "policy": COUNTERFACTUAL_SAFE_V1,
        "decision": APPLY_PCTIS if apply else KEEP_BASELINE,
        "intervention_applied": bool(apply),
        "rejection_reasons": [] if apply else unique_reasons,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def commit_counterfactual_decision(
    *,
    decision: Mapping[str, Any],
    base_solver: Mapping[str, Any],
    proposal_solver: Mapping[str, Any],
    base_matrix: np.ndarray,
    proposal_matrix: np.ndarray,
    base_target_uid: str | None,
    proposal_target_uid: str | None,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit exactly one already-solved assignment at the runtime point."""

    name = str(decision.get("decision"))
    base = np.asarray(base_matrix, dtype=np.float64)
    proposal = np.asarray(proposal_matrix, dtype=np.float64)
    if base.shape != proposal.shape or base.ndim != 2 or not np.isfinite(base).all() or not np.isfinite(proposal).all():
        raise ValueError("committed matrices must be aligned finite rank-2 arrays")
    if name == APPLY_PCTIS:
        committed_solver = proposal_solver
        committed_matrix = proposal
        committed_target_uid = None if proposal_target_uid is None else str(proposal_target_uid)
        applied = True
    elif name == KEEP_BASELINE:
        committed_solver = base_solver
        committed_matrix = base
        committed_target_uid = None if base_target_uid is None else str(base_target_uid)
        applied = False
    else:
        raise ValueError(f"unknown counterfactual decision: {name}")
    proposal_selected_uid = selection.get("selected_candidate_uid")
    committed_selected_uid = (
        str(proposal_selected_uid)
        if (
            proposal_selected_uid is not None
            and committed_target_uid is not None
            and str(proposal_selected_uid) == str(committed_target_uid)
        )
        else None
    )
    return {
        "decision": name,
        "committed_solver": committed_solver,
        "committed_matrix": committed_matrix,
        "committed_target_uid": committed_target_uid,
        "intervention_applied": applied,
        "proposal_selected_uid": None if proposal_selected_uid is None else str(proposal_selected_uid),
        "committed_selected_uid": committed_selected_uid,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


__all__ = [
    "KEEP_BASELINE",
    "APPLY_PCTIS",
    "COUNTERFACTUAL_SAFE_V1",
    "EPSILON",
    "solver_public_assignment_map",
    "solver_candidate_public_map",
    "compare_global_assignments",
    "build_counterfactual_audit",
    "decide_counterfactual_safe_v1",
    "commit_counterfactual_decision",
]
