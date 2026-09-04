"""Target-scoped candidate merge and public-assignment constraints for N72R6.

The main SAM candidate stream is copied without relabelling.  A target-session
candidate carries an explicit correction scope and no public ID.  The only
cross-stream operation here is concatenation plus a public-state domain mask:
main candidates cannot claim the corrected public state, while target-session
candidates can claim that state or explicit ``NONE``.  This module never
infers a public identity from a native ID or from a candidate position.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


TARGET_CANDIDATE_KIND = "TARGET_CORRECTION_SESSION_CANDIDATE"
FORBIDDEN_SCORE = -1.0e6
FEATURE_DIM = 512


def _public_id(runtime: Any, state: Any) -> int:
    record = runtime.get_identity_by_state_id(int(state.pid))
    if record is None:
        raise ValueError(f"state {state.pid} has no persistent public authority")
    return int(record.public_id)


def _candidate_uid(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_uid")
    if value in (None, ""):
        raise ValueError("candidate is missing candidate_uid")
    return str(value)


def _unit_feature(value: Any, *, label: str) -> np.ndarray:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)):
        raise ValueError(f"{label}: expected finite {FEATURE_DIM}-D feature")
    norm = float(np.linalg.norm(feature))
    if norm <= 1.0e-6:
        raise ValueError(f"{label}: zero-norm feature")
    return feature / norm


def apply_human_anchor_verification_gate(
    target_candidates: Sequence[Mapping[str, Any]],
    human_anchor: Any,
    *,
    threshold: float,
    event_id: str,
    frame: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accept only target-session rows consistent with the explicit anchor.

    This is a target-session provenance gate, not an identity inference path:
    the public target is supplied by the human event and the gate can only
    keep the already-scoped candidate or reject it to ``NONE``.  It never
    considers future GT and never substitutes a main-session candidate.
    """

    value = float(threshold)
    if not np.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("human-anchor verification threshold must be finite in [-1, 1]")
    anchor = _unit_feature(human_anchor, label=f"{event_id}:{frame}:human_anchor")
    candidates = [deepcopy(dict(item)) for item in target_candidates]
    if len(candidates) > 1:
        raise ValueError(f"target-session candidate cardinality exceeds one: {event_id}:{frame}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        uid = _candidate_uid(candidate)
        feature = _unit_feature(candidate.get("feature"), label=f"{event_id}:{frame}:{uid}")
        cosine = float(np.dot(feature, anchor))
        finite_pass = bool(np.isfinite(cosine))
        keep = bool(finite_pass and cosine >= value)
        audit_item = {
            "candidate_uid": uid,
            "frame": int(frame),
            "human_anchor_cosine": cosine,
            "threshold": value,
            "accepted": keep,
            "runtime_future_gt_used": False,
            "public_id_inference": False,
        }
        if keep:
            accepted.append(candidate)
        else:
            audit_item["rejection_reason"] = (
                "NONFINITE_ANCHOR_COSINE" if not finite_pass else "HUMAN_ANCHOR_COSINE_BELOW_THRESHOLD"
            )
            rejected.append(audit_item)
    return accepted, {
        "schema_version": "N72R6_HUMAN_ANCHOR_VERIFICATION_GATE_V1",
        "enabled": True,
        "event_id": str(event_id),
        "frame": int(frame),
        "threshold": value,
        "input_candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted),
        "rejected_candidate_count": len(rejected),
        "accepted": accepted[0].get("candidate_uid") if accepted else None,
        "rejected": rejected,
        "rejected_to_explicit_none": True,
        "main_candidate_fallback": False,
        "runtime_future_gt_used": False,
        "public_id_inference": False,
    }


def _validate_target_candidate(
    candidate: Mapping[str, Any],
    *,
    event_id: str,
    frame: int,
    target_public_id: int,
    target_session_scope: str,
) -> None:
    if candidate.get("candidate_kind") != TARGET_CANDIDATE_KIND:
        raise ValueError(f"target candidate has wrong kind: {_candidate_uid(candidate)}")
    if candidate.get("public_id") is not None:
        raise ValueError(f"target candidate must not carry public_id: {_candidate_uid(candidate)}")
    if str(candidate.get("event_id", event_id)) != str(event_id):
        raise ValueError(f"target candidate event mismatch: {_candidate_uid(candidate)}")
    if int(candidate.get("frame", frame)) != int(frame):
        raise ValueError(f"target candidate frame mismatch: {_candidate_uid(candidate)}")
    if int(candidate.get("human_target_scope_public_id", -1)) != int(target_public_id):
        raise ValueError(f"target candidate public scope mismatch: {_candidate_uid(candidate)}")
    if str(candidate.get("target_session_scope")) != str(target_session_scope):
        raise ValueError(f"target candidate native scope mismatch: {_candidate_uid(candidate)}")
    for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
        if candidate.get(flag, False) is not False:
            raise ValueError(f"target candidate {flag} is not false: {_candidate_uid(candidate)}")


def merge_main_and_target_candidates(
    main_candidates: Sequence[Mapping[str, Any]],
    target_candidates: Sequence[Mapping[str, Any]],
    *,
    event_id: str,
    frame: int,
    target_public_id: int,
    target_session_scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a lossless main+target stream merge for one global frame."""

    main = [deepcopy(dict(item)) for item in main_candidates]
    target = [deepcopy(dict(item)) for item in target_candidates]
    main_uids = [_candidate_uid(item) for item in main]
    if len(main_uids) != len(set(main_uids)):
        raise ValueError(f"duplicate main candidate UID at {event_id}:{frame}")
    target_uids = [_candidate_uid(item) for item in target]
    if len(target_uids) != len(set(target_uids)):
        raise ValueError(f"duplicate target candidate UID at {event_id}:{frame}")
    if set(main_uids) & set(target_uids):
        raise ValueError(f"main/target candidate UID collision at {event_id}:{frame}")
    if len(target) > 1:
        raise ValueError(f"target session exposed more than one candidate at {event_id}:{frame}")
    for item in target:
        _validate_target_candidate(
            item,
            event_id=str(event_id),
            frame=int(frame),
            target_public_id=int(target_public_id),
            target_session_scope=str(target_session_scope),
        )
    next_index = max([int(item.get("candidate_index", -1)) for item in main] or [-1]) + 1
    for item in target:
        item["target_session_candidate_index"] = int(item.get("candidate_index", 0))
        # Candidate indices are an association-row axis, not an identity
        # authority.  Give the appended row a collision-free index while
        # retaining its original target-session index above.
        item["candidate_index"] = next_index
        next_index += 1
    merged = main + target
    audit = {
        "schema_version": "N72R6_TARGET_SCOPED_MERGE_V1",
        "event_id": str(event_id),
        "frame": int(frame),
        "main_candidate_count": len(main),
        "target_candidate_count": len(target),
        "merged_candidate_count": len(merged),
        "main_candidate_uids": main_uids,
        "target_candidate_uids": target_uids,
        "target_public_id": int(target_public_id),
        "target_session_scope": str(target_session_scope),
        "main_rows_copied_without_relabel": True,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
    }
    return merged, audit


def apply_target_exclusive_constraints(
    base_candidate_state: np.ndarray,
    states: Sequence[Any],
    candidates: Sequence[Mapping[str, Any]],
    runtime: Any,
    *,
    target_public_id: int,
    force_target_uid: str | None = None,
    shadowed_main_uids: Sequence[str] | None = None,
    fallback_main_uid: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask the target public column to the target-session domain.

    ``base_candidate_state`` is candidate×state.  The returned matrix is the
    same orientation and is the only matrix submitted to the exact public
    solver.  ``NONE`` remains the solver's explicit per-candidate alternative;
    the caller decides whether NONE can create an outer birth.
    """

    matrix = np.asarray(base_candidate_state, dtype=np.float64).copy()
    if matrix.shape != (len(candidates), len(states)):
        raise ValueError(f"target exclusive matrix shape mismatch: {matrix.shape}")
    public_axis = [_public_id(runtime, state) for state in states]
    if int(target_public_id) not in public_axis:
        raise ValueError(f"target public is absent from state axis: {target_public_id}")
    target_col = public_axis.index(int(target_public_id))
    target_uids = [
        _candidate_uid(candidate)
        for candidate in candidates
        if candidate.get("candidate_kind") == TARGET_CANDIDATE_KIND
    ]
    shadowed_uids = {str(value) for value in (shadowed_main_uids or ())}
    fallback_uid = None if fallback_main_uid is None else str(fallback_main_uid)
    if fallback_uid is not None and fallback_uid not in shadowed_uids:
        raise ValueError("main fallback must be selected from shadowed target rows")
    if shadowed_uids & set(target_uids):
        raise ValueError("a target-session candidate cannot also be a shadowed main candidate")
    unknown_shadowed = shadowed_uids - {
        _candidate_uid(candidate) for candidate in candidates
    }
    if unknown_shadowed:
        raise ValueError(f"shadowed main candidate is not in merged stream: {sorted(unknown_shadowed)}")
    if len(target_uids) > 1:
        raise ValueError("merged stream contains more than one target-session candidate")
    for row, candidate in enumerate(candidates):
        uid = _candidate_uid(candidate)
        is_target = uid in target_uids
        if uid in shadowed_uids:
            # This main row was the frozen C0 carrier of target_public_id.
            # Once target-scoped correction is active it is a shadow of the
            # target-session domain, not a fallback observation.  Force its
            # only solver option to explicit NONE so the outer birth policy
            # cannot turn it into a duplicate or absorb it into a protected
            # public identity.
            matrix[row, :] = FORBIDDEN_SCORE
            if fallback_uid == uid and not target_uids:
                # A fallback is still target-scoped: only the frozen B0 row
                # that previously carried this target public may be assigned
                # to the same public, or explicit NONE.  No protected main
                # row can enter this branch.
                matrix[row, target_col] = float(base_candidate_state[row, target_col])
            continue
        if is_target:
            matrix[row, :] = FORBIDDEN_SCORE
            # A future target row may still choose explicit NONE if its score
            # is insufficient.  At the event frame the caller supplies a
            # deterministic authority force below.
            matrix[row, target_col] = float(base_candidate_state[row, target_col])
            if force_target_uid is not None and uid == str(force_target_uid):
                matrix[row, :] = FORBIDDEN_SCORE
                matrix[row, target_col] = -FORBIDDEN_SCORE
        else:
            matrix[row, target_col] = FORBIDDEN_SCORE
    if not np.isfinite(matrix).all():
        raise ValueError("target exclusive matrix contains non-finite values")
    audit = {
        "schema_version": "N72R6_TARGET_EXCLUSIVE_CONSTRAINT_V1",
        "target_public_id": int(target_public_id),
        "target_state_column": int(target_col),
        "target_candidate_uids": target_uids,
        "shadowed_main_candidate_uids": sorted(shadowed_uids),
        "fallback_main_candidate_uid": fallback_uid,
        "force_target_uid": None if force_target_uid is None else str(force_target_uid),
        "main_rows_blocked_from_target_public": sum(
            uid not in target_uids and uid != fallback_uid
            for uid in (_candidate_uid(item) for item in candidates)
        ),
        "fallback_main_row_allowed_target_public_or_none": fallback_uid is not None,
        "shadowed_main_rows_forced_none": len(shadowed_uids),
        "target_rows_allowed_target_public_or_none": True,
        "target_row_non_target_states_blocked": True,
        "explicit_none_preserved": True,
        "runtime_future_gt_used": False,
        "public_id_inference": False,
    }
    return matrix, audit


__all__ = [
    "FORBIDDEN_SCORE",
    "TARGET_CANDIDATE_KIND",
    "apply_human_anchor_verification_gate",
    "apply_target_exclusive_constraints",
    "merge_main_and_target_candidates",
]
