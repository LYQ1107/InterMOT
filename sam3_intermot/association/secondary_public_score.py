"""Shared secondary candidate-to-public score construction for N72R11R4.

The secondary interaction worker and the exact on-policy corpus builder must
submit exactly the same candidate x public score matrix to the frozen global
solver.  This helper owns only the supplemental target-column edges; it does
not solve assignments and it never creates public identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sam3_intermot.reacquisition.target_candidate_pool import (
    FUTURE_FRAME_REQUERY,
    MAIN_B0_CANDIDATE,
    TARGET_SESSION_CURRENT_RAW,
)


TARGET_PUBLIC_EDGE_FUTURE = 4.50
TARGET_PUBLIC_EDGE_CURRENT = 4.00
_ALLOWED_SUPPLEMENTAL_SOURCES = {
    TARGET_SESSION_CURRENT_RAW,
    FUTURE_FRAME_REQUERY,
}


@dataclass(frozen=True)
class SecondaryPublicScoreFrame:
    """One solver input frame with explicit state/public axes."""

    state_axis: list[int]
    public_axis: list[int]
    target_column: int
    matrix: np.ndarray
    target_edge_by_uid: dict[str, float]


def _unit(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 512 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite 512-D")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError(f"{label} has zero norm")
    return array / norm


def _box_iou(left: Any, right: Any) -> float:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.size != 4 or second.size != 4 or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("secondary score boxes must be finite XYXY values")
    if first[2] <= first[0] or first[3] <= first[1] or second[2] <= second[0] or second[3] <= second[1]:
        return 0.0
    intersection = max(0.0, min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0]))) * max(
        0.0,
        min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])),
    )
    first_area = float(first[2] - first[0]) * float(first[3] - first[1])
    second_area = float(second[2] - second[0]) * float(second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def supplemental_target_edge(
    candidate: Mapping[str, Any],
    *,
    anchor_feature: Sequence[float],
    predicted_box: Sequence[float],
    source: str,
) -> float:
    """Return the frozen supplemental edge used by the secondary worker."""

    if str(source) not in _ALLOWED_SUPPLEMENTAL_SOURCES:
        raise ValueError(f"unsupported secondary supplemental source: {source}")
    feature = candidate.get("feature")
    cosine = 0.0 if feature is None else float(np.dot(_unit(feature, "candidate feature"), _unit(anchor_feature, "anchor feature")))
    presence = float(candidate.get("presence_score", candidate.get("confidence", 0.0)) or 0.0)
    if not np.isfinite(presence):
        raise ValueError("candidate presence is non-finite")
    base = TARGET_PUBLIC_EDGE_CURRENT if str(source) == TARGET_SESSION_CURRENT_RAW else TARGET_PUBLIC_EDGE_FUTURE
    return float(
        base
        + cosine
        + 0.20 * np.clip(presence, 0.0, 1.0)
        + 0.20 * _box_iou(candidate["box_xyxy"], predicted_box)
    )


def _explicit_axes(c0_row: Mapping[str, Any], target_public_id: int) -> tuple[list[int], list[int], np.ndarray]:
    solver = c0_row.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("C0 row lacks solver audit")
    public_axis = [int(value) for value in solver.get("public_id_axis", [])]
    state_axis = [int(value) for value in solver.get("association_state_axis", [])]
    if len(public_axis) != len(state_axis) or len(public_axis) != len(set(public_axis)) or len(state_axis) != len(set(state_axis)):
        raise ValueError("C0 public/state axes are not explicit and unique")
    base = np.asarray(c0_row.get("base_score_matrix"), dtype=np.float64)
    if base.ndim != 2:
        raise ValueError(f"C0 base score matrix must be rank-2, got {base.shape}")
    main_count = len(c0_row.get("candidate_rows", []))
    if base.shape != (main_count, len(public_axis)):
        raise ValueError(
            f"C0 base score matrix shape {base.shape} != {(main_count, len(public_axis))}"
        )
    if not np.all(np.isfinite(base)):
        raise ValueError("C0 base score matrix is non-finite")
    if int(target_public_id) not in public_axis:
        target_identity = next(
            (
                row
                for row in c0_row.get("identity_rows", [])
                if row.get("public_id") is not None
                and int(row["public_id"]) == int(target_public_id)
                and row.get("association_state_id") is not None
            ),
            None,
        )
        if target_identity is None:
            raise ValueError(f"target public authority {target_public_id} is absent from C0 axes")
        target_state = int(target_identity["association_state_id"])
        if target_state in state_axis:
            raise ValueError("target state collides with an active C0 state axis")
        public_axis.append(int(target_public_id))
        state_axis.append(target_state)
        base = np.concatenate([base, np.zeros((base.shape[0], 1), dtype=np.float64)], axis=1)
    return state_axis, public_axis, base


def build_secondary_public_score_frame(
    *,
    c0_row: Mapping[str, Any],
    main_candidates: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    target_public_id: int,
    anchor_feature: Sequence[float],
    predicted_box: Sequence[float],
    event_id: str,
    frame: int,
) -> SecondaryPublicScoreFrame:
    """Construct the complete candidate x public matrix with frozen semantics."""

    del event_id, frame  # retained in the public signature for audit callers
    state_axis, public_axis, base = _explicit_axes(c0_row, int(target_public_id))
    target_column = public_axis.index(int(target_public_id))
    main_uid_to_index = {
        str(row["candidate_uid"]): index for index, row in enumerate(main_candidates)
    }
    if len(main_uid_to_index) != len(main_candidates):
        raise ValueError("main candidate UID axis is not unique")
    pool_uids = [str(row.get("candidate_uid")) for row in pool]
    if any(uid in {"None", ""} for uid in pool_uids) or len(pool_uids) != len(set(pool_uids)):
        raise ValueError("secondary pool candidate UID axis is not unique")
    matrix = np.zeros((len(pool), len(public_axis)), dtype=np.float64)
    target_edges: dict[str, float] = {}
    for index, candidate in enumerate(pool):
        uid = str(candidate["candidate_uid"])
        main_index = main_uid_to_index.get(uid)
        if main_index is not None:
            matrix[index] = base[main_index]
            target_edges[uid] = float(matrix[index, target_column])
            continue
        source = str(candidate.get("candidate_source", candidate.get("source_kind", "")))
        if source == MAIN_B0_CANDIDATE or source not in _ALLOWED_SUPPLEMENTAL_SOURCES:
            raise ValueError(f"secondary candidate is not in the frozen source axis: {uid}:{source}")
        edge = supplemental_target_edge(
            candidate,
            anchor_feature=anchor_feature,
            predicted_box=predicted_box,
            source=source,
        )
        matrix[index, target_column] = edge
        target_edges[uid] = edge
    if not np.all(np.isfinite(matrix)):
        raise ValueError("secondary public score matrix is non-finite")
    return SecondaryPublicScoreFrame(
        state_axis=state_axis,
        public_axis=public_axis,
        target_column=int(target_column),
        matrix=matrix,
        target_edge_by_uid=target_edges,
    )


__all__ = [
    "TARGET_PUBLIC_EDGE_FUTURE",
    "TARGET_PUBLIC_EDGE_CURRENT",
    "SecondaryPublicScoreFrame",
    "supplemental_target_edge",
    "build_secondary_public_score_frame",
]
