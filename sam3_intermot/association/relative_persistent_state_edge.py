"""Relative, row-maximum-preserving persistent-state edge for N72R15.

The edge changes candidate-to-public identity competition while preserving the
maximum score in each candidate row.  Since the exact public solver has an
explicit NONE column with score zero, this prevents the state residual from
silently changing the candidate-vs-NONE evidence axis.  The solver itself is
still :func:`solve_effect_assignment`; this module only builds its matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .trusted_persistent_public_state import (
    TrustedPersistentPublicAssociationBank,
    TrustedPublicState,
    _feature,
    _raw,
    _scope,
)


STATE_EDGE_SCALE = 1.0
EPSILON = 1.0e-10


def _box(candidate: Mapping[str, Any]) -> np.ndarray:
    value = candidate.get("box_xyxy", candidate.get("box"))
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != 4 or not np.isfinite(result).all():
        raise ValueError("candidate box is not finite xyxy")
    return result


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _predicted_iou(state: TrustedPublicState, candidate_box: Sequence[float], frame: int) -> float:
    if state.trusted_last_box is None or state.trusted_last_seen_frame is None:
        return 0.0
    gap = max(0, int(frame) - int(state.trusted_last_seen_frame))
    predicted = np.asarray(state.trusted_last_box, dtype=np.float64).copy()
    if gap > 0:
        predicted[[0, 2]] += float(state.trusted_velocity[0]) * gap
        predicted[[1, 3]] += float(state.trusted_velocity[1]) * gap
    return float(_box_iou(predicted, candidate_box))


def _native_continuity(state: TrustedPublicState, candidate: Mapping[str, Any]) -> float:
    candidate_raw, candidate_scope = _raw(candidate), _scope(candidate)
    if state.trusted_raw_sam_id is None or state.trusted_native_scope is None:
        return 0.0
    if candidate_raw is None or candidate_scope is None:
        return 0.0
    return float(
        int(candidate_raw) == int(state.trusted_raw_sam_id)
        and str(candidate_scope) == str(state.trusted_native_scope)
    )


def _zero(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def fuse_row_max_preserving(
    base_matrix: Sequence[Sequence[float]],
    relative_delta: Sequence[Sequence[float]],
    *,
    state_edge_scale: float = STATE_EDGE_SCALE,
) -> dict[str, Any]:
    """Fuse the relative residual while preserving every row maximum."""

    base = np.asarray(base_matrix, dtype=np.float64)
    delta = np.asarray(relative_delta, dtype=np.float64)
    if base.ndim != 2 or delta.shape != base.shape:
        raise ValueError(f"base/delta shape mismatch: {base.shape} vs {delta.shape}")
    if not np.isfinite(base).all() or not np.isfinite(delta).all() or not np.isfinite(float(state_edge_scale)):
        raise ValueError("base/delta/scale must be finite")
    n_rows = int(base.shape[0])
    if base.shape[1] == 0:
        return {
            "proposal": base.copy(),
            "fused": base.copy(),
            "row_max_before": [],
            "row_max_after": [],
            "row_shift": [],
            "row_max_preserved": True,
            "state_edge_scale": float(state_edge_scale),
        }
    proposal = base + float(state_edge_scale) * delta
    before = np.max(base, axis=1)
    proposal_max = np.max(proposal, axis=1)
    shift = before - proposal_max
    fused = proposal + shift[:, None]
    after = np.max(fused, axis=1)
    preserved = bool(np.allclose(before, after, atol=1.0e-10, rtol=0.0))
    if not preserved:
        raise RuntimeError("row maximum preservation failed")
    return {
        "proposal": proposal,
        "fused": fused,
        "row_max_before": before,
        "row_max_after": after,
        "row_shift": shift,
        "row_max_preserved": preserved,
        "state_edge_scale": float(state_edge_scale),
    }


def build_relative_persistent_state_edge_matrix(
    *,
    bank: TrustedPersistentPublicAssociationBank,
    candidate_rows: Sequence[Mapping[str, Any]],
    public_id_axis: Sequence[int],
    frame: int,
    target_public_id: int | None = None,
    state_scope: str = "TRUSTED_GLOBAL",
) -> dict[str, Any]:
    """Build raw and centered evidence for H1 or G2."""

    publics = [int(value) for value in public_id_axis]
    if len(publics) != len(set(publics)):
        raise ValueError("public_id_axis contains duplicates")
    uids = [str(row.get("candidate_uid")) for row in candidate_rows]
    if any(uid in ("None", "") for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("candidate rows contain missing/duplicate candidate_uid")
    if state_scope not in {"HUMAN_TARGET_ONLY", "TRUSTED_GLOBAL"}:
        raise ValueError(f"unknown state scope: {state_scope}")
    if state_scope == "HUMAN_TARGET_ONLY" and target_public_id is None:
        raise ValueError("HUMAN_TARGET_ONLY requires target_public_id")
    frame = int(frame)
    shape = (len(candidate_rows), len(publics))
    appearance = _zero(shape)
    appearance_prototype = _zero(shape)
    appearance_positive = _zero(shape)
    appearance_negative = _zero(shape)
    motion = _zero(shape)
    native = _zero(shape)
    gap = _zero(shape)
    raw = _zero(shape)
    state_available = np.zeros(shape, dtype=bool)
    feature_available = np.zeros(len(candidate_rows), dtype=bool)
    evidence_available = np.zeros(shape, dtype=bool)
    for row_index, candidate in enumerate(candidate_rows):
        candidate_feature = _feature(candidate.get("feature", candidate.get("embedding")), "candidate feature")
        feature_available[row_index] = candidate_feature is not None
        candidate_box = _box(candidate)
        for public_index, public in enumerate(publics):
            state = bank.motion_states.get(public)
            state_available[row_index, public_index] = state is not None
            enabled = state_scope == "TRUSTED_GLOBAL" or public == int(target_public_id)
            if state is None or not enabled:
                continue
            evidence_available[row_index, public_index] = True
            if candidate_feature is not None:
                components = bank.appearance_memory._score_components(
                    public, candidate_feature, frame
                )
                appearance_prototype[row_index, public_index] = float(components["prototype"])
                appearance_positive[row_index, public_index] = float(components["positive"])
                appearance_negative[row_index, public_index] = float(components["negative"])
                appearance[row_index, public_index] = float(components["total"])
            motion[row_index, public_index] = _predicted_iou(state, candidate_box, frame)
            native[row_index, public_index] = _native_continuity(state, candidate)
            if state.trusted_last_seen_frame is None:
                gap[row_index, public_index] = 1.0
            else:
                gap[row_index, public_index] = min(
                    1.0,
                    max(0, frame - int(state.trusted_last_seen_frame)) / 200.0,
                )
            raw[row_index, public_index] = (
                appearance[row_index, public_index]
                + motion[row_index, public_index]
                + 0.5 * native[row_index, public_index]
                - 0.1 * gap[row_index, public_index]
            )

    row_centered = raw - (np.mean(raw, axis=1, keepdims=True) if raw.size else raw)
    column_centered = raw - (np.mean(raw, axis=0, keepdims=True) if raw.size else raw)
    relative = 0.5 * (row_centered + column_centered)
    relative_delta = np.tanh(relative)
    arrays = (
        appearance,
        appearance_prototype,
        appearance_positive,
        appearance_negative,
        motion,
        native,
        gap,
        raw,
        row_centered,
        column_centered,
        relative,
        relative_delta,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("relative persistent state evidence contains non-finite values")
    return {
        "schema_version": "N72R15_RELATIVE_PERSISTENT_STATE_EDGE_V1",
        "frame": frame,
        "candidate_uids": uids,
        "public_id_axis": publics,
        "shape": [shape[0], shape[1]],
        "state_scope": state_scope,
        "target_public_id": None if target_public_id is None else int(target_public_id),
        "state_edge_scale": float(STATE_EDGE_SCALE),
        "feature_available_by_candidate": feature_available.tolist(),
        "state_available": state_available.tolist(),
        "evidence_available": evidence_available.tolist(),
        "appearance": appearance.tolist(),
        "appearance_prototype": appearance_prototype.tolist(),
        "appearance_positive": appearance_positive.tolist(),
        "appearance_negative": appearance_negative.tolist(),
        "motion": motion.tolist(),
        "native_continuity": native.tolist(),
        "gap": gap.tolist(),
        "raw": raw.tolist(),
        "row_centered": row_centered.tolist(),
        "column_centered": column_centered.tolist(),
        "relative": relative.tolist(),
        "relative_delta": relative_delta.tolist(),
        "runtime_future_gt_used": False,
    }


__all__ = [
    "EPSILON",
    "STATE_EDGE_SCALE",
    "build_relative_persistent_state_edge_matrix",
    "fuse_row_max_preserving",
]
