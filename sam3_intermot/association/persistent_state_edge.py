"""Candidate-by-public persistent-state edge matrix for N72R14."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .online_associator import predicted_iou
from .persistent_public_state import PersistentPublicAssociationBank, PublicAssociationMotionState


STATE_EDGE_SCALE = 1.0


def _raw(candidate: Mapping[str, Any]) -> int | None:
    for key in (
        "official_raw_sam_id",
        "raw_sam_id",
        "raw_native_id",
        "native_tid",
        "adapter_external_id",
    ):
        value = candidate.get(key)
        if value is not None:
            return int(value)
    return None


def _scope(candidate: Mapping[str, Any]) -> str | None:
    value = candidate.get("native_scope", candidate.get("native_tid_scope"))
    return None if value in (None, "") else str(value)


def _box(candidate: Mapping[str, Any]) -> np.ndarray:
    value = candidate.get("box_xyxy", candidate.get("box"))
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != 4 or not np.isfinite(result).all():
        raise ValueError("candidate box is not finite xyxy")
    return result


def _feature(candidate: Mapping[str, Any]) -> np.ndarray | None:
    value = candidate.get("feature", candidate.get("embedding"))
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size == 0:
        return None
    if result.size != 512 or not np.isfinite(result).all():
        raise ValueError("candidate feature is not finite 512-D")
    if float(np.linalg.norm(result)) <= 1.0e-6:
        return None
    return result


def _native_continuity(state: PublicAssociationMotionState | None, candidate: Mapping[str, Any]) -> float:
    """Require both raw ID and native scope; bare numeric ID is insufficient."""

    if state is None:
        return 0.0
    candidate_raw, candidate_scope = _raw(candidate), _scope(candidate)
    if state.previous_raw_sam_id is None or state.previous_native_scope is None:
        return 0.0
    if candidate_raw is None or candidate_scope is None:
        return 0.0
    return float(
        int(candidate_raw) == int(state.previous_raw_sam_id)
        and str(candidate_scope) == str(state.previous_native_scope)
    )


def build_persistent_state_edge_matrix(
    *,
    bank: PersistentPublicAssociationBank,
    candidate_rows: Sequence[Mapping[str, Any]],
    public_id_axis: Sequence[int],
    frame: int,
) -> dict[str, Any]:
    """Build a complete candidate × public-ID matrix and component audit.

    The formula is intentionally fixed and untrained::

        raw = appearance + 1.0 * motion + 0.5 * native_continuity - 0.1 * gap
        delta = tanh(STATE_EDGE_SCALE * raw)

    Appearance is evaluated through :class:`AppearanceMemory`; a missing
    feature contributes zero and is recorded as unavailable rather than being
    replaced with a zero feature vector.
    """

    publics = [int(value) for value in public_id_axis]
    if len(publics) != len(set(publics)):
        raise ValueError("public_id_axis contains duplicates")
    uids = [str(row.get("candidate_uid")) for row in candidate_rows]
    if any(uid in {"None", ""} for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("candidate rows contain missing/duplicate candidate_uid")
    frame = int(frame)
    n_rows, n_public = len(candidate_rows), len(publics)
    appearance = np.zeros((n_rows, n_public), dtype=np.float64)
    motion = np.zeros_like(appearance)
    native = np.zeros_like(appearance)
    gap = np.zeros_like(appearance)
    raw = np.zeros_like(appearance)
    delta = np.zeros_like(appearance)
    feature_available = np.zeros(n_rows, dtype=bool)
    state_available = np.zeros((n_rows, n_public), dtype=bool)
    for row_index, candidate in enumerate(candidate_rows):
        candidate_feature = _feature(candidate)
        feature_available[row_index] = candidate_feature is not None
        candidate_box = _box(candidate)
        for public_index, public in enumerate(publics):
            state = bank.motion_states.get(public)
            state_available[row_index, public_index] = state is not None
            if candidate_feature is not None:
                appearance[row_index, public_index] = float(
                    bank.appearance_memory.score(public, candidate_feature, frame)
                )
            if state is not None:
                motion[row_index, public_index] = float(predicted_iou(state, candidate_box, frame))
                native[row_index, public_index] = _native_continuity(state, candidate)
                if state.last_seen_frame is None:
                    gap[row_index, public_index] = 1.0
                else:
                    gap[row_index, public_index] = min(
                        1.0,
                        max(0, frame - int(state.last_seen_frame)) / 200.0,
                    )
            raw[row_index, public_index] = (
                appearance[row_index, public_index]
                + motion[row_index, public_index]
                + 0.5 * native[row_index, public_index]
                - 0.1 * gap[row_index, public_index]
            )
            delta[row_index, public_index] = float(np.tanh(STATE_EDGE_SCALE * raw[row_index, public_index]))
    matrices = {
        "appearance": appearance,
        "motion": motion,
        "native_continuity": native,
        "gap": gap,
        "raw": raw,
        "delta": delta,
    }
    if not all(np.isfinite(value).all() for value in matrices.values()):
        raise RuntimeError("persistent state edge matrix contains non-finite values")
    return {
        "schema_version": "N72R14_PERSISTENT_STATE_EDGE_V1",
        "frame": frame,
        "candidate_uids": uids,
        "public_id_axis": publics,
        "shape": [n_rows, n_public],
        "state_edge_scale": float(STATE_EDGE_SCALE),
        "feature_available_by_candidate": feature_available.tolist(),
        "state_available": state_available.tolist(),
        "appearance": appearance.tolist(),
        "motion": motion.tolist(),
        "native_continuity": native.tolist(),
        "gap": gap.tolist(),
        "raw": raw.tolist(),
        "delta": delta.tolist(),
        "runtime_future_gt_used": False,
    }


__all__ = ["STATE_EDGE_SCALE", "build_persistent_state_edge_matrix"]
