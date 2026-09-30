"""Candidate-public residual edge for the frozen N72R20 solver."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .learned_identity_memory import LearnedIdentityMemoryBank
from .relative_persistent_state_edge import fuse_row_max_preserving


STATE_EDGE_SCALE = 1.0


def build_learned_identity_state_edge_matrix(
    *,
    bank: LearnedIdentityMemoryBank,
    candidate_rows: Sequence[Mapping[str, Any]],
    public_id_axis: Sequence[int],
    frame: int,
    state_scope: str = "HUMAN_INITIALIZED_ONLY",
    base_candidate_public_scores: Sequence[Sequence[float]] | None = None,
    state_edge_scale: float = STATE_EDGE_SCALE,
) -> dict[str, Any]:
    """Build learned relative evidence and optionally fuse it into base scores.

    The returned score matrix is candidate-row × public-ID.  Centering is
    performed on the learned signal only; the frozen base matrix is passed to
    the existing row-max-preserving fusion function.  Thus the maximum public
    score for every candidate row, and consequently its competition with the
    explicit NONE columns, is unchanged.
    """

    if state_scope not in {"HUMAN_INITIALIZED_ONLY", "ALL_INITIALIZED"}:
        raise ValueError(f"unknown state_scope {state_scope}")
    score = bank.score_matrix(
        candidate_rows=candidate_rows,
        public_id_axis=public_id_axis,
        frame=int(frame),
    )
    state_candidate = np.asarray(score["state_candidate_scores"], dtype=np.float64)
    if state_scope == "HUMAN_INITIALIZED_ONLY":
        available = np.asarray(score["state_available_by_public"], dtype=bool)
        state_candidate = state_candidate * available[:, None]
    candidate_public = state_candidate.T
    if candidate_public.size:
        row_centered = candidate_public - np.mean(candidate_public, axis=1, keepdims=True)
        column_centered = candidate_public - np.mean(candidate_public, axis=0, keepdims=True)
    else:
        row_centered = candidate_public.copy()
        column_centered = candidate_public.copy()
    relative = 0.5 * (row_centered + column_centered)
    relative_delta = np.tanh(relative)
    fused = None
    fusion = None
    if base_candidate_public_scores is not None:
        base = np.asarray(base_candidate_public_scores, dtype=np.float64)
        if base.shape != relative_delta.shape:
            raise ValueError(f"base/learned score shape mismatch: {base.shape} vs {relative_delta.shape}")
        fusion = fuse_row_max_preserving(base, relative_delta, state_edge_scale=float(state_edge_scale))
        fused = np.asarray(fusion["fused"], dtype=np.float64)
    arrays = (state_candidate, candidate_public, row_centered, column_centered, relative, relative_delta)
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("learned identity edge contains non-finite values")
    return {
        "schema_version": "N72R20_LEARNED_IDENTITY_STATE_EDGE_V1",
        "frame": int(frame),
        "candidate_uids": [str(row["candidate_uid"]) for row in candidate_rows],
        "public_id_axis": [int(value) for value in public_id_axis],
        "shape": [int(candidate_public.shape[0]), int(candidate_public.shape[1])],
        "state_scope": state_scope,
        "state_candidate_scores": state_candidate.tolist(),
        "candidate_public_scores": candidate_public.tolist(),
        "row_centered": row_centered.tolist(),
        "column_centered": column_centered.tolist(),
        "relative": relative.tolist(),
        "relative_delta": relative_delta.tolist(),
        "state_edge_scale": float(state_edge_scale),
        "fused_candidate_public_scores": None if fused is None else fused.tolist(),
        "row_max_before": None if fusion is None else np.asarray(fusion["row_max_before"]).tolist(),
        "row_max_after": None if fusion is None else np.asarray(fusion["row_max_after"]).tolist(),
        "row_max_preserved": None if fusion is None else bool(fusion["row_max_preserved"]),
        "runtime_future_gt_used": False,
    }


__all__ = ["STATE_EDGE_SCALE", "build_learned_identity_state_edge_matrix"]
