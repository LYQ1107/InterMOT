"""Causal candidate selection and exact target-column interface for N72R11R3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


EDGE_MODE_BASE = "BASE"
EDGE_MODE_LEGACY_INJECTION = "LEGACY_INJECTION"
EDGE_MODE_BRIDGE = "CORRECTED_TARGET_EDGE_BRIDGE"
ADMISSION_SCORE = 0.50
ADMISSION_MARGIN = 0.20
INJECTION_SCALE = 1.0


def select_candidate_from_logits(
    candidate_logits: Sequence[float],
    none_logit: float,
    candidate_uids: Sequence[str],
) -> dict[str, Any]:
    """Select a candidate with the preregistered score/margin admission rule."""

    values = np.asarray(candidate_logits, dtype=np.float64).reshape(-1)
    if len(candidate_uids) != values.size or values.size < 1:
        raise ValueError("candidate logit/UID axes must be non-empty and aligned")
    none = float(none_logit)
    if not np.isfinite(values).all() or not np.isfinite(none):
        raise ValueError("candidate and NONE logits must be finite")
    scores = values - none
    order = sorted(range(values.size), key=lambda index: (-float(scores[index]), str(candidate_uids[index])))
    best_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else None
    best_score = float(scores[best_index])
    second_score = None if second_index is None else float(scores[second_index])
    margin = float(best_score - max(0.0, second_score or 0.0))
    accepted = bool(best_score >= ADMISSION_SCORE and margin >= ADMISSION_MARGIN)
    selected_uid = str(candidate_uids[best_index]) if accepted else None
    return {
        "selected_candidate_uid": selected_uid,
        "selected_index": best_index if accepted else len(candidate_uids),
        "selected_score": best_score if accepted else None,
        "best_candidate_uid": str(candidate_uids[best_index]),
        "best_candidate_index": best_index,
        "best_score": best_score,
        "second_candidate_uid": None if second_index is None else str(candidate_uids[second_index]),
        "second_candidate_index": second_index,
        "second_score": second_score,
        "best_minus_second_margin": margin,
        "none_logit": none,
        "candidate_scores": [float(value) for value in scores],
        "ranked_candidates": [
            {"candidate_uid": str(candidate_uids[index]), "index": int(index), "score": float(scores[index])}
            for index in order
        ],
        "admission_score_threshold": ADMISSION_SCORE,
        "admission_margin_threshold": ADMISSION_MARGIN,
        "accepted": accepted,
        "runtime_future_gt_used": False,
        "public_id_inference": False,
    }


def apply_legacy_injection(
    base_matrix: np.ndarray,
    *,
    target_column: int,
    candidate_uids: Sequence[str],
    selection: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    """Apply the frozen N72R9 target-column injection, leaving other edges intact."""

    matrix = np.asarray(base_matrix, dtype=np.float64).copy()
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("base score matrix must be finite 2-D")
    if len(candidate_uids) != matrix.shape[0] or not 0 <= int(target_column) < matrix.shape[1]:
        raise ValueError("legacy injection axes are invalid")
    selected_uid = selection.get("selected_candidate_uid")
    if selected_uid is None:
        return matrix, 0.0
    try:
        index = next(index for index, uid in enumerate(candidate_uids) if str(uid) == str(selected_uid))
    except StopIteration as exc:
        raise ValueError("selected candidate is absent from legacy injection axis") from exc
    selected_score = selection.get("selected_score")
    if selected_score is None or not np.isfinite(float(selected_score)):
        raise ValueError("selected score must be finite for legacy injection")
    delta = float(INJECTION_SCALE * max(float(selected_score), 0.0))
    matrix[index, int(target_column)] += delta
    return matrix, delta
__all__ = [
    "EDGE_MODE_BASE",
    "EDGE_MODE_LEGACY_INJECTION",
    "EDGE_MODE_BRIDGE",
    "ADMISSION_SCORE",
    "ADMISSION_MARGIN",
    "INJECTION_SCALE",
    "select_candidate_from_logits",
    "apply_legacy_injection",
]
