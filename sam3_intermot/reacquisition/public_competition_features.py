"""Explicit public-competition features for the N72R11R4 scorer.

These six values are derived from the current frozen legacy score matrix and
the explicit exact-solver result.  They contain no future labels and are kept
separate from the historical 530-D local candidate feature contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


PUBLIC_COMPETITION_FEATURE_DIM = 6
PUBLIC_COMPETITION_FEATURE_SCHEMA = (
    "tanh_legacy_target_score",
    "tanh_legacy_best_other_score",
    "tanh_target_minus_best_other",
    "incumbent_is_target",
    "incumbent_is_other",
    "base_solver_assigns_target",
)


def build_public_competition_feature(
    candidate: Mapping[str, Any],
    *,
    candidate_uid: str,
    target_public_id: int,
    legacy_target_score: float,
    legacy_best_other_score: float,
    base_target_uid: str | None,
) -> np.ndarray:
    """Build the fixed six-dimensional current-frame competition token."""

    target_score = float(legacy_target_score)
    best_other = float(legacy_best_other_score)
    if not np.isfinite(target_score) or not np.isfinite(best_other):
        raise ValueError("legacy public competition scores must be finite")
    incumbent = candidate.get("incumbent_public_id_if_any")
    values = np.asarray(
        [
            np.tanh(target_score),
            np.tanh(best_other),
            np.tanh(target_score - best_other),
            float(incumbent == int(target_public_id)),
            float(incumbent is not None and incumbent != int(target_public_id)),
            float(base_target_uid is not None and str(candidate_uid) == str(base_target_uid)),
        ],
        dtype=np.float32,
    )
    if values.shape != (PUBLIC_COMPETITION_FEATURE_DIM,) or not np.all(np.isfinite(values)):
        raise RuntimeError("public competition feature construction produced an invalid six-dimensional vector")
    return values


__all__ = [
    "PUBLIC_COMPETITION_FEATURE_DIM",
    "PUBLIC_COMPETITION_FEATURE_SCHEMA",
    "build_public_competition_feature",
]
