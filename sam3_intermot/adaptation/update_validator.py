"""Exactness and safety checks for N28 live challenger updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class UpdateValidatorConfig:
    support_margin: float = 0.01
    protected_drop: float = 0.0
    max_update_norm: float = 100.0


@dataclass(frozen=True)
class UpdateValidation:
    accepted: bool
    finite: bool
    support_min_margin: float
    protected_max_drop: float
    update_norm: float
    reasons: tuple[str, ...]


class UpdateValidator:
    """Validate a prospective checkpoint before it can become live state."""

    def __init__(self, config: Optional[UpdateValidatorConfig] = None) -> None:
        self.config = config or UpdateValidatorConfig()

    def validate(
        self,
        *,
        before_scores: Mapping[Any, np.ndarray],
        after_scores: Mapping[Any, np.ndarray],
        support_pairs: Iterable[tuple[Any, int, Optional[int]]],
        protected_pairs: Iterable[tuple[Any, int]] = (),
        update_norm: float = 0.0,
    ) -> UpdateValidation:
        reasons: list[str] = []
        finite = True
        for values in list(before_scores.values()) + list(after_scores.values()):
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                finite = False
                break
        if not finite:
            reasons.append("non_finite_score")

        margins: list[float] = []
        for identity_id, positive, rejected in support_pairs:
            scores = np.asarray(after_scores[identity_id], dtype=float).reshape(-1)
            positive = int(positive)
            if positive < 0 or positive >= len(scores):
                reasons.append(f"support_positive_out_of_range:{identity_id}")
                continue
            if rejected is None:
                alternatives = np.delete(scores, positive)
                margin = float(scores[positive] - np.max(alternatives)) if len(alternatives) else float("inf")
            else:
                rejected = int(rejected)
                if rejected < 0 or rejected >= len(scores):
                    reasons.append(f"support_rejected_out_of_range:{identity_id}")
                    continue
                margin = float(scores[positive] - scores[rejected])
            margins.append(margin)
            if margin < self.config.support_margin:
                reasons.append(f"support_margin:{identity_id}:{margin:.8f}")

        drops: list[float] = []
        for identity_id, column in protected_pairs:
            before = np.asarray(before_scores[identity_id], dtype=float).reshape(-1)
            after = np.asarray(after_scores[identity_id], dtype=float).reshape(-1)
            column = int(column)
            if column < 0 or column >= len(before) or column >= len(after):
                reasons.append(f"protected_out_of_range:{identity_id}")
                continue
            drop = float(before[column] - after[column])
            drops.append(drop)
            if drop > self.config.protected_drop:
                reasons.append(f"protected_drop:{identity_id}:{drop:.8f}")

        update_norm = float(update_norm)
        if not np.isfinite(update_norm):
            reasons.append("non_finite_update_norm")
        elif update_norm > self.config.max_update_norm:
            reasons.append(f"trust_region:{update_norm:.8f}")

        return UpdateValidation(
            accepted=not reasons,
            finite=finite,
            support_min_margin=min(margins) if margins else float("nan"),
            protected_max_drop=max(drops) if drops else 0.0,
            update_norm=update_norm,
            reasons=tuple(reasons),
        )


def exact_zero(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Return true only for elementwise exact equality, not an epsilon test."""
    return np.array_equal(np.asarray(reference), np.asarray(candidate))
