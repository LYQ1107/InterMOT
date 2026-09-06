"""Research-only target-column bridge for N72R11.

This module does not replace either exact public-ID solver.  It produces a
calibrated score for the target public column for every candidate; all other
public columns and the exact ``NONE`` score remain untouched.  The feature
builder is public-ID aware only through the explicit target axis supplied by
the caller and never infers authority from a candidate row or from GT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import torch
from torch import nn


BRIDGE_INPUT_DIM = 14
SOURCE_NAMES = (
    "MAIN_B0_CANDIDATE",
    "TARGET_SESSION_CURRENT_RAW",
    "STATIC_EVENT_REQUERY",
    "FUTURE_FRAME_REQUERY",
    "UNKNOWN",
)
SOURCE_FEATURE_DIM = len(SOURCE_NAMES)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _candidate_source(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_source", candidate.get("source_kind", "UNKNOWN"))
    return str(value) if str(value) in SOURCE_NAMES else "UNKNOWN"


def _numeric_candidate_value(candidate: Mapping[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = candidate.get(key)
        if value is not None:
            return _finite(value, key)
    return float(default)


def build_target_edge_feature(
    candidate: Mapping[str, Any],
    *,
    candidate_logit: float,
    none_logit: float,
    legacy_target_score: float,
    legacy_public_scores: Sequence[float] | Mapping[int, float],
    target_public_id: int,
) -> list[float]:
    """Build the frozen 14-D candidate feature without posthoc fields."""

    candidate_logit = _finite(candidate_logit, "candidate_logit")
    none_logit = _finite(none_logit, "none_logit")
    target_score = _finite(legacy_target_score, "legacy_target_score")
    if isinstance(legacy_public_scores, Mapping):
        other_values = [
            _finite(value, f"legacy_public_scores[{key}]")
            for key, value in legacy_public_scores.items()
            if int(key) != int(target_public_id)
        ]
    else:
        other_values = [_finite(value, "legacy_public_scores") for value in legacy_public_scores]
    best_other = max(other_values, default=0.0)
    incumbent = candidate.get("incumbent_public_id_if_any")
    incumbent_target = float(incumbent is not None and int(incumbent) == int(target_public_id))
    incumbent_other = float(incumbent is not None and int(incumbent) != int(target_public_id))
    source_one_hot = [float(_candidate_source(candidate) == name) for name in SOURCE_NAMES]
    feature = [
        candidate_logit - none_logit,
        target_score,
        best_other,
        target_score - best_other,
        incumbent_target,
        incumbent_other,
        _numeric_candidate_value(candidate, ("confidence", "score")),
        _numeric_candidate_value(candidate, ("presence_score", "confidence")),
        _numeric_candidate_value(candidate, ("motion_iou", "motion_iou_to_incumbent", "raw_continuity")),
        *source_one_hot,
    ]
    if len(feature) != BRIDGE_INPUT_DIM or not np.isfinite(np.asarray(feature, dtype=np.float64)).all():
        raise RuntimeError("target-edge bridge feature construction produced an invalid 14-D vector")
    return feature


def fit_residual_scale(legacy_target: Sequence[float], legacy_best_other: Sequence[float]) -> float:
    """Fit the train-only residual scale required by the N72R11 protocol."""

    target = np.asarray(legacy_target, dtype=np.float64).reshape(-1)
    other = np.asarray(legacy_best_other, dtype=np.float64).reshape(-1)
    if target.size == 0 or target.shape != other.shape or not np.isfinite(target).all() or not np.isfinite(other).all():
        raise ValueError("train legacy target/other arrays must be finite and equally sized")
    # The protocol uses the train-only p95 legacy target-vs-best-other scale,
    # with a one-unit floor.  Do not cap a larger observed scale: doing so
    # would silently change the preregistered residual parameterization.
    return float(max(np.percentile(np.abs(target - other), 95.0), 1.0))


class TargetEdgeBridge(nn.Module):
    """Small bounded residual bridge applied only to the target edge."""

    def __init__(self, *, residual_scale: float) -> None:
        super().__init__()
        scale = _finite(residual_scale, "residual_scale")
        if scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = float(scale)
        self.network = nn.Sequential(
            nn.Linear(BRIDGE_INPUT_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        legacy_target_edge: torch.Tensor | Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != BRIDGE_INPUT_DIM:
            raise ValueError(f"bridge features must end in {BRIDGE_INPUT_DIM}, got {tuple(features.shape)}")
        legacy = torch.as_tensor(legacy_target_edge, dtype=features.dtype, device=features.device)
        raw = self.network(features).squeeze(-1)
        delta = float(self.residual_scale) * torch.tanh(raw)
        calibrated = legacy + delta
        return delta, calibrated

    @torch.no_grad()
    def audit_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "N72R11_TARGET_EDGE_BRIDGE_V1",
            "input_dim": BRIDGE_INPUT_DIM,
            "source_names": list(SOURCE_NAMES),
            "architecture": ["Linear(14,64)", "GELU", "Linear(64,64)", "GELU", "Linear(64,1)"],
            "residual": "residual_scale*tanh(raw)",
            "residual_scale": float(self.residual_scale),
            "target_column_only": True,
            "other_public_columns_untouched": True,
            "none_score_untouched": True,
        }


__all__ = [
    "BRIDGE_INPUT_DIM",
    "SOURCE_NAMES",
    "TargetEdgeBridge",
    "build_target_edge_feature",
    "fit_residual_scale",
]
