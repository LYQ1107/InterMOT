"""Correction-supervised recursive least-squares challenger (C-RLS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class CRLSConfig:
    feature_dim: int = 10
    ridge: float = 1.0
    residual_scale: float = 2.0
    forgetting: float = 1.0


@dataclass
class _RLSState:
    weight: np.ndarray
    precision: np.ndarray

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.weight.copy(), self.precision.copy()

    def restore(self, snapshot: tuple[np.ndarray, np.ndarray]) -> None:
        self.weight[...] = snapshot[0]
        self.precision[...] = snapshot[1]


class CorrectionRLS:
    """Identity-scoped ridge RLS with labels restricted to human events."""

    def __init__(self, config: Optional[CRLSConfig] = None) -> None:
        self.config = config or CRLSConfig()
        if self.config.feature_dim <= 0 or self.config.ridge <= 0:
            raise ValueError("feature_dim and ridge must be positive")
        self._states: dict[Any, _RLSState] = {}

    def ensure_identity(self, identity_id: Any) -> _RLSState:
        if identity_id not in self._states:
            dim = self.config.feature_dim
            self._states[identity_id] = _RLSState(
                weight=np.zeros(dim, dtype=np.float64),
                precision=np.eye(dim, dtype=np.float64) / self.config.ridge,
            )
        return self._states[identity_id]

    def _features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.config.feature_dim}, got {values.shape[-1]}"
            )
        return values

    def delta(self, identity_id: Any, features: np.ndarray) -> np.ndarray:
        values = self._features(features)
        state = self.ensure_identity(identity_id)
        if not np.any(state.weight):
            return np.zeros(values.shape[:-1], dtype=np.float32)
        return (values @ state.weight * self.config.residual_scale).astype(np.float32)

    def delta_batch(self, features: np.ndarray, identity_ids: list[Any]) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 3 or len(identity_ids) != values.shape[0]:
            raise ValueError("features must be [identity, candidate, dimension]")
        return np.stack(
            [self.delta(identity_id, values[index]) for index, identity_id in enumerate(identity_ids)]
        ) if len(identity_ids) else np.zeros(values.shape[:2], dtype=np.float32)

    def update(self, identity_id: Any, features: np.ndarray, labels: np.ndarray) -> None:
        """Apply sequential Sherman–Morrison updates for one legal event."""
        x = self._features(features).reshape(-1, self.config.feature_dim)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if len(x) != len(y):
            raise ValueError("features and labels must have the same number of rows")
        state = self.ensure_identity(identity_id)
        forgetting = float(self.config.forgetting)
        if not (0.0 < forgetting <= 1.0):
            raise ValueError("forgetting must be in (0, 1]")
        for row, label in zip(x, y):
            precision_row = state.precision @ row
            denominator = forgetting + float(row @ precision_row)
            gain = precision_row / max(denominator, 1e-12)
            residual = float(label - row @ state.weight)
            state.weight += gain * residual
            state.precision = (state.precision - np.outer(gain, row @ state.precision)) / forgetting
            state.precision = 0.5 * (state.precision + state.precision.T)
        if not np.isfinite(state.weight).all() or not np.isfinite(state.precision).all():
            raise FloatingPointError("non-finite C-RLS state")

    def snapshot(self, identity_ids: Iterable[Any]) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
        return {identity_id: self.ensure_identity(identity_id).snapshot() for identity_id in identity_ids}

    def restore(self, snapshot: dict[Any, tuple[np.ndarray, np.ndarray]]) -> None:
        for identity_id, values in snapshot.items():
            self.ensure_identity(identity_id).restore(values)

    def reset(self, identity_ids: Optional[Iterable[Any]] = None) -> None:
        ids = list(self._states) if identity_ids is None else list(identity_ids)
        for identity_id in ids:
            state = self.ensure_identity(identity_id)
            state.weight.fill(0.0)
            state.precision[...] = np.eye(self.config.feature_dim) / self.config.ridge

    def state_norm(self, identity_ids: Iterable[Any]) -> float:
        values = [np.sum(self.ensure_identity(identity_id).weight ** 2) for identity_id in identity_ids]
        return float(np.sqrt(np.sum(values))) if values else 0.0
