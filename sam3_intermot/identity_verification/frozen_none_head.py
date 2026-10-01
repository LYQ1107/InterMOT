"""Small NONE calibration head for the frozen R3R2 identity metric."""

from __future__ import annotations

import torch
from torch import Tensor, nn


NONE_FEATURE_DIMENSION = 9


class FrozenMetricNoneHead(nn.Module):
    """Map nine runtime-safe metric/context features to a NONE score.

    Candidate/query representation towers are deliberately not part of this
    module.  The returned score is on the same raw cosine-score scale as the
    frozen adapter and is concatenated with candidate scores by the caller.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        if not 1 <= int(hidden_dim) <= 32:
            raise ValueError("hidden_dim must be in 1..32")
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(NONE_FEATURE_DIMENSION, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != NONE_FEATURE_DIMENSION:
            raise ValueError(f"NONE features must have shape [B,{NONE_FEATURE_DIMENSION}")
        if not torch.isfinite(features).all():
            raise ValueError("NONE features must be finite")
        return self.network(features).squeeze(-1)


__all__ = ["FrozenMetricNoneHead", "NONE_FEATURE_DIMENSION"]
