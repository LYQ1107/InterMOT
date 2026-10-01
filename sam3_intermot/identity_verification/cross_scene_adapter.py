"""Small generic metric adapter for N72R20R3R2.

The adapter maps frozen 512-D OSNet-derived identity observations and frozen
N72R18 causal states into a shared identity space.  It has no sequence, GT,
track-ID, solver, or public-ID inputs and remains a metric function at
inference time.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FEATURE_DIMENSION = 512
BOTTLENECK_DIMENSION = 128


class ResidualIdentityTower(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIMENSION, bottleneck_dim: int = BOTTLENECK_DIMENSION) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, bottleneck_dim)
        self.output_projection = nn.Linear(bottleneck_dim, feature_dim)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        transformed = self.output_projection(F.gelu(self.input_projection(self.layer_norm(value))))
        return F.normalize(residual + transformed, dim=-1, eps=1.0e-8)


class CrossSceneIdentityAdapter(nn.Module):
    """Dual-tower residual metric adapter with a fixed 0.07 temperature."""

    def __init__(self, feature_dim: int = FEATURE_DIMENSION, bottleneck_dim: int = BOTTLENECK_DIMENSION) -> None:
        super().__init__()
        self.query_tower = ResidualIdentityTower(feature_dim, bottleneck_dim)
        self.candidate_tower = ResidualIdentityTower(feature_dim, bottleneck_dim)
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.temperature = 0.07

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_query(self, query_state: Tensor) -> Tensor:
        return self.query_tower(query_state)

    def encode_candidates(self, candidates: Tensor) -> Tensor:
        return self.candidate_tower(candidates)

    def forward(self, query_state: Tensor, candidates: Tensor, candidate_mask: Tensor | None = None) -> dict[str, Tensor]:
        if query_state.ndim != 2 or candidates.ndim != 3:
            raise ValueError("query_state and candidates must have shapes [B,512] and [B,N,512]")
        if query_state.shape[0] != candidates.shape[0] or query_state.shape[-1] != self.feature_dim or candidates.shape[-1] != self.feature_dim:
            raise ValueError("query/candidate shape mismatch")
        if not torch.isfinite(query_state).all() or not torch.isfinite(candidates).all():
            raise ValueError("non-finite adapter input")
        if candidate_mask is None:
            candidate_mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        if candidate_mask.shape != candidates.shape[:2]:
            raise ValueError("candidate_mask shape mismatch")
        query_embedding = self.encode_query(query_state)
        candidate_embedding = self.encode_candidates(candidates)
        scores = torch.einsum("bd,bnd->bn", query_embedding, candidate_embedding)
        scores = scores.masked_fill(~candidate_mask, -1.0e9)
        return {"query_embedding": query_embedding, "candidate_embedding": candidate_embedding, "scores": scores}


__all__ = ["BOTTLENECK_DIMENSION", "CrossSceneIdentityAdapter", "FEATURE_DIMENSION", "ResidualIdentityTower"]
