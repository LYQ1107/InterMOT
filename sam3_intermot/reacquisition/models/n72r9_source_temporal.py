"""Independent N72R9 source-aware temporal identity model.

The module is intentionally additive and research-only.  It does not alter
SAM3, candidate generation, the frozen public-ID solver, or production
association code.  Candidate tokens carry an explicit source one-hot vector;
the context consumes causal trusted and distractor memory, a neighbor summary,
and temporal uncertainty features.  Runtime callers must supply only sealed
candidate observations and causal state.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CANDIDATE_FEATURE_DIM = 530
SOURCE_FEATURE_DIM = 4
TEMPORAL_FEATURE_DIM = 8
MEMORY_FEATURE_DIM = 512
TRUSTED_MEMORY_SLOTS = 4
DISTRACTOR_MEMORY_SLOTS = 4


class N72R9SourceAwareTemporalIdentityModel(nn.Module):
    """Set scorer with explicit source and causal memory channels."""

    def __init__(
        self,
        *,
        candidate_feature_dim: int = CANDIDATE_FEATURE_DIM,
        source_feature_dim: int = SOURCE_FEATURE_DIM,
        temporal_feature_dim: int = TEMPORAL_FEATURE_DIM,
        trusted_slots: int = TRUSTED_MEMORY_SLOTS,
        distractor_slots: int = DISTRACTOR_MEMORY_SLOTS,
        hidden_dim: int = 96,
        layers: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if layers < 1 or trusted_slots < 1 or distractor_slots < 1:
            raise ValueError("layers and memory slots must be positive")
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.source_feature_dim = int(source_feature_dim)
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.trusted_slots = int(trusted_slots)
        self.distractor_slots = int(distractor_slots)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.candidate_projection = nn.Sequential(
            nn.Linear(self.candidate_feature_dim + self.source_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(MEMORY_FEATURE_DIM * 4 + self.temporal_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.none_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        weights = mask.unsqueeze(-1).to(dtype=values.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denominator

    @staticmethod
    def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
        result = masked.max(dim=1).values
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            result = result.clone()
            result[empty] = 0.0
        return result

    def forward(
        self,
        candidate_features: Tensor,
        candidate_mask: Tensor,
        source_features: Tensor,
        trusted_memory: Tensor,
        trusted_mask: Tensor,
        distractor_memory: Tensor,
        distractor_mask: Tensor,
        neighbor_feature: Tensor,
        temporal_features: Tensor,
    ) -> Tensor:
        if candidate_features.ndim != 3 or candidate_mask.ndim != 2:
            raise ValueError("candidate_features/mask must be [batch,candidates,...]")
        if candidate_features.shape[:2] != candidate_mask.shape:
            raise ValueError("candidate mask does not align with candidates")
        if source_features.shape[:2] != candidate_features.shape[:2] or source_features.shape[-1] != self.source_feature_dim:
            raise ValueError("source feature shape does not align with candidates")
        batch = candidate_features.shape[0]
        if candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature width mismatch")
        if trusted_memory.shape != (batch, self.trusted_slots, MEMORY_FEATURE_DIM):
            raise ValueError("trusted memory shape mismatch")
        if distractor_memory.shape != (batch, self.distractor_slots, MEMORY_FEATURE_DIM):
            raise ValueError("distractor memory shape mismatch")
        if trusted_mask.shape != (batch, self.trusted_slots) or distractor_mask.shape != (batch, self.distractor_slots):
            raise ValueError("memory mask shape mismatch")
        if neighbor_feature.shape != (batch, MEMORY_FEATURE_DIM):
            raise ValueError("neighbor feature shape mismatch")
        if temporal_features.shape != (batch, self.temporal_feature_dim):
            raise ValueError("temporal feature shape mismatch")
        if not all(bool(torch.isfinite(value).all()) for value in (
            candidate_features, source_features, trusted_memory, distractor_memory,
            neighbor_feature, temporal_features,
        )):
            raise ValueError("N72R9 model inputs must be finite")
        candidate_tokens = self.candidate_projection(torch.cat([candidate_features, source_features], dim=-1))
        valid = candidate_mask.to(dtype=torch.bool)
        if candidate_tokens.shape[1] < 1:
            raise ValueError("candidate dimension must be positive")
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if bool(empty.any()):
            safe_valid[empty, 0] = True
            candidate_tokens = candidate_tokens.clone()
            candidate_tokens[empty, 0] = 0.0
        encoded = self.set_encoder(candidate_tokens, src_key_padding_mask=~safe_valid)
        trusted_mean = self._masked_mean(trusted_memory, trusted_mask)
        trusted_max = self._masked_max(trusted_memory, trusted_mask)
        distractor_mean = self._masked_mean(distractor_memory, distractor_mask)
        context_values = torch.cat(
            [trusted_mean, trusted_max, distractor_mean, neighbor_feature, temporal_features], dim=-1
        )
        context = self.context_projection(context_values)
        combined = torch.cat([encoded, context.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1)
        logits = self.candidate_head(combined).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        none_logit = self.none_head(context)
        return torch.cat([logits, none_logit], dim=1)


def n72r9_loss(
    logits: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
    *,
    pairwise_weight: float = 0.15,
    pairwise_margin: float = 0.20,
    example_weight: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Cross entropy plus a fixed hard-negative ranking term."""

    if logits.ndim != 2 or labels.ndim != 1 or candidate_mask.ndim != 2:
        raise ValueError("invalid N72R9 loss shapes")
    candidates = candidate_mask.shape[1]
    if logits.shape != (labels.shape[0], candidates + 1):
        raise ValueError("N72R9 logit/label shape mismatch")
    labels = labels.to(dtype=torch.long)
    if bool((labels < 0).any()) or bool((labels > candidates).any()):
        raise ValueError("N72R9 label outside candidate/NONE range")
    if example_weight is None:
        weights = torch.ones(labels.shape[0], dtype=logits.dtype, device=logits.device)
    else:
        weights = example_weight.to(device=logits.device, dtype=logits.dtype)
        if weights.shape != labels.shape or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("N72R9 example weights must be finite and positive")
    ce_values = F.cross_entropy(logits, labels, reduction="none")
    ce = (ce_values * weights).sum() / weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    positive = labels < candidates
    pairwise = logits.new_zeros(())
    if bool(positive.any()):
        target = labels[positive]
        values = logits[positive, :candidates]
        other_mask = candidate_mask[positive].clone().to(dtype=torch.bool)
        other_mask.scatter_(1, target.unsqueeze(1), False)
        other = values.masked_fill(~other_mask, torch.finfo(values.dtype).min).max(dim=1).values
        valid_negative = other_mask.any(dim=1)
        if bool(valid_negative.any()):
            target_value = values[torch.arange(len(target), device=values.device), target]
            pair_values = F.relu(pairwise_margin - target_value[valid_negative] + other[valid_negative])
            pair_weights = weights[positive][valid_negative]
            pairwise = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    total = ce + float(pairwise_weight) * pairwise
    return total, {"cross_entropy": ce.detach(), "hard_negative_ranking": pairwise.detach()}


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "SOURCE_FEATURE_DIM",
    "TEMPORAL_FEATURE_DIM",
    "MEMORY_FEATURE_DIM",
    "TRUSTED_MEMORY_SLOTS",
    "DISTRACTOR_MEMORY_SLOTS",
    "N72R9SourceAwareTemporalIdentityModel",
    "n72r9_loss",
]
