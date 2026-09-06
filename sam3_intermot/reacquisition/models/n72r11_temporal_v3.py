"""N72R11 causal temporal identity scorer.

This is an isolated successor to the N72R10 scorer.  It keeps separate recent
and long-term trusted memory, exposes an explicit human anchor, and reserves a
larger distractor bank.  Memory admission is owned by the caller and must be
based only on causal runtime assignment confidence; this module contains no
GT or posthoc path.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CANDIDATE_FEATURE_DIM = 530
SOURCE_FEATURE_DIM = 5
TEMPORAL_FEATURE_DIM = 8
MEMORY_FEATURE_DIM = 512
RECENT_TRUSTED_SLOTS = 4
LONG_TERM_TRUSTED_SLOTS = 4
DISTRACTOR_SLOTS = 8
HIDDEN_DIM = 256
LAYERS = 3
HEADS = 8
DROPOUT = 0.0


class N72R11TemporalIdentityModel(nn.Module):
    """Set scorer with human, recent, long-term, distractor and temporal context."""

    def __init__(
        self,
        *,
        candidate_feature_dim: int = CANDIDATE_FEATURE_DIM,
        source_feature_dim: int = SOURCE_FEATURE_DIM,
        temporal_feature_dim: int = TEMPORAL_FEATURE_DIM,
        recent_trusted_slots: int = RECENT_TRUSTED_SLOTS,
        long_term_trusted_slots: int = LONG_TERM_TRUSTED_SLOTS,
        distractor_slots: int = DISTRACTOR_SLOTS,
        hidden_dim: int = HIDDEN_DIM,
        layers: int = LAYERS,
        heads: int = HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(heads):
            raise ValueError("hidden_dim must be divisible by heads")
        if min(int(recent_trusted_slots), int(long_term_trusted_slots), int(distractor_slots), int(layers)) < 1:
            raise ValueError("memory slots and layers must be positive")
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.source_feature_dim = int(source_feature_dim)
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.recent_trusted_slots = int(recent_trusted_slots)
        self.long_term_trusted_slots = int(long_term_trusted_slots)
        self.distractor_slots = int(distractor_slots)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.candidate_projection = nn.Sequential(
            nn.Linear(self.candidate_feature_dim + self.source_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        context_dim = MEMORY_FEATURE_DIM * 8 + self.temporal_feature_dim
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        self.candidate_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.none_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        valid = mask.to(dtype=torch.bool)
        weights = valid.unsqueeze(-1).to(dtype=values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
        valid = mask.to(dtype=torch.bool)
        masked = values.masked_fill(~valid.unsqueeze(-1), torch.finfo(values.dtype).min)
        result = masked.max(dim=1).values
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            result = result.clone()
            result[empty] = 0.0
        return result

    def forward(
        self,
        candidate_features: Tensor,
        candidate_mask: Tensor,
        source_features: Tensor,
        human_anchor: Tensor,
        recent_trusted_memory: Tensor,
        recent_trusted_mask: Tensor,
        long_term_trusted_memory: Tensor,
        long_term_trusted_mask: Tensor,
        distractor_memory: Tensor,
        distractor_mask: Tensor,
        neighbor_feature: Tensor,
        temporal_features: Tensor,
    ) -> Tensor:
        if candidate_features.ndim != 3 or candidate_mask.ndim != 2:
            raise ValueError("candidate_features/mask must be [batch,candidates,...]")
        batch, count = candidate_features.shape[:2]
        if candidate_mask.shape != (batch, count):
            raise ValueError("candidate mask does not align with candidates")
        if candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature width mismatch")
        if source_features.shape != (batch, count, self.source_feature_dim):
            raise ValueError("source feature shape mismatch")
        expected_memory_shapes = (
            (batch, self.recent_trusted_slots, MEMORY_FEATURE_DIM),
            (batch, self.long_term_trusted_slots, MEMORY_FEATURE_DIM),
            (batch, self.distractor_slots, MEMORY_FEATURE_DIM),
        )
        if recent_trusted_memory.shape != expected_memory_shapes[0] or long_term_trusted_memory.shape != expected_memory_shapes[1] or distractor_memory.shape != expected_memory_shapes[2]:
            raise ValueError("temporal memory shape mismatch")
        if recent_trusted_mask.shape != (batch, self.recent_trusted_slots) or long_term_trusted_mask.shape != (batch, self.long_term_trusted_slots) or distractor_mask.shape != (batch, self.distractor_slots):
            raise ValueError("temporal memory mask shape mismatch")
        if human_anchor.shape != (batch, MEMORY_FEATURE_DIM) or neighbor_feature.shape != (batch, MEMORY_FEATURE_DIM):
            raise ValueError("human/neighbor feature shape mismatch")
        if temporal_features.shape != (batch, self.temporal_feature_dim):
            raise ValueError("temporal feature shape mismatch")
        tensors = (
            candidate_features,
            source_features,
            human_anchor,
            recent_trusted_memory,
            long_term_trusted_memory,
            distractor_memory,
            neighbor_feature,
            temporal_features,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("N72R11 model inputs must be finite")
        tokens = self.candidate_projection(torch.cat([candidate_features, source_features], dim=-1))
        valid = candidate_mask.to(dtype=torch.bool)
        if count < 1:
            raise ValueError("candidate dimension must be positive")
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if bool(empty.any()):
            safe_valid[empty, 0] = True
            tokens = tokens.clone()
            tokens[empty, 0] = 0.0
        encoded = self.set_encoder(tokens, src_key_padding_mask=~safe_valid)
        context_values = torch.cat(
            [
                human_anchor,
                self._masked_mean(recent_trusted_memory, recent_trusted_mask),
                self._masked_max(recent_trusted_memory, recent_trusted_mask),
                self._masked_mean(long_term_trusted_memory, long_term_trusted_mask),
                self._masked_max(long_term_trusted_memory, long_term_trusted_mask),
                self._masked_mean(distractor_memory, distractor_mask),
                self._masked_max(distractor_memory, distractor_mask),
                neighbor_feature,
                temporal_features,
            ],
            dim=-1,
        )
        context = self.context_projection(context_values)
        combined = torch.cat([encoded, context.unsqueeze(1).expand(-1, count, -1)], dim=-1)
        logits = self.candidate_head(combined).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        return torch.cat([logits, self.none_head(context)], dim=1)


def n72r11_loss(
    logits: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
    *,
    none_weight: float = 1.0,
    pairwise_weight: float = 0.15,
    pairwise_margin: float = 0.20,
    example_weight: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted CE plus hard-negative ranking; NONE weight is train-only."""

    if logits.ndim != 2 or labels.ndim != 1 or candidate_mask.ndim != 2:
        raise ValueError("invalid N72R11 loss shapes")
    candidates = int(candidate_mask.shape[1])
    if logits.shape != (labels.shape[0], candidates + 1):
        raise ValueError("N72R11 logit/label shape mismatch")
    if not torch.isfinite(logits).all():
        raise ValueError("N72R11 logits must be finite")
    labels = labels.to(dtype=torch.long)
    if bool((labels < 0).any()) or bool((labels > candidates).any()):
        raise ValueError("N72R11 label outside candidate/NONE range")
    if not float(none_weight) > 0.0:
        raise ValueError("none_weight must be positive")
    weights = torch.ones(labels.shape[0], dtype=logits.dtype, device=logits.device)
    weights = torch.where(labels == candidates, weights * float(none_weight), weights)
    if example_weight is not None:
        extra = example_weight.to(device=logits.device, dtype=logits.dtype)
        if extra.shape != labels.shape or not torch.isfinite(extra).all() or bool((extra <= 0).any()):
            raise ValueError("example_weight must be finite and positive")
        weights = weights * extra
    ce_values = F.cross_entropy(logits, labels, reduction="none")
    ce = (ce_values * weights).sum() / weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    positive = labels < candidates
    pairwise = logits.new_zeros(())
    if bool(positive.any()):
        target = labels[positive]
        values = logits[positive, :candidates]
        other_mask = candidate_mask[positive].to(dtype=torch.bool)
        other_mask.scatter_(1, target.unsqueeze(1), False)
        valid_negative = other_mask.any(dim=1)
        if bool(valid_negative.any()):
            other = values.masked_fill(~other_mask, torch.finfo(values.dtype).min).max(dim=1).values
            target_value = values[torch.arange(len(target), device=values.device), target]
            pair_values = F.relu(float(pairwise_margin) - target_value[valid_negative] + other[valid_negative])
            pair_weights = weights[positive][valid_negative]
            pairwise = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    total = ce + float(pairwise_weight) * pairwise
    return total, {"cross_entropy": ce.detach(), "hard_negative_ranking": pairwise.detach()}


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "SOURCE_FEATURE_DIM",
    "TEMPORAL_FEATURE_DIM",
    "MEMORY_FEATURE_DIM",
    "RECENT_TRUSTED_SLOTS",
    "LONG_TERM_TRUSTED_SLOTS",
    "DISTRACTOR_SLOTS",
    "N72R11TemporalIdentityModel",
    "n72r11_loss",
]
