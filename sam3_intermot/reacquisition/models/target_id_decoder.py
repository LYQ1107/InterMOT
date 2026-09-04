"""Human-conditioned target candidate decoder for the N72R7 branch.

The module predicts a candidate index or an explicit ``NONE`` decision for a
single persistent target.  It is deliberately public-ID agnostic: public IDs
are assigned only by the frozen exact global solver after this module returns.
The model is kept in a new research namespace and never changes the pinned
SAM3 candidate generator or the production association formula.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CANDIDATE_FEATURE_DIM = 530
CONTEXT_FEATURE_DIM = 522


class HumanConditionedTargetIDDecoder(nn.Module):
    """Set decoder over current candidates plus an explicit NONE class.

    ``candidate_features`` has shape ``[batch, candidates, 530]`` and
    ``candidate_mask`` marks real candidates.  ``context_features`` has shape
    ``[batch, 522]`` and contains only the current/past target context.  The
    candidate logits are followed by the NONE logit in the returned tensor.
    """

    def __init__(
        self,
        *,
        candidate_feature_dim: int = CANDIDATE_FEATURE_DIM,
        context_feature_dim: int = CONTEXT_FEATURE_DIM,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if layers < 1:
            raise ValueError("at least one set layer is required")
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.context_feature_dim = int(context_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.candidate_projection = nn.Sequential(
            nn.Linear(self.candidate_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(self.context_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.context_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.none_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_features: Tensor,
        candidate_mask: Tensor,
        context_features: Tensor,
    ) -> Tensor:
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must be [batch,candidates,features]")
        if candidate_mask.ndim != 2 or candidate_mask.shape[:2] != candidate_features.shape[:2]:
            raise ValueError("candidate_mask must align with candidate_features")
        if context_features.ndim != 2 or context_features.shape[0] != candidate_features.shape[0]:
            raise ValueError("context_features must be [batch,features]")
        if candidate_features.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature width does not match decoder")
        if context_features.shape[-1] != self.context_feature_dim:
            raise ValueError("context feature width does not match decoder")
        mask = candidate_mask.to(dtype=torch.bool)
        candidates = self.candidate_projection(candidate_features)
        # Transformer attention cannot represent an all-padding row.  A zero
        # sentinel is admitted only for that numerical corner; its logit is
        # masked below and it cannot become a selected candidate.
        safe_mask = mask.clone()
        if safe_mask.shape[1] == 0:
            raise ValueError("candidate dimension must be positive; pad empty sets to one sentinel")
        empty = ~safe_mask.any(dim=1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
            candidates = candidates.clone()
            candidates[empty, 0] = 0.0
        encoded = self.set_encoder(candidates, src_key_padding_mask=~safe_mask)
        context = self.context_projection(context_features).unsqueeze(1)
        attended, _ = self.context_attention(
            context,
            encoded,
            encoded,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        context = self.context_norm(context + attended)
        context_vector = context[:, 0]
        pooled_denominator = safe_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        pooled = (encoded * safe_mask.unsqueeze(-1).to(encoded.dtype)).sum(dim=1) / pooled_denominator
        candidate_context = torch.cat(
            [encoded, context_vector.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1
        )
        candidate_logits = self.candidate_head(candidate_context).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~mask, torch.finfo(candidate_logits.dtype).min)
        none_logit = self.none_head(torch.cat([context_vector, pooled], dim=-1))
        return torch.cat([candidate_logits, none_logit], dim=1)


def set_decoder_loss(
    logits: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
    *,
    pairwise_weight: float = 0.15,
    pairwise_margin: float = 0.20,
    example_weight: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Set CE plus a small hard-negative ranking auxiliary loss.

    Labels are candidate indices or ``candidate_count`` for NONE.  The
    auxiliary term is evaluated only when the label names a real candidate;
    no future-effect metric is used by this training objective.
    """
    if logits.ndim != 2 or labels.ndim != 1 or candidate_mask.ndim != 2:
        raise ValueError("invalid decoder loss shapes")
    candidate_count = candidate_mask.shape[1]
    if logits.shape != (labels.shape[0], candidate_count + 1):
        raise ValueError("logit/label shape mismatch")
    labels = labels.to(dtype=torch.long)
    if bool((labels < 0).any()) or bool((labels > candidate_count).any()):
        raise ValueError("decoder label is outside candidate/NONE range")
    if example_weight is None:
        weights = torch.ones(labels.shape[0], dtype=logits.dtype, device=logits.device)
    else:
        weights = example_weight.to(device=logits.device, dtype=logits.dtype)
        if weights.ndim != 1 or weights.shape[0] != labels.shape[0]:
            raise ValueError("example_weight must align with labels")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("example_weight must be finite and strictly positive")
    ce_values = F.cross_entropy(logits, labels, reduction="none")
    ce = (ce_values * weights).sum() / weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    positive = labels < candidate_count
    pairwise = logits.new_zeros(())
    if bool(positive.any()):
        target_indices = labels[positive]
        candidate_logits = logits[positive, :candidate_count]
        other_mask = candidate_mask[positive].clone().to(dtype=torch.bool)
        other_mask.scatter_(1, target_indices.unsqueeze(1), False)
        other_logits = candidate_logits.masked_fill(~other_mask, torch.finfo(candidate_logits.dtype).min)
        hard_negative = other_logits.max(dim=1).values
        valid_negative = other_mask.any(dim=1)
        if bool(valid_negative.any()):
            target_logits = candidate_logits[
                torch.arange(len(target_indices), device=logits.device), target_indices
            ]
            pairwise_values = F.relu(
                pairwise_margin - target_logits[valid_negative] + hard_negative[valid_negative]
            )
            positive_weights = weights[positive][valid_negative]
            pairwise = (pairwise_values * positive_weights).sum() / positive_weights.sum().clamp_min(
                torch.finfo(logits.dtype).eps
            )
    total = ce + float(pairwise_weight) * pairwise
    return total, {"cross_entropy": ce.detach(), "hard_negative_ranking": pairwise.detach()}


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "CONTEXT_FEATURE_DIM",
    "HumanConditionedTargetIDDecoder",
    "set_decoder_loss",
]
