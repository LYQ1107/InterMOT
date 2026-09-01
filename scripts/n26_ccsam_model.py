#!/usr/bin/env python3
"""N26 local Correction-Conditioned Set Association Model (CC-SAM)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CCSAMConfig:
    identity_dim: int = 1280
    scalar_dim: int = 50
    memory_meta_dim: int = 10
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1
    max_candidates: int = 5
    max_memory: int = 17

    def to_dict(self) -> dict:
        return asdict(self)


class CCSAM(nn.Module):
    """Set encoder with target-specific positive/negative correction memory.

    SAM3 and CLIP-ReID are deliberately absent from the optimizer: this module
    consumes their frozen, candidate-aligned features.
    """

    def __init__(self, config: CCSAMConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.identity_projection = nn.Sequential(
            nn.LayerNorm(config.identity_dim),
            nn.Linear(config.identity_dim, d, bias=False),
            nn.LayerNorm(d),
        )
        self.candidate_scalar_projection = nn.Sequential(
            nn.LayerNorm(config.scalar_dim),
            nn.Linear(config.scalar_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.memory_meta_projection = nn.Sequential(
            nn.LayerNorm(config.memory_meta_dim),
            nn.Linear(config.memory_meta_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.rank_embedding = nn.Embedding(config.max_candidates, d)
        self.none_token = nn.Parameter(torch.empty(1, 1, d))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers, norm=nn.LayerNorm(d))
        self.memory_attention = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.logit_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.none_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.existence_head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.risk_head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.positive_scale_raw = nn.Parameter(torch.tensor(0.5))
        self.negative_scale_raw = nn.Parameter(torch.tensor(0.5))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.none_token, std=0.02)

    @staticmethod
    def _mode_mask(memory_mask: torch.Tensor, memory_kind: torch.Tensor, mode: str) -> torch.Tensor:
        root = memory_kind.eq(0)
        positive = memory_kind.eq(1)
        explicit_negative = memory_kind.eq(2)
        hard_negative = memory_kind.eq(3)
        if mode == "off":
            selected = root
        elif mode == "positive":
            selected = root | positive
        elif mode == "negative":
            selected = root | explicit_negative
        elif mode == "positive_negative":
            selected = root | positive | explicit_negative
        elif mode == "hard_negative":
            selected = root | positive | hard_negative
        elif mode == "all":
            selected = root | positive | explicit_negative | hard_negative
        else:
            raise ValueError(f"unknown memory mode {mode}")
        selected = memory_mask & selected
        # The builder guarantees ROOT, but retain a deterministic fallback for
        # device-side ablations that supply a narrower mask.
        selected[:, 0] = True
        return selected

    def forward(
        self,
        candidate_clip: torch.Tensor,
        candidate_scalar: torch.Tensor,
        candidate_mask: torch.Tensor,
        memory_clip: torch.Tensor,
        memory_meta: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_kind: torch.Tensor,
        *,
        memory_mode: str = "positive_negative",
        disable_none: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, candidates, _ = candidate_clip.shape
        rank = torch.arange(candidates, device=candidate_clip.device).unsqueeze(0).expand(batch, -1)
        candidate_token = self.identity_projection(candidate_clip.float())
        candidate_token = candidate_token + self.candidate_scalar_projection(candidate_scalar.float()) + self.rank_embedding(rank)
        none = self.none_token.expand(batch, -1, -1)
        set_token = torch.cat([candidate_token, none], dim=1)
        set_mask = torch.cat([candidate_mask.bool(), torch.ones(batch, 1, dtype=torch.bool, device=candidate_mask.device)], dim=1)
        encoded = self.set_encoder(set_token, src_key_padding_mask=~set_mask)

        active_memory = self._mode_mask(memory_mask.bool(), memory_kind, memory_mode)
        memory_token = self.identity_projection(memory_clip.float()) + self.memory_meta_projection(memory_meta.float())
        attended, attention = self.memory_attention(encoded, memory_token, memory_token, key_padding_mask=~active_memory, need_weights=True)
        fused = self.cross_norm(encoded + attended)
        pooled_candidates = (fused[:, :candidates] * candidate_mask.unsqueeze(-1)).sum(dim=1)
        pooled_candidates = pooled_candidates / candidate_mask.sum(dim=1, keepdim=True).clamp_min(1)
        global_token = torch.cat([pooled_candidates, fused[:, -1]], dim=-1)

        candidate_logits = self.logit_head(fused[:, :candidates]).squeeze(-1)
        raw_candidate = F.normalize(candidate_clip.float(), dim=-1)
        raw_memory = F.normalize(memory_clip.float(), dim=-1)
        similarity = torch.einsum("bkd,bmd->bkm", raw_candidate, raw_memory)
        positive_mask = active_memory & (memory_kind.eq(0) | memory_kind.eq(1))
        if memory_mode == "hard_negative":
            negative_mask = active_memory & memory_kind.eq(3)
        else:
            negative_mask = active_memory & memory_kind.eq(2)
        positive_similarity = similarity.masked_fill(~positive_mask[:, None, :], -1e4).max(dim=-1).values
        negative_similarity = similarity.masked_fill(~negative_mask[:, None, :], -1e4).max(dim=-1).values
        negative_available = negative_mask.any(dim=1, keepdim=True)
        penalty = F.relu(negative_similarity - positive_similarity + 0.02) * negative_available
        candidate_logits = candidate_logits + F.softplus(self.positive_scale_raw) * positive_similarity
        candidate_logits = candidate_logits - F.softplus(self.negative_scale_raw) * penalty
        candidate_logits = candidate_logits.masked_fill(~candidate_mask.bool(), -1e4)

        none_logit = self.none_head(fused[:, -1]).squeeze(-1)
        if disable_none:
            none_logit = torch.full_like(none_logit, -1e4)
        logits = torch.cat([candidate_logits, none_logit[:, None]], dim=1)
        pooled = pooled_candidates[:, None, :].expand(-1, candidates, -1)
        risk_logits = self.risk_head(torch.cat([fused[:, :candidates], pooled], dim=-1)).squeeze(-1)
        risk_logits = risk_logits.masked_fill(~candidate_mask.bool(), -1e4)
        existence_logit = self.existence_head(global_token).squeeze(-1)
        return {
            "logits": logits,
            "candidate_logits": candidate_logits,
            "existence_logit": existence_logit,
            "risk_logits": risk_logits,
            "attention": attention,
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "negative_penalty": penalty,
            "active_memory": active_memory,
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
