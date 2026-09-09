"""N72R11R4 public-competition-aware temporal identity scorer.

PCTIS keeps the N72R11 V3 capacity and all causal state channels.  The only
architectural addition is a six-dimensional, current-frame public-competition
token appended to the existing 530-D candidate token.  This module contains
no GT or posthoc path; labels are accepted only by the offline loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sam3_intermot.reacquisition.public_competition_features import (
    PUBLIC_COMPETITION_FEATURE_DIM,
    PUBLIC_COMPETITION_FEATURE_SCHEMA,
)
from sam3_intermot.reacquisition.models.n72r11_temporal_v3 import (
    CANDIDATE_FEATURE_DIM,
    DISTRACTOR_SLOTS,
    HEADS,
    HIDDEN_DIM,
    LAYERS,
    LONG_TERM_TRUSTED_SLOTS,
    MEMORY_FEATURE_DIM,
    RECENT_TRUSTED_SLOTS,
    SOURCE_FEATURE_DIM,
    TEMPORAL_FEATURE_DIM,
)


DROPOUT = 0.0


class PublicCompetitionTemporalIdentityModel(nn.Module):
    """V3-capacity temporal scorer with explicit public competition input."""

    def __init__(
        self,
        *,
        candidate_feature_dim: int = CANDIDATE_FEATURE_DIM,
        source_feature_dim: int = SOURCE_FEATURE_DIM,
        public_competition_feature_dim: int = PUBLIC_COMPETITION_FEATURE_DIM,
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
        self.public_competition_feature_dim = int(public_competition_feature_dim)
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.recent_trusted_slots = int(recent_trusted_slots)
        self.long_term_trusted_slots = int(long_term_trusted_slots)
        self.distractor_slots = int(distractor_slots)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.candidate_projection = nn.Sequential(
            nn.Linear(
                self.candidate_feature_dim + self.source_feature_dim + self.public_competition_feature_dim,
                self.hidden_dim,
            ),
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
        public_competition_features: Tensor,
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
        if public_competition_features.shape != (batch, count, self.public_competition_feature_dim):
            raise ValueError("public competition feature shape mismatch")
        expected_memory_shapes = (
            (batch, self.recent_trusted_slots, MEMORY_FEATURE_DIM),
            (batch, self.long_term_trusted_slots, MEMORY_FEATURE_DIM),
            (batch, self.distractor_slots, MEMORY_FEATURE_DIM),
        )
        if (
            recent_trusted_memory.shape != expected_memory_shapes[0]
            or long_term_trusted_memory.shape != expected_memory_shapes[1]
            or distractor_memory.shape != expected_memory_shapes[2]
        ):
            raise ValueError("temporal memory shape mismatch")
        if (
            recent_trusted_mask.shape != (batch, self.recent_trusted_slots)
            or long_term_trusted_mask.shape != (batch, self.long_term_trusted_slots)
            or distractor_mask.shape != (batch, self.distractor_slots)
        ):
            raise ValueError("temporal memory mask shape mismatch")
        if human_anchor.shape != (batch, MEMORY_FEATURE_DIM) or neighbor_feature.shape != (batch, MEMORY_FEATURE_DIM):
            raise ValueError("human/neighbor feature shape mismatch")
        if temporal_features.shape != (batch, self.temporal_feature_dim):
            raise ValueError("temporal feature shape mismatch")
        finite_tensors = (
            candidate_features,
            source_features,
            public_competition_features,
            human_anchor,
            recent_trusted_memory,
            long_term_trusted_memory,
            distractor_memory,
            neighbor_feature,
            temporal_features,
        )
        if not all(bool(torch.isfinite(value).all()) for value in finite_tensors):
            raise ValueError("PCTIS model inputs must be finite")
        tokens = self.candidate_projection(
            torch.cat([candidate_features, source_features, public_competition_features], dim=-1)
        )
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


def pctis_model_config() -> dict[str, Any]:
    return {
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "source_feature_dim": SOURCE_FEATURE_DIM,
        "public_competition_feature_dim": PUBLIC_COMPETITION_FEATURE_DIM,
        "temporal_feature_dim": TEMPORAL_FEATURE_DIM,
        "recent_trusted_slots": RECENT_TRUSTED_SLOTS,
        "long_term_trusted_slots": LONG_TERM_TRUSTED_SLOTS,
        "distractor_slots": DISTRACTOR_SLOTS,
        "hidden_dim": HIDDEN_DIM,
        "layers": LAYERS,
        "heads": HEADS,
        "dropout": DROPOUT,
    }


def initialize_from_v3_checkpoint(model: PublicCompetitionTemporalIdentityModel, v3_checkpoint: str | Path) -> dict[str, Any]:
    """Copy a V3 checkpoint, zeroing only the six new input columns."""

    path = Path(v3_checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise RuntimeError(f"invalid V3 checkpoint for PCTIS initialization: {path}")
    old_state = payload["state_dict"]
    new_state = model.state_dict()
    copied: list[str] = []
    adapted: list[str] = []
    missing: list[str] = []
    unexpected = sorted(set(str(key) for key in old_state) - set(new_state))
    for key, destination in new_state.items():
        if key not in old_state:
            missing.append(key)
            continue
        source = old_state[key]
        if key == "candidate_projection.0.weight":
            expected_old_shape = (destination.shape[0], destination.shape[1] - PUBLIC_COMPETITION_FEATURE_DIM)
            if tuple(source.shape) != expected_old_shape:
                raise RuntimeError(f"V3 candidate projection shape {tuple(source.shape)} != {expected_old_shape}")
            destination.zero_()
            destination[:, : source.shape[1]].copy_(source)
            adapted.append(key)
            continue
        if tuple(source.shape) != tuple(destination.shape):
            raise RuntimeError(f"same-capacity parameter shape mismatch for {key}: {tuple(source.shape)} != {tuple(destination.shape)}")
        destination.copy_(source)
        copied.append(key)
    if missing or unexpected:
        raise RuntimeError(f"PCTIS V3 initialization key mismatch: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(new_state, strict=True)
    return {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": _sha256(path),
        "copied_key_count": len(copied),
        "adapted_key_count": len(adapted),
        "adapted_keys": adapted,
        "zero_initialized_competition_columns": PUBLIC_COMPETITION_FEATURE_DIM,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_same_capacity_copy": True,
    }


def public_competition_temporal_loss(
    logits: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
    legacy_target_scores: Tensor,
    legacy_best_other_scores: Tensor,
    protected_candidate_mask: Tensor,
    *,
    none_weight: float = 2.0,
    pairwise_weight: float = 0.15,
    pairwise_margin: float = 0.20,
    positive_boundary_weight: float = 0.25,
    protected_boundary_weight: float = 0.25,
    delta_l2_weight: float = 0.01,
    public_margin: float = 0.20,
    example_weight: Optional[Tensor] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Fixed CE + ranking + public-boundary loss used only for offline labels."""

    if logits.ndim != 2 or labels.ndim != 1 or candidate_mask.ndim != 2:
        raise ValueError("invalid PCTIS loss shapes")
    batch, candidates = candidate_mask.shape
    if logits.shape != (batch, candidates + 1) or labels.shape != (batch,):
        raise ValueError("PCTIS logit/label shape mismatch")
    expected = (batch, candidates)
    for name, values in (
        ("legacy_target_scores", legacy_target_scores),
        ("legacy_best_other_scores", legacy_best_other_scores),
        ("protected_candidate_mask", protected_candidate_mask),
    ):
        if tuple(values.shape) != expected:
            raise ValueError(f"{name} shape mismatch")
    if not all(bool(torch.isfinite(value).all()) for value in (logits, legacy_target_scores, legacy_best_other_scores)):
        raise ValueError("PCTIS loss inputs must be finite")
    labels = labels.to(dtype=torch.long)
    if bool((labels < 0).any()) or bool((labels > candidates).any()):
        raise ValueError("PCTIS label outside candidate/NONE range")
    if not float(none_weight) > 0.0:
        raise ValueError("none_weight must be positive")
    weights = torch.ones(batch, dtype=logits.dtype, device=logits.device)
    weights = torch.where(labels == candidates, weights * float(none_weight), weights)
    if example_weight is not None:
        extra = example_weight.to(device=logits.device, dtype=logits.dtype)
        if extra.shape != labels.shape or not torch.isfinite(extra).all() or bool((extra <= 0).any()):
            raise ValueError("example_weight must be finite and positive")
        weights = weights * extra
    ce_values = F.cross_entropy(logits, labels, reduction="none")
    ce = (ce_values * weights).sum() / weights.sum().clamp_min(torch.finfo(logits.dtype).eps)

    positive = labels < candidates
    candidate_logits = logits[:, :candidates]
    pairwise = logits.new_zeros(())
    if bool(positive.any()):
        target = labels[positive]
        values = candidate_logits[positive]
        valid_negative_mask = candidate_mask[positive].to(dtype=torch.bool).clone()
        valid_negative_mask.scatter_(1, target.unsqueeze(1), False)
        valid_negative = valid_negative_mask.any(dim=1)
        if bool(valid_negative.any()):
            other = values.masked_fill(~valid_negative_mask, torch.finfo(values.dtype).min).max(dim=1).values
            target_value = values[torch.arange(len(target), device=values.device), target]
            pair_values = F.relu(float(pairwise_margin) - target_value[valid_negative] + other[valid_negative])
            pair_weights = weights[positive][valid_negative]
            pairwise = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)

    none_logit = logits[:, candidates : candidates + 1]
    model_score = candidate_logits - none_logit
    injection_delta = torch.relu(model_score)
    fused_target_edge = legacy_target_scores + injection_delta
    positive_boundary = logits.new_zeros(())
    if bool(positive.any()):
        positive_rows = torch.nonzero(positive, as_tuple=False).reshape(-1)
        positive_labels = labels[positive_rows]
        target_edge = fused_target_edge[positive_rows, positive_labels]
        other_edge = legacy_best_other_scores[positive_rows, positive_labels]
        positive_boundary = torch.relu(other_edge + float(public_margin) - target_edge).mean()

    protected_mask = protected_candidate_mask.to(dtype=torch.bool).clone()
    if bool(positive.any()):
        positive_rows = torch.nonzero(positive, as_tuple=False).reshape(-1)
        protected_mask[positive_rows, labels[positive_rows]] = False
    violation = torch.relu(fused_target_edge - legacy_best_other_scores + float(public_margin))
    protected_denominator = protected_mask.to(dtype=logits.dtype).sum().clamp_min(1.0)
    protected_boundary = ((violation * protected_mask.to(dtype=logits.dtype)) ** 2).sum() / protected_denominator
    delta_denominator = candidate_mask.to(dtype=logits.dtype).sum().clamp_min(1.0)
    delta_l2 = ((injection_delta * candidate_mask.to(dtype=logits.dtype)) ** 2).sum() / delta_denominator
    total = (
        ce
        + float(pairwise_weight) * pairwise
        + float(positive_boundary_weight) * positive_boundary
        + float(protected_boundary_weight) * protected_boundary
        + float(delta_l2_weight) * delta_l2
    )
    if not torch.isfinite(total):
        raise ValueError("PCTIS loss is non-finite")
    return total, {
        "cross_entropy": ce.detach(),
        "hard_negative_ranking": pairwise.detach(),
        "positive_public_boundary": positive_boundary.detach(),
        "protected_public_boundary": protected_boundary.detach(),
        "delta_l2": delta_l2.detach(),
    }


def _sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pctis_checkpoint(path: str | Path, device: torch.device) -> PublicCompetitionTemporalIdentityModel:
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise RuntimeError(f"invalid PCTIS checkpoint: {checkpoint}")
    if payload.get("public_competition_feature_schema") != list(PUBLIC_COMPETITION_FEATURE_SCHEMA):
        raise RuntimeError(f"PCTIS competition feature schema mismatch: {checkpoint}")
    model = PublicCompetitionTemporalIdentityModel(**pctis_model_config()).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "SOURCE_FEATURE_DIM",
    "PUBLIC_COMPETITION_FEATURE_DIM",
    "PUBLIC_COMPETITION_FEATURE_SCHEMA",
    "TEMPORAL_FEATURE_DIM",
    "RECENT_TRUSTED_SLOTS",
    "LONG_TERM_TRUSTED_SLOTS",
    "DISTRACTOR_SLOTS",
    "PublicCompetitionTemporalIdentityModel",
    "pctis_model_config",
    "initialize_from_v3_checkpoint",
    "public_competition_temporal_loss",
    "load_pctis_checkpoint",
]
