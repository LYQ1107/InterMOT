"""Evidence-based observation selection on top of a frozen N72R18 GRU."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import normalize
from .updater import HIIM, VARIANT_GRU, build_updater

FEATURE_NAMES: tuple[str, ...] = (
    "memory_similarity",
    "anchor_similarity",
    "assigned_score",
    "best_competitor_score",
    "competition_margin",
    "top1_top2_margin",
    "assigned_rank_normalized",
    "competition_entropy",
    "prospective_state_change",
    "anchor_drift_candidate",
    "temporal_gap_normalized",
    "memory_age_normalized",
    "recent_state_stability",
)

FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def load_frozen_n72r18_gru(checkpoint_path: str | Path, device: torch.device | str) -> HIIM:
    """Load the proven N72R18 GRU with an explicit strict state-dict check."""

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if str(checkpoint.get("stage")) != "N72R18":
        raise ValueError("R1 base checkpoint must be an N72R18 checkpoint")
    if str(checkpoint.get("variant")) != VARIANT_GRU:
        raise ValueError("R1 base checkpoint must be the ungated N72R18 GRU")
    config = dict(checkpoint.get("config", {}))
    feature_dim = int(config.get("feature_dimension", 512))
    model = build_updater(VARIANT_GRU, feature_dim=feature_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


class ObservationSelector(nn.Module):
    """Small selector operating on explicit evidence features only."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dims: Sequence[int] = (64, 32),
        feature_indices: Iterable[int] | None = None,
    ):
        super().__init__()
        if feature_indices is None:
            feature_indices = range(input_dim)
        self.feature_indices = tuple(int(index) for index in feature_indices)
        if not self.feature_indices:
            raise ValueError("ObservationSelector needs at least one evidence feature")
        if any(index < 0 or index >= input_dim for index in self.feature_indices):
            raise ValueError("feature index is outside the evidence vector")
        dims = [len(self.feature_indices), *[int(value) for value in hidden_dims], 1]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(left, right), nn.GELU()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        selected = features[..., list(self.feature_indices)]
        return torch.sigmoid(self.network(selected)).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TrustedState:
    """Online-only numeric state kept alongside one identity memory."""

    def __init__(self, anchor: torch.Tensor, frame: int):
        self.anchor = normalize(anchor.detach())
        self.last_trusted_update_frame = int(frame)
        self.trusted_update_count = 0
        self.recent_stabilities: deque[float] = deque(maxlen=5)

    @property
    def recent_state_stability(self) -> float:
        if not self.recent_stabilities:
            return 0.0
        return float(sum(self.recent_stabilities) / len(self.recent_stabilities))

    def observe_accept(self, previous: torch.Tensor, current: torch.Tensor, frame: int) -> None:
        drift = float(1.0 - torch.dot(normalize(previous), normalize(current)).item())
        self.recent_stabilities.append(max(0.0, drift))
        self.last_trusted_update_frame = int(frame)
        self.trusted_update_count += 1


def _candidate_scores(
    state: torch.Tensor,
    observation: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    state = normalize(state)
    observation = normalize(observation)
    valid = torch.nonzero(competitor_mask, as_tuple=False).flatten()
    candidate_vectors = [observation]
    if valid.numel():
        candidate_vectors.append(normalize(competitors[valid]))
    candidates = torch.cat([item.unsqueeze(0) if item.ndim == 1 else item for item in candidate_vectors], dim=0)
    scores = torch.matmul(candidates, state)
    hard_index = int(valid[int(torch.argmax(scores[1:]).item())].item()) if valid.numel() else None
    return scores, valid, hard_index


def build_evidence_features(
    state: torch.Tensor,
    human_anchor: torch.Tensor,
    observation: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
    *,
    frame: int,
    last_trusted_update_frame: int,
    trusted_update_count: int,
    recent_state_stability: float,
    frozen_gru: HIIM,
    entropy_temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float | int | None], torch.Tensor]:
    """Compute all selector evidence before any state write."""

    state = normalize(state)
    human_anchor = normalize(human_anchor)
    observation = normalize(observation)
    candidate_state, _, _ = frozen_gru(state.unsqueeze(0), observation.unsqueeze(0))
    candidate_state = candidate_state.squeeze(0)
    scores, valid, _ = _candidate_scores(state, observation, competitors, competitor_mask)
    assigned_score = float(scores[0].item())
    competitor_scores = scores[1:]
    if competitor_scores.numel():
        best_competitor_score = float(competitor_scores.max().item())
    else:
        best_competitor_score = -1.0
    sorted_scores = torch.sort(scores, descending=True).values
    top1_top2_margin = float((sorted_scores[0] - sorted_scores[1]).item()) if sorted_scores.numel() > 1 else 0.0
    assigned_rank = 1 + int((scores[1:] > scores[0] + 1e-12).sum().item())
    entropy_distribution = F.softmax(scores / entropy_temperature, dim=0)
    entropy = -(entropy_distribution * torch.log(entropy_distribution.clamp_min(1e-12))).sum()
    entropy = entropy / max(float(torch.log(torch.tensor(float(scores.numel()))).item()), 1e-12) if scores.numel() > 1 else torch.zeros_like(entropy)
    state_change = 1.0 - float(torch.dot(state, candidate_state).item())
    anchor_drift = 1.0 - float(torch.dot(human_anchor, candidate_state).item())
    temporal_gap = min(max(int(frame) - int(last_trusted_update_frame), 0), 100) / 100.0
    memory_age = min(float(torch.log1p(torch.tensor(float(trusted_update_count))).item()), 5.0) / 5.0
    feature_values = [
        assigned_score,
        float(torch.dot(human_anchor, observation).item()),
        assigned_score,
        best_competitor_score,
        assigned_score - best_competitor_score,
        top1_top2_margin,
        assigned_rank / max(float(scores.numel()), 1.0),
        float(entropy.item()),
        state_change,
        anchor_drift,
        temporal_gap,
        memory_age,
        float(recent_state_stability),
    ]
    features = torch.tensor(feature_values, dtype=state.dtype, device=state.device)
    info: dict[str, float | int | None] = {
        "memory_similarity": assigned_score,
        "anchor_similarity": feature_values[1],
        "assigned_score": assigned_score,
        "best_competitor_score": best_competitor_score,
        "competition_margin": feature_values[4],
        "top1_top2_margin": top1_top2_margin,
        "assigned_rank": assigned_rank,
        "assigned_rank_normalized": feature_values[6],
        "competition_entropy": feature_values[7],
        "prospective_state_change": state_change,
        "anchor_drift_candidate": anchor_drift,
        "temporal_gap_normalized": temporal_gap,
        "memory_age_normalized": memory_age,
        "recent_state_stability": float(recent_state_stability),
        "candidate_competitor_count": int(valid.numel()),
    }
    return features, info, candidate_state


def build_evidence_batch(
    state: torch.Tensor,
    human_anchor: torch.Tensor,
    observations: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
    *,
    frames: torch.Tensor,
    last_trusted_update_frames: torch.Tensor,
    trusted_update_counts: torch.Tensor,
    recent_state_stability: torch.Tensor,
    frozen_gru: HIIM,
    entropy_temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Vectorized evidence computation for replay batches.

    The first candidate in every row is the proposed observation; the remaining
    candidates are the visible competitors. Padding is masked before entropy,
    rank, and hard-negative computations.
    """

    state = normalize(state)
    human_anchor = normalize(human_anchor)
    observations = normalize(observations)
    candidate_state, _, _ = frozen_gru(state, observations)
    assigned_score = (state * observations).sum(dim=-1)
    competitor_scores = torch.einsum("bd,bcd->bc", state, normalize(competitors))
    competitor_scores = competitor_scores.masked_fill(~competitor_mask, -torch.inf)
    all_scores = torch.cat((assigned_score.unsqueeze(-1), competitor_scores), dim=-1)
    all_mask = torch.cat(
        (torch.ones((state.shape[0], 1), dtype=torch.bool, device=state.device), competitor_mask), dim=-1
    )
    masked_scores = all_scores.masked_fill(~all_mask, -torch.inf)
    sorted_scores = torch.sort(masked_scores, descending=True, dim=-1).values
    top1_top2_margin = torch.where(
        all_mask.sum(dim=-1) > 1,
        sorted_scores[:, 0] - sorted_scores[:, 1],
        torch.zeros_like(assigned_score),
    )
    has_competitor = competitor_mask.any(dim=-1)
    best_competitor_score = competitor_scores.max(dim=-1).values
    best_competitor_score = torch.where(
        has_competitor, best_competitor_score, torch.full_like(best_competitor_score, -1.0)
    )
    assigned_rank = 1 + (competitor_scores > assigned_score.unsqueeze(-1) + 1e-12).sum(dim=-1)
    assigned_rank_normalized = assigned_rank.to(state.dtype) / all_mask.sum(dim=-1).clamp_min(1).to(state.dtype)
    probabilities = torch.softmax(masked_scores / entropy_temperature, dim=-1)
    entropy_terms = torch.where(
        all_mask,
        probabilities * torch.log(probabilities.clamp_min(1e-12)),
        torch.zeros_like(probabilities),
    )
    entropy = -entropy_terms.sum(dim=-1)
    denominator = torch.log(all_mask.sum(dim=-1).to(state.dtype).clamp_min(2.0))
    competition_entropy = torch.where(all_mask.sum(dim=-1) > 1, entropy / denominator, torch.zeros_like(entropy))
    anchor_similarity = (human_anchor * observations).sum(dim=-1)
    prospective_state_change = 1.0 - (state * candidate_state).sum(dim=-1)
    anchor_drift_candidate = 1.0 - (human_anchor * candidate_state).sum(dim=-1)
    temporal_gap_normalized = ((frames - last_trusted_update_frames).clamp_min(0).clamp_max(100).to(state.dtype) / 100.0)
    memory_age_normalized = torch.log1p(trusted_update_counts.to(state.dtype)).clamp_max(5.0) / 5.0
    features = torch.stack(
        (
            assigned_score,
            anchor_similarity,
            assigned_score,
            best_competitor_score,
            assigned_score - best_competitor_score,
            top1_top2_margin,
            assigned_rank_normalized,
            competition_entropy,
            prospective_state_change,
            anchor_drift_candidate,
            temporal_gap_normalized,
            memory_age_normalized,
            recent_state_stability.to(state.dtype),
        ),
        dim=-1,
    )
    info = {
        "memory_similarity": assigned_score,
        "anchor_similarity": anchor_similarity,
        "assigned_score": assigned_score,
        "best_competitor_score": best_competitor_score,
        "competition_margin": assigned_score - best_competitor_score,
        "top1_top2_margin": top1_top2_margin,
        "assigned_rank": assigned_rank,
        "assigned_rank_normalized": assigned_rank_normalized,
        "competition_entropy": competition_entropy,
        "prospective_state_change": prospective_state_change,
        "anchor_drift_candidate": anchor_drift_candidate,
        "temporal_gap_normalized": temporal_gap_normalized,
        "memory_age_normalized": memory_age_normalized,
        "recent_state_stability": recent_state_stability.to(state.dtype),
        "candidate_competitor_count": competitor_mask.sum(dim=-1),
    }
    return features, candidate_state, info, competitor_scores


def update_state_hard(
    previous_state: torch.Tensor,
    candidate_state: torch.Tensor,
    accepted: bool,
) -> torch.Tensor:
    return candidate_state if bool(accepted) else previous_state


def update_state_soft(
    previous_state: torch.Tensor,
    candidate_state: torch.Tensor,
    selection_score: torch.Tensor,
) -> torch.Tensor:
    return normalize(selection_score.unsqueeze(-1) * candidate_state + (1.0 - selection_score).unsqueeze(-1) * previous_state)


def feature_indices_for_ablation(name: str) -> tuple[int, ...]:
    groups = {
        "A1_memory_similarity_only": ("memory_similarity",),
        "A2_memory_plus_human_anchor": ("memory_similarity", "anchor_similarity"),
        "A3_plus_competition_margin": ("memory_similarity", "anchor_similarity", "competition_margin"),
        "A4_plus_prospective_state_drift": (
            "memory_similarity",
            "anchor_similarity",
            "competition_margin",
            "prospective_state_change",
        ),
        "A5_full_observation_evidence": FEATURE_NAMES,
    }
    if name not in groups:
        raise ValueError(f"unknown R1 ablation: {name}")
    return tuple(FEATURE_INDEX[item] for item in groups[name])


__all__ = [
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "ObservationSelector",
    "TrustedState",
    "build_evidence_features",
    "build_evidence_batch",
    "feature_indices_for_ablation",
    "load_frozen_n72r18_gru",
    "update_state_hard",
    "update_state_soft",
]
