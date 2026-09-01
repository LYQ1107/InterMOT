#!/usr/bin/env python3
"""Anchor-Preserving Correction Residual (APCR-S).

The module has no independent candidate scorer.  It receives frozen B10
scores and produces only a bounded residual.  Positive and explicit-negative
channels have structurally fixed signs and are monotone in their respective
memory similarities.  The context gate deliberately excludes identity
similarities, sequence IDs, candidate rank, and camera coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


CONTEXT_NAMES = (
    "detector_score", "candidate_count_fraction", "has_positive", "has_negative", "has_hard",
    "positive_count_fraction", "negative_count_fraction", "hard_count_fraction",
    "positive_age_log", "negative_age_log", "hard_age_log",
)
CONTEXT_DIM = len(CONTEXT_NAMES)


@dataclass(frozen=True)
class APCRConfig:
    hidden: int = 24
    positive_bound: float = 0.03
    negative_bound: float = 0.03
    hard_bound: float = 0.03
    temperature: float = 0.05


class _Gate(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CONTEXT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.tau_raw = nn.Parameter(torch.tensor(0.0))
        self.log_slope = nn.Parameter(torch.tensor(1.0))

    def forward(self, context: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
        context_gate = torch.sigmoid(self.network(context)).squeeze(-1)
        tau = torch.tanh(self.tau_raw)
        slope = F.softplus(self.log_slope) + 1e-4
        monotone_response = torch.sigmoid(slope * (similarity - tau))
        return context_gate * monotone_response


class APCRS(nn.Module):
    """Small bounded residual over a frozen B10 score."""

    def __init__(self, config: APCRConfig | None = None) -> None:
        super().__init__()
        self.config = config or APCRConfig()
        self.positive_gate = _Gate(self.config.hidden)
        self.negative_gate = _Gate(self.config.hidden)
        self.hard_gate = _Gate(self.config.hidden)

    @staticmethod
    def context(features: dict[str, torch.Tensor]) -> torch.Tensor:
        # Every input is event/candidate context or a detector score.  No
        # positive/negative similarity enters this gate, preserving the
        # explicit monotonicity proof for the memory channels.
        values = [features[name] for name in CONTEXT_NAMES]
        return torch.stack(values, dim=-1)

    def residual(self, features: dict[str, torch.Tensor], mode: str = "both") -> torch.Tensor:
        context = self.context(features)
        zeros = torch.zeros_like(features["b10_score"])
        positive = zeros
        negative = zeros
        hard = zeros
        if mode in {"both", "positive_only"}:
            positive_mask = features["has_positive"] if features["has_positive"].ndim == 2 else features["has_positive"].unsqueeze(-1).expand_as(features["positive_similarity"])
            positive = self.config.positive_bound * self.positive_gate(context, features["positive_similarity"]) * positive_mask
        if mode in {"both", "negative_only"}:
            negative_mask = features["has_negative"] if features["has_negative"].ndim == 2 else features["has_negative"].unsqueeze(-1).expand_as(features["negative_similarity"])
            negative = self.config.negative_bound * self.negative_gate(context, features["negative_similarity"]) * negative_mask
        if mode == "hard_negative":
            hard_mask = features["has_hard"] if features["has_hard"].ndim == 2 else features["has_hard"].unsqueeze(-1).expand_as(features["hard_similarity"])
            hard = self.config.hard_bound * self.hard_gate(context, features["hard_similarity"]) * hard_mask
        result = positive - negative - hard
        return result * features["candidate_mask"].to(result.dtype)

    def forward(self, features: dict[str, torch.Tensor], mode: str = "both", residual_off: bool = False) -> dict[str, torch.Tensor]:
        if residual_off:
            delta = torch.zeros_like(features["b10_score"])
        else:
            delta = self.residual(features, mode=mode)
        scores = features["b10_score"] + delta
        scores = scores.masked_fill(~features["candidate_mask"], -1e4)
        return {"delta": delta, "scores": scores, "b10_scores": features["b10_score"]}


def feature_tensors(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert one stored batch to the model's named tensor interface."""
    detector = batch["detector_score"].float()

    def expand(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return value.unsqueeze(-1).expand_as(detector) if value.ndim == 1 else value

    return {
        "candidate_mask": batch["candidate_mask"].bool(),
        "b10_score": batch["b10_score"].float(),
        "positive_similarity": batch["positive_similarity"].float(),
        "negative_similarity": batch["negative_similarity"].float(),
        "hard_similarity": batch["hard_similarity"].float(),
        "detector_score": detector,
        "candidate_count_fraction": expand(batch["candidate_count"]),
        "has_positive": expand(batch["has_positive"]),
        "has_negative": expand(batch["has_negative"]),
        "has_hard": expand(batch["has_hard"]),
        "positive_count_fraction": expand(batch["positive_count"]),
        "negative_count_fraction": expand(batch["negative_count"]),
        "hard_count_fraction": expand(batch["hard_count"]),
        "positive_age_log": expand(batch["positive_age"]),
        "negative_age_log": expand(batch["negative_age"]),
        "hard_age_log": expand(batch["hard_age"]),
    }


def counterfactual_tensors(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Make the exact latest-correction-removed feature view."""
    view = dict(batch)
    view["b10_score"] = batch["cf_b10_score"]
    view["positive_similarity"] = batch["cf_positive_similarity"]
    view["negative_similarity"] = batch["cf_negative_similarity"]
    view["has_positive"] = batch["cf_has_positive"]
    view["has_negative"] = batch["cf_has_negative"]
    view["positive_count"] = batch["cf_positive_count"]
    view["negative_count"] = batch["cf_negative_count"]
    view["positive_age"] = batch["cf_positive_age"]
    view["negative_age"] = batch["cf_negative_age"]
    return feature_tensors(view)
