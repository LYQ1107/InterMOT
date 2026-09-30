"""Hard-negative contrastive and memory-stability losses for N72R18."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .encoder import normalize


def hard_negative_info_nce(
    state: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute the declared positive-vs-all-visible-competitors InfoNCE.

    ``negative_mask`` marks real competitors in the padded candidate tensor.
    The returned tensor is one loss per batch item; callers choose which
    items are valid competitive frames.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    state = normalize(state)
    positive = normalize(positive)
    negatives = normalize(negatives)
    positive_score = (state * positive).sum(dim=-1, keepdim=True)
    if negatives.shape[1] == 0:
        return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
    negative_scores = torch.einsum("bd,bcd->bc", state, negatives)
    negative_scores = negative_scores.masked_fill(~negative_mask, -1e4)
    logits = torch.cat((positive_score, negative_scores), dim=1) / temperature
    return -F.log_softmax(logits, dim=1)[:, 0]


def stability_loss(previous_state: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
    """Return one ``L2`` state-change value per sample."""

    return torch.linalg.vector_norm(new_state - previous_state, dim=-1)


__all__ = ["hard_negative_info_nce", "stability_loss"]
