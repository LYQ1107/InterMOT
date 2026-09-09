"""Uniform scorer adapters for N72R11R4 formal replay.

The replay owns candidate construction, legacy injection and the exact public
solver.  These adapters only translate the common NumPy runtime tensors to the
V3-compatible or public-competition-aware model forward call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class _BaseScorerAdapter:
    model_name = "UNSPECIFIED"

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    def _tensor(self, value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _finish(self, output: torch.Tensor) -> np.ndarray:
        if output.ndim != 2 or output.shape[0] != 1:
            raise RuntimeError(f"{self.model_name} must return [batch,candidates+NONE], got {tuple(output.shape)}")
        values = output[0].detach().float().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{self.model_name} returned non-finite logits")
        return values


class V3TemporalScorerAdapter(_BaseScorerAdapter):
    """Adapter for the frozen N72R11 V3 model."""

    model_name = "N72R11_V3"

    @torch.no_grad()
    def score(
        self,
        *,
        candidate_features,
        candidate_mask,
        source_features,
        public_competition_features,
        human_anchor,
        recent_trusted_memory,
        recent_trusted_mask,
        long_term_trusted_memory,
        long_term_trusted_mask,
        distractor_memory,
        distractor_mask,
        neighbor_feature,
        temporal_features,
    ) -> np.ndarray:
        del public_competition_features
        output = self.model(
            self._tensor(candidate_features, dtype=torch.float32),
            self._tensor(candidate_mask, dtype=torch.bool),
            self._tensor(source_features, dtype=torch.float32),
            self._tensor(human_anchor, dtype=torch.float32),
            self._tensor(recent_trusted_memory, dtype=torch.float32),
            self._tensor(recent_trusted_mask, dtype=torch.bool),
            self._tensor(long_term_trusted_memory, dtype=torch.float32),
            self._tensor(long_term_trusted_mask, dtype=torch.bool),
            self._tensor(distractor_memory, dtype=torch.float32),
            self._tensor(distractor_mask, dtype=torch.bool),
            self._tensor(neighbor_feature, dtype=torch.float32),
            self._tensor(temporal_features, dtype=torch.float32),
        )
        return self._finish(output)


class PublicCompetitionScorerAdapter(_BaseScorerAdapter):
    """Adapter for the N72R11R4 public-competition-aware scorer."""

    model_name = "N72R11R4_PCTIS"

    @torch.no_grad()
    def score(
        self,
        *,
        candidate_features,
        candidate_mask,
        source_features,
        public_competition_features,
        human_anchor,
        recent_trusted_memory,
        recent_trusted_mask,
        long_term_trusted_memory,
        long_term_trusted_mask,
        distractor_memory,
        distractor_mask,
        neighbor_feature,
        temporal_features,
    ) -> np.ndarray:
        output = self.model(
            self._tensor(candidate_features, dtype=torch.float32),
            self._tensor(candidate_mask, dtype=torch.bool),
            self._tensor(source_features, dtype=torch.float32),
            self._tensor(public_competition_features, dtype=torch.float32),
            self._tensor(human_anchor, dtype=torch.float32),
            self._tensor(recent_trusted_memory, dtype=torch.float32),
            self._tensor(recent_trusted_mask, dtype=torch.bool),
            self._tensor(long_term_trusted_memory, dtype=torch.float32),
            self._tensor(long_term_trusted_mask, dtype=torch.bool),
            self._tensor(distractor_memory, dtype=torch.float32),
            self._tensor(distractor_mask, dtype=torch.bool),
            self._tensor(neighbor_feature, dtype=torch.float32),
            self._tensor(temporal_features, dtype=torch.float32),
        )
        return self._finish(output)


__all__ = ["V3TemporalScorerAdapter", "PublicCompetitionScorerAdapter"]
