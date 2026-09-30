"""Small, frozen candidate-to-identity bridge for N72R20.

The bridge intentionally stops at feature extraction, cosine scoring, and
ranking.  It does not assign public identities, read ground truth, update the
SAM3 state, or train an encoder/memory model.  A caller may provide an
existing frozen OSNet crop extractor; precomputed ``embedding`` values are
also accepted for cache/evaluation replay.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .encoder import FEATURE_DIMENSION


FeatureExtractor = Callable[[Sequence[Any]], torch.Tensor | np.ndarray]

_FRAME_FIELDS = ("frame_id", "frame", "frame_idx")
_BOX_FIELDS = ("bbox", "box_xyxy")
_SCORE_FIELDS = ("sam_score", "presence", "presence_score", "confidence")
_FORBIDDEN_RUNTIME_FIELDS = frozenset(
    {
        "gt_id",
        "gt_ids",
        "gt_box",
        "gt_boxes",
        "identity_id",
        "public_id",
        "future_identity",
        "future_candidate",
    }
)


def _field(candidate: Any, names: Sequence[str], *, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        for name in names:
            if name in candidate:
                return candidate[name]
        return default
    for name in names:
        if hasattr(candidate, name):
            return getattr(candidate, name)
    return default


class CandidateIdentityMatcher:
    """Rank real SAM3 candidates against one frozen identity state.

    ``feature_extractor`` is deliberately a narrow callback.  In the real
    stream it should wrap the already frozen N72R16 OSNet box-crop encoder; it
    must return one 512-D vector per candidate and must not use GT.  During
    cache replay, candidates can instead carry a float16/float32
    ``embedding`` field, which is converted to float32 for scoring.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        *,
        feature_dim: int = FEATURE_DIMENSION,
        device: torch.device | str | None = None,
    ) -> None:
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_extractor = feature_extractor
        self.feature_dim = int(feature_dim)
        self.device = None if device is None else torch.device(device)

    def _validate_candidates(self, candidates: Sequence[Any]) -> None:
        for index, candidate in enumerate(candidates):
            if isinstance(candidate, Mapping):
                forbidden = _FORBIDDEN_RUNTIME_FIELDS.intersection(candidate)
                if forbidden:
                    names = ", ".join(sorted(forbidden))
                    raise ValueError(f"candidate {index} contains forbidden GT/identity fields: {names}")
            if _field(candidate, _FRAME_FIELDS) is None:
                raise ValueError(f"candidate {index} is missing frame_id/frame")
            if _field(candidate, _BOX_FIELDS) is None:
                raise ValueError(f"candidate {index} is missing bbox/box_xyxy")
            if _field(candidate, _SCORE_FIELDS) is None:
                raise ValueError(f"candidate {index} is missing SAM score/confidence")

    def extract_candidate_features(self, candidates: Sequence[Any]) -> torch.Tensor:
        """Return normalized float32 features in candidate order.

        The method does not persist crops, masks, decoder tokens, or dense
        feature maps.  Those fields are candidate provenance only; the
        identity bridge consumes the compact OSNet embedding.
        """

        items = tuple(candidates)
        self._validate_candidates(items)
        if not items:
            return torch.empty((0, self.feature_dim), dtype=torch.float32, device=self.device)

        embedded = [_field(item, ("embedding",)) for item in items]
        has_embedded = [value is not None for value in embedded]
        if any(has_embedded) and not all(has_embedded):
            raise ValueError("either all candidates or none may provide precomputed embeddings")
        if all(has_embedded):
            raw_features: Any = np.stack([np.asarray(value) for value in embedded], axis=0)
        elif self.feature_extractor is not None:
            raw_features = self.feature_extractor(items)
        else:
            raise ValueError("a frozen feature_extractor or precomputed candidate embeddings is required")

        features = raw_features if isinstance(raw_features, torch.Tensor) else torch.as_tensor(raw_features)
        if features.ndim != 2 or tuple(features.shape) != (len(items), self.feature_dim):
            raise ValueError(
                f"candidate features must have shape ({len(items)}, {self.feature_dim}), got {tuple(features.shape)}"
            )
        features = features.to(dtype=torch.float32)
        if self.device is not None:
            features = features.to(self.device)
        if not torch.isfinite(features).all():
            raise ValueError("candidate features contain non-finite values")
        if torch.any(torch.linalg.vector_norm(features, dim=-1) <= 1.0e-12):
            raise ValueError("candidate features must be non-zero")
        return F.normalize(features, p=2, dim=-1, eps=1.0e-12)

    def score_candidates(
        self,
        identity_state: torch.Tensor | np.ndarray,
        candidate_features: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Return cosine scores for the identity state in candidate order."""

        state = identity_state if isinstance(identity_state, torch.Tensor) else torch.as_tensor(identity_state)
        features = candidate_features if isinstance(candidate_features, torch.Tensor) else torch.as_tensor(candidate_features)
        if state.ndim != 1 or state.shape[0] != self.feature_dim:
            raise ValueError(f"identity_state must have shape ({self.feature_dim},), got {tuple(state.shape)}")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"candidate_features must have shape (N, {self.feature_dim}), got {tuple(features.shape)}"
            )
        state = state.to(dtype=torch.float32)
        features = features.to(dtype=torch.float32, device=state.device)
        if not torch.isfinite(state).all() or not torch.isfinite(features).all():
            raise ValueError("identity state and candidate features must be finite")
        if torch.linalg.vector_norm(state) <= 1.0e-12:
            raise ValueError("identity_state must be non-zero")
        if features.shape[0] and torch.any(torch.linalg.vector_norm(features, dim=-1) <= 1.0e-12):
            raise ValueError("candidate features must be non-zero")
        return F.normalize(features, p=2, dim=-1, eps=1.0e-12) @ F.normalize(
            state, p=2, dim=0, eps=1.0e-12
        )

    def rank_candidates(self, scores: torch.Tensor | np.ndarray) -> list[int]:
        """Return stable descending candidate indices; ties keep input order."""

        values = scores.detach().cpu().reshape(-1) if isinstance(scores, torch.Tensor) else torch.as_tensor(scores).reshape(-1)
        if not torch.isfinite(values).all():
            raise ValueError("candidate scores contain non-finite values")
        return sorted(range(int(values.numel())), key=lambda index: (-float(values[index]), index))


__all__ = ["CandidateIdentityMatcher", "FeatureExtractor"]
