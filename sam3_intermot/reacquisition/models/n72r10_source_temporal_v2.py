"""Isolated N72R10 source-aware temporal model.

The architecture reuses the audited N72R9 scorer while adding a fifth,
explicit source channel for genuinely future-frame re-query candidates.  It
is research-only: it does not modify the N72R9 model, the SAM3 backend, the
candidate generator, or the public-ID solver.
"""

from __future__ import annotations

from sam3_intermot.reacquisition.models.n72r9_source_temporal import (
    CANDIDATE_FEATURE_DIM,
    DISTRACTOR_MEMORY_SLOTS,
    MEMORY_FEATURE_DIM,
    N72R9SourceAwareTemporalIdentityModel,
    TEMPORAL_FEATURE_DIM,
    TRUSTED_MEMORY_SLOTS,
    n72r9_loss,
)


SOURCE_FEATURE_DIM = 5


class N72R10SourceAwareTemporalIdentityModel(N72R9SourceAwareTemporalIdentityModel):
    """N72R9 scorer with an explicit FUTURE_FRAME_REQUERY source input."""

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
        if int(source_feature_dim) != SOURCE_FEATURE_DIM:
            raise ValueError(
                f"N72R10 requires source_feature_dim={SOURCE_FEATURE_DIM}, got {source_feature_dim}"
            )
        super().__init__(
            candidate_feature_dim=int(candidate_feature_dim),
            source_feature_dim=int(source_feature_dim),
            temporal_feature_dim=int(temporal_feature_dim),
            trusted_slots=int(trusted_slots),
            distractor_slots=int(distractor_slots),
            hidden_dim=int(hidden_dim),
            layers=int(layers),
            heads=int(heads),
            dropout=float(dropout),
        )


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "DISTRACTOR_MEMORY_SLOTS",
    "MEMORY_FEATURE_DIM",
    "SOURCE_FEATURE_DIM",
    "TEMPORAL_FEATURE_DIM",
    "TRUSTED_MEMORY_SLOTS",
    "N72R10SourceAwareTemporalIdentityModel",
    "n72r9_loss",
]
