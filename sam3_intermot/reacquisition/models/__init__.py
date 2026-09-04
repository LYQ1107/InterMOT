"""Learned target-candidate models for the N72R7 research branch."""

from .target_id_decoder import (
    CANDIDATE_FEATURE_DIM,
    CONTEXT_FEATURE_DIM,
    HumanConditionedTargetIDDecoder,
    set_decoder_loss,
)

__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "CONTEXT_FEATURE_DIM",
    "HumanConditionedTargetIDDecoder",
    "set_decoder_loss",
]
