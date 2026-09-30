"""HIIM model wrapper and variant construction."""

from __future__ import annotations

import torch
from torch import nn

from .encoder import FEATURE_DIMENSION
from .memory import GRUMemoryUpdater

VARIANT_GRU = "GRU_MEMORY"
VARIANT_GRU_GATE = "GRU_RELIABILITY_GATE"


class HIIM(nn.Module):
    """Human-Initialized Identity Memory Network (HIIM)."""

    def __init__(self, variant: str, feature_dim: int = FEATURE_DIMENSION, gate_hidden_dim: int = 256):
        super().__init__()
        if variant not in {VARIANT_GRU, VARIANT_GRU_GATE}:
            raise ValueError(f"unknown N72R18 variant: {variant}")
        self.variant = variant
        self.feature_dim = feature_dim
        self.memory = GRUMemoryUpdater(
            feature_dim=feature_dim,
            use_gate=variant == VARIANT_GRU_GATE,
            gate_hidden_dim=gate_hidden_dim,
        )

    @property
    def use_gate(self) -> bool:
        return self.memory.use_gate

    def forward(self, previous_state: torch.Tensor, observation: torch.Tensor):
        return self.memory(previous_state, observation)


def build_updater(variant: str, feature_dim: int = FEATURE_DIMENSION, gate_hidden_dim: int = 256) -> HIIM:
    return HIIM(variant=variant, feature_dim=feature_dim, gate_hidden_dim=gate_hidden_dim)


__all__ = ["HIIM", "VARIANT_GRU", "VARIANT_GRU_GATE", "build_updater"]
