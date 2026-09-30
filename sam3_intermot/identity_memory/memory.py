"""Small learned identity-memory primitives for N72R18."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import FEATURE_DIMENSION, normalize


class ReliabilityGate(nn.Module):
    """Predict an observation reliability value in ``[0, 1]``.

    The gate sees only the previous identity state and the current frozen
    observation.  Difference features make the comparison explicit while
    keeping the gate independent of any identity label or future frame.
    """

    def __init__(self, feature_dim: int = FEATURE_DIMENSION, hidden_dim: int = 256):
        super().__init__()
        input_dim = feature_dim * 4
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, previous_state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        previous_state = normalize(previous_state)
        observation = normalize(observation)
        delta = observation - previous_state
        features = torch.cat((previous_state, observation, delta, delta.abs()), dim=-1)
        return torch.sigmoid(self.network(features)).squeeze(-1)


class GRUMemoryUpdater(nn.Module):
    """GRU identity update with an optional learned scalar reliability gate."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIMENSION,
        use_gate: bool = False,
        gate_hidden_dim: int = 256,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_gate = use_gate
        self.gru = nn.GRUCell(input_size=feature_dim, hidden_size=feature_dim)
        self.gate = ReliabilityGate(feature_dim, gate_hidden_dim) if use_gate else None

    def forward(
        self,
        previous_state: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(new_state, reliability, candidate_state)``.

        The state is normalized after the gated interpolation.  A gate value
        of zero preserves the previous state; a value of one uses the GRU
        candidate.  No current observation is scored by this method.
        """

        previous_state = normalize(previous_state)
        observation = normalize(observation)
        candidate = F.normalize(self.gru(observation, previous_state), dim=-1, eps=1e-12)
        if self.gate is None:
            reliability = torch.ones(previous_state.shape[:-1], device=previous_state.device, dtype=previous_state.dtype)
        else:
            reliability = self.gate(previous_state, observation)
        new_state = F.normalize(
            reliability.unsqueeze(-1) * candidate
            + (1.0 - reliability).unsqueeze(-1) * previous_state,
            dim=-1,
            eps=1e-12,
        )
        return new_state, reliability, candidate


def state_stability_loss(previous_state: torch.Tensor, new_state: torch.Tensor) -> torch.Tensor:
    """The declared ``||z_t-z_{t-1}||`` stability penalty."""

    return torch.linalg.vector_norm(new_state - previous_state, dim=-1)


__all__ = ["GRUMemoryUpdater", "ReliabilityGate", "state_stability_loss"]
