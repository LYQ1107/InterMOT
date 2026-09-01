"""Small causal candidate selector for N31 Path B.

The selector receives only correction-time geometry, official predicted-IoU
metadata, mask area and a candidate-type encoding.  It has no public-ID,
future-frame, or future-ground-truth input.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CorrectionStateSelector(nn.Module):
    """MLP scorer whose logits are compared only among one episode's candidates."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.network(features)
        return values.squeeze(-1)

    @staticmethod
    def listwise_loss(scores: torch.Tensor, rewards: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Soft-label listwise loss for one episode's candidate list."""

        target = torch.softmax((rewards - rewards.max()) / max(float(temperature), 1.0e-4), dim=0)
        return -(target * torch.log_softmax(scores, dim=0)).sum()

    def select(self, features: torch.Tensor) -> int:
        with torch.no_grad():
            scores = self(features)
        return int(torch.argmax(scores).item())


def selector_state_dict(model: CorrectionStateSelector, *, mean: Any, scale: Any) -> dict[str, Any]:
    return {
        "input_dim": int(model.input_dim),
        "hidden_dim": int(model.hidden_dim),
        "state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_scale": scale,
        "future_gt_used_for_input": False,
        "public_id_used_for_input": False,
    }


__all__ = ["CorrectionStateSelector", "selector_state_dict"]
