"""Fixed small MLP for strategy-level correction application selection."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CorrectionApplicationSelector(nn.Module):
    """Return three logits in the fixed K0/K1/K2 order."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    @staticmethod
    def listwise_kl(logits: torch.Tensor, rewards: torch.Tensor, temperature: float) -> torch.Tensor:
        target = torch.softmax((rewards - rewards.max()) / max(float(temperature), 1.0e-4), dim=-1)
        return torch.sum(target * (torch.log(target.clamp_min(1.0e-8)) - torch.log_softmax(logits, dim=-1)), dim=-1).mean()

    @staticmethod
    def pairwise_margin(logits: torch.Tensor, rewards: torch.Tensor, epsilon: float, margin: float) -> torch.Tensor:
        losses = []
        for i in range(logits.shape[-1]):
            for j in range(i + 1, logits.shape[-1]):
                delta = rewards[..., i] - rewards[..., j]
                mask = delta.abs() >= float(epsilon)
                if bool(mask.any()):
                    sign = torch.sign(delta[mask])
                    losses.append(torch.relu(float(margin) - sign * (logits[..., i][mask] - logits[..., j][mask])).mean())
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0

    def loss(self, logits: torch.Tensor, rewards: torch.Tensor, *, temperature: float, epsilon: float, margin: float, pairwise_weight: float, weights: torch.Tensor | None = None) -> torch.Tensor:
        if logits.ndim == 2:
            listwise = []
            pairwise = []
            for index in range(logits.shape[0]):
                listwise.append(self.listwise_kl(logits[index:index + 1], rewards[index:index + 1], temperature))
                pairwise.append(self.pairwise_margin(logits[index:index + 1], rewards[index:index + 1], epsilon, margin))
            value = torch.stack(listwise).squeeze(-1) + float(pairwise_weight) * torch.stack(pairwise).squeeze(-1)
            if weights is not None:
                value = value * weights
            return value.mean()
        return self.listwise_kl(logits, rewards, temperature) + float(pairwise_weight) * self.pairwise_margin(logits, rewards, epsilon, margin)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


__all__ = ["CorrectionApplicationSelector", "parameter_count"]
