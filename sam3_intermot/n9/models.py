"""Lightweight N9 association models (pairwise MLP, set-level cross-attention)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseMLP(nn.Module):
    """Same-identity probability from (memory feature, row feature, motion)."""

    def __init__(self, feat_dim: int = 512, motion_dim: int = 10, hidden: int = 256):
        super().__init__()
        self.net = MLP(feat_dim * 2 + motion_dim, hidden, 1, depth=3)

    def forward(
        self,
        mem_feat: torch.Tensor,
        row_feat: torch.Tensor,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([mem_feat, row_feat, motion], dim=-1)
        return self.net(x).squeeze(-1)


class SetAssociator(nn.Module):
    """Set-level cross-attention association with one-to-one via Hungarian."""

    def __init__(
        self,
        feat_dim: int = 512,
        motion_dim: int = 10,
        d: int = 128,
        layers: int = 2,
        heads: int = 2,
        max_objects: int = 24,
    ):
        super().__init__()
        self.d = d
        self.mem_proj = MLP(feat_dim + motion_dim, d, d, depth=2)
        self.row_proj = MLP(feat_dim + motion_dim, d, d, depth=2)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(d, heads, batch_first=True)
                for _ in range(layers)
            ]
        )
        self.out_proj = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        self.motion_net = nn.Sequential(
            nn.Linear(motion_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.max_objects = max_objects

    def forward(
        self,
        mem_feat: torch.Tensor,
        row_feat: torch.Tensor,
        mem_motion: torch.Tensor,
        row_motion: torch.Tensor,
        mem_mask: Optional[torch.Tensor] = None,
        row_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns logits [B, R, M] (row x memory)."""
        m = self.mem_proj(torch.cat([mem_feat, mem_motion], dim=-1))
        r = self.row_proj(torch.cat([row_feat, row_motion], dim=-1))
        for layer in self.layers:
            attn_mask = None
            if mem_mask is not None:
                attn_mask = torch.zeros(
                    (r.shape[1], m.shape[1]), dtype=torch.bool, device=r.device
                )
                attn_mask = mem_mask.unsqueeze(1).expand(-1, r.shape[1], -1).contiguous()
                attn_mask = attn_mask.reshape(-1, m.shape[1])
                attn_mask = ~attn_mask
            r2, _ = layer(r, m, m, key_padding_mask=attn_mask if False else None)
            r = r + r2
        r = self.out_proj(r)
        logits = torch.bmm(r, m.transpose(1, 2)) * self.logit_scale
        # pairwise motion logits (row-memory context)
        B, R, M = logits.shape
        rmv = row_motion[:, :, None, :].expand(B, R, M, -1)
        mmv = mem_motion[:, None, :, :].expand(B, R, M, -1)
        logits = logits + self.motion_net(torch.cat([rmv, mmv], dim=-1)).squeeze(-1)
        if row_mask is not None:
            logits = logits.masked_fill(~row_mask.unsqueeze(2), -1e9)
        if mem_mask is not None:
            logits = logits.masked_fill(~mem_mask.unsqueeze(1), -1e9)
        return logits


def hungarian_assignment(logits: torch.Tensor) -> torch.Tensor:
    """One-to-one assignment maximizing total logits (rows x memories)."""
    n = logits.shape[0]
    m = logits.shape[1]
    size = max(n, m)
    cost = -logits.detach().cpu().float().numpy()
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    rows, cols = linear_sum_assignment(np.nan_to_num(cost, nan=1e9))
    return torch.as_tensor(np.stack([rows, cols], axis=1))
