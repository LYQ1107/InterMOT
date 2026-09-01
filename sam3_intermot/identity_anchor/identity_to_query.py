"""Identity-to-Query (I2Q) generators for N15.

H_i (pretrained identity anchor) -> Q_i (SAM3 dynamic detector query).
Linear baseline first; an optional gated contextual projector is trained only
if the linear baseline fails the Query Swap Test.
"""

import torch
import torch.nn as nn


class LinearI2Q(nn.Module):
    """Linear projector from a pretrained identity embedding to the detector
    query space.  Output is normalized to match the frozen query-embed
    distribution scale used by the decoder (unit norm)."""

    def __init__(self, in_dim: int, d_model: int, hidden: int = 1024):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # zero-init the last layer: start at the mean query (safe warm start)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(h.float()), dim=-1)


class GatedContextualI2Q(nn.Module):
    """H_i + current-frame SAM ROI context -> Q_i (small gated projector).

    Identity anchor is authoritative; the context only modulates the query via
    a gate so motion/geometry cannot override identity."""

    def __init__(self, in_dim: int, d_model: int, context_dim: int = 256,
                 hidden: int = 512):
        super().__init__()
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        a = self.anchor_proj(h.float())
        c = self.context_proj(ctx.float())
        g = torch.sigmoid(self.scale + self.gate(torch.cat([a, c], dim=-1)))
        q = a + g * c
        return torch.nn.functional.normalize(q, dim=-1)


def load_i2q(path: str, in_dim: int, d_model: int, variant: str = "linear"):
    cls = LinearI2Q if variant == "linear" else GatedContextualI2Q
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = cls(in_dim=in_dim, d_model=d_model)
    model.load_state_dict(ck["state"])
    return model

