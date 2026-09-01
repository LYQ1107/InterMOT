"""Per-slot detector head adapter (N14, working name).

Only active on reserved persistent-query slots.  With an empty query bank the
adapter is bypassed, so official (frozen) outputs stay byte-identical.
"""

import torch
import torch.nn as nn


class SlotHeadAdapter(nn.Module):
    """Per-slot head: box = human reference + learned bounded delta;
    score = identity similarity between the persistent query q (appearance
    written at the human frame) and the future-frame ROI feature.
    The dynamic slot's head is *replaced* by these MLPs; the frozen shared
    heads keep producing all other 199 queries' outputs."""

    def __init__(self, d_model: int = 256, hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.match_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.score_scale = nn.Parameter(torch.tensor(8.0))
        self.score_bias = nn.Parameter(torch.tensor(-4.0))
        in_dim = d_model + d_model + 4
        self.box_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        # Zero-init: box starts at the reference (safe for B0 and early steps).
        nn.init.zeros_(self.box_net[-1].weight)
        nn.init.zeros_(self.box_net[-1].bias)

    def forward(self, q, roi_cand, roi_anchor, ref):
        """q: [bs, d_model], roi_cand: [bs, d_model] (identity candidate),
        roi_anchor: [bs, d_model] (reference-location appearance) ->
        box_delta [bs,4] (bounded), score_logit [bs,1]"""
        qn = torch.nn.functional.normalize(q, dim=-1)
        roi_n = torch.nn.functional.normalize(roi_cand, dim=-1)
        f = torch.nn.functional.normalize(
            self.match_proj(roi_n), dim=-1
        )
        dscore = (
            self.score_scale * (f * qn).sum(-1, keepdim=True) + self.score_bias
        )
        anchor_n = torch.nn.functional.normalize(roi_anchor, dim=-1)
        x = torch.cat([qn, anchor_n, ref], dim=-1)
        dbox = 0.5 * torch.tanh(self.box_net(x))
        return dbox, dscore
