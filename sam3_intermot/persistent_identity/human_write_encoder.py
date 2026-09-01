"""HumanWriteEncoder: ROI feature -> detector query embedding Q_i^det."""

import torch
import torch.nn as nn


def roi_pool_feature(
    memory: torch.Tensor,
    encoder_out: dict,
    box_norm,
) -> torch.Tensor:
    """Mean-pool level-0 encoder memory inside a normalized box [x0,y0,x1,y1]."""
    mem = memory[:, 0, :]  # bs=1
    ls = encoder_out["level_start_index"].cpu().numpy()
    ss = encoder_out["spatial_shapes"].cpu().numpy()
    h0, w0 = int(ss[0, 0]), int(ss[0, 1])
    end0 = int(ls[1]) if len(ls) > 1 else mem.shape[0]
    feats = mem[:end0]
    x0, y0, x1, y1 = (float(v) for v in box_norm)
    yy = torch.arange(h0, device=mem.device)
    xx = torch.arange(w0, device=mem.device)
    ym = (yy[:, None] >= y0 * h0) & (yy[:, None] <= y1 * h0)
    xm = (xx[None, :] >= x0 * w0) & (xx[None, :] <= x1 * w0)
    sel = (ym & xm).reshape(-1)
    if int(sel.sum().item()) == 0:
        return feats.mean(0)
    return feats[sel].mean(0)


class HumanWriteEncoder(nn.Module):
    """Small MLP mapping an ROI feature into the detector query space."""

    def __init__(self, d_model: int = 256, hidden: int = 512):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        out = self.net(roi_feat)
        return torch.nn.functional.normalize(out, dim=-1)
