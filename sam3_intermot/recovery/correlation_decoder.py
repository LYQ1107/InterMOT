"""HCRD-v1: dense correlation-based target-conditioned re-detection head.

Template: SAM3 level-0 features inside the human box (resized to 8x8) plus
the pretrained identity anchor.  Search: current-frame level-0 features.
Cosine correlation between the pooled template and every search location
produces a targetness heatmap; a small conv head predicts the box and a
presence score.  Dense supervision gives strong gradients and mirrors
SOT-style localization over the full frame (global search).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def level0_grid(enc, device, hw=(72, 72)):
    mem = enc["encoder_hidden_states"].to(device)  # [h*w, bs, d]
    pos = enc["pos_embed"].to(device)
    d = mem.shape[-1]
    nt = mem.shape[0]
    H = W = int(round(nt ** 0.5))
    feat = (mem + pos).permute(1, 2, 0).reshape(1, d, H, W)
    if (H, W) != hw:
        feat = F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)
    return feat


class CorrelationRecoveryDecoder(nn.Module):
    def __init__(self, anchor_dim: int = 1280, d_model: int = 256,
                 template_hw: int = 8, hidden: int = 512,
                 grid_hw: int = 36):
        super().__init__()
        self.d_model = d_model
        self.template_hw = template_hw
        self.grid_hw = grid_hw
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(anchor_dim), nn.Linear(anchor_dim, d_model), nn.GELU()
        )
        self.template_proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.box_head = nn.Conv2d(d_model, 4, 1)
        self.pres_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.normal_(self.box_head.weight, std=0.01)
        nn.init.zeros_(self.box_head.bias)
        nn.init.normal_(self.pres_head[-1].weight, std=0.01)
        nn.init.zeros_(self.pres_head[-1].bias)

    def forward(self, h: torch.Tensor, roi: torch.Tensor, box_norm: torch.Tensor,
                enc_t, enc_f):
        """h: [bs,1280]; roi: [bs,256] (reference ROI feature); box_norm: [bs,4].
        enc_t: reference encoder dict; enc_f: search encoder dict."""
        bs = h.shape[0]
        grid = self.grid_hw
        tfeat = level0_grid(enc_t, h.device, hw=(grid, grid))  # [bs,d,72,72]
        sfeat = level0_grid(enc_f, h.device, hw=(grid, grid))
        # template patch inside the human box
        x1, y1, x2, y2 = box_norm.unbind(-1)
        hh, ww = grid, grid
        ys = torch.linspace(0, 1, hh, device=h.device)[:, None]
        xs = torch.linspace(0, 1, ww, device=h.device)[None, :]
        mask = ((xs >= x1[:, None, None]) & (xs <= x2[:, None, None])
                & (ys >= y1[:, None, None]) & (ys <= y2[:, None, None])).float()
        patch = tfeat * mask.unsqueeze(1)
        patch = F.interpolate(patch, size=(self.template_hw, self.template_hw),
                              mode="bilinear", align_corners=False)
        # identity-conditioned correlation: CLIP anchor modulates both sides
        a = self.anchor_proj(h.float())  # [bs,d]
        t_vec = patch.flatten(2).mean(dim=2).float()  # [bs,d]
        t_vec = self.template_proj(t_vec) + a
        t_vec = F.normalize(t_vec, dim=-1)
        sfeat_n = F.normalize(
            sfeat.permute(0, 2, 3, 1).float() + a.unsqueeze(1).unsqueeze(1), dim=-1
        )  # [bs,72,72,d]
        obj = torch.matmul(sfeat_n, t_vec.unsqueeze(-1)).squeeze(-1)  # [bs,72,72]
        # box/presence conditioning: concat search feature + broadcast template
        hb = a.unsqueeze(1).unsqueeze(1).expand(-1, grid, grid, -1)
        merged = self.merge(torch.cat([sfeat.permute(0, 2, 3, 1).float(), hb], dim=-1))
        merged = merged.permute(0, 3, 1, 2)
        box = self.box_head(merged)  # [bs,4,72,72]
        # presence from top-k aggregated features
        topk = merged.flatten(2).topk(8, dim=2).values.mean(dim=2)
        presence = self.pres_head(topk).squeeze(-1)  # [bs]
        return {"obj": obj, "box": box, "presence": presence}

    def decode_boxes(self, obj, box, ref_box=None):
        """obj: [bs,72,72]; box: [bs,4,72,72] (unused when ref_box given);
        ref_box: [bs,4] cxcywh normalized -> top-K xyxy boxes using the
        human-box size (correlation-peak + reference-scale localization)."""
        obj = obj.float()
        box = box.float()
        bs = obj.shape[0]
        grid = self.grid_hw
        flat = obj.reshape(bs, -1)
        idx = torch.argsort(-flat, dim=1)[:, :5]
        out = []
        for b in range(bs):
            boxes = []
            for i in idx[b]:
                cx = ((i % grid).float() + 0.5) / grid
                cy = ((i // grid).float() + 0.5) / grid
                if ref_box is not None:
                    w = ref_box[b, 2]
                    hh = ref_box[b, 3]
                    cxx, cyy = cx, cy
                else:
                    d = box[b, :, i // grid, i % grid]
                    cxx = torch.sigmoid(d[0]).clamp(0, 1)
                    cyy = torch.sigmoid(d[1]).clamp(0, 1)
                    w = torch.sigmoid(d[2]) * 0.3
                    hh = torch.sigmoid(d[3]) * 0.5
                    cx, cy = cxx, cyy
                boxes.append(torch.stack([cxx - w / 2, cyy - hh / 2,
                                          cxx + w / 2, cyy + hh / 2]))
            out.append(torch.stack(boxes))
        return torch.stack(out)
