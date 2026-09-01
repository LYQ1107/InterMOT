"""HTD-v1: Human-Conditioned Transformer Detector.

Reference tokens (SAM3 ROI patch + CLIP anchor + geometry) initialize K
target queries; a multi-layer cross-attention decoder reads the full-frame
SAM3 encoder memory and outputs K proposals with box, targetness, presence
and identity-compatibility heads.  Trained on HCRED episodes (cached
features); no future GT at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pos(hw=72, d=256, device="cpu"):
    ys = torch.arange(hw, device=device).float() / hw
    xs = torch.arange(hw, device=device).float() / hw
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx, gy], dim=-1)  # [72,72,2]
    inv = 10000.0 ** (-torch.arange(0, d, 4, device=device).float()[: d // 4] / d)
    pos = []
    for i in range(d // 4):
        pos.append(torch.sin(coords[..., 0:1] * inv[i]))
        pos.append(torch.cos(coords[..., 0:1] * inv[i]))
        pos.append(torch.sin(coords[..., 1:2] * inv[i]))
        pos.append(torch.cos(coords[..., 1:2] * inv[i]))
    return torch.cat(pos, dim=-1)[:d]  # [72,72,d]


class HTD(nn.Module):
    def __init__(self, anchor_dim=1280, d_model=256, n_queries=4,
                 n_layers=3, n_heads=4, hidden=512, grid=72, patch=4,
                 modulate=True):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.grid = grid
        self.patch = patch
        self.modulate = modulate
        self.anchor_proj = nn.Sequential(
            nn.LayerNorm(anchor_dim), nn.Linear(anchor_dim, d_model), nn.GELU()
        )
        self.ref_encoder = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=hidden,
            batch_first=True, dropout=0.0,
        )
        self.query_embed = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=hidden,
                batch_first=True, dropout=0.0,
            ),
            num_layers=n_layers,
        )
        self.box_head = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                      nn.Linear(hidden, 4))
        self.tgt_head = nn.Linear(d_model, 1)
        self.pres_head = nn.Sequential(nn.LayerNorm(d_model),
                                       nn.Linear(d_model, hidden), nn.GELU(),
                                       nn.Linear(hidden, 1))
        self.id_head = nn.Linear(d_model, d_model)
        self.register_buffer("pos", sinusoidal_pos(grid, d_model))
        nn.init.normal_(self.box_head[-1].weight, std=0.01)
        nn.init.zeros_(self.box_head[-1].bias)
        nn.init.normal_(self.tgt_head.weight, std=0.05)
        nn.init.zeros_(self.tgt_head.bias)
        nn.init.normal_(self.pres_head[-1].weight, std=0.05)
        nn.init.zeros_(self.pres_head[-1].bias)

    def reference_tokens(self, ref_mem, ref_box, h):
        """ref_mem: [bs,72,72,d]; ref_box: [bs,4] cxcywh; h: [bs,1280]."""
        bs = ref_mem.shape[0]
        x1, y1, x2, y2 = ref_box.unbind(-1)
        g = self.grid
        ys = torch.arange(g, device=ref_mem.device).float()[:, None]
        xs = torch.arange(g, device=ref_mem.device).float()[None, :]
        mask = ((xs >= x1[:, None, None] * g) & (xs <= x2[:, None, None] * g)
                & (ys >= y1[:, None, None] * g) & (ys <= y2[:, None, None] * g)).float()
        patch = ref_mem * mask.unsqueeze(-1)  # [bs,72,72,d]
        patch = patch.permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=(self.patch, self.patch),
                              mode="bilinear", align_corners=False)
        patch = patch.flatten(2).permute(0, 2, 1)  # [bs,16,d]
        mean_tok = patch.mean(dim=1, keepdim=True)
        h_tok = self.anchor_proj(h.float()).unsqueeze(1)  # [bs,1,d]
        refs = torch.cat([patch, mean_tok, h_tok], dim=1)  # [bs,18,d]
        q = self.query_embed.unsqueeze(0).expand(bs, -1, -1)
        z = self.ref_encoder(q, refs)
        return z

    def forward(self, ref_mem, ref_box, h, search_mem):
        """search_mem: [bs,72,72,d].  Returns proposals."""
        bs = search_mem.shape[0]
        z = self.reference_tokens(ref_mem, ref_box, h)
        s = search_mem + self.pos.unsqueeze(0)
        if self.modulate:
            s = s + self.anchor_proj(h.float()).unsqueeze(1).unsqueeze(1)
        s = s.flatten(1, 2)  # [bs,5184,d]
        out = self.decoder(z, s)  # [bs,K,d]
        box = torch.sigmoid(self.box_head(out))  # [bs,K,4] cxcywh norm
        tgt = self.tgt_head(out).squeeze(-1)  # [bs,K]
        presence = self.pres_head(out.mean(dim=1)).squeeze(-1)  # [bs]
        idf = F.normalize(self.id_head(out), dim=-1)  # [bs,K,d]
        hn = F.normalize(self.anchor_proj(h.float()), dim=-1)
        id_sim = (idf * hn.unsqueeze(1)).sum(-1)  # [bs,K]
        return {"boxes": box, "targetness": tgt, "presence": presence,
                "id_sim": id_sim}

    def decode_boxes(self, boxes):
        """boxes: [bs,K,4] cxcywh -> xyxy normalized."""
        cx, cy, w, hh = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2], dim=-1)
