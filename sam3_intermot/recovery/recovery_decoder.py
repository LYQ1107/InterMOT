"""Human-Conditioned Recovery Decoder (HCRD-v0).

Target tokens are initialized from the pretrained identity anchor H_i plus the
SAM3 ROI feature at the human box; they cross-attend the full multi-scale
SAM3 encoder features of the current frame and produce K target-specific
proposals (box, targetness, presence).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_features(enc, device):
    """Return [n_tokens, bs, d] memory + positional embeddings from SAM3 enc."""
    mem = enc["encoder_hidden_states"].to(device)
    pos = enc["pos_embed"]
    # pos_embed has the same [n_tokens, bs, d] layout as memory
    return mem + pos.to(device)


class RecoveryDecoder(nn.Module):
    def __init__(self, anchor_dim: int = 1280, roi_dim: int = 256,
                 d_model: int = 256, n_queries: int = 4, n_layers: int = 2,
                 n_heads: int = 4, hidden: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(anchor_dim + roi_dim + 4),
            nn.Linear(anchor_dim + roi_dim + 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_embed = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        layers = []
        for _ in range(n_layers):
            layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "norm1": nn.LayerNorm(d_model),
                        "cross_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
                        ),
                        "norm3": nn.LayerNorm(d_model),
                    }
                )
            )
        self.layers = nn.ModuleList(layers)
        self.box_head = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 4))
        self.tgt_head = nn.Linear(d_model, 1)
        self.presence_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)
        nn.init.zeros_(self.tgt_head.weight)
        nn.init.zeros_(self.tgt_head.bias)
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.zeros_(self.presence_head[-1].bias)

    def forward(self, h: torch.Tensor, roi: torch.Tensor, box: torch.Tensor,
                enc_f):
        """h: [bs, anchor_dim]; roi: [bs, roi_dim]; box: [bs, 4] cxcywh norm;
        enc_f: current-frame encoder dict.  Returns dict of proposals."""
        bs = h.shape[0]
        if os.environ.get("N16_DEBUG"):
            print("N16_DBG h", tuple(h.shape), "roi", tuple(roi.shape),
                  "enc_hist", tuple(enc_f["encoder_hidden_states"].shape),
                  "pos", tuple(enc_f["pos_embed"].shape), flush=True)
        cond = torch.cat([h.float(), roi.float(), box.float()], dim=-1)
        c = self.cond_proj(cond)  # [bs, d]
        q = self.query_embed.unsqueeze(0).expand(bs, -1, -1) + c.unsqueeze(1)
        mem = flatten_features(enc_f, h.device)
        mem = mem.permute(1, 0, 2)  # [bs, n_tokens, d]
        if os.environ.get("N16_DEBUG"):
            print("N16_DBG mem", tuple(mem.shape), flush=True)
        for layer in self.layers:
            q2 = layer["self_attn"](q, q, q, need_weights=False)[0]
            q = layer["norm1"](q + q2)
            q2 = layer["cross_attn"](q, mem, mem, need_weights=False)[0]
            q = layer["norm2"](q + q2)
            q = layer["norm3"](q + layer["ffn"](q))
        box = torch.sigmoid(self.box_head(q))  # [bs, nq, 4] cxcywh norm
        tgt = self.tgt_head(q).squeeze(-1)  # [bs, nq]
        presence = self.presence_head(q.mean(dim=1)).squeeze(-1)  # [bs]
        return {"boxes": box, "targetness": tgt, "presence": presence}
