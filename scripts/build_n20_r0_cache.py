#!/usr/bin/env python
"""Precompute R0 gallery/query embeddings per sequence (CPU) for N20."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402

CACHE = ROOT / "outputs/n18/route_c/gfn_cache"
OUT = ROOT / "outputs/n20/gfn_cache_r0"


def head_embed(model, f4, f5):
    with torch.inference_mode():
        emb, _ = model.roi_heads.embedding_head(
            {"feat_res4": f4, "feat_res5": f5})
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)


def main():
    split = json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]
    seqs = sorted(set(split["calibration10"]) | set(split["train30"]))
    gfn, _, _, _, _ = load_model("cpu")
    gfn.eval()
    r0_path = ROOT / "outputs/n18/route_c/models/r0_best.pt"
    if r0_path.exists():
        gfn.roi_heads.embedding_head.load_state_dict(
            torch.load(r0_path, map_location="cpu"))
        gfn.roi_heads.embedding_head.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        p = OUT / f"{seq}.npz"
        if p.exists():
            print(f"skip {seq}", flush=True)
            continue
        z = np.load(CACHE / f"{seq}.npz")
        qz = np.load(CACHE / f"{seq}_queries.npz")
        f4 = torch.from_numpy(z["feat4"].astype(np.float32))
        f5 = torch.from_numpy(z["feat5"].astype(np.float32))
        qf4 = torch.from_numpy(qz["qfeat4"].astype(np.float32))
        qf5 = torch.from_numpy(qz["qfeat5"].astype(np.float32))
        with torch.inference_mode():
            r0g = head_embed(gfn, f4, f5).numpy()
            r0q = head_embed(gfn, qf4, qf5).numpy()
        np.savez(p, r0g=r0g, r0q=r0q)
        z.close()
        qz.close()
        print(f"r0 {seq} {r0g.shape} {r0q.shape}", flush=True)
    print("R0_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
