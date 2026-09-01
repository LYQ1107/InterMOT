#!/usr/bin/env python
"""Load GFN (SeqNeXt) official CUHK-SYSU ConvNeXt-B checkpoint for inference.

The checkpoint's saved config uses slightly older key names than the current
repo code; we overlay the repo default.yaml and rename a few keys. Only the
official checkpoint and the official repo source are used.
"""

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(".")
GFN = ROOT / "third_party/GFN"
CKPT = ROOT / "outputs/n18/checkpoints/gfn_cuhk_convnext_pytorch.pt"

sys.path.insert(0, str(GFN / "src"))

RENAMES = {
    "use_gfn_image_lut": "gfn_use_image_lut",
    "gfn_temp": "gfn_train_temp",
    "gfn_pool_size": "gfn_scene_pool_size",
}


def build_config():
    default = yaml.safe_load((GFN / "configs/default.yaml").read_text())
    ckpt = torch.load(str(CKPT), map_location="cpu")
    saved = dict(ckpt["config"])
    cfg = dict(default)
    cfg.update(saved)
    for old, new in RENAMES.items():
        if old in cfg:
            cfg[new] = cfg.pop(old)
    cfg.setdefault("gfn_se_temp", 0.2)
    # The checkpoint was saved with an older config schema; the state dict is
    # a ConvNeXt-B SeqNeXt (backbone.body.*, 27-block stage5), so rebuild with
    # the current builder names and load all weights from the checkpoint.
    cfg["model"] = "convnext"
    cfg["backbone_arch"] = "convnext_base"
    cfg["pretrained"] = False
    return cfg, ckpt["model"], ckpt["epoch"]


def load_model(device="cuda:0"):
    from osr.models.seqnext import SeqNeXt

    cfg, state_dict, epoch = build_config()
    # CUHK-SYSU training split has 5532 identities; this only sizes the OIM
    # buffers, which are removed from the state dict before loading anyway.
    model = SeqNeXt(cfg, oim_lut_size=5532, device=device)
    for key in (
        "roi_heads.reid_loss.lut", "roi_heads.reid_loss.cq",
        "gfn.reid_loss.lut", "gfn.reid_loss.cq",
        "roi_heads.gfn.reid_loss.lut", "roi_heads.gfn.reid_loss.cq",
    ):
        state_dict.pop(key, None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, epoch, missing, unexpected


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg, epoch, missing, unexpected = load_model(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded epoch={epoch} device={dev} params={n/1e6:.1f}M")
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("missing sample:", missing[:8])
    if unexpected:
        print("unexpected sample:", unexpected[:8])
