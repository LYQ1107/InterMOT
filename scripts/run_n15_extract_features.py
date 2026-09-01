#!/usr/bin/env python
"""Extract pretrained identity features for the N15 Human Seed Identity Benchmark.

Usage:
  python scripts/run_n15_extract_features.py --backbone osnet|clipreid|dinov2
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

ROOT = Path(".")
BENCH = ROOT / "outputs/n15/identity_benchmark/benchmark.json"
OUT_DIR = ROOT / "outputs/n15/features"
DT_ROOT = Path("/path/to/dancetrack")


def crop_image(seq: str, split: str, frame: int, box) -> np.ndarray:
    img = Image.open(DT_ROOT / split / seq / "img1" / f"{frame + 1:08d}.jpg").convert("RGB")
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    return np.asarray(img.crop((x1, y1, x2, y2)), dtype=np.uint8)


class ClipReidViT(torch.nn.Module):
    """CLIP ViT-B/16 visual tower matching Syliz517/CLIP-ReID vanilla config."""

    def __init__(self):
        super().__init__()
        width, layers, heads, out_dim = 768, 12, 12, 512
        h_res, w_res = 16, 8
        self.conv1 = torch.nn.Conv2d(3, width, kernel_size=16, stride=16, bias=False)
        scale = width ** -0.5
        self.class_embedding = torch.nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = torch.nn.Parameter(
            scale * torch.randn(h_res * w_res + 1, width)
        )
        self.ln_pre = torch.nn.LayerNorm(width)
        blocks = []
        for _ in range(layers):
            blocks.append(
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.MultiheadAttention(width, heads),
                        "ln_1": torch.nn.LayerNorm(width),
                        "mlp": _MLP(width),
                        "ln_2": torch.nn.LayerNorm(width),
                    }
                )
            )
        self.transformer = _Transformer(blocks)
        self.ln_post = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Parameter(scale * torch.randn(width, out_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        for b in self.transformer.resblocks[:11]:
            x = self._block(b, x)
        x11 = x
        b12 = self.transformer.resblocks[11]
        x12 = self._block(b12, x)
        x11 = x11.permute(1, 0, 2)
        x12 = x12.permute(1, 0, 2)
        x12 = self.ln_post(x12)
        xproj = x12 @ self.proj
        return x11, x12, xproj

    @staticmethod
    def _block(b, x):
        x = x + b["attn"](b["ln_1"](x), b["ln_1"](x), b["ln_1"](x), need_weights=False)[0]
        x = x + b["mlp"](b["ln_2"](x))
        return x


class _MLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = torch.nn.Linear(width, width * 4)
        self.gelu = _QuickGELU()
        self.c_proj = torch.nn.Linear(width * 4, width)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class _QuickGELU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class _Transformer(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(blocks)


def build_clipreid(path: str, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ClipReidViT().to(device)
    prefix = "image_encoder."
    keys = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(keys, strict=False)
    print(f"clipreid missing={len(missing)} unexpected={len(unexpected)}")
    return model.eval()


def extract(args):
    payload = json.loads(BENCH.read_text(encoding="utf-8"))
    crops = payload["crops"]
    n = len(crops)
    if args.limit:
        n = min(n, args.limit)
        crops = crops[:n]
    seq_split = {q["seq"]: q["split"] for q in payload["queries"]}
    device = "cuda"

    if args.backbone == "osnet":
        from torchreid.reid.utils.feature_extractor import FeatureExtractor

        ext = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"),
            image_size=(256, 128),
            device="cuda",
            verbose=False,
        )
        tf = None
    elif args.backbone == "clipreid":
        model = build_clipreid(
            str(ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"),
            device,
        )
        tf = T.Compose(
            [
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    elif args.backbone == "dinov2":
        import timm

        model = (
            timm.create_model(
                "vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224
            )
            .to(device)
            .eval()
        )
        tf = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError(args.backbone)

    dims = {"osnet": 512, "clipreid": 1280, "dinov2": 768}
    feats = np.zeros((n, dims[args.backbone]), dtype=np.float32)
    batch = 64
    t0 = time.time()
    for start in range(0, n, batch):
        chunk = crops[start : start + batch]
        images = [
            crop_image(c["seq"], seq_split.get(c["seq"], "train"), c["frame"], c["box"])
            for c in chunk
        ]
        if args.backbone == "osnet":
            with torch.no_grad():
                out = ext(images).cpu().numpy()
        else:
            xs = torch.stack([tf(Image.fromarray(im)) for im in images]).to(device)
            with torch.no_grad():
                if args.backbone == "clipreid":
                    _, x12, xproj = model(xs)
                    out = torch.cat([x12[:, 0], xproj[:, 0]], dim=1).cpu().numpy()
                else:
                    f = model.forward_features(xs)
                    if isinstance(f, dict):
                        f = f["x_norm_clstoken"]
                    elif f.ndim == 3:
                        f = f[:, 0]
                    out = f.cpu().numpy()
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
        feats[start : start + len(chunk)] = out
        if (start // batch) % 20 == 0:
            print(f"{start}/{n} elapsed={time.time()-t0:.1f}s", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{args.backbone}.npy", feats)
    print(f"saved {args.backbone}.npy shape={feats.shape} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["osnet", "clipreid", "dinov2"])
    ap.add_argument("--limit", type=int, default=0, help="smoke-test limit")
    args = ap.parse_args()
    extract(args)
