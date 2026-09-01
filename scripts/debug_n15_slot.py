#!/usr/bin/env python
"""Probe the injected-query slot output on one event/frame."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")
CLIP_CKPT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"


def clone_find_input(fin, img_id: int):
    out = copy.copy(fin)
    for field in fin.__dataclass_fields__:
        v = getattr(out, field)
        if isinstance(v, torch.Tensor):
            setattr(out, field, v.clone())
        elif isinstance(v, list) and v and all(isinstance(x, torch.Tensor) for x in v):
            setattr(out, field, [x.clone() for x in v])
    out.img_ids = torch.tensor([img_id], dtype=torch.long, device="cuda")
    return out


def cxcywh_norm(box, iw, ih):
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.asarray(
        [(x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, (x2 - x1) / iw, (y2 - y1) / ih],
        dtype=float,
    )


def cxcywh_to_xyxy(cx, cy, w, h):
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="outputs/n15/models/i2q_linear_v1.pt")
    ap.add_argument("--slot", type=int, default=199)
    ap.add_argument("--seq", default="dancetrack0074")
    ap.add_argument("--t", type=int, default=6)
    ap.add_argument("--gid", type=int, default=1)
    ap.add_argument("--f", type=int, default=7)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.identity_anchor.identity_to_query import LinearI2Q
    from sam3_intermot.persistent_identity import install_query_patch
    from sam3.model.geometry_encoders import Prompt
    from scripts.run_n15_extract_features import build_clipreid
    import torchvision.transforms as T

    ck = torch.load(ROOT / args.model, map_location="cpu", weights_only=False)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.use_batched_grounding = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    image = model.detector
    d_model = image.transformer.decoder.query_embed.weight.shape[1]
    i2q = LinearI2Q(in_dim=1280, d_model=d_model, hidden=1024).cuda().eval()
    i2q.load_state_dict(ck["state"])
    clip = build_clipreid(str(CLIP_CKPT), "cuda")
    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def clip_feat(img_path, box):
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        x = clip_tf(img.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
        with torch.no_grad():
            _, x12, xproj = clip(x)
            fv = torch.cat([x12[:, 0], xproj[:, 0]], dim=1)
        return F.normalize(fv, dim=-1)

    ds = DanceTrackDataset(str(DT), sequences=None, split="train")
    backend.start_video(str(DT / "train" / args.seq / "img1"))
    iw, ih = backend._frame_w, backend._frame_h
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    ib = state["input_batch"]

    def enc_for(f):
        fin = clone_find_input(ib.find_inputs[f], img_id=0)
        img_t = ib.img_batch.tensors[f].unsqueeze(0).clone().to("cuda")
        tx = model.detector.backbone.forward_text(["person"], device="cuda")
        bo = {
            "img_batch_all_stages": img_t,
            "language_features": tx["language_features"].clone(),
            "language_mask": tx["language_mask"].clone(),
        }
        geo = Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device="cuda"),
            box_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
            box_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
            point_embeddings=torch.zeros(0, 1, 2, device="cuda"),
            point_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
        )
        with torch.no_grad():
            prompt, pmask, bo2 = model.detector._encode_prompt(bo, fin, geo)
            bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)
        return enc, prompt, pmask

    gt = ds.load_gt(args.seq)
    hb = np.asarray(gt[args.t].boxes[gt[args.t].gt_ids.index(args.gid)], dtype=float)
    fb = np.asarray(gt[args.f].boxes[gt[args.f].gt_ids.index(args.gid)], dtype=float)
    hf = clip_feat(f"{DT}/train/{args.seq}/img1/{args.t + 1:08d}.jpg", hb)
    q = i2q(hf)[0]
    ref = torch.as_tensor(cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda")
    enc, prompt, pmask = enc_for(args.f)
    for name, query in [
        ("trained_i2q", q),
        ("zero", torch.zeros_like(q)),
        ("random", F.normalize(torch.randn_like(q), dim=0)),
        ("static_query_embed", image.transformer.decoder.query_embed.weight[args.slot].detach()),
    ]:
        uninstall = install_query_patch(image, lambda qq=query: ([qq], [ref]), [args.slot])
        try:
            out = {"encoder_hidden_states": enc["encoder_hidden_states"]}
            with torch.no_grad():
                out, _ = image._run_decoder(
                    pos_embed=enc["pos_embed"],
                    memory=enc["encoder_hidden_states"],
                    src_mask=enc["padding_mask"],
                    out=out,
                    prompt=prompt,
                    prompt_mask=pmask,
                    encoder_out=enc,
                )
            sl = out["pred_logits"][0, args.slot, 0].item()
            sb = out["pred_boxes"][0, args.slot].detach().cpu().tolist()
            sb_px = cxcywh_to_xyxy(*sb) * np.asarray([iw, ih, iw, ih])
            gt_norm = cxcywh_norm(fb, iw, ih)
            print(
                f"{name}: score={torch.sigmoid(torch.tensor(sl)).item():.4f} "
                f"slot_px={np.round(sb_px,1)} gt_px={np.round(fb,1)} "
                f"ref_px={np.round(hb,1)} ||q||={float(torch.norm(query)):.3f}",
                flush=True,
            )
        finally:
            uninstall()


if __name__ == "__main__":
    main()

