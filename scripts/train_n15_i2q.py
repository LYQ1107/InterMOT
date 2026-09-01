#!/usr/bin/env python
"""Train N15 Linear I2Q: pretrained H_i -> SAM3 detector query Q_i.

Supervision: the injected query slot's frozen-decoder output at a future frame
must have high score and a box close to the target GT (positive), and low
score when the identity is absent (negative).  No future GT at inference.
"""

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "outputs/n15"
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


def deep_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return [deep_clone(v) for v in x]
    if isinstance(x, dict):
        return {k: deep_clone(v) for k, v in x.items()}
    return x


def cxcywh_norm(box, iw, ih):
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.asarray(
        [(x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, (x2 - x1) / iw, (y2 - y1) / ih],
        dtype=float,
    )


def cxcywh_to_xyxy(cx, cy, w, h):
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def clear_model_caches(model) -> int:
    n = 0
    for m in model.modules():
        for k in list(vars(m)):
            v = getattr(m, k, None)
            if k == "cache" and isinstance(v, dict):
                setattr(m, k, {})
                n += 1
            elif k == "coord_cache" and isinstance(v, dict):
                setattr(m, k, {})
                n += 1
            elif k == "compilable_cord_cache":
                setattr(m, k, None)
                n += 1
            if isinstance(v, dict):
                for kk, vv in list(v.items()):
                    if isinstance(vv, torch.Tensor) and torch.is_inference(vv):
                        v[kk] = vv.clone()
                        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--box-weight", type=float, default=5.0)
    ap.add_argument("--score-weight", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--slot", type=int, default=199)
    ap.add_argument("--max-samples", type=int, default=1200)
    ap.add_argument("--out", default="i2q_linear_v1")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.identity_anchor.identity_to_query import LinearI2Q
    from sam3_intermot.persistent_identity import (
        install_query_patch,
        roi_pool_feature,
    )
    from sam3.model.geometry_encoders import Prompt

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
    print("d_model", d_model, flush=True)

    # CLIP-ReID extractor (frozen)
    sys.path.insert(0, str(ROOT))
    from scripts.run_n15_extract_features import build_clipreid

    clip = build_clipreid(str(CLIP_CKPT), "cuda")
    import torchvision.transforms as T

    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def anchor_feat(seq, frame, box):
        img = Image.open(DT / "train" / seq / "img1" / f"{frame + 1:08d}.jpg").convert("RGB")
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        crop = img.crop((x1, y1, x2, y2))
        x = clip_tf(crop).unsqueeze(0).cuda()
        with torch.no_grad():
            _, x12, xproj = clip(x)
            fv = torch.cat([x12[:, 0], xproj[:, 0]], dim=1)
        return F.normalize(fv, dim=-1)

    i2q = LinearI2Q(in_dim=1280, d_model=d_model, hidden=args.hidden).cuda()
    opt = torch.optim.AdamW(i2q.parameters(), lr=args.lr, weight_decay=1e-4)

    rows = []
    with (OUT / "i2q_train_manifest.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if args.max_samples:
        rng = np.random.default_rng(42)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
    print(f"samples={len(rows)}", flush=True)

    ds_video = {}
    frame_cache = {}
    text_out = None
    empty_geo = None

    def text_features():
        nonlocal text_out
        if text_out is None:
            with torch.no_grad():
                tx = model.detector.backbone.forward_text(["person"], device="cuda")
                text_out = {
                    "language_features": tx["language_features"].clone(),
                    "language_mask": tx["language_mask"].clone(),
                }
        return text_out

    def encoder_features(seq, f):
        key = (seq, f)
        if key in frame_cache:
            return frame_cache[key]
        if ds_video.get("seq") != seq:
            if ds_video.get("seq") is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            backend.start_video(str(DT / "train" / seq / "img1"))
            ds_video["seq"] = seq
            clear_model_caches(model)
        state = backend._predictor._all_inference_states[backend._session_id]["state"]
        ib = state["input_batch"]
        fin = clone_find_input(ib.find_inputs[f], img_id=0)
        img_t = ib.img_batch.tensors[f].unsqueeze(0).clone().to("cuda")
        tx = text_features()
        bo = {
            "img_batch_all_stages": img_t,
            "language_features": tx["language_features"],
            "language_mask": tx["language_mask"],
        }
        nonlocal empty_geo
        if empty_geo is None:
            empty_geo = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device="cuda"),
                box_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                box_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
                point_embeddings=torch.zeros(0, 1, 2, device="cuda"),
                point_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                point_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
            )
        with torch.no_grad():
            prompt, pmask, bo2 = model.detector._encode_prompt(bo, fin, empty_geo)
            bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)
        feat = {
            "enc": deep_clone(enc),
            "prompt": prompt.clone(),
            "pmask": pmask.clone(),
        }
        # move to CPU to protect GPU memory; to_cuda() moves back per sample
        feat = {
            k: ({kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv)
                 for kk, vv in v.items()} if isinstance(v, dict)
                else (v.cpu() if isinstance(v, torch.Tensor) else v))
            for k, v in feat.items()
        }
        frame_cache[key] = feat
        if len(frame_cache) > 64:
            frame_cache.pop(next(iter(frame_cache)))
        return feat

    def to_cuda(feat):
        out = {}
        for k, v in feat.items():
            if isinstance(v, dict):
                out[k] = {kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv)
                          for kk, vv in v.items()}
            elif isinstance(v, torch.Tensor):
                out[k] = v.cuda()
            else:
                out[k] = v
        return out

    iw, ih = 1920, 1080
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        losses, ious, scores = [], [], []
        for r in rows:
            seq = r["sequence"]
            t, f = int(r["human_frame"]), int(r["future_frame"])
            hb = np.asarray(json.loads(r["human_box"]), dtype=float)
            fb = np.asarray(json.loads(r["future_box"]), dtype=float)
            ft = to_cuda(encoder_features(seq, t))
            ff = to_cuda(encoder_features(seq, f))
            h = anchor_feat(seq, t, hb)
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                q = i2q(h)[0]
                ref = torch.as_tensor(
                    cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda"
                )
                bank = lambda: ([q], [ref])
                uninstall = install_query_patch(image, bank, [args.slot])
                try:
                    out = {"encoder_hidden_states": ff["enc"]["encoder_hidden_states"]}
                    out, _ = image._run_decoder(
                        pos_embed=ff["enc"]["pos_embed"],
                        memory=ff["enc"]["encoder_hidden_states"],
                        src_mask=ff["enc"]["padding_mask"],
                        out=out,
                        prompt=ff["prompt"],
                        prompt_mask=ff["pmask"],
                        encoder_out=ff["enc"],
                    )
                    slot_logit = out["pred_logits"][0, args.slot, 0]
                    slot_box = out["pred_boxes"][0, args.slot]
                finally:
                    uninstall()
                if int(r["visible"]) == 1:
                    loss = args.score_weight * F.binary_cross_entropy_with_logits(
                        slot_logit.reshape(1), torch.ones(1, device="cuda")
                    )
                    gt_cxcywh = torch.as_tensor(
                        cxcywh_norm(fb, iw, ih), dtype=torch.float32, device="cuda"
                    )
                    loss = loss + args.box_weight * F.l1_loss(slot_box, gt_cxcywh)
                    sb = cxcywh_to_xyxy(*slot_box.detach().cpu().tolist())
                    ious.append(iou_xyxy(sb * np.asarray([iw, ih, iw, ih]), fb))
                else:
                    loss = args.score_weight * F.binary_cross_entropy_with_logits(
                        slot_logit.reshape(1), torch.zeros(1, device="cuda")
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(i2q.parameters(), 5.0)
                opt.step()
                losses.append(float(loss))
                scores.append(float(torch.sigmoid(slot_logit).item()))
            if len(losses) % 200 == 0:
                print(
                    f"ep{epoch} step={len(losses)} loss={float(np.mean(losses)):.4f} "
                    f"iou={float(np.mean(ious)):.4f} score={float(np.mean(scores)):.4f} "
                    f"elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
        print(
            f"EPOCH {epoch}: loss={float(np.mean(losses)):.4f} "
            f"iou={float(np.mean(ious)):.4f} score={float(np.mean(scores)):.4f}",
            flush=True,
        )
    torch.save(
        {"state": i2q.state_dict(), "args": vars(args), "d_model": d_model,
         "in_dim": 1280},
        OUT / "models" / f"{args.out}.pt",
    )
    print("SAVED", args.out, flush=True)
    try:
        runner.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
