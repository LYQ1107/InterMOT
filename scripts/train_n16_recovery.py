#!/usr/bin/env python
"""Train HCRD-v0: human-conditioned recovery decoder."""

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "outputs/n16"
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


def giou_xyxy(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    iou = inter / (union + 1e-9)
    cx1, cy1 = min(a[0], b[0]), min(a[1], b[1])
    cx2, cy2 = max(a[2], b[2]), max(a[3], b[3])
    c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return iou - (c_area - union) / (c_area + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--out", default="hcrd_v0")
    ap.add_argument("--n-queries", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--arch", default="correlation", choices=["decoder", "correlation"])
    ap.add_argument("--manifest", default="hcred_manifest.csv")
    ap.add_argument("--finetune", type=int, default=0,
                    help="F1: unfreeze SAM3 transformer encoder for search frames")
    ap.add_argument("--finetune-res", type=int, default=0,
                    help="resize encoder input to HxH when fine-tuning (speed)")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.persistent_identity import roi_pool_feature
    from sam3_intermot.recovery.recovery_decoder import RecoveryDecoder
    from sam3.model.geometry_encoders import Prompt
    from scripts.run_n15_extract_features import build_clipreid
    import torchvision.transforms as T

    rows = []
    with (OUT / args.manifest).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if args.seqs and r["sequence"] not in args.seqs.split(","):
                continue
            rows.append(r)
    if args.max_samples:
        rng = np.random.default_rng(7)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
    # group by sequence and shuffle within groups to keep the backend warm
    from collections import OrderedDict, defaultdict
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["sequence"]].append(r)
    rows = []
    for seq in sorted(by_seq):
        rng.shuffle(by_seq[seq])
        rows.extend(by_seq[seq])
    print(f"samples={len(rows)}", flush=True)

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
    clip = build_clipreid(str(CLIP_CKPT), "cuda")
    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    if args.arch == "correlation":
        from sam3_intermot.recovery.correlation_decoder import CorrelationRecoveryDecoder
        model_rec = CorrelationRecoveryDecoder(
            anchor_dim=1280, d_model=d_model, template_hw=8, grid_hw=36
        ).cuda()
    else:
        model_rec = RecoveryDecoder(
            anchor_dim=1280, roi_dim=d_model, d_model=d_model,
            n_queries=args.n_queries, n_layers=args.n_layers,
        ).cuda()
    enc_params = []
    if args.finetune:
        enc = model.detector.transformer.encoder
        for p in enc.parameters():
            p.requires_grad_(True)
        enc_params = list(enc.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": model_rec.parameters(), "lr": args.lr, "weight_decay": 1e-4},
            {"params": enc_params, "lr": args.lr * 0.01, "weight_decay": 1e-4},
        ]
        if enc_params else model_rec.parameters(),
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

    def encoder_features(seq, f, want_grad=False):
        need_grad = want_grad
        key = (seq, f)
        if need_grad:
            # search-frame features must be differentiable; compute fresh
            return encoder_features_grad(seq, f)
        if key in frame_cache:
            return frame_cache[key]
        cache_dir = OUT / "enc_cache"
        cache_path = cache_dir / f"{seq}_{f}.npz"
        if cache_path.exists():
            z = np.load(cache_path)
            enc = {
                "encoder_hidden_states": torch.from_numpy(z["mem"].astype(np.float32)),
                "pos_embed": torch.from_numpy(z["pos"].astype(np.float32)),
                "spatial_shapes": torch.from_numpy(z["shapes"]),
                "level_start_index": torch.from_numpy(z["starts"]),
            }
            feat = {"enc": enc}
            frame_cache[key] = feat
            return feat
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
        if args.finetune_res:
            img_t = torch.nn.functional.interpolate(
                img_t, size=(args.finetune_res, args.finetune_res),
                mode="bilinear", align_corners=False,
            )
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
        feat = {
            k: ({kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv)
                 for kk, vv in v.items()} if isinstance(v, dict)
                else (v.cpu() if isinstance(v, torch.Tensor) else v))
            for k, v in feat.items()
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            mem=enc["encoder_hidden_states"].detach().cpu().to(torch.float16).numpy(),
            pos=enc["pos_embed"].detach().cpu().to(torch.float16).numpy(),
            shapes=enc["spatial_shapes"].detach().cpu().numpy(),
            starts=enc["level_start_index"].detach().cpu().numpy(),
        )
        frame_cache[key] = feat
        if len(frame_cache) > 96:
            frame_cache.pop(next(iter(frame_cache)))
        return feat

    def encoder_features_grad(seq, f):
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
        if args.finetune_res:
            img_t = torch.nn.functional.interpolate(
                img_t, size=(args.finetune_res, args.finetune_res),
                mode="bilinear", align_corners=False,
            )
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
        with torch.enable_grad():
            prompt, pmask, bo2 = model.detector._encode_prompt(bo, fin, empty_geo)
            bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)
        return {"enc": enc}

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
        losses, ious, pres = [], [], []
        for r in rows:
            seq = r["sequence"]
            t, f = int(r["t"]), int(r["f"])
            hb = np.asarray(json.loads(r["human_box"]), dtype=float)
            present = int(r["target_present"]) == 1
            fb = np.asarray(json.loads(r["target_box"]), dtype=float) if present else None
            enc_t = to_cuda(encoder_features(seq, t))
            enc_f = encoder_features(seq, f, want_grad=bool(args.finetune))
            box_norm = torch.as_tensor(
                cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda"
            )
            with torch.no_grad():
                roi = roi_pool_feature(
                    enc_t["enc"]["encoder_hidden_states"], enc_t["enc"], box_norm.cpu().numpy()
                ).float().to("cuda")
                h = anchor_feat(seq, t, hb)
            opt.zero_grad(set_to_none=True)
            if args.finetune:
                clear_model_caches(model)
            if args.arch == "correlation":
                pred = model_rec(h, roi.unsqueeze(0), box_norm.unsqueeze(0),
                                 enc_t["enc"], enc_f["enc"])
                obj = pred["obj"][0]
                boxd = pred["box"][0]
                presence = pred["presence"][0]
                grid = model_rec.grid_hw
                ys = torch.linspace(0, 1, grid, device="cuda")[:, None]
                xs = torch.linspace(0, 1, grid, device="cuda")[None, :]
                if present:
                    gt_cx = (fb[0] + fb[2]) / 2 / iw
                    gt_cy = (fb[1] + fb[3]) / 2 / ih
                    gt_w = (fb[2] - fb[0]) / iw
                    gt_h = (fb[3] - fb[1]) / ih
                    sigma = max(gt_w, gt_h) / 2 * grid
                    dist2 = ((xs - gt_cx) * grid) ** 2 + ((ys - gt_cy) * grid) ** 2
                    target = torch.exp(-dist2 / (2 * sigma ** 2)).to("cuda")
                else:
                    target = torch.zeros(grid, grid, device="cuda")
                obj_loss = F.binary_cross_entropy_with_logits(
                    obj, target, pos_weight=torch.tensor(5.0, device="cuda")
                )
                presence_loss = F.binary_cross_entropy_with_logits(
                    presence.reshape(1),
                    torch.tensor([1.0], device="cuda") if present
                    else torch.zeros(1, device="cuda"),
                )
                loss = obj_loss + presence_loss
                with torch.no_grad():
                    pboxes = model_rec.decode_boxes(
                        pred["obj"], pred["box"], ref_box=box_norm.unsqueeze(0)
                    )[0].detach().cpu().numpy()
                    if present:
                        best = max(
                            iou_xyxy(pb * np.asarray([iw, ih, iw, ih]), fb)
                            for pb in pboxes
                        )
                        ious.append(best)
            else:
                pred = model_rec(h, roi.unsqueeze(0), box_norm.unsqueeze(0), enc_f["enc"])
                boxes = pred["boxes"][0]
                tgt = pred["targetness"][0]
                presence = pred["presence"][0]
                loss = F.binary_cross_entropy_with_logits(
                    presence.reshape(1), torch.tensor([1.0], device="cuda")
                ) if present else F.binary_cross_entropy_with_logits(
                    presence.reshape(1), torch.zeros(1, device="cuda")
                )
                if present:
                    gt_box = torch.as_tensor(
                        cxcywh_norm(fb, iw, ih), dtype=torch.float32, device="cuda"
                    )
                    gt_xyxy = torch.as_tensor(
                        [fb[0] / iw, fb[1] / ih, fb[2] / iw, fb[3] / ih],
                        dtype=torch.float32, device="cuda",
                    )
                    cx, cy, w, hh = boxes.unbind(-1)
                    px1 = (cx - w / 2).clamp(0, 1)
                    py1 = (cy - hh / 2).clamp(0, 1)
                    px2 = (cx + w / 2).clamp(0, 1)
                    py2 = (cy + hh / 2).clamp(0, 1)
                    ix1 = torch.maximum(px1, gt_xyxy[0])
                    iy1 = torch.maximum(py1, gt_xyxy[1])
                    ix2 = torch.minimum(px2, gt_xyxy[2])
                    iy2 = torch.minimum(py2, gt_xyxy[3])
                    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                    ua = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
                    ub = (gt_xyxy[2] - gt_xyxy[0]) * (gt_xyxy[3] - gt_xyxy[1])
                    iou = inter / (ua + ub - inter + 1e-9)
                    match = int(torch.argmax(iou).item())
                    match_loss = F.l1_loss(boxes[match], gt_box) + 0.5 * (1 - iou[match])
                    tgt_loss = F.binary_cross_entropy_with_logits(
                        tgt, torch.zeros_like(tgt)
                    )
                    tgt_loss = tgt_loss + F.binary_cross_entropy_with_logits(
                        tgt[match].reshape(1), torch.ones(1, device="cuda")
                    ) - F.binary_cross_entropy_with_logits(
                        tgt[match].reshape(1), torch.zeros(1, device="cuda")
                    )
                    loss = loss + 5.0 * match_loss + tgt_loss
                    ious.append(float(iou[match].item()))
                else:
                    tgt_loss = F.binary_cross_entropy_with_logits(tgt, torch.zeros_like(tgt))
                    loss = loss + tgt_loss
            loss.backward()
            if os.environ.get("N16_DEBUG"):
                gn = sum(p.grad.abs().sum().item() for p in model_rec.parameters()
                         if p.grad is not None)
                wn = sum(p.abs().sum().item() for p in model_rec.parameters())
                dev = next(model_rec.parameters()).device
                outv = float(pred["obj"][0, 0, 0]) if args.arch == "correlation" else float(pred["targetness"][0, 0])
                w0 = float(next(model_rec.parameters()).abs().sum().item())
                p0 = next(iter(model_rec.parameters()))
                opt_p0 = opt.param_groups[0]["params"][0]
                print(f"N16_GRAD gn={gn:.3f} wn={wn:.3f} loss={float(loss):.4f} "
                      f"dev={dev} out={outv:.6f} w0={w0:.4f} "
                      f"same_param={p0 is opt_p0} "
                      f"nz={sum(1 for p in model_rec.parameters() if p.grad is not None and p.grad.abs().sum().item()>0)}",
                      flush=True)
                if os.environ.get("N16_DEBUG_NAMES"):
                    names = [n for n, p in model_rec.named_parameters()
                             if p.grad is not None and p.grad.abs().sum().item() > 0]
                    print("N16_NZ", names, flush=True)
            torch.nn.utils.clip_grad_norm_(model_rec.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
            pres.append(float(torch.sigmoid(presence).item()))
            if len(losses) % 200 == 0:
                print(
                    f"ep{epoch} step={len(losses)} loss={float(np.mean(losses)):.4f} "
                    f"iou={float(np.mean(ious)) if ious else 0:.4f} "
                    f"pres={float(np.mean(pres)):.4f} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
        print(
            f"EPOCH {epoch}: loss={float(np.mean(losses)):.4f} "
            f"iou={float(np.mean(ious)) if ious else 0:.4f} "
            f"pres={float(np.mean(pres)):.4f}",
            flush=True,
        )
    torch.save(
        {
            "state": model_rec.state_dict(), "args": vars(args), "d_model": d_model,
            "encoder_state": (
                model.detector.transformer.encoder.state_dict()
                if args.finetune else None
            ),
        },
        OUT / "models" / f"{args.out}.pt",
    )
    print("SAVED", args.out, flush=True)
    try:
        runner.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
