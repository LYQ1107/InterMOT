#!/usr/bin/env python
"""Train HTD-v1 on cached HCRED episodes (no SAM3 backend needed)."""

import argparse
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
OUT = ROOT / "outputs/n17"
DT = Path("/path/to/dancetrack")
CLIP_CKPT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"
CACHE = OUT / "enc_cache"


def load_mem(seq, f):
    p = CACHE / f"{seq}_{f}.npy"
    if not p.exists():
        return None
    return torch.from_numpy(np.load(p)).float().reshape(72, 72, 256)


def giou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ua = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    ub = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    union = ua + ub - inter
    iou = inter / (union + 1e-9)
    cx1 = torch.minimum(ax1, bx1)
    cy1 = torch.minimum(ay1, by1)
    cx2 = torch.maximum(ax2, bx2)
    cy2 = torch.maximum(ay2, by2)
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
    return iou - (c_area - union) / (c_area + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="htd_v1")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    from scripts.run_n15_extract_features import build_clipreid
    import torchvision.transforms as T

    clip = build_clipreid(str(CLIP_CKPT), device)
    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    with (OUT / "train_episodes.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["target_present"] == "1"
                or r["target_present"] == "0"]
    if args.max_episodes:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_episodes]
    shard_rows = rows[args.shard:: args.nshards]
    print(f"shard {args.shard}/{args.nshards} episodes={len(shard_rows)}", flush=True)

    from sam3_intermot.recovery.htd import HTD
    model = HTD().to(device)
    if args.resume:
        ck = torch.load(ROOT / args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        print("resumed from", args.resume, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bs = args.batch_size
    t0 = time.time()
    for epoch in range(args.epochs):
        losses, ious, pres, hits = [], [], [], []
        for start in range(0, len(shard_rows), bs):
            chunk = shard_rows[start:start + bs]
            ref_mems, search_mems, ref_boxes, Hs, presents, gt_boxes = [], [], [], [], [], []
            ok = True
            for r in chunk:
                seq = r["sequence"]
                t, f = int(r["t"]), int(r["f"])
                rm = load_mem(seq, t)
                sm = load_mem(seq, f)
                if rm is None or sm is None:
                    ok = False
                    break
                hb = np.asarray(json.loads(r["human_box"]), dtype=float)
                iw, ih = 1920.0, 1080.0
                box = torch.tensor(
                    [
                        (hb[0] + hb[2]) / 2 / iw, (hb[1] + hb[3]) / 2 / ih,
                        (hb[2] - hb[0]) / iw, (hb[3] - hb[1]) / ih,
                    ],
                    dtype=torch.float32,
                )
                img = Image.open(DT / "train" / seq / "img1" / f"{t + 1:08d}.jpg").convert("RGB")
                x1, y1, x2, y2 = [int(round(v)) for v in hb]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                crop = img.crop((x1, y1, x2, y2))
                present = int(r["target_present"]) == 1
                ref_mems.append(rm)
                search_mems.append(sm)
                ref_boxes.append(box)
                Hs.append(clip_tf(crop))
                presents.append(1.0 if present else 0.0)
                if present:
                    fb = np.asarray(json.loads(r["target_box"]), dtype=float)
                    gt_boxes.append(
                        torch.tensor(
                            [
                                (fb[0] + fb[2]) / 2 / iw, (fb[1] + fb[3]) / 2 / ih,
                                (fb[2] - fb[0]) / iw, (fb[3] - fb[1]) / ih,
                            ],
                            dtype=torch.float32,
                        )
                    )
                else:
                    gt_boxes.append(torch.zeros(4))
            if not ok:
                continue
            ref_mem = torch.stack(ref_mems).to(device)
            search_mem = torch.stack(search_mems).to(device)
            ref_box = torch.stack(ref_boxes).to(device)
            x = torch.stack(Hs).to(device)
            with torch.no_grad():
                _, x12, xproj = clip(x)
                h = F.normalize(torch.cat([x12[:, 0], xproj[:, 0]], dim=1), dim=-1)
            present_t = torch.tensor(presents, device=device)
            gt = torch.stack(gt_boxes).to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(ref_mem, ref_box, h, search_mem)
            tgt = pred["targetness"]
            presence = pred["presence"]
            boxes = pred["boxes"]
            loss = 2.0 * F.binary_cross_entropy_with_logits(presence, present_t)
            tgt_target = torch.zeros_like(tgt)
            box_loss = torch.zeros((), device=device)
            id_loss = torch.zeros((), device=device)
            iou_sum = 0.0
            n_p = int(present_t.sum().item())
            for i in range(len(chunk)):
                if present_t[i] == 0:
                    continue
                xyxy = model.decode_boxes(boxes[i])  # [K,4]
                gxyxy = torch.stack(
                    [
                        gt[i, 0] - gt[i, 2] / 2, gt[i, 1] - gt[i, 3] / 2,
                        gt[i, 0] + gt[i, 2] / 2, gt[i, 1] + gt[i, 3] / 2,
                    ],
                    dim=-1,
                ).unsqueeze(0)
                iou = torch.stack([giou_xyxy(xyxy[j:j + 1], gxyxy) for j in range(xyxy.shape[0])])
                m = int(torch.argmax(iou).item())
                tgt_target[i, m] = 1.0
                box_loss = box_loss + F.l1_loss(boxes[i, m], gt[i]) + (1 - iou[m])
                id_loss = id_loss + (1 - pred["id_sim"][i, m])
                id_loss = id_loss + F.relu(pred["id_sim"][i, torch.arange(model.n_queries, device=device) != m]).sum() * 0.1
                iou_sum += float(torch.clamp(iou[m], 0, 1).item())
            loss = loss + F.binary_cross_entropy_with_logits(tgt, tgt_target)
            if n_p:
                loss = loss + 5.0 * box_loss / n_p + 0.1 * id_loss / n_p
                ious.append(iou_sum / n_p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
            pres.append(float(torch.sigmoid(presence).mean().item()))
            if len(losses) % 100 == 0:
                print(
                    f"ep{epoch} step={len(losses)} loss={np.mean(losses):.4f} "
                    f"iou={np.mean(ious) if ious else 0:.4f} "
                    f"pres={np.mean(pres):.4f} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )
        print(
            f"EPOCH {epoch}: loss={np.mean(losses):.4f} "
            f"iou={np.mean(ious) if ious else 0:.4f} "
            f"pres={np.mean(pres):.4f}",
            flush=True,
        )
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "args": vars(args),
            "episodes": len(shard_rows),
        },
        OUT / "models" / f"{args.out}_shard{args.shard}.pt",
    )
    print("SAVED", args.out, flush=True)


if __name__ == "__main__":
    main()
