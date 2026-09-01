#!/usr/bin/env python
"""N18 RouteC.8: R1 partial-backbone temporal adaptation.

Unfreezes the GFN identity head stack: ``roi_heads.box_head`` (= ConvNeXt
features[6] + [7] of the ReID branch) plus ``roi_heads.embedding_head``.
The detection branch (``prop_head`` deep copy, RPN, predictors) and backbone
features[0..5] stay frozen, so detection outputs are unchanged by
construction.

Crops are processed through the official model with ``inference_mode='gt'``
(GT boxes are offline training supervision only), then trained with the same
InfoNCE + hard-negative margin objective as R0.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402

OUT = ROOT / "outputs/n18/route_c"
MODELS = OUT / "models"
DT = Path("/path/to/dancetrack")


def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def crop_with_origin(img, box, margin=0.2):
    W, H = img.size
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ox = max(0.0, x1 - margin * w)
    oy = max(0.0, y1 - margin * h)
    x2c = min(float(W), x2 + margin * w)
    y2c = min(float(H), y2 + margin * h)
    crop = img.crop((int(ox), int(oy), int(x2c), int(y2c)))
    return crop, float(ox), float(oy)


def box_in_crop(box, ox, oy):
    return [box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy]


def gt_emb(model, img, box):
    """Embedding of an exact box in a crop through the 'gt' inference path."""
    t = F.to_tensor(img).cuda()
    targets = [{"boxes": torch.tensor([box], dtype=torch.float32,
                                      device="cuda:0")}]
    with torch.inference_mode():
        out = model([t], targets, inference_mode="gt")[0]
    e = out["gt_emb"].float()
    return e / (e.norm(dim=1, keepdim=True) + 1e-8)


def load_rows():
    return list(csv.DictReader(open(
        OUT / "temporal_pairs_train.csv", encoding="utf-8")))


def build_sample_store(rows, seed):
    pos = [i for i, r in enumerate(rows)
           if int(r["target_present"]) and
           int(r["detector_contains_target"])]
    neg = [i for i, r in enumerate(rows) if not int(r["target_present"])]
    rng = np.random.RandomState(seed)
    return rows, np.asarray(pos, np.int64), np.asarray(neg, np.int64), rng


def sample_crops(rows, pos, neg, rng, batch_size, gt_store, k_neg=2):
    """Return list of (kind, gid, img, box) crop tasks per batch."""
    tasks = []
    for _ in range(batch_size):
        idx = int(rng.choice(pos))
        r = rows[idx]
        seq = r["sequence"]
        gid = int(r["gid"])
        gt = gt_store[seq]
        qf = int(r["query_frame"])
        qbox = np.asarray(gt[qf].boxes[gt[qf].gt_ids.index(gid)],
                          dtype=float)
        p = DT / "train" / seq / "img1" / f"{qf + 1:08d}.jpg"
        qimg = Image.open(p).convert("RGB")
        qcrop, qox, qoy = crop_with_origin(qimg, qbox)
        qbox_c = box_in_crop(qbox, qox, qoy)
        tasks.append(("q", gid, qcrop, qbox_c))
        tf = int(r["gallery_frame"])
        gbox = np.asarray(gt[tf].boxes[gt[tf].gt_ids.index(gid)],
                          dtype=float) \
            if gid in gt[tf].gt_ids else None
        if gbox is not None:
            p = DT / "train" / seq / "img1" / f"{tf + 1:08d}.jpg"
            gimg = Image.open(p).convert("RGB")
            gcrop, gox, goy = crop_with_origin(gimg, gbox)
            gbox_c = box_in_crop(gbox, gox, goy)
            tasks.append(("p", gid, gcrop, gbox_c))
            # hard negatives: nearest spatial other persons at gallery frame
            gf = gt[tf]
            c0 = ((gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2)
            others = []
            for oid, ob in zip(gf.gt_ids, gf.boxes):
                if oid == gid:
                    continue
                ob = np.asarray(ob, dtype=float)
                oc = ((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2)
                others.append((np.hypot(oc[0] - c0[0], oc[1] - c0[1]), ob))
            for _, ob in sorted(others, key=lambda x: x[0])[:k_neg]:
                ncrop, nox, noy = crop_with_origin(gimg, ob)
                nob_c = box_in_crop(ob, nox, noy)
                tasks.append(("n", gid, ncrop, nob_c))
    return tasks


def forward_tasks(model, tasks):
    imgs = [F.to_tensor(img) for _, _, img, _ in tasks]
    sizes = torch.tensor([[t.shape[-2], t.shape[-1]] for t in imgs],
                         dtype=torch.float32)
    max_h = int(sizes[:, 0].max())
    max_w = int(sizes[:, 1].max())
    padded = torch.zeros(len(imgs), 3, max_h, max_w)
    for i, t in enumerate(imgs):
        padded[i, :, : t.shape[-2], : t.shape[-1]] = t
    boxes = torch.tensor([box for _, _, _, box in tasks],
                         dtype=torch.float32)
    emb = model(padded.cuda(), boxes.cuda(), sizes.cuda())
    embs, kinds, gids = [], [], []
    for (kind, gid, _, _), e in zip(tasks, emb):
        e = e.float() / (e.norm() + 1e-8)
        embs.append(e)
        kinds.append(kind)
        gids.append(gid)
    return embs, kinds, gids


class CropEmbedder(torch.nn.Module):
    """backbone -> RoI pool -> box_head -> embedding_head on padded crops."""

    def __init__(self, backbone, roi_pool, box_head, emb_head):
        super().__init__()
        self.backbone = backbone
        self.roi_pool = roi_pool
        self.box_head = box_head
        self.emb_head = emb_head

    def forward(self, images, boxes, sizes):
        n = images.shape[0]
        bb = self.backbone(images)
        box_list = [boxes[i:i + 1] for i in range(n)]
        size_list = [(int(sizes[i, 0]), int(sizes[i, 1]))
                     for i in range(n)]
        pooled = self.roi_pool(bb, box_list, size_list)
        h = self.box_head(pooled)
        emb, _ = self.emb_head(h)
        return emb


def r1_loss(zq, zp, zn, gids, tau=0.07, margin=0.25, w_margin=0.5):
    B = zq.shape[0]
    logits_pos = (zq * zp).sum(1) / tau
    same = torch.tensor(gids).unsqueeze(1) == torch.tensor(gids).unsqueeze(0)
    same = same.to(zq.device)
    sim_other = (zq @ zp.T / tau).masked_fill(same, -1e9)
    sim_other = sim_other - torch.eye(B, device=zq.device) * 1e9
    zn_flat = zn.reshape(-1, zn.shape[-1])
    parts = [logits_pos.unsqueeze(1), sim_other]
    if zn_flat.shape[0]:
        parts.append(zq @ zn_flat.T / tau)
    logits = torch.cat(parts, dim=1)
    loss = Fn.cross_entropy(
        logits, torch.zeros(B, dtype=torch.long, device=zq.device))
    if zn_flat.shape[0]:
        hard = (zq @ zn_flat.T / tau).max(dim=1).values
        loss = loss + w_margin * Fn.relu(
            margin - (logits_pos - hard)).mean()
    return loss


def validate(train_model, base_model, sample_per_bin=4):
    base_model.eval()
    rows = list(csv.DictReader(open(
        OUT / "temporal_pairs_cal.csv", encoding="utf-8")))
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["gap_bin"]].append(i)
    rng = np.random.RandomState(7)
    picked = []
    for b in sorted(by, key=lambda x: (x == "480+", int(x)
                                        if x != "480+" else 1e9)):
        idx = by[b]
        picked.extend(rng.choice(
            idx, size=min(len(idx), sample_per_bin), replace=False).tolist())
    gt_store, frame_cache = {}, {}
    hits = defaultdict(int)
    n = 0
    core = base_model
    with torch.inference_mode():
        for i in picked:
            r = rows[i]
            seq = r["sequence"]
            if seq not in gt_store:
                gt_store[seq] = load_gt(seq)
            gt = gt_store[seq]
            gid = int(r["gid"])
            qf = int(r["query_frame"])
            qbox = np.asarray(gt[qf].boxes[gt[qf].gt_ids.index(gid)],
                              dtype=float)
            qimg = Image.open(DT / "train" / seq / "img1" /
                              f"{qf + 1:08d}.jpg").convert("RGB")
            qcrop, qox, qoy = crop_with_origin(qimg, qbox)
            qe = gt_emb(core, qcrop, box_in_crop(qbox, qox, qoy))
            tf = int(r["gallery_frame"])
            if (seq, tf) not in frame_cache:
                fimg = Image.open(DT / "train" / seq / "img1" /
                                  f"{tf + 1:08d}.jpg").convert("RGB")
                with torch.inference_mode():
                    out = core([F.to_tensor(fimg).cuda()], None,
                               inference_mode="det")[0]
                ge = out["det_emb"].float()
                ge = ge / (ge.norm(dim=1, keepdim=True) + 1e-8)
                frame_cache[(seq, tf)] = (
                    out["det_boxes"].float().cpu().numpy().reshape(-1, 4),
                    ge)
            boxes_d, ge = frame_cache[(seq, tf)]
            sims = (ge @ qe.T)[:, 0]
            if int(r["target_present"]) and int(
                    r["detector_contains_target"]) and \
                    int(r["det_idx"]) >= 0:
                n += 1
                gf = gt[tf]
                gbox = np.asarray(gf.boxes[gf.gt_ids.index(gid)],
                                  dtype=float)
                ious = np.asarray([iou(b, gbox) for b in boxes_d])
                best = int(np.argmax(ious))
                rank = int((sims > sims[best]).sum()) + 1
                for k in (1, 3):
                    hits[k] += int(rank <= k)
    train_model.train()
    return {"n": n,
            "top1": hits[1] / n if n else None,
            "top3": hits[3] / n if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-box", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--w-margin", type=float, default=0.5)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()

    device_ids = [int(x) for x in args.gpus.split(",")]
    torch.cuda.set_device(device_ids[0])
    model, _, _, _, _ = load_model("cuda:0")
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in (model.roi_heads.box_head, model.roi_heads.embedding_head):
        for p in m.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    embedder = CropEmbedder(
        model.backbone, model.roi_heads.reid_roi_pool,
        model.roi_heads.box_head, model.roi_heads.embedding_head)
    base_model = model
    model = torch.nn.DataParallel(embedder, device_ids=device_ids)
    model.train()

    head_params = [p for n, p in model.named_parameters()
                   if "embedding_head" in n]
    box_params = [p for n, p in model.named_parameters()
                  if "box_head" in n]
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr_head},
        {"params": box_params, "lr": args.lr_box},
    ], weight_decay=1e-4)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

    rows, pos, neg, rng = build_sample_store(load_rows(), args.seed)
    gt_store = {}
    for seq in {r["sequence"] for r in rows}:
        gt_store[seq] = load_gt(seq)

    log_path = OUT / f"{args.tag}_training_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "epoch", "loss", "val_n",
                                "val_top1", "val_top3"])
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            tasks = sample_crops(rows, pos, neg, rng, args.batch_size,
                                 gt_store)
            embs, kinds, gids = forward_tasks(model, tasks)
            # organize: one query + positive + up to k_neg negatives/sample
            qs, ps = [], []
            qg = []
            neg_lists = []
            cur = None
            for e, k, g in zip(embs, kinds, gids):
                if k == "q":
                    cur = len(qs)
                    qs.append(e)
                    qg.append(g)
                    neg_lists.append([])
                elif k == "p":
                    ps.append(e)
                else:
                    neg_lists[cur].append(e)
            zq = torch.stack(qs)
            zp = torch.stack(ps)
            zn = torch.zeros(len(qs), 2, 2048, device=zq.device)
            for s, lst in enumerate(neg_lists):
                for j, e in enumerate(lst[:2]):
                    zn[s, j] = e
            loss = r1_loss(zq, zp, zn, qg, args.tau, args.margin,
                           args.w_margin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.val_every == 0:
                val = validate(model, base_model)
                with open(log_path, "a", newline="",
                          encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [step, epoch, round(float(loss), 4), val["n"],
                         val["top1"], val["top3"]])
                print(json.dumps({"step": step, "loss": float(loss),
                                  **val}, ensure_ascii=False), flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "box_head": model.module.box_head.state_dict(),
        "embedding_head": model.module.emb_head.state_dict(),
    }, MODELS / f"{args.tag}_last.pt")
    (MODELS / f"{args.tag}_config.json").write_text(json.dumps({
        "tag": args.tag, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size, "lr_head": args.lr_head,
        "lr_box": args.lr_box, "tau": args.tau, "margin": args.margin,
        "seed": args.seed, "runtime_s": round(time.time() - t0, 1),
        "trainable": "roi_heads.box_head + roi_heads.embedding_head",
        "frozen": "backbone features[0..5], RPN, prop_head, predictors",
    }, indent=1))
    print("R1_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
