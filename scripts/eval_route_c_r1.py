#!/usr/bin/env python
"""N18 RouteC.9: R1 partial-backbone evaluation on the real V0 attempt
distribution. Gallery/query embeddings are recomputed through the full model
because R1 changed box_head; the frozen baseline is replayed from the cache.
Also checks that the frozen detection path is unchanged."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from run_n18_full_loop_v0 import crop_query, load_gt  # noqa: E402
from eval_route_c_r0 import (  # noqa: E402
    load_transactions, summarize, stratify, age_bin_of, iou)

OUT = ROOT / "outputs/n18"
RC = ROOT / "outputs/n18/route_c"
CACHE = RC / "gfn_cache"
DT = Path("/path/to/dancetrack")


def frame_img(seq, f):
    p = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
    return Image.open(p).convert("RGB") if p.exists() else None


def gallery(model, seq, f, device):
    img = frame_img(seq, f)
    if img is None:
        return None
    with torch.inference_mode():
        out = model([F.to_tensor(img).to(device)], None,
                    inference_mode="det")[0]
    boxes = out["det_boxes"].float().cpu().numpy().reshape(-1, 4)
    scores = out["det_scores"].float().cpu().numpy().reshape(-1)
    emb = out["det_emb"].float()
    if emb.shape[0]:
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    return boxes, scores, emb.cpu().numpy()


def query(model, seq, f, box, device):
    img = frame_img(seq, f)
    if img is None:
        return None
    qcrop = crop_query(img, box)
    with torch.inference_mode():
        out = model([F.to_tensor(qcrop).to(device)], None,
                    inference_mode="det")[0]
    if out["det_emb"].shape[0] == 0:
        return None
    qi = int(torch.argmax(out["det_scores"].float()))
    qe = out["det_emb"].float()[qi].reshape(1, -1)
    return qe / (qe.norm(dim=1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-tag", default="r1")
    ap.add_argument("--max-attempts", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    model, _, _, _, _ = load_model(device)
    model.eval()
    sd = torch.load(args.model, map_location="cpu")
    if "box_head" in sd:
        model.roi_heads.box_head.load_state_dict(sd["box_head"])
    model.roi_heads.embedding_head.load_state_dict(
        sd.get("embedding_head", sd))

    tx = load_transactions()
    if args.max_attempts:
        tx = tx[: args.max_attempts]
    by_seq = defaultdict(list)
    for t in tx:
        by_seq[t["sequence"]].append(t)

    rows = []
    det_mismatch = 0
    det_checked = 0
    for seq in sorted(by_seq):
        gt = load_gt(seq)
        first_app = {}
        for f0 in sorted(gt):
            for g0 in gt[f0].gt_ids:
                if g0 not in first_app:
                    first_app[g0] = f0
        z = np.load(CACHE / f"{seq}.npz")
        zmat = {"frames": z["frames"], "offsets": z["offsets"],
                "boxes": z["boxes"], "emb": z["emb"]}
        qz = np.load(CACHE / f"{seq}_queries.npz")
        qmat = {"gids": qz["gids"], "qframe": qz["qframe"],
                "qbox": qz["qbox"], "qemb": qz["qemb"]}
        z.close()
        qz.close()
        frames = zmat["frames"]
        offsets = zmat["offsets"]
        qidx = {int(g): i for i, g in enumerate(qmat["gids"])}
        gal_cache = {}
        q_cache = {}
        for t in by_seq[seq]:
            f, gid = int(t["frame"]), int(t["gid"])
            af = first_app.get(gid, f)
            age = f - af
            if (seq, f) not in gal_cache:
                gal_cache[(seq, f)] = gallery(model, seq, f, device)
            gal = gal_cache[(seq, f)]
            if gal is None:
                continue
            boxes_d, scores_d, emb_new = gal
            o = np.searchsorted(frames, f)
            lo = int(offsets[o - 1]) if o > 0 else 0
            hi = int(offsets[o])
            # detection-side effect check: box_head feeds score_predictor, so
            # R1 can change the post-NMS candidate set
            det_checked += 1
            if len(boxes_d) != hi - lo or (
                    hi - lo > 0 and
                    np.abs(boxes_d - zmat["boxes"][lo:hi]).max() > 1e-4):
                det_mismatch += 1
            crowd = len(boxes_d)
            row = {
                "sequence": seq, "frame": f, "gid": gid,
                "anchor_frame": af, "anchor_age": age,
                "age_bin": age_bin_of(age), "gap_bin": age_bin_of(age),
                "crowd_size": crowd,
                "gt_present": 0, "detector_contains_target": 0,
                "best_det_iou": None,
                "nearest_distractor_distance": None,
                "rank_stale": None, "sim_stale_best": None,
                "top1_sim_stale": None,
                "rank_frozen": None, "sim_frozen_best": None,
                "top1_sim_frozen": None,
                "rank_fresh": None, "sim_fresh_best": None,
                "top1_sim_fresh": None,
            }
            gf = gt.get(f)
            gbox = None
            if gf is not None and gid in gf.gt_ids:
                gbox = np.asarray(gf.boxes[gf.gt_ids.index(gid)],
                                  dtype=float)
                row["gt_present"] = 1
                c0 = ((gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2)
                dists = []
                for oid, ob in zip(gf.gt_ids, gf.boxes):
                    if oid == gid:
                        continue
                    ob = np.asarray(ob, dtype=float)
                    oc = ((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2)
                    dists.append(np.hypot(oc[0] - c0[0], oc[1] - c0[1]))
                row["nearest_distractor_distance"] = (
                    round(float(min(dists)), 2) if dists else None)
            if gid in qidx:
                qi = qidx[gid]
                qf = int(qmat["qframe"][qi])
                qbox = qmat["qbox"][qi]
                if (seq, gid) not in q_cache:
                    q_cache[(seq, gid)] = query(model, seq, qf, qbox,
                                                device)
                q_new = q_cache[(seq, gid)]
                if q_new is not None and len(boxes_d):
                    qn = q_new[0].cpu().numpy()
                    sim_new = emb_new @ qn
                    q_frozen = qmat["qemb"][qi].astype(np.float32)
                    q_frozen = q_frozen / (
                        np.linalg.norm(q_frozen) + 1e-8)
                    emb_f = zmat["emb"][lo:hi].astype(np.float32) \
                        if hi > lo else np.zeros((0, 2048), np.float32)
                    if emb_f.shape[0]:
                        emb_f = emb_f / (np.linalg.norm(
                            emb_f, axis=1, keepdims=True) + 1e-8)
                    sim_fr = emb_f @ q_frozen
                    if gbox is not None:
                        # R1 ranking over the R1 candidate set
                        ious_new = np.asarray([iou(b, gbox)
                                               for b in boxes_d])
                        best = int(np.argmax(ious_new))
                        row["best_det_iou"] = round(
                            float(ious_new[best]), 4)
                        row["detector_contains_target"] = int(
                            ious_new[best] >= 0.5)
                        row["sim_stale_best"] = round(float(sim_new[best]), 4)
                        row["rank_stale"] = int(
                            (sim_new > sim_new[best]).sum()) + 1
                        # frozen ranking over the frozen cached candidate set
                        if emb_f.shape[0]:
                            boxes_f = zmat["boxes"][lo:hi]
                            ious_f = np.asarray([iou(b, gbox)
                                                 for b in boxes_f])
                            best_f = int(np.argmax(ious_f))
                            row["sim_frozen_best"] = round(
                                float(sim_fr[best_f]), 4)
                            row["rank_frozen"] = int(
                                (sim_fr > sim_fr[best_f]).sum()) + 1
                    row["top1_sim_stale"] = round(float(sim_new.max()), 4)
                    if emb_f.shape[0]:
                        row["top1_sim_frozen"] = round(
                            float(sim_fr.max()), 4)
            rows.append(row)
        print(f"eval {seq} attempts={len(by_seq[seq])}", flush=True)

    fname = RC / f"{args.out_tag}_calibration_retrieval.csv"
    fields = list(rows[0].keys())
    with fname.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, tag=args.out_tag)
    summary["detector_preservation"] = {
        "frames_checked": det_checked, "mismatch_frames": det_mismatch}
    (RC / f"{args.out_tag}_retrieval_summary.json").write_text(
        json.dumps(summary, indent=1))
    stratify(rows, args.out_tag)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("R1_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
