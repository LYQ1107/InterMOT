#!/usr/bin/env python
"""N18 RouteC.10 eval: detection-independent MLP canonicalizer on the real
V0 attempt distribution, using only the cached frozen features."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_route_c_r0 import (  # noqa: E402
    load_transactions, summarize, stratify, age_bin_of)
from train_route_c_r0 import run_validation  # noqa: E402
from train_route_c_upgrade import Canonicalizer  # noqa: E402

OUT = ROOT / "outputs/n18"
RC = ROOT / "outputs/n18/route_c"
CACHE = RC / "gfn_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-tag", default="upgrade")
    ap.add_argument("--max-attempts", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    head = Canonicalizer().to(device)
    head.load_state_dict(torch.load(args.model, map_location=device))
    head.eval()

    tx = load_transactions()
    if args.max_attempts:
        tx = tx[: args.max_attempts]
    by_seq = defaultdict(list)
    for t in tx:
        by_seq[t["sequence"]].append(t)

    rows = []
    with torch.inference_mode():
        for seq in sorted(by_seq):
            gt = None
            from run_n18_full_loop_v0 import load_gt
            gt = load_gt(seq)
            first_app = {}
            for f0 in sorted(gt):
                for g0 in gt[f0].gt_ids:
                    if g0 not in first_app:
                        first_app[g0] = f0
            z = np.load(CACHE / f"{seq}.npz")
            zmat = {"frames": z["frames"], "offsets": z["offsets"],
                    "boxes": z["boxes"], "emb": z["emb"],
                    "feat4": z["feat4"], "feat5": z["feat5"]}
            qz = np.load(CACHE / f"{seq}_queries.npz")
            qmat = {"gids": qz["gids"], "qemb": qz["qemb"],
                    "qfeat4": qz["qfeat4"], "qfeat5": qz["qfeat5"]}
            z.close()
            qz.close()
            frames = zmat["frames"]
            offsets = zmat["offsets"]
            qidx = {int(g): i for i, g in enumerate(qmat["gids"])}
            for t in by_seq[seq]:
                f, gid = int(t["frame"]), int(t["gid"])
                af = first_app.get(gid, f)
                age = f - af
                o = np.searchsorted(frames, f)
                lo = int(offsets[o - 1]) if o > 0 else 0
                hi = int(offsets[o])
                crowd = hi - lo
                row = {
                    "sequence": seq, "frame": f, "gid": gid,
                    "anchor_frame": af, "anchor_age": age,
                    "age_bin": age_bin_of(age),
                    "gap_bin": age_bin_of(age),
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
                        dists.append(np.hypot(oc[0] - c0[0],
                                              oc[1] - c0[1]))
                    row["nearest_distractor_distance"] = (
                        round(float(min(dists)), 2) if dists else None)
                if hi > lo and gid in qidx:
                    qi = qidx[gid]
                    f4 = torch.from_numpy(zmat["feat4"][lo:hi]
                                          .astype(np.float32)).to(device)
                    f5 = torch.from_numpy(zmat["feat5"][lo:hi]
                                          .astype(np.float32)).to(device)
                    emb_new = head(f4, f5).cpu().numpy()
                    qf4 = torch.from_numpy(qmat["qfeat4"][qi]
                                           .astype(np.float32)
                                           ).unsqueeze(0).to(device)
                    qf5 = torch.from_numpy(qmat["qfeat5"][qi]
                                           .astype(np.float32)
                                           ).unsqueeze(0).to(device)
                    q_new = head(qf4, qf5)[0].cpu().numpy()
                    sim_new = emb_new @ q_new
                    q_frozen = qmat["qemb"][qi].astype(np.float32)
                    q_frozen = q_frozen / (np.linalg.norm(q_frozen) + 1e-8)
                    emb_f = zmat["emb"][lo:hi].astype(np.float32)
                    emb_f = emb_f / (np.linalg.norm(
                        emb_f, axis=1, keepdims=True) + 1e-8)
                    sim_fr = emb_f @ q_frozen
                    if gbox is not None:
                        boxes_d = zmat["boxes"][lo:hi]
                        ious = np.asarray([float(iou2(b, gbox))
                                           for b in boxes_d])
                        best = int(np.argmax(ious))
                        row["best_det_iou"] = round(float(ious[best]), 4)
                        row["detector_contains_target"] = int(
                            ious[best] >= 0.5)
                        row["sim_stale_best"] = round(float(sim_new[best]), 4)
                        row["rank_stale"] = int(
                            (sim_new > sim_new[best]).sum()) + 1
                        row["sim_frozen_best"] = round(float(sim_fr[best]), 4)
                        row["rank_frozen"] = int(
                            (sim_fr > sim_fr[best]).sum()) + 1
                    row["top1_sim_stale"] = round(float(sim_new.max()), 4)
                    row["top1_sim_frozen"] = round(float(sim_fr.max()), 4)
                rows.append(row)
            print(f"eval {seq} attempts={len(by_seq[seq])}", flush=True)

    fname = RC / f"{args.out_tag}_calibration_retrieval.csv"
    with fname.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, tag=args.out_tag)
    (RC / f"{args.out_tag}_retrieval_summary.json").write_text(
        json.dumps(summary, indent=1))
    stratify(rows, args.out_tag)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("UPGRADE_EVAL_DONE", flush=True)


def iou2(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


if __name__ == "__main__":
    main()
