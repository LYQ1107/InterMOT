#!/usr/bin/env python
"""N18 RouteC.5: stale-human-anchor retrieval on the real FULL_LOOP_V0
attempt distribution (cal10), frozen GFN vs R0 temporal head.

The evaluation replays the exact recorded causal attempts; no result is fed
back into the loop. Fresh last-visible-GT anchors are an offline upper bound
diagnostic only.
"""

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
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from run_n18_full_loop_v0 import crop_query, load_gt  # noqa: E402

OUT = ROOT / "outputs/n18"
RC = ROOT / "outputs/n18/route_c"
CACHE = RC / "gfn_cache"
DT = Path("/path/to/dancetrack")

AGE_BINS = [30, 60, 120, 240, 480, float("inf")]


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


def age_bin_of(age):
    for b in AGE_BINS:
        if age <= b:
            return b
    return "480+"


def load_transactions():
    rows = []
    for p in sorted(OUT.glob(
            "reactivation_transactions_full_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def head_embed(model, f4, f5):
    with torch.inference_mode():
        emb, _ = model.roi_heads.embedding_head({
            "feat_res4": f4, "feat_res5": f5})
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)


def frame_img(seq, f):
    p = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
    return Image.open(p).convert("RGB") if p.exists() else None


def fresh_query_emb(model, device, seq, f, gid, gt):
    last = None
    for f2 in range(f - 1, -1, -1):
        gf = gt.get(f2)
        if gf is not None and gid in gf.gt_ids:
            last = (f2, np.asarray(
                gf.boxes[gf.gt_ids.index(gid)], dtype=float))
            break
    if last is None:
        return None, None
    af, abox = last
    img = frame_img(seq, af)
    if img is None:
        return None, None
    qcrop = crop_query(img, abox)
    with torch.inference_mode():
        qout = model([F.to_tensor(qcrop).to(device)], None,
                     inference_mode="det")[0]
    if qout["det_emb"].shape[0] == 0:
        return None, None
    qi = int(torch.argmax(qout["det_scores"].float()))
    qe = qout["det_emb"].float()[qi].reshape(1, -1)
    return qe / (qe.norm(dim=1, keepdim=True) + 1e-8), af


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="frozen",
                    help="path to r0 head state dict, or 'frozen'")
    ap.add_argument("--out-tag", default="r0")
    ap.add_argument("--fresh-oracle", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    model, _, _, _, _ = load_model(device)
    model.eval()
    if args.model != "frozen":
        head = model.roi_heads.embedding_head
        sd = torch.load(args.model, map_location="cpu")
        head.load_state_dict(sd)
        head.eval()

    tx = load_transactions()
    if args.max_attempts:
        tx = tx[: args.max_attempts]
    by_seq = defaultdict(list)
    for t in tx:
        by_seq[t["sequence"]].append(t)

    gt_store, cache_store, q_store = {}, {}, {}
    rows = []
    for seq in sorted(by_seq):
        gt_store[seq] = load_gt(seq)
        first_app = {}
        for f0 in sorted(gt_store[seq]):
            gf0 = gt_store[seq][f0]
            for g0 in gf0.gt_ids:
                if g0 not in first_app:
                    first_app[g0] = f0
        z = np.load(CACHE / f"{seq}.npz")
        qz = np.load(CACHE / f"{seq}_queries.npz")
        zmat = {
            "frames": z["frames"], "offsets": z["offsets"],
            "boxes": z["boxes"], "emb": z["emb"],
            "feat4": z["feat4"], "feat5": z["feat5"],
        }
        z.close()
        qmat = {"gids": qz["gids"], "qframe": qz["qframe"],
                "qbox": qz["qbox"], "qemb": qz["qemb"],
                "qfeat4": qz["qfeat4"], "qfeat5": qz["qfeat5"]}
        qz.close()
        cache_store[seq] = zmat
        q_store[seq] = ({int(g): i for i, g in enumerate(qmat["gids"])},
                        qmat)
        frames = zmat["frames"]
        offsets = zmat["offsets"]
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
                "crowd_size": crowd,
                "gap_bin": age_bin_of(age),
                "gt_present": 0,
                "detector_contains_target": 0,
                "best_det_iou": None,
                "nearest_distractor_distance": None,
                "rank_stale": None, "sim_stale_best": None,
                "top1_sim_stale": None,
                "rank_frozen": None, "sim_frozen_best": None,
                "top1_sim_frozen": None,
                "rank_fresh": None, "sim_fresh_best": None,
                "top1_sim_fresh": None,
            }
            gf = gt_store[seq].get(f)
            gbox = None
            if gf is not None and gid in gf.gt_ids:
                gbox = np.asarray(
                    gf.boxes[gf.gt_ids.index(gid)], dtype=float)
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
            if hi > lo:
                boxes_d = zmat["boxes"][lo:hi]
                emb_f = zmat["emb"][lo:hi].astype(np.float32)
                emb_f = emb_f / (np.linalg.norm(
                    emb_f, axis=1, keepdims=True) + 1e-8)
                f4 = torch.from_numpy(
                    zmat["feat4"][lo:hi].astype(np.float32)).to(device)
                f5 = torch.from_numpy(
                    zmat["feat5"][lo:hi].astype(np.float32)).to(device)
                emb_new = head_embed(model, f4, f5).cpu().numpy()
                # stale first-appearance query
                qi = q_store[seq][0].get(gid)
                if qi is not None:
                    qf4 = torch.from_numpy(q_store[seq][1]["qfeat4"][qi]
                                           .astype(np.float32)).unsqueeze(0).to(device)
                    qf5 = torch.from_numpy(q_store[seq][1]["qfeat5"][qi]
                                           .astype(np.float32)).unsqueeze(0).to(device)
                    q_new = head_embed(model, qf4, qf5)[0].cpu().numpy()
                    q_frozen = q_store[seq][1]["qemb"][qi].astype(np.float32)
                    q_frozen = q_frozen / (np.linalg.norm(q_frozen) + 1e-8)
                    sim_new = emb_new @ q_new
                    sim_fr = emb_f @ q_frozen
                    if gbox is not None and len(boxes_d):
                        ious = np.asarray([iou(b, gbox) for b in boxes_d])
                        best = int(np.argmax(ious))
                        row["best_det_iou"] = round(float(ious[best]), 4)
                        row["detector_contains_target"] = int(
                            ious[best] >= 0.5)
                        row["sim_stale_best"] = round(float(sim_new[best]), 4)
                        row["sim_frozen_best"] = round(float(sim_fr[best]), 4)
                        row["rank_stale"] = int((sim_new > sim_new[best]).sum()) + 1
                        row["rank_frozen"] = int((sim_fr > sim_fr[best]).sum()) + 1
                    row["top1_sim_stale"] = round(float(sim_new.max()), 4)
                    row["top1_sim_frozen"] = round(float(sim_fr.max()), 4)
                if args.fresh_oracle and gbox is not None and len(boxes_d):
                    qf_emb, qf_f = fresh_query_emb(
                        model, device, seq, f, gid, gt_store[seq])
                    if qf_emb is not None:
                        sim_fresh = (emb_new @ qf_emb.cpu().numpy().T)[:, 0]
                        ious = np.asarray([iou(b, gbox) for b in boxes_d])
                        best = int(np.argmax(ious))
                        row["sim_fresh_best"] = round(
                            float(sim_fresh[best]), 4)
                        row["rank_fresh"] = int(
                            (sim_fresh > sim_fresh[best]).sum()) + 1
            rows.append(row)
        print(f"eval {seq} attempts={len(by_seq[seq])}", flush=True)

    fname = RC / f"{args.out_tag}_calibration_retrieval.csv"
    fields = list(rows[0].keys())
    with fname.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows, tag=args.out_tag)
    (RC / f"{args.out_tag}_retrieval_summary.json").write_text(
        json.dumps(summary, indent=1))
    stratify(rows, args.out_tag)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("R0_EVAL_DONE", flush=True)


def summarize(rows, tag):
    out = {"tag": tag, "n_attempts": len(rows)}
    for mode in ("stale", "frozen", "fresh"):
        sub = [r for r in rows if r[f"rank_{mode}"] is not None]
        hits = {k: sum(r[f"rank_{mode}"] <= k for r in sub)
                for k in (1, 3, 5, 10)}
        mrr = float(np.mean([1.0 / r[f"rank_{mode}"] for r in sub])) if sub else None
        out[mode] = {
            "n_with_detector_target": len(sub),
            "top1": hits[1] / len(sub) if sub else None,
            "top3": hits[3] / len(sub) if sub else None,
            "top5": hits[5] / len(sub) if sub else None,
            "top10": hits[10] / len(sub) if sub else None,
            "mrr": None if mrr is None else round(mrr, 4),
        }
        present = [r for r in rows
                   if r["gt_present"] and r[f"sim_{mode}_best"] is not None]
        if present:
            y = np.asarray([r["detector_contains_target"] for r in present])
            s = np.asarray([r[f"sim_{mode}_best"] for r in present],
                           dtype=float)
            if len(np.unique(y)) > 1:
                out[mode]["auc"] = round(roc_auc_score(y, s), 4)
                out[mode]["pr_auc"] = round(
                    average_precision_score(y, s), 4)
        absent = [r for r in rows
                  if not r["gt_present"] and
                  r[f"top1_sim_{mode}"] is not None]
        out[mode]["n_absent"] = len(absent)
        for thr in (0.4, 0.5, 0.6):
            out[mode][f"absent_top1_sim_ge_{thr}"] = (
                round(float(np.mean(
                    [r[f"top1_sim_{mode}"] >= thr for r in absent])), 4)
                if absent else None)
    return out


def stratify(rows, tag):
    def agg(key, out_csv, metric="rank_stale"):
        groups = defaultdict(lambda: [0, 0])
        for r in rows:
            v = r[key]
            if v is None or r[metric] is None:
                continue
            groups[str(v)][0] += 1
            groups[str(v)][1] += int(r[metric] <= 1)
        with (RC / out_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([key, "n", "top1"])
            for k in sorted(groups):
                w.writerow([k, groups[k][0],
                            round(groups[k][1] / groups[k][0], 4)])
    agg("age_bin", f"{tag}_anchor_age_analysis.csv")
    agg("gap_bin", f"{tag}_gap_analysis.csv")
    agg("crowd_size", f"{tag}_crowd_analysis.csv")
    # crowd bins
    rows2 = []
    for r in rows:
        rr = dict(r)
        c = r["crowd_size"]
        rr["crowd_bin"] = "low" if c < 8 else "medium" if c < 20 else "high"
        rows2.append(rr)
    agg = None
    groups = defaultdict(lambda: [0, 0])
    for r in rows2:
        if r["rank_stale"] is None:
            continue
        k = r["crowd_bin"]
        groups[k][0] += 1
        groups[k][1] += int(r["rank_stale"] <= 1)
    with (RC / f"{tag}_crowd_analysis.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crowd_bin", "n", "top1"])
        for k in ("low", "medium", "high"):
            w.writerow([k, groups[k][0],
                        round(groups[k][1] / groups[k][0], 4)
                        if groups[k][0] else None])
    # hard distractor: nearest person distance at attempt frame
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        d = r["nearest_distractor_distance"]
        if d is None or r["rank_stale"] is None:
            continue
        k = "near" if d < 60 else "mid" if d < 150 else "far"
        groups[k][0] += 1
        groups[k][1] += int(r["rank_stale"] <= 1)
    with (RC / f"{tag}_hard_distractor_analysis.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distractor_bin", "n", "top1"])
        for k in ("near", "mid", "far"):
            w.writerow([k, groups[k][0],
                        round(groups[k][1] / groups[k][0], 4)
                        if groups[k][0] else None])
    # absent analysis
    with (RC / f"{tag}_absent_analysis.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "false_reactivation_rate_stale",
                    "false_reactivation_rate_frozen"])
        absent = [r for r in rows
                  if not r["gt_present"] and
                  r["top1_sim_stale"] is not None]
        for thr in (0.4, 0.5, 0.6):
            fr = [r["top1_sim_frozen"] >= thr for r in absent
                  if r["top1_sim_frozen"] is not None]
            w.writerow([
                thr,
                round(float(np.mean([r["top1_sim_stale"] >= thr
                                     for r in absent])), 4) if absent else None,
                round(float(np.mean(fr)), 4) if fr else None,
            ])


if __name__ == "__main__":
    main()
