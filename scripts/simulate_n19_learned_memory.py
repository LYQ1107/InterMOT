#!/usr/bin/env python
"""N19.9/N19.10: offline learned-memory simulation on the cal10 trace.

Replays the recorded deliveries and applies the trained writer to maintain
K-slot memory; computes write safety (TP/FP/FN/TN), memory purity, anchor
age at V0 recovery attempts, and offline retrieval top1 with the learned
memory as the query. No SAM3 reactivation (FULL_LOOP_N19 does that).
"""

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

from eval_route_c_r0 import iou  # noqa: E402
from gfn_recovery_model import load_model  # noqa: E402
from n19_writer_features import feature_names, to_feature_vec  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402
from train_n19_writer import WriterMLP  # noqa: E402

OUT = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"


def head_embed(model, f4, f5):
    with torch.inference_mode():
        emb, _ = model.roi_heads.embedding_head(
            {"feat_res4": f4, "feat_res5": f5})
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer",
                    default=str(ROOT / "outputs/n19/models/writer_v0/writer_v0.pt"))
    ap.add_argument("--writer-config",
                    default=str(ROOT / "outputs/n19/models/writer_v0/writer_config.json"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--events-prefix", default="full_loop_v0_events_oracle_n19")
    ap.add_argument("--transactions-prefix",
                    default="reactivation_transactions_full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="sim")
    args = ap.parse_args()

    wcfg = json.loads(Path(args.writer_config).read_text())
    writer = WriterMLP(len(feature_names()), hidden=wcfg["hidden"])
    writer.load_state_dict(torch.load(args.writer, map_location="cpu"))
    writer.eval()
    mean = np.asarray(wcfg["scaler_mean"], dtype=np.float32)
    std = np.asarray(wcfg["scaler_std"], dtype=np.float32)
    gfn, _, _, _, _ = load_model(args.device)
    gfn.eval()
    if Path(ROOT / "outputs/n18/route_c/models/r0_best.pt").exists():
        gfn.roi_heads.embedding_head.load_state_dict(torch.load(
            ROOT / "outputs/n18/route_c/models/r0_best.pt",
            map_location="cpu"))
        gfn.roi_heads.embedding_head.eval()
    threshold = args.threshold
    if threshold is None:
        tj = N19 / "models/writer_v0/writer_threshold.json"
        if tj.exists():
            threshold = float(json.loads(tj.read_text())["threshold"])
        else:
            threshold = 0.5
    ks = [int(x) for x in args.ks.split(",")]

    ev = []
    for p in sorted(OUT.glob(f"{args.events_prefix}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                ev.append(json.loads(line))
    tx = []
    for p in sorted(OUT.glob(f"{args.transactions_prefix}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                tx.append(json.loads(line))

    by_seq = defaultdict(list)
    for e in ev:
        by_seq[e["sequence"]].append(e)
    tx_by_seq = defaultdict(list)
    for t in tx:
        tx_by_seq[t["sequence"]].append(t)

    rows = []
    agg = {k: {"writes": 0, "tp": 0, "fp": 0, "fn": 0,
               "purity_sum": 0.0, "purity_n": 0,
               "anchor_age": [], "retrieval_top1": [0, 0],
               "retrieval_top3": [0, 0]}
            for k in ks}
    r0_cache = {}
    for seq in sorted(by_seq):
        z = np.load(CACHE / f"{seq}.npz")
        qz = np.load(CACHE / f"{seq}_queries.npz")
        frames = z["frames"]
        offsets = z["offsets"]
        dets_all = z["boxes"]
        emb_all = z["emb"].astype(np.float32)
        emb_all = emb_all / (np.linalg.norm(emb_all, axis=1,
                                            keepdims=True) + 1e-8)
        qgids = [int(g) for g in qz["gids"]]
        qemb = qz["qemb"].astype(np.float32)
        qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-8)
        qidx = {g: i for i, g in enumerate(qgids)}
        f4 = torch.from_numpy(z["feat4"].astype(np.float32))
        f5 = torch.from_numpy(z["feat5"].astype(np.float32))
        qf4 = torch.from_numpy(qz["qfeat4"].astype(np.float32))
        qf5 = torch.from_numpy(qz["qfeat5"].astype(np.float32))
        with torch.inference_mode():
            r0g = head_embed(gfn, f4, f5).numpy()
            r0q = head_embed(gfn, qf4, qf5).numpy()
        r0_cache[seq] = (r0g, r0q, qidx)
        z.close()
        qz.close()
        gt = load_gt(seq)
        first_app = {}
        for f0 in sorted(gt):
            for g0 in gt[f0].gt_ids:
                first_app.setdefault(g0, f0)
        hist = defaultdict(list)
        for e in by_seq[seq]:
            hist[e["gid"]].append(e)
        attempts = defaultdict(list)
        for t in tx_by_seq[seq]:
            attempts[t["gid"]].append((int(t["frame"]),
                                       t.get("recovery_box")))
        for gid, evs in hist.items():
            qi = qidx.get(gid)
            if qi is None:
                continue
            qe_h = qemb[qi]
            qe_r0 = r0_cache[seq][1][qi]
            r0g = r0_cache[seq][0]
            slots = [
                (first_app.get(gid, 0), qe_h.copy(), qe_r0.copy(), 1)
            ]  # human root seeds the memory
            heur = []
            prev_box = None
            for e in sorted(evs, key=lambda x: int(x["frame"])):
                f0 = int(e["frame"])
                if not e.get("delivered") or e.get("delivered_box") is None:
                    prev_box = None
                    continue
                box = np.asarray(e["delivered_box"], dtype=float)
                o = int(np.searchsorted(frames, f0))
                lo = int(offsets[o - 1]) if o > 0 else 0
                hi = int(offsets[o])
                if hi == lo:
                    prev_box = box
                    continue
                ious = np.asarray([iou(b, box) for b in dets_all[lo:hi]])
                best = int(np.argmax(ious))
                if ious[best] < 0.5:
                    prev_box = box
                    continue
                gi = lo + best
                gfn_e = emb_all[gi]
                r0_e = r0g[gi]
                src = e.get("source", "")
                dscore = e.get("delivery_score")
                correct = int(e.get("correct", 0))
                if src in ("p0_tid", "p0") and dscore is not None and \
                        float(dscore) >= 0.5 and prev_box is not None and \
                        iou(box, prev_box) >= 0.5:
                    heur.append((f0, gfn_e.copy()))
                    if len(heur) > 8:
                        heur.pop(0)
                feats = {
                    "gfn_sim_human_root": float(gfn_e @ qe_h),
                    "r0_sim_human_root": float(r0_e @ qe_r0),
                    "gfn_sim_oracle_last": "",
                    "gfn_sim_oracle_max": "",
                    "r0_sim_oracle_max": "",
                    "gfn_sim_heur_last": "",
                    "gfn_sim_heur_max": "",
                    "gfn_margin_h": "",
                    "det_score": dscore if dscore is not None else 0.0,
                    "box_area": float((box[2] - box[0]) *
                                      (box[3] - box[1])),
                    "temporal_iou": float(iou(box, prev_box))
                    if prev_box is not None else 0.0,
                    "center_delta": "",
                    "consecutive_delivered": 0,
                    "missing_streak": 0,
                    "crowd": hi - lo,
                    "overlap_max": "",
                    "nearest_det_distance": "",
                    "heur_memory_age": "",
                    "oracle_memory_age": "",
                    "candidate_age": f0 - first_app.get(gid, f0),
                    "slots_oracle_count": min(len(slots), 2),
                    "slots_heur_count": len(heur),
                    "source": src,
                }
                if slots:
                    mem = slots[-2:]
                    gsims = [s[1] @ gfn_e for s in mem]
                    rsims = [s[2] @ r0_e for s in mem]
                    feats["gfn_sim_oracle_last"] = float(gsims[-1])
                    feats["gfn_sim_oracle_max"] = float(max(gsims))
                    feats["r0_sim_oracle_max"] = float(max(rsims))
                    feats["oracle_memory_age"] = f0 - slots[-1][0]
                if heur:
                    hs = np.stack([s[1] for s in heur[-8:]]) @ gfn_e
                    feats["gfn_sim_heur_last"] = float(hs[-1])
                    feats["gfn_sim_heur_max"] = float(hs.max())
                    feats["heur_memory_age"] = f0 - heur[-1][0]
                sims_all = emb_all[lo:hi] @ qe_h
                if len(sims_all) > 1:
                    order = np.argsort(-sims_all)
                    feats["gfn_margin_h"] = float(
                        sims_all[order[0]] - sims_all[order[1]])
                if hi - lo > 1:
                    other = np.delete(dets_all[lo:hi], best, axis=0)
                    feats["overlap_max"] = float(max(
                        iou(b, box) for b in other))
                    cc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    dc = np.stack([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                                   for b in other])
                    dd = np.hypot(dc[:, 0] - cc[0], dc[:, 1] - cc[1])
                    feats["nearest_det_distance"] = float(np.min(dd))
                if prev_box is not None:
                    c0 = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    c1 = ((prev_box[0] + prev_box[2]) / 2,
                          (prev_box[1] + prev_box[3]) / 2)
                    feats["center_delta"] = float(
                        np.hypot(c0[0] - c1[0], c0[1] - c1[1]))
                x = to_feature_vec(feats)
                x = (x - mean) / std
                with torch.inference_mode():
                    p = float(torch.sigmoid(
                        writer(torch.from_numpy(x[None]))).item())
                write = int(p >= threshold)
                rows.append({
                    "sequence": seq, "gid": gid, "frame": f0,
                    "source": src, "correct": correct,
                    "writer_score": round(p, 4), "write": write,
                    "memory_k": len(slots),
                })
                if write:
                    slots.append((f0, gfn_e.copy(), r0_e.copy(), correct))
                for k in ks:
                    st = agg[k]
                    mem = slots[-k:] if k > 0 else []
                    if write:
                        st["writes"] += 1
                        if correct:
                            st["tp"] += 1
                        else:
                            st["fp"] += 1
                    elif correct and not any(s[3] for s in mem):
                        st["fn"] += 1
                    if mem:
                        st["purity_sum"] += float(
                            np.mean([s[3] for s in mem]))
                        st["purity_n"] += 1
                # offline retrieval at V0 attempts using learned memory
                for af, rbox in attempts[gid]:
                    if af <= f0:
                        continue
                    for k in ks:
                        mem = slots[-k:]
                        if not mem:
                            continue
                        st = agg[k]
                        st["anchor_age"].append(af - mem[-1][0])
                        if rbox is None:
                            continue
                        oa = int(np.searchsorted(frames, af))
                        loa = int(offsets[oa - 1]) if oa > 0 else 0
                        hia = int(offsets[oa])
                        if hia == loa:
                            continue
                        gf = gt.get(af)
                        if gf is None or gid not in gf.gt_ids:
                            continue
                        tgt = np.asarray(
                            gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                        db = dets_all[loa:hia]
                        ious2 = np.asarray([iou(b, tgt) for b in db])
                        ti = int(np.argmax(ious2))
                        if ious2[ti] < 0.5:
                            continue
                        sims = np.maximum.reduce(
                            [emb_all[loa:hia] @ s[1] for s in mem])
                        rank = int((sims > sims[ti]).sum()) + 1
                        st["retrieval_top1"][0] += 1
                        st["retrieval_top1"][1] += int(rank <= 1)
                        st["retrieval_top3"][0] += 1
                        st["retrieval_top3"][1] += int(rank <= 3)
                prev_box = box
        print(f"sim {seq} rows={len(rows)}", flush=True)

    (N19 / f"write_safety_metrics_{args.tag}.csv").write_text(
        "k,writes,tp,fp,fn,precision,recall,purity\n" + "\n".join(
            f"{k},{st['writes']},{st['tp']},{st['fp']},{st['fn']},"
            f"{st['tp']/max(st['writes'],1):.4f},"
            f"{st['tp']/max(st['tp']+st['fn'],1):.4f},"
            f"{st['purity_sum']/max(st['purity_n'],1):.4f}"
            for k, st in sorted(agg.items())) + "\n", encoding="utf-8")
    with (N19 / f"purity_freshness_{args.tag}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "n_retrieval", "top1", "top3",
                    "median_anchor_age", "p90_anchor_age"])
        for k, st in sorted(agg.items()):
            ages = st["anchor_age"]
            w.writerow([
                k,
                st["retrieval_top1"][0],
                round(st["retrieval_top1"][1] /
                      max(st["retrieval_top1"][0], 1), 4),
                round(st["retrieval_top3"][1] /
                      max(st["retrieval_top3"][0], 1), 4),
                round(float(np.median(ages)), 1) if ages else "",
                round(float(np.percentile(ages, 90)), 1) if ages else "",
            ])
    with (N19 / f"learned_memory_simulation_{args.tag}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("SIM_DONE threshold=", threshold, flush=True)


if __name__ == "__main__":
    main()
