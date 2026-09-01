#!/usr/bin/env python
"""N19.1/N19.2: Oracle causal refresh replay on the real FULL_LOOP_V0 attempt
distribution.

Memory slots are the GT-correct delivered observations recorded by the V0
run (frame <= attempt frame; strictly causal). The slot embedding is the
frozen GFN embedding of the cached detection best matching the delivered
box (fallback: the identity's first-appearance H_i query when no correct
delivery exists). Reader variants Last/MaxSim/Mean/AgeWeighted are compared
for K=1/2/4/8. This is an offline upper-bound diagnostic only.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_route_c_r0 import load_transactions, iou  # noqa: E402

OUT = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"


def load_trace():
    rows = []
    for p in sorted(OUT.glob("full_loop_v0_events_full_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--out-tag", default="oracle")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    N19.mkdir(parents=True, exist_ok=True)

    trace = load_trace()
    tx = load_transactions()
    by_seq = defaultdict(list)
    for t in tx:
        by_seq[t["sequence"]].append(t)
    cache_store = {}
    q_store = {}
    gt_store = {}
    deliveries = defaultdict(list)
    for seq in sorted(by_seq):
        from run_n18_full_loop_v0 import load_gt
        gt_store[seq] = load_gt(seq)
        for e in trace:
            if e.get("sequence") != seq:
                continue
            if e.get("delivered") == 1 and e.get("correct") == 1:
                f0, g0 = int(e["frame"]), int(e["gid"])
                gf0 = gt_store[seq].get(f0)
                if gf0 is not None and g0 in gf0.gt_ids:
                    bx = np.asarray(
                        gf0.boxes[gf0.gt_ids.index(g0)], dtype=float)
                    deliveries[(seq, g0)].append((f0, bx))
        z = np.load(CACHE / f"{seq}.npz")
        cache_store[seq] = {
            "frames": z["frames"], "offsets": z["offsets"],
            "boxes": z["boxes"], "emb": z["emb"]}
        z.close()
        qz = np.load(CACHE / f"{seq}_queries.npz")
        q_store[seq] = ({int(g): i for i, g in enumerate(qz["gids"])},
                        {"qemb": qz["qemb"], "qframe": qz["qframe"]})
        qz.close()

    def det_emb(seq, frame, box):
        z = cache_store[seq]
        o = np.searchsorted(z["frames"], frame)
        lo = int(z["offsets"][o - 1]) if o > 0 else 0
        hi = int(z["offsets"][o])
        if hi == lo:
            return None
        boxes_d = z["boxes"][lo:hi]
        ious = np.asarray([iou(b, box) for b in boxes_d])
        best = int(np.argmax(ious))
        if ious[best] < 0.5:
            return None
        e = z["emb"][lo + best].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-8)

    def fallback_emb(seq, gid):
        qi = q_store[seq][0].get(gid)
        if qi is None:
            return None
        e = q_store[seq][1]["qemb"][qi].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-8)

    rows = []
    agg = {}
    for k in ks:
        for reader in ("last", "max", "mean", "agew"):
            agg[(k, reader)] = {
                "n_present_det": 0, "hits": {1: 0, 3: 0, 5: 0, 10: 0},
                "mrr_sum": 0.0, "absent": [], "ages": [],
                "fallback": 0, "n_attempts": 0,
                "overall_hits": {1: 0, 3: 0, 5: 0, 10: 0},
                "overall_mrr": 0.0, "overall_n": 0,
            }
    for seq in sorted(by_seq):
        z = cache_store[seq]
        frames, offsets = z["frames"], z["offsets"]
        for t in by_seq[seq]:
            f, gid = int(t["frame"]), int(t["gid"])
            o = np.searchsorted(frames, f)
            lo = int(offsets[o - 1]) if o > 0 else 0
            hi = int(offsets[o])
            if hi == lo:
                continue
            boxes_d = z["boxes"][lo:hi]
            emb_d = z["emb"][lo:hi].astype(np.float32)
            emb_d = emb_d / (np.linalg.norm(
                emb_d, axis=1, keepdims=True) + 1e-8)
            row = {
                "sequence": seq, "frame": f, "gid": gid,
                "gt_present": 0, "detector_contains_target": 0,
                "best_det_iou": None, "rank_static": None,
                "rank_k4_max": None, "top1_sim_k4_max": None,
                "n_slots": 0, "anchor_age": None, "fallback": 1,
            }
            gf = None
            gt = gt_store.get(seq)
            gf = gt.get(f)
            gbox = None
            if gf is not None and gid in gf.gt_ids:
                gbox = np.asarray(gf.boxes[gf.gt_ids.index(gid)],
                                  dtype=float)
                row["gt_present"] = 1
                ious = np.asarray([iou(b, gbox) for b in boxes_d])
                best = int(np.argmax(ious))
                row["best_det_iou"] = round(float(ious[best]), 4)
                row["detector_contains_target"] = int(ious[best] >= 0.5)
            # static H_i ranking
            qs = fallback_emb(seq, gid)
            if qs is not None and gbox is not None:
                sims = emb_d @ qs
                row["rank_static"] = int((sims > sims[best]).sum()) + 1
            # oracle slots
            slots_all = [(fr, bx) for fr, bx in deliveries[(seq, gid)]
                         if fr <= f]
            slot_embs_by_k = {}
            for k in ks:
                es = []
                for fr, bx in slots_all[-k:]:
                    e = det_emb(seq, fr, bx)
                    if e is not None:
                        es.append(e)
                slot_embs_by_k[k] = es
            slot_embs = slot_embs_by_k.get(4, [])
            row["n_slots"] = len(slot_embs)
            if slot_embs:
                row["fallback"] = 0
                row["anchor_age"] = f - slots_all[-1][0]
                sim_max = np.maximum.reduce(
                    [emb_d @ e for e in slot_embs])
                row["top1_sim_k4_max"] = round(float(sim_max.max()), 4)
                if gbox is not None:
                    row["rank_k4_max"] = int(
                        (sim_max > sim_max[best]).sum()) + 1
            for k in ks:
                for reader in ("last", "max", "mean", "agew"):
                    es = slot_embs_by_k[k]
                    st = agg[(k, reader)]
                    st["n_attempts"] += 1
                    if not es:
                        st["fallback"] += 1
                        rk = row["rank_static"]
                        if rk not in (None, ""):
                            x = int(float(rk))
                            st["overall_n"] += 1
                            st["overall_mrr"] += 1.0 / x
                            for h in (1, 3, 5, 10):
                                st["overall_hits"][h] += int(x <= h)
                        continue
                    if reader == "last":
                        q = es[-1]
                        sims = emb_d @ q
                    elif reader == "max":
                        sims = np.maximum.reduce([emb_d @ e for e in es])
                    elif reader == "mean":
                        q = np.mean(np.stack(es), axis=0)
                        q = q / (np.linalg.norm(q) + 1e-8)
                        sims = emb_d @ q
                    else:
                        ages = np.asarray([f - fr for fr, _ in slots_all[-k:]
                                           ][:len(es)], dtype=float)
                        w = np.exp(-ages / 120.0)
                        w = w / (w.sum() + 1e-8)
                        q = np.sum(np.stack(es) * w[:, None], axis=0)
                        q = q / (np.linalg.norm(q) + 1e-8)
                        sims = emb_d @ q
                    if gbox is not None:
                        rank = int((sims > sims[best]).sum()) + 1
                        st["n_present_det"] += 1
                        for h in (1, 3, 5, 10):
                            st["hits"][h] += int(rank <= h)
                        st["mrr_sum"] += 1.0 / rank
                        st["ages"].append(f - slots_all[-1][0])
                        st["overall_n"] += 1
                        st["overall_mrr"] += 1.0 / rank
                        for h in (1, 3, 5, 10):
                            st["overall_hits"][h] += int(rank <= h)
                    else:
                        st["absent"].append(float(sims.max()))
            rows.append(row)
        print(f"oracle {seq} attempts={len(by_seq[seq])}", flush=True)

    fields = list(rows[0].keys())
    with (N19 / f"{args.out_tag}_refresh.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    agg_rows = []
    for (k, reader), st in sorted(agg.items()):
        n = st["n_present_det"]
        ages = st["ages"]
        agg_rows.append({
            "K": k, "reader": reader, "n_present_det": n,
            "top1": round(st["hits"][1] / n, 4) if n else None,
            "top3": round(st["hits"][3] / n, 4) if n else None,
            "top5": round(st["hits"][5] / n, 4) if n else None,
            "top10": round(st["hits"][10] / n, 4) if n else None,
            "mrr": round(st["mrr_sum"] / n, 4) if n else None,
            "absent_fp_0.5": round(
                float(np.mean([x >= 0.5 for x in st["absent"]])), 4)
            if st["absent"] else None,
            "absent_fp_0.6": round(
                float(np.mean([x >= 0.6 for x in st["absent"]])), 4)
            if st["absent"] else None,
            "mean_anchor_age": round(float(np.mean(ages)), 1) if ages else None,
            "median_anchor_age": round(float(np.median(ages)), 1)
            if ages else None,
            "p90_anchor_age": round(float(np.percentile(ages, 90)), 1)
            if ages else None,
            "fallback_rate": round(
                st["fallback"] / st["n_attempts"], 4)
            if st["n_attempts"] else None,
            "overall_top1": round(
                st["overall_hits"][1] / st["overall_n"], 4)
            if st["overall_n"] else None,
            "overall_top3": round(
                st["overall_hits"][3] / st["overall_n"], 4)
            if st["overall_n"] else None,
            "overall_top5": round(
                st["overall_hits"][5] / st["overall_n"], 4)
            if st["overall_n"] else None,
            "overall_top10": round(
                st["overall_hits"][10] / st["overall_n"], 4)
            if st["overall_n"] else None,
            "overall_mrr": round(
                st["overall_mrr"] / st["overall_n"], 4)
            if st["overall_n"] else None,
        })
    with (N19 / f"{args.out_tag}_memory_k.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print("ORACLE_REFRESH_DONE rows=", len(rows), flush=True)


gt_store = {}


if __name__ == "__main__":
    main()
