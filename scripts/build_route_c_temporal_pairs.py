#!/usr/bin/env python
"""N18 RouteC.2: build the THAR-style temporal query->future pair dataset.

Query: the identity's first human-like GT crop (same H_i protocol as
FULL_LOOP_V0). Gallery: future frames t_q + delta. Positive = same identity
in the gallery frame; hard negatives = nearest spatial person / highest
frozen-GFN-similarity wrong person / highest-score wrong detection. Absent
gallery frames are kept as explicit negatives so the model learns "no match".

TRAIN = train30, CAL = calibration10. val25 is never read here.
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
OUT = ROOT / "outputs/n18/route_c"
CACHE = OUT / "gfn_cache"

GAP_BINS = [1, 3, 5, 10, 30, 60, 120, 240, 480, float("inf")]
ANCHOR_AGE_BINS = [30, 60, 120, 240, 480, float("inf")]
K_NEG = 4


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


def load_gt(seq):
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    DT = Path("/path/to/dancetrack")
    return DanceTrackDataset(str(DT), sequences=[], split="train").load_gt(seq)


def gap_bin_of(delta):
    for b in GAP_BINS:
        if delta <= b:
            return "480+" if b == float("inf") else b
    return "480+"


def age_bin_of(age):
    for b in ANCHOR_AGE_BINS:
        if age <= b:
            return "480+" if b == float("inf") else b
    return "480+"


def build_seq(seq, split, sample_per_bin, seed):
    cache = np.load(CACHE / f"{seq}.npz")
    qc = np.load(CACHE / f"{seq}_queries.npz")
    frames = cache["frames"]
    offsets = cache["offsets"]
    boxes = cache["boxes"]
    scores = cache["scores"]
    emb = cache["emb"]
    frame_of_det = {}
    for i, f in enumerate(frames):
        lo = int(offsets[i - 1]) if i > 0 else 0
        for d in range(lo, int(offsets[i])):
            frame_of_det[d] = int(f)
    # det indices per frame
    dets_at = defaultdict(list)
    for i, f in enumerate(frames):
        lo = int(offsets[i - 1]) if i > 0 else 0
        dets_at[int(f)] = list(range(lo, int(offsets[i])))

    gt = load_gt(seq)
    gt_at = {f: gf for f, gf in gt.items()}
    gid_frames = defaultdict(list)
    for f in sorted(gt):
        for gid in gt[f].gt_ids:
            gid_frames[gid].append(f)
    all_frames = sorted({int(f) for f in frames})

    qidx = {int(g): i for i, g in enumerate(qc["gids"])}
    rng = np.random.RandomState(seed)
    rows = []
    stats = defaultdict(int)

    for qi, gid in enumerate(qc["gids"]):
        gid = int(gid)
        qf = int(qc["qframe"][qi])
        qemb = qc["qemb"][qi].astype(np.float32)
        qemb = qemb / (np.linalg.norm(qemb) + 1e-8)
        fs = gid_frames.get(gid, [])
        fs_set = set(fs)
        for gi in range(len(GAP_BINS) - 1):
            lo, hi = GAP_BINS[gi], GAP_BINS[gi + 1]
            pres = [t for t in fs if t > qf and t - qf >= lo
                    and (hi == float("inf") or t - qf <= hi)]
            absn = [t for t in all_frames
                    if t > qf and t - qf >= lo
                    and (hi == float("inf") or t - qf <= hi)
                    and t not in fs_set]
            picked_p = sorted(rng.choice(
                pres, size=min(len(pres), sample_per_bin),
                replace=False).tolist())
            picked_a = sorted(rng.choice(
                absn, size=min(len(absn), sample_per_bin),
                replace=False).tolist())
            for tg in picked_p + picked_a:
                delta = tg - qf
                dids = dets_at.get(tg, [])
                crowd = len(dids)
                gbox = None
                gf = gt_at.get(tg)
                if gf is not None and gid in gf.gt_ids:
                    gbox = np.asarray(
                        gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                present = gbox is not None
                if not present and not dids:
                    continue
                det_idx = -1
                rank = -1
                det_has = 0
                area = None
                near_dist = None
                negs = []
                if dids and qemb.size:
                    sims = emb[dids].astype(np.float32) @ qemb
                    order = np.argsort(-sims)
                else:
                    sims = np.zeros(len(dids), dtype=np.float32)
                    order = np.argsort(-sims)
                if present:
                    ious = np.asarray([iou(boxes[d], gbox) for d in dids])
                    if len(ious):
                        best = int(np.argmax(ious))
                        det_idx = dids[best]
                        det_has = int(ious[best] >= 0.5)
                        rank = int(np.where(order == best)[0][0]) + 1
                    w, h = gbox[2] - gbox[0], gbox[3] - gbox[1]
                    area = float(w * h)
                    c0 = (gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2
                    others = []
                    if gf is not None:
                        for oid, ob in zip(gf.gt_ids, gf.boxes):
                            if oid == gid:
                                continue
                            ob = np.asarray(ob, dtype=float)
                            oc = ((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2)
                            others.append(np.hypot(oc[0] - c0[0],
                                                   oc[1] - c0[1]))
                    near_dist = float(min(others)) if others else None
                    # hard negatives: wrong persons only
                    pool = [d for d in dids
                            if iou(boxes[d], gbox) < 0.3 and d != det_idx]
                    if pool:
                        sim_rank = sorted(pool, key=lambda d: -sims[dids.index(d)])
                        score_rank = sorted(
                            pool, key=lambda d: -scores[d])
                        cen = {d: np.hypot(
                            (boxes[d][0] + boxes[d][2]) / 2 - c0[0],
                            (boxes[d][1] + boxes[d][3]) / 2 - c0[1])
                            for d in pool}
                        near_rank = sorted(pool, key=lambda d: cen[d])
                        chosen = []
                        for cand in (near_rank, sim_rank, score_rank,
                                     sim_rank):
                            for d in cand:
                                if d not in chosen:
                                    chosen.append(d)
                                    break
                        negs = chosen[:K_NEG]
                else:
                    # absent: all dets are potential false matches
                    sim_rank = sorted(dids, key=lambda d: -sims[dids.index(d)])
                    score_rank = sorted(dids, key=lambda d: -scores[d])
                    chosen = []
                    for d in sim_rank + score_rank:
                        if d not in chosen:
                            chosen.append(d)
                        if len(chosen) >= K_NEG:
                            break
                    negs = chosen[:K_NEG]
                stats["rows"] += 1
                stats["pos_rows" if (present and det_has) else
                      "present_nodet_rows" if present else "absent_rows"] += 1
                stats[f"gap_{gap_bin_of(delta)}"] += 1
                rows.append({
                    "split": split, "sequence": seq, "gid": gid,
                    "query_frame": qf, "gallery_frame": tg,
                    "delta": delta, "anchor_age": delta,
                    "gap_bin": gap_bin_of(delta),
                    "anchor_age_bin": age_bin_of(delta),
                    "crowd_size": crowd, "target_present": int(present),
                    "detector_contains_target": det_has,
                    "det_idx": det_idx,
                    "frozen_gfn_rank": rank,
                    "bbox_area": None if area is None else round(area, 1),
                    "nearest_distractor_distance":
                    None if near_dist is None else round(near_dist, 2),
                    "hard_neg_idxs": json.dumps([int(x) for x in negs]),
                    "n_hard_neg": len(negs),
                })
    return rows, dict(stats), len(qc["gids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-per-bin", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split = json.loads((ROOT / "outputs/n15/n15_frozen.json").read_text())
    out_rows = []
    totals = defaultdict(int)
    per_seq = []
    for name, seqs in [("train", split["split"]["train30"]),
                       ("cal", split["split"]["calibration10"])]:
        for i, seq in enumerate(sorted(seqs)):
            rows, st, nids = build_seq(seq, name, args.sample_per_bin,
                                       args.seed + i)
            out_rows.extend(rows)
            for k, v in st.items():
                totals[k] += v
            per_seq.append({"split": name, "sequence": seq,
                            "n_identities": nids, "n_rows": len(rows)})
            print(f"{name} {seq} ids={nids} rows={len(rows)}", flush=True)

    fieldnames = list(out_rows[0].keys())
    for name in ("train", "cal"):
        sub = [r for r in out_rows if r["split"] == name]
        with (OUT / f"temporal_pairs_{name}.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sub)
    (OUT / "temporal_dataset_stats.json").write_text(
        json.dumps({
            "n_rows": len(out_rows),
            "n_train_rows": totals.get("rows", 0),
            "pos_rows": totals["pos_rows"],
            "present_nodet_rows": totals["present_nodet_rows"],
            "absent_rows": totals["absent_rows"],
            "n_sequences": len(per_seq),
            "n_identities": sum(s["n_identities"] for s in per_seq),
            "per_sequence": per_seq,
            "gap_distribution": {k: totals[k] for k in sorted(totals)
                                 if k.startswith("gap_")},
            "sample_per_bin": args.sample_per_bin, "seed": args.seed,
        }, indent=1))
    with (OUT / "gap_distribution.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gap_bin", "n_rows"])
        for k in sorted(totals):
            if k.startswith("gap_"):
                w.writerow([k[4:], totals[k]])
    with (OUT / "anchor_age_distribution.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anchor_age_bin", "n_rows"])
        age = defaultdict(int)
        for r in out_rows:
            age[r["anchor_age_bin"]] += 1
        for k in sorted(age, key=lambda x: (x == "480+", x)):
            w.writerow([k, age[k]])
    with (OUT / "hard_negative_stats.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "n_rows", "n_with_hard_neg", "mean_n_hard"])
        by = defaultdict(lambda: [0, 0, 0])
        for r in out_rows:
            s = by[r["sequence"]]
            s[0] += 1
            s[1] += int(r["n_hard_neg"] > 0)
            s[2] += r["n_hard_neg"]
        for seq in sorted(by):
            a, b, c = by[seq]
            w.writerow([seq, a, b, round(c / max(1, a), 2)])
    print("PAIRS_DONE total_rows=", len(out_rows), flush=True)


if __name__ == "__main__":
    main()
