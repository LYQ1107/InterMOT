#!/usr/bin/env python
"""N19.5b: Future-recovery utility labels for write candidates.

For each delivered candidate (seq, gid, frame f), replay future recovery
attempts of the same identity at horizons 10/30/60/120/240/480 frames and
measure whether using this candidate as the query anchor (latest slot,
offline) improves the target rank versus the static H_i query. Strictly an
offline TRAIN/CAL label computation: future GT/attempts are used only to
score, never as an inference input.
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

from eval_route_c_r0 import iou  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402

OUT = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"
HORIZONS = [10, 30, 60, 120, 240, 480]


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(N19 / "write_dataset_cal10.csv"))
    ap.add_argument("--transactions-prefix",
                    default="reactivation_transactions_full")
    ap.add_argument("--events-prefix",
                    default="full_loop_v0_events_oracle_n19")
    ap.add_argument("--out", default=str(N19 / "future_utility_cal10.csv"))
    args = ap.parse_args()

    rows = load_csv(Path(args.dataset))
    tx = []
    for p in sorted(OUT.glob(f"{args.transactions_prefix}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                tx.append(json.loads(line))
    ev = []
    for p in sorted(OUT.glob(f"{args.events_prefix}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                ev.append(json.loads(line))

    by_seq = defaultdict(lambda: {"cand": [], "attempts": []})
    for r in rows:
        if r.get("gfn_sim_human_root") in (None, ""):
            continue
        by_seq[r["sequence"]]["cand"].append(
            (int(r["frame"]), int(r["gid"])))
    for t in tx:
        by_seq[t["sequence"]]["attempts"].append(
            (int(t["frame"]), int(t["gid"])))
    deliv_box = defaultdict(dict)
    for e in ev:
        if e.get("delivered") == 1 and e.get("delivered_box") is not None:
            deliv_box[e["sequence"]][(int(e["frame"]), int(e["gid"]))] = \
                np.asarray(e["delivered_box"], dtype=float)

    out_fields = ["sequence", "gid", "frame"] + [
        f"improved_{h}" for h in HORIZONS] + [
        f"n_attempts_{h}" for h in HORIZONS] + [
        f"any_improve_{h}" for h in HORIZONS]
    result = []
    acc = defaultdict(lambda: {h: {"n": 0, "improved": 0}
                               for h in HORIZONS})

    for seq, data in sorted(by_seq.items()):
        gt = load_gt(seq)
        z = np.load(CACHE / f"{seq}.npz")
        qz = np.load(CACHE / f"{seq}_queries.npz")
        frames = z["frames"]
        offsets = z["offsets"]
        det_boxes = z["boxes"]
        emb = z["emb"].astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        qgids = [int(g) for g in qz["gids"]]
        qemb = qz["qemb"].astype(np.float32)
        qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-8)
        qidx = {g: i for i, g in enumerate(qgids)}
        z.close()
        qz.close()

        # candidate query embeddings keyed by (gid, frame)
        cand_q = defaultdict(dict)
        for f0, gid in data["cand"]:
            o = int(np.searchsorted(frames, f0))
            lo = int(offsets[o - 1]) if o > 0 else 0
            hi = int(offsets[o])
            if hi == lo:
                continue
            box = deliv_box[seq].get((f0, gid))
            if box is None:
                continue
            db = det_boxes[lo:hi]
            ious = np.asarray([iou(b, box) for b in db])
            best = int(np.argmax(ious))
            if ious[best] < 0.5:
                continue
            cand_q[gid][f0] = emb[lo + best]

        # group attempts by gid, sorted by frame
        att_by_gid = defaultdict(list)
        for f0, gid in data["attempts"]:
            att_by_gid[gid].append(f0)
        for gid, atts in att_by_gid.items():
            atts = sorted(set(atts))
            qi = qidx.get(gid)
            if qi is None:
                continue
            qh = qemb[qi]
            cands = sorted(cand_q[gid].items())
            cidx = 0
            for a in atts:
                o = int(np.searchsorted(frames, a))
                lo = int(offsets[o - 1]) if o > 0 else 0
                hi = int(offsets[o])
                if hi == lo:
                    continue
                gf = gt.get(a)
                if gf is None or gid not in gf.gt_ids:
                    continue
                tgt = np.asarray(
                    gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                db = det_boxes[lo:hi]
                ious = np.asarray([iou(b, tgt) for b in db])
                ti = int(np.argmax(ious))
                if ious[ti] < 0.5:
                    continue  # target not in gallery: no ranking possible
                G = emb[lo:hi]  # m x D
                rank_static = int((G @ qh > G[ti] @ qh).sum()) + 1
                # advance candidate window: candidates within [a-480, a)
                while cidx < len(cands) and cands[cidx][0] < a - 480:
                    cidx += 1
                window = [c for c in cands[cidx:]
                          if c[0] < a]
                if not window:
                    continue
                cfs = np.asarray([c[0] for c in window])
                Q = np.stack([c[1] for c in window])  # n x D
                S = G @ Q.T  # m x n
                ranks = (S > S[ti][None, :]).sum(axis=0) + 1
                improved = ranks < rank_static
                # accumulate into per-candidate horizon buckets
                for j, (cf, imp) in enumerate(zip(cfs, improved)):
                    for h in HORIZONS:
                        if a - cf <= h:
                            key = (seq, gid, int(cf))
                            acc[key][h]["n"] += 1
                            acc[key][h]["improved"] += int(imp)

        # build output rows
        all_cands = sorted((g, f) for g, m in cand_q.items() for f in m)
        for gid, f0 in all_cands:
            row = {"sequence": seq, "gid": gid, "frame": f0}
            for h in HORIZONS:
                a = acc[(seq, gid, f0)][h]
                row[f"n_attempts_{h}"] = a["n"]
                row[f"improved_{h}"] = (
                    round(a["improved"] / a["n"], 4) if a["n"] else "")
                row[f"any_improve_{h}"] = int(a["improved"] > 0)
            result.append(row)
        print(f"utility {seq} cands={len(result)}", flush=True)

    with Path(args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(result)
    print(f"FUTURE_UTILITY_DONE rows={len(result)} file={args.out}",
          flush=True)


if __name__ == "__main__":
    main()
