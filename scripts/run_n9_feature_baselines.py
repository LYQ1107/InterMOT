"""Feature-feasibility baselines on per-frame decision episodes (CPU)."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import match_boxes, read_mot_rows
from sam3_intermot.n9.relink_benchmark import load_decision_csv


ROOT = Path(".")
SPLIT = os.environ.get("N9_SPLIT", "train")
SEQS = sorted(os.environ.get("N9_SEQS", "").split())
BENCH = ROOT / "outputs/n9/benchmark" / SPLIT
FEAT = ROOT / "outputs/n9/features"
OUT = ROOT / "outputs/n9/tables"


def load_feats(seq):
    p = FEAT / f"{seq}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    key = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(d["frame"], d["tid"]))}
    return key, d["feat"].astype(np.float32)


def feat_at(key, feats, frame, tid):
    i = key.get((int(frame), int(tid)))
    if i is None:
        return None
    v = feats[i]
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def mem_feat(key, feats, gid_to_tid, gid, mem_frames):
    vals = []
    for f in mem_frames:
        tid = gid_to_tid.get(f, {}).get(gid)
        if tid is None:
            continue
        v = feat_at(key, feats, f, tid)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    v = np.mean(vals, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def motion_score(hist_box, hist_vel, gap, cur_box):
    h = np.asarray(hist_box, float)
    n = np.asarray(cur_box, float)
    if hist_vel is None or gap is None:
        pred = h
    else:
        v = np.asarray(hist_vel, float)
        pred = h + np.asarray([v[0] * gap, v[1] * gap, v[0] * gap, v[1] * gap])
    ix1, iy1 = max(pred[0], n[0]), max(pred[1], n[1])
    ix2, iy2 = min(pred[2], n[2]), min(pred[3], n[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (pred[2] - pred[0]) * (pred[3] - pred[1]) + (n[2] - n[0]) * (n[3] - n[1]) - inter
    iou = inter / union if union > 0 else 0.0
    c1 = np.asarray([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
    c2 = np.asarray([(n[0] + n[2]) / 2, (n[1] + n[3]) / 2])
    dist = float(np.linalg.norm(c1 - c2))
    return iou, dist


def rank_metrics(rows):
    if not rows:
        return 0.0, 0.0
    r1 = r5 = 0
    for r in rows:
        scores = [r["pos"]] + [s for s, *_ in r["negs"]]
        rank = sum(1 for s in scores if s > r["pos"] + 1e-9) + 1
        r1 += rank == 1
        r5 += rank <= 5
    return r1 / len(rows), r5 / len(rows)


def main():
    if not SEQS:
        SEQS.extend(sorted(p.stem for p in BENCH.glob("*.csv") if p.stem != "summary"))
    OUT.mkdir(parents=True, exist_ok=True)
    ds = DanceTrackDataset("/path/to/dancetrack", split=SPLIT)
    results = {}
    for wm in (0.0, 1.0, 2.0):
        rows_all = []
        for seq in SEQS:
            csvp = BENCH / f"{seq}.csv"
            key, feats = load_feats(seq)
            if not csvp.exists() or key is None:
                continue
            p0 = read_mot_rows(
                (ROOT / "outputs/n9/p0_train" if SPLIT == "train" else ROOT / "outputs/n5/integrity/canonical_mot_results/b0")
                / f"{seq}.txt"
            )
            gt = ds.load_gt(seq)
            gid_to_tid = {}
            for f, fr in p0.items():
                gtf = gt.get(f)
                if gtf is None or not gtf.boxes or not fr:
                    continue
                ms = match_boxes(
                    [np.asarray(b, float) for b in gtf.boxes],
                    [np.asarray(b, float) for _, b in fr],
                    0.5,
                )
                gid_to_tid[f] = {
                    gtf.gt_ids[gi]: int(fr[pi][0]) for gi, pi, _ in ms
                }
            for e in load_decision_csv(csvp):
                if e["miss"] or not e["mem_frames"] or e["pos_tid"] is None:
                    continue
                mf = mem_feat(key, feats, gid_to_tid, e["gid"], e["mem_frames"])
                pf = feat_at(key, feats, e["frame"], e["pos_tid"])
                if mf is None or pf is None:
                    continue
                last_frame = e["mem_frames"][-1]
                last_tid = gid_to_tid.get(last_frame, {}).get(e["gid"])
                hist_box = None
                if last_tid is not None:
                    for t, b in p0.get(last_frame, []):
                        if int(t) == int(last_tid):
                            hist_box = b
                            break
                pos_box = None
                for t, b in p0.get(e["frame"], []):
                    if int(t) == int(e["pos_tid"]):
                        pos_box = b
                        break
                iou_pos, _ = (
                    motion_score(hist_box, None, e["gap"] or 0, pos_box)
                    if hist_box is not None and pos_box is not None
                    else (0.0, 1e9)
                )
                pos_score = float(np.dot(mf, pf)) + wm * iou_pos
                negs = []
                for nt in e["neg_tids"]:
                    nf = feat_at(key, feats, e["frame"], nt)
                    if nf is not None:
                        nb = None
                        for t, b in p0.get(e["frame"], []):
                            if int(t) == int(nt):
                                nb = b
                                break
                        iou_n, _ = (
                            motion_score(hist_box, None, e["gap"] or 0, nb)
                            if hist_box is not None and nb is not None
                            else (0.0, 1e9)
                        )
                        negs.append((float(np.dot(mf, nf)) + wm * iou_n, nt))
                rows_all.append(
                    {
                        "pos": pos_score + wm * 0.0,
                        "negs": negs,
                        "gap": e["gap"] or 0,
                        "crowd": e["crowd"],
                    }
                )
        if not rows_all:
            continue
        pos = [r["pos"] for r in rows_all]
        neg = [s for r in rows_all for s, *_ in r["negs"]]
        auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg) if pos and neg else float("nan")
        r1, r5 = rank_metrics(rows_all)
        by_gap = {}
        for lo, hi, name in (
            (0, 5, "0-5"),
            (6, 15, "6-15"),
            (16, 30, "16-30"),
            (31, 60, "31-60"),
            (61, 10**9, ">60"),
        ):
            sub = [r for r in rows_all if lo <= r["gap"] <= hi]
            a, b = rank_metrics(sub)
            by_gap[name] = {"r1": round(a, 4), "r5": round(b, 4), "n": len(sub)}
        by_crowd = {}
        for name, lo, hi in (("<=4", 0, 4), ("5-8", 5, 8), ("9-12", 9, 12), (">12", 13, 10**9)):
            sub = [r for r in rows_all if lo <= r["crowd"] <= hi]
            a, b = rank_metrics(sub)
            by_crowd[name] = {"r1": round(a, 4), "r5": round(b, 4), "n": len(sub)}
        results[f"reid_wm{int(wm)}"] = {
            "auc": round(auc, 4),
            "r1": round(r1, 4),
            "r5": round(r5, 4),
            "n": len(rows_all),
            "by_gap": by_gap,
            "by_crowd": by_crowd,
        }
    (OUT / "feature_feasibility.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
