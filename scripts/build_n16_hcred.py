#!/usr/bin/env python
"""Build HCRED: Human-Correction Re-Detection Episodes from DanceTrack.

Reference frame t (simulated human correction) -> future frame t+delta with
labels: target present/absent, generic SAM3 candidate present/absent (from
the frozen AUTO P0 output), crowd, distractor proximity, scale change.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_intermot.identity_anchor.identity_benchmark import load_gt

ROOT = Path(".")
DT = Path("/path/to/dancetrack")
P0 = ROOT / "outputs/n9/p0_train"

TRAIN30 = json.loads((ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]["train30"]
CAL10 = json.loads((ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]["calibration10"]


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_p0(seq):
    p = P0 / f"{seq}.txt"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame = int(float(parts[0])) - 1
        tid = int(float(parts[1]))
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        out.setdefault(frame, []).append((tid, np.asarray([x, y, x + w, y + h], float)))
    return out


def main() -> None:
    deltas = (1, 3, 5, 10, 30, 60)
    rows = []
    miss_stats = []
    rng = np.random.default_rng(42)
    for seq in TRAIN30 + CAL10:
        split = "train" if seq in TRAIN30 else "calibration"
        gt = load_gt(DT / "train" / seq)
        p0 = load_p0(seq)
        frames = sorted(gt.keys())
        if not frames:
            continue
        max_f = max(frames)
        per_delta = {d: [0, 0] for d in deltas}  # miss, present
        for t in frames:
            if t + max(deltas) > max_f:
                continue
            rows_t = gt[t]
            for gid, hb in rows_t:
                area = (hb[2] - hb[0]) * (hb[3] - hb[1])
                if area < 1200:
                    continue
                for d in deltas:
                    f = t + d
                    if f not in gt:
                        continue
                    target_present = any(g == gid for g, _ in gt[f])
                    if not target_present:
                        continue
                    fb = next(b for g, b in gt[f] if g == gid)
                    cands = p0.get(f, [])
                    cand_present = any(iou_xyxy(cb, fb) >= 0.5 for _, cb in cands)
                    generic_miss = not cand_present
                    per_delta[d][0] += int(generic_miss)
                    per_delta[d][1] += 1
                    if rng.random() < 0.35:
                        # distractor: nearest other GT person at f
                        others = [b for g, b in gt[f] if g != gid]
                        dists = [
                            np.linalg.norm((fb[:2] + fb[2:]) / 2 - (b[:2] + b[2:]) / 2)
                            for b in others
                        ] if others else [1e9]
                        rows.append(
                            {
                                "sequence": seq, "split": split, "t": t, "gid": gid,
                                "human_box": json.dumps([round(float(v), 1) for v in hb]),
                                "f": f, "delta": d, "target_present": 1,
                                "target_box": json.dumps([round(float(v), 1) for v in fb]),
                                "generic_candidate_present": int(cand_present),
                                "generic_miss": int(generic_miss),
                                "crowd": len(rows_t), "distractor_dist": round(float(min(dists)), 1),
                                "area": round(area, 1),
                            }
                        )
        miss_stats.append(
            {
                "sequence": seq, "split": split,
                **{f"miss_d{d}": per_delta[d][0] for d in deltas},
                **{f"present_d{d}": per_delta[d][1] for d in deltas},
            }
        )
    # add absent episodes (target not present at f) for presence training
    n_absent = 0
    for seq in TRAIN30:
        gt = load_gt(DT / "train" / seq)
        frames = sorted(gt.keys())
        max_f = max(frames)
        for t in frames[::3]:
            if t + 60 > max_f:
                continue
            for gid, hb in gt[t]:
                area = (hb[2] - hb[0]) * (hb[3] - hb[1])
                if area < 1200:
                    continue
                for d in (5, 10, 30):
                    f = t + d
                    if f in gt and any(g == gid for g, _ in gt[f]):
                        continue
                    if f not in gt:
                        continue
                    rows.append(
                        {
                            "sequence": seq, "split": "train", "t": t, "gid": gid,
                            "human_box": json.dumps([round(float(v), 1) for v in hb]),
                            "f": f, "delta": d, "target_present": 0,
                            "target_box": "", "generic_candidate_present": "",
                            "generic_miss": "", "crowd": len(gt[t]),
                            "distractor_dist": "", "area": round(area, 1),
                        }
                    )
                    n_absent += 1
                    if n_absent >= 2000:
                        break
                if n_absent >= 2000:
                    break
            if n_absent >= 2000:
                break
    out = ROOT / "outputs/n16"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "hcred_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (out / "generic_miss_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(miss_stats[0].keys()))
        w.writeheader()
        w.writerows(miss_stats)
    print(f"episodes={len(rows)} absent={n_absent} "
          f"miss={sum(1 for r in rows if r.get('generic_miss') == 1)}")


if __name__ == "__main__":
    main()
