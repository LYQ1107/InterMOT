#!/usr/bin/env python
"""Sample N17 training/calibration episodes and the unique frame list."""

import csv
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n17"


def load_rows():
    with (ROOT / "outputs/n16/hcred_manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_absent(rows, seq, t, gid, hb, gt, rng, n=3, horizon=90):
    added = 0
    for f in range(t + 1, min(t + horizon + 1, max(gt) + 1)):
        if f in gt and any(g == gid for g, _ in gt[f]):
            continue
        rows.append(
            {
                "sequence": seq, "split": "train", "t": t, "gid": gid,
                "human_box": json.dumps([round(float(v), 1) for v in hb]),
                "f": f, "delta": f - t, "target_present": 0,
                "target_box": "", "generic_candidate_present": "",
                "generic_miss": "", "crowd": len(gt.get(t, [])),
                "distractor_dist": "", "area": round(
                    (hb[2] - hb[0]) * (hb[3] - hb[1]), 1
                ),
            }
        )
        added += 1
        if added >= n:
            break
    return added


def main():
    rng = random.Random(17)
    rows = load_rows()
    train = [r for r in rows if r["split"] == "train"]
    cal = [r for r in rows if r["split"] == "calibration"]
    train_miss = [r for r in train if r["generic_miss"] == "1"]
    train_pres = [r for r in train if r["generic_miss"] == "0"]
    cal_miss = [r for r in cal if r["generic_miss"] == "1"]
    cal_pres = [r for r in cal if r["generic_miss"] == "0"]

    # training sample: miss 20k, present (hard) 12k, absent 10k
    miss_pick = rng.sample(train_miss, min(20000, len(train_miss)))
    hard = [r for r in train_pres
            if (r["distractor_dist"] != "" and float(r["distractor_dist"]) < 100)
            or (r["crowd"] != "" and int(r["crowd"]) >= 8)]
    pres_pick = rng.sample(hard if len(hard) >= 12000 else train_pres,
                           min(12000, len(hard if len(hard) >= 12000 else train_pres)))
    # absent from GT (same helper as N16)
    from sam3_intermot.identity_anchor.identity_benchmark import load_gt
    sys.path.insert(0, str(ROOT))
    DT = Path("/path/to/dancetrack")
    absent_pick = []
    pool = rng.sample(train_miss + train_pres, min(20000, len(train_miss + train_pres)))
    seqs_gt = {}
    for r in pool:
        seq = r["sequence"]
        if seq not in seqs_gt:
            seqs_gt[seq] = load_gt(DT / "train" / seq)
        hb = np.asarray(json.loads(r["human_box"]), dtype=float)
        add_absent(absent_pick, seq, int(r["t"]), int(r["gid"]), hb,
                   seqs_gt[seq], rng, n=3, horizon=120)
        if len(absent_pick) >= 12000:
            break
    train_pick = miss_pick + pres_pick + absent_pick
    rng.shuffle(train_pick)
    with (OUT / "train_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(train_pick)

    # calibration eval sample: miss 1500, present 1000, absent 500
    cal_pick = rng.sample(cal_miss, min(1500, len(cal_miss))) + rng.sample(
        cal_pres, min(1000, len(cal_pres))
    )
    cal_absent = []
    pool = rng.sample(cal_miss, min(500, len(cal_miss)))
    seqs_gt2 = {}
    for r in pool:
        seq = r["sequence"]
        if seq not in seqs_gt2:
            seqs_gt2[seq] = load_gt(DT / "train" / seq)
        hb = np.asarray(json.loads(r["human_box"]), dtype=float)
        add_absent(cal_absent, seq, int(r["t"]), int(r["gid"]), hb,
                   seqs_gt2[seq], rng, n=1)
        if len(cal_absent) >= 500:
            break
    for r in cal_absent:
        r["split"] = "calibration"
    cal_pick += cal_absent
    rng.shuffle(cal_pick)
    with (OUT / "cal_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(cal_pick)

    # unique frame list
    frames = set()
    for r in train_pick + cal_pick:
        frames.add((r["sequence"], int(r["t"])))
        frames.add((r["sequence"], int(r["f"])))
    with (OUT / "unique_frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "frame"])
        for seq, fr in sorted(frames):
            w.writerow([seq, fr])
    print(
        f"train={len(train_pick)} (miss={len(miss_pick)} pres={len(pres_pick)} "
        f"absent={len(absent_pick)}) cal={len(cal_pick)} unique_frames={len(frames)}"
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
