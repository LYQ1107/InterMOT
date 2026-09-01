#!/usr/bin/env python
"""Build the N15 I2Q training manifest from the 30 train sequences (GT only)."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_intermot.identity_anchor.identity_benchmark import load_gt

ROOT = Path(".")
DT = Path("/path/to/dancetrack")
TRAIN30 = json.loads(
    (ROOT / "outputs/n15/n15_frozen.json").read_text()
)["split"]["train30"]
OUT = ROOT / "outputs/n15/i2q_train_manifest.csv"


def main() -> None:
    deltas = (1, 3, 5, 10, 30)
    rng = np.random.default_rng(42)
    rows = []
    n_pos = 0
    n_neg = 0
    for seq in TRAIN30:
        gt = load_gt(DT / "train" / seq)
        frames = sorted(gt.keys())
        max_f = max(frames)
        for t in frames:
            if t + max(deltas) > max_f:
                continue
            candidates = [
                (g, b) for g, b in gt[t]
                if (b[2] - b[0]) * (b[3] - b[1]) >= 1200
            ]
            rng.shuffle(candidates)
            for gid, hb in candidates[:2]:
                for d in deltas:
                    f = t + d
                    if f not in gt:
                        continue
                    pos = [b for g, b in gt[f] if g == gid]
                    if not pos:
                        continue
                    fb = pos[0]
                    rows.append(
                        {
                            "sequence": seq, "human_frame": t, "gid": gid,
                            "human_box": [round(float(v), 2) for v in hb],
                            "future_frame": f, "visible": 1,
                            "future_box": [round(float(v), 2) for v in fb],
                            "neg_boxes": [],
                        }
                    )
                    n_pos += 1
                # negative samples: same anchor, future frames where gid absent
                absent = [f2 for f2 in range(t + 1, min(t + 91, max_f + 1))
                          if f2 not in gt or gid not in [g for g, _ in gt.get(f2, [])]]
                rng.shuffle(absent)
                for f2 in absent[:4]:
                    rows.append(
                        {
                            "sequence": seq, "human_frame": t, "gid": gid,
                            "human_box": [round(float(v), 2) for v in hb],
                            "future_frame": f2, "visible": 0,
                            "future_box": [],
                            "neg_boxes": [],
                        }
                    )
                    n_neg += 1
        if n_pos >= 1800:
            break
    # attach negatives (other identities at the future frame) for a subset
    by_key = {}
    for seq in TRAIN30:
        by_key[seq] = load_gt(DT / "train" / seq)
    for r in rows:
        f = r["future_frame"]
        negs = [
            [round(float(v), 2) for v in b]
            for g, b in by_key[r["sequence"]].get(f, [])
            if g != r["gid"]
        ]
        rng.shuffle(negs)
        r["neg_boxes"] = negs[:3]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["sequence", "human_frame", "gid", "human_box",
                            "future_frame", "visible", "future_box", "neg_boxes"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "human_box": json.dumps(r["human_box"]),
                    "future_box": json.dumps(r["future_box"]),
                    "neg_boxes": json.dumps(r["neg_boxes"]),
                }
            )
    print(f"rows={len(rows)} positives={n_pos} negatives={n_neg}")


if __name__ == "__main__":
    main()
