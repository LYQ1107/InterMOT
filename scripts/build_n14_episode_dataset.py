"""Build N14 training episodes (sequence-disjoint from calibration events).

Each episode: one identity, human frame t with GT box (authoritative write),
future frames t+1..t+8 with target-visible labels.  GT is training-label data
only; it is never used at inference time.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(".")
DT = Path("/path/to/dancetrack")
CALIB_SEQS = {
    "dancetrack0074", "dancetrack0075", "dancetrack0080", "dancetrack0082",
    "dancetrack0083", "dancetrack0086", "dancetrack0087", "dancetrack0096",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--future-horizon", type=int, default=8)
    ap.add_argument("--min-visible", type=int, default=12)
    ap.add_argument("--max-ids-per-seq", type=int, default=8)
    ap.add_argument("--out", default="outputs/n14/episode_manifest.csv")
    args = ap.parse_args()

    from sam3_intermot.datasets.dancetrack import DanceTrackDataset

    ds = DanceTrackDataset(str(DT), sequences=None, split="train")
    seqs = [
        s for s in sorted(ds.sequences)
        if s not in CALIB_SEQS
    ][: args.n_seq]
    print("train sequences:", seqs, flush=True)

    rows = []
    neg_added = 0
    for seq in seqs:
        gt = ds.load_gt(seq)
        ids = sorted({gid for entry in gt.values() for gid in entry.gt_ids})
        used = 0
        for gid in ids:
            if used >= args.max_ids_per_seq:
                break
            visible = sorted(f for f, e in gt.items() if gid in e.gt_ids)
            if len(visible) < args.min_visible:
                continue
            t = visible[1]
            if t + 2 > visible[-1]:
                continue
            pos = 0
            for k in range(1, args.future_horizon + 1):
                f = t + k
                entry = gt.get(f)
                vis = entry is not None and gid in entry.gt_ids
                box = np.zeros(4, dtype=float)
                if vis:
                    box = np.asarray(
                        entry.boxes[entry.gt_ids.index(gid)], dtype=float
                    )
                    pos += 1
                    others = [
                        (o, np.asarray(b, dtype=float))
                        for o, b in zip(entry.gt_ids, entry.boxes)
                        if o != gid
                    ]
                    ac = np.asarray(
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    )
                    others.sort(
                        key=lambda ob: np.linalg.norm(
                            np.asarray(
                                [(ob[1][0] + ob[1][2]) / 2,
                                 (ob[1][1] + ob[1][3]) / 2]
                            )
                            - ac
                        )
                    )
                    neg_boxes = [ob[1].tolist() for ob in others[:4]]
                else:
                    neg_boxes = []
                hb = np.asarray(
                    gt[t].boxes[gt[t].gt_ids.index(gid)], dtype=float
                )
                rows.append(
                    {
                        "episode_id": f"{seq}_{gid}",
                        "split": "train_f0",
                        "sequence": seq,
                        "gid": gid,
                        "human_frame": t,
                        "human_box_x1": hb[0], "human_box_y1": hb[1],
                        "human_box_x2": hb[2], "human_box_y2": hb[3],
                        "future_frame": f,
                        "target_visible": int(vis),
                        "future_box_x1": box[0], "future_box_y1": box[1],
                        "future_box_x2": box[2], "future_box_y2": box[3],
                        "distractor_gid": "",
                        "neg_boxes": json.dumps(neg_boxes),
                    }
                )
            if pos >= 2:
                used += 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    episodes = len({r["episode_id"] for r in rows})
    print(
        f"WROTE {out_path}: {len(rows)} samples ({neg_added} negatives), "
        f"{episodes} episodes, "
        f"seqs={len(seqs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
