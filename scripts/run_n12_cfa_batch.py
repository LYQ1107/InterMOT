#!/usr/bin/env python
"""N12 CFA no-update batch: one box prompt per TRUE_MISS_NEW event,
measure Target Recall@1/3/5/10/30 on the official SAM3.1 full pipeline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/n12"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")


def load_events(seq: str, event_type: str = "TRUE_MISS_NEW", budget: str = "b8"):
    path = ROOT / "outputs/n10/real" / f"human_{budget}" / seq / "interaction_events.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("accepted") and e.get("event_type") == event_type:
            events.append(e)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--event-idx", type=int, default=0)
    ap.add_argument("--out", default="cfa_baselines.csv")
    ap.add_argument("--seqs", default=(
        "dancetrack0074 dancetrack0075 dancetrack0080 dancetrack0082 "
        "dancetrack0083 dancetrack0086 dancetrack0087 dancetrack0096"
    ))
    args = ap.parse_args()

    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner,
        recall_at,
    )
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset

    import torch
    torch.cuda.set_device(args.gpu)

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    ds = DanceTrackDataset(
        "/path/to/dancetrack", sequences=[], split="train"
    )
    seqs = args.seqs.split()
    rows = []
    for seq in seqs:
        events = load_events(seq)
        if not events:
            print(f"no events: {seq}", flush=True)
            continue
        ev = events[args.event_idx % len(events)]
        box = np.asarray(ev["gt_box"], dtype=float)
        ep = runner.run_episode(
            sequence=seq,
            frame_idx=int(ev["frame"]),
            event_type=ev["event_type"],
            gid=int(ev["dataset_gt_id"]),
            human_box=box,
            horizon=args.horizon,
        )
        gt = ds.load_gt(seq)
        rows.append(
            {
                "sequence": ep.sequence,
                "frame": ep.frame,
                "event_type": ep.event_type,
                "gid": ep.gid,
                "prompt_had_output": int(ep.prompt_had_output),
                "recall_1": round(recall_at(ep, gt, 1), 3),
                "recall_3": round(recall_at(ep, gt, 3), 3),
                "recall_5": round(recall_at(ep, gt, 5), 3),
                "recall_10": round(recall_at(ep, gt, 10), 3),
                "recall_30": round(recall_at(ep, gt, 30), 3),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    runner.close()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / args.out
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "prompt_had_output": sum(r["prompt_had_output"] for r in rows),
                "mean_recall_1": round(
                    np.mean([r["recall_1"] for r in rows]), 3
                ),
                "mean_recall_3": round(
                    np.mean([r["recall_3"] for r in rows]), 3
                ),
                "mean_recall_5": round(
                    np.mean([r["recall_5"] for r in rows]), 3
                ),
                "mean_recall_10": round(
                    np.mean([r["recall_10"] for r in rows]), 3
                ),
                "mean_recall_30": round(
                    np.mean([r["recall_30"] for r in rows]), 3
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
