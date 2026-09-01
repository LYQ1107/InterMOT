#!/usr/bin/env python
"""N12 CFA no-update baseline on official SAM3.1 full pipeline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/n12"


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
    ap.add_argument("--seq", default="dancetrack0074")
    ap.add_argument("--event-idx", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out", default="cfa_baselines.csv")
    args = ap.parse_args()

    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner,
        recall_at,
    )
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset

    events = load_events(args.seq)
    if not events:
        raise SystemExit(f"no accepted TRUE_MISS_NEW events in {args.seq}")
    ev = events[args.event_idx % len(events)]
    box = np.asarray(ev["gt_box"], dtype=float)

    runner = CFABackendRunner(
        checkpoint_path=str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"),
        split="train",
    )
    # ensure the backend runs on the requested GPU
    import torch
    torch.cuda.set_device(args.gpu)
    ep = runner.run_episode(
        sequence=args.seq,
        frame_idx=int(ev["frame"]),
        event_type=ev["event_type"],
        gid=int(ev["dataset_gt_id"]),
        human_box=box,
        horizon=args.horizon,
    )
    gt = DanceTrackDataset(
        "/path/to/dancetrack", sequences=[], split="train"
    ).load_gt(args.seq)
    row = {
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
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / args.out
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
