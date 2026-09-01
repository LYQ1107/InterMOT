#!/usr/bin/env python
"""Convert the live trajdump JSONL (attempt -> ranked trajectories) into
the N20 all-candidate cache format consumed by
build_n20_kplus1_dataset.py and build_n21_tracklet_identity_dataset.py.
CPU-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_n18_full_loop_v0 import load_gt  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import iou  # noqa: E402

SRC = ROOT / "outputs/n21/train30_true_onpolicy_trajdump"
DST = ROOT / "outputs/n20/live_traj_cache"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    n_rec = 0
    gt_cache = {}
    for p in sorted(SRC.glob("trajectories_*.jsonl")):
        seq = p.stem.replace("trajectories_", "")
        if seq not in gt_cache:
            gt_cache[seq] = load_gt(seq)
        gt = gt_cache[seq]
        out = DST / f"{seq}.jsonl"
        with p.open(encoding="utf-8") as f, out.open(
                "w", encoding="utf-8") as fo:
            for line in f:
                r = json.loads(line)
                for tr in r["trajectories"]:
                    frames = []
                    for k, v in sorted(tr["frames"].items(),
                                       key=lambda kv: int(kv[0])):
                        corr = 0
                        g = gt.get(int(k))
                        if v is not None and g is not None and \
                                r["gid"] in g.gt_ids:
                            corr = int(iou(
                                v, g.boxes[g.gt_ids.index(r["gid"])]) >= 0.5)
                        frames.append({"frame": int(k), "box": v,
                                       "correct": corr})
                    if not frames:
                        continue
                    start_box = frames[0]["box"]
                    is_correct = 0
                    g = gt.get(r["frame"])
                    if start_box is not None and g is not None and \
                            r["gid"] in g.gt_ids:
                        gbox = g.boxes[g.gt_ids.index(r["gid"])]
                        is_correct = int(
                            iou(start_box, gbox) >= 0.5)
                    fo.write(json.dumps({
                        "sequence": r["sequence"],
                        "frame": r["frame"],
                        "gid": r["gid"],
                        "candidate_rank": tr["rank"],
                        "is_correct": is_correct,
                        "start_box": start_box,
                        "runtime_s": 0.0,
                        "traj_len": len(frames),
                        "frames": frames,
                    }, ensure_ascii=False) + "\n")
                    n_rec += 1
        print(f"{seq}: {n_rec} hypotheses so far", flush=True)
    print(f"CONVERT_DONE total_hypotheses={n_rec}", flush=True)


if __name__ == "__main__":
    main()
