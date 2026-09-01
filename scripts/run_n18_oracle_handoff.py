#!/usr/bin/env python
"""N18.6 Oracle Recovery -> SAM3 handoff (recovered box at frame f).

For each GFN-recoverable event, seed SAM3 with the oracle-selected GFN box at
frame f, propagate, and measure same-id retention at f+1/3/5/10/30/60 vs A0
(frozen P0 output). Video readers are reused within a sequence to avoid
re-indexing per event.
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

OUT = ROOT / "outputs/n18"
DT = Path("/path/to/dancetrack")
P0 = ROOT / "outputs/n9/p0_train"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
HORIZONS = (1, 3, 5, 10, 30, 60)


def iou(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def load_gt(seq):
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    return DanceTrackDataset(str(DT), sequences=[], split="train").load_gt(seq)


def load_p0(seq):
    out = defaultdict(list)
    p = P0 / f"{seq}.txt"
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        f0 = int(float(parts[0])) - 1
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]),
                      float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        out[f0].append(np.asarray([x, y, x + w, y + h], dtype=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=0)
    args = ap.parse_args()
    import torch
    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner, parse_raw_outputs)
    from sam3_intermot.detection_query.prompt_replay import (
        _best_delivery, invalidate_detector_prefetch,
        set_frame_geometric_prompt)

    events = []
    seen = set()
    for p in sorted(OUT.glob("gfn_recovery_boxes_s*.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["sequence"], r["f"], r["gid"])
                if key in seen:
                    continue
                seen.add(key)
                events.append(r)
    src = {}
    with (OUT.parent / "n17/cal_episodes.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src[(r["sequence"], r["t"], r["gid"], r["f"])] = r
    events = [r for r in events
              if src.get((r["sequence"], r["t"], r["gid"], r["f"]),
                         {}).get("generic_miss") == "1"]
    # Resume: skip events already recovered from previous shard logs.
    done = set()
    for p in sorted(OUT.glob("oracle_handoff_s*.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["sequence"], int(r["t"]), int(r["f"]),
                          int(r["gid"])))
    events = [r for r in events
              if (r["sequence"], int(r["t"]), int(r["f"]),
                  int(r["gid"])) not in done]
    events.sort(key=lambda r: (r["sequence"], int(r["f"])))
    # Shard at sequence granularity so each process re-indexes each video once.
    seqs = sorted({r["sequence"] for r in events})
    my_seqs = set(seqs[args.shard:: args.nshards])
    events = [r for r in events if r["sequence"] in my_seqs]
    if args.max_events:
        events = events[: args.max_events]
    print(f"events={len(events)}", flush=True)

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    current_seq = None
    gt_cache = {}
    p0_cache = {}
    rows = []
    for ev in events:
        seq = ev["sequence"]
        if seq != current_seq:
            backend.start_video(str(DT / "train" / seq / "img1"))
            current_seq = seq
        if seq not in gt_cache:
            gt_cache[seq] = load_gt(seq)
            p0_cache[seq] = load_p0(seq)
        t = int(ev["f"])
        gid = int(ev["gid"])
        box = np.asarray(json.loads(ev["recovered_box"]), dtype=float)
        iw, ih = backend._frame_w, backend._frame_h
        x1, y1, x2, y2 = box
        req = dict(
            type="add_prompt",
            session_id=backend._session_id,
            frame_index=t,
            text="person",
            bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw,
                             (y2 - y1) / ih]],
            bounding_box_labels=[1],
            clear_old_boxes=True,
        )
        backend._predictor.handle_request(req)
        state = backend._predictor._all_inference_states[
            backend._session_id]["state"]
        num_frames = state["num_frames"]
        prev = box.copy()
        records = {t: prev.copy()}
        if t + 1 < num_frames:
            set_frame_geometric_prompt(runner, t + 1, None)
        req2 = dict(
            type="propagate_in_video",
            session_id=backend._session_id,
            propagation_direction="forward",
            start_frame_index=t,
            max_frame_num_to_track=None,
        )
        for response in backend._predictor.handle_stream_request(request=req2):
            f = int(response["frame_index"])
            cands = parse_raw_outputs(response, frame_size=(iw, ih))
            cand_boxes = [np.asarray(b, dtype=float) for _, b in cands]
            delivered = _best_delivery(prev, cand_boxes)
            if delivered is not None:
                prev = delivered.copy()
            records[f] = delivered
            if f >= t + 60:
                break
            if f + 1 < num_frames:
                set_frame_geometric_prompt(runner, f + 1, None)
            invalidate_detector_prefetch(runner, f)
        row = {"sequence": seq, "t": ev["t"], "f": t, "gid": gid,
               "recovered_iou": ev["recovered_iou"]}
        gt = gt_cache[seq]
        p0 = p0_cache[seq]
        for h in HORIZONS:
            f = t + h
            entry = gt.get(f)
            if entry is None or gid not in entry.gt_ids:
                row[f"gt_present_{h}"] = 0
                row[f"a0_{h}"] = ""
                row[f"react_{h}"] = ""
                continue
            target = np.asarray(entry.boxes[entry.gt_ids.index(gid)],
                                dtype=float)
            row[f"gt_present_{h}"] = 1
            a0_hit = any(iou(b, target) >= 0.5 for b in p0.get(f, []))
            row[f"a0_{h}"] = int(a0_hit)
            rb = records.get(f)
            row[f"react_{h}"] = int(rb is not None and iou(rb, target) >= 0.5)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    out_path = OUT / f"oracle_handoff{tag}.csv"
    existing = {}
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[(r["sequence"], int(r["t"]), int(r["f"]),
                          int(r["gid"]))] = r
    merged = list(existing.values()) + rows
    if not merged:
        print(f"shard{args.shard} nothing to do", flush=True)
        runner.close()
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    agg = []
    for h in HORIZONS:
        sub = [r for r in merged if r[f"gt_present_{h}"] == 1]
        a0 = np.mean([r[f"a0_{h}"] for r in sub]) if sub else float("nan")
        rc = np.mean([r[f"react_{h}"] for r in sub]) if sub else float("nan")
        agg.append({"horizon": h, "n": len(sub), "A0": round(float(a0), 4),
                    "react": round(float(rc), 4)})
    with (OUT / f"oracle_handoff_retention{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "n", "A0", "react"])
        w.writeheader()
        w.writerows(agg)
    print("AGG", json.dumps(agg), flush=True)
    runner.close()


if __name__ == "__main__":
    main()
