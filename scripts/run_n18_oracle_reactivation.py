#!/usr/bin/env python
"""N18.1 Oracle Reactivation / Handoff Benchmark.

Find real "SAM3 lost identity" events on calibration sequences (P0 AUTO has
no box matching GT at frame t while GT is present), then compare:
  A0:     keep AUTO (no recovery), measured from the frozen P0 output
  ONE-SHOT: correct only frame t's delivered output (no tracker re-init)
  ORACLE-REACTIVATION: add_prompt(GT box at t) -> SAM3 new/refreshed track
                       -> automatic propagation from t+1
GT is used only to define the event and seed the oracle box; t+1.. are
evaluated without any future GT.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "outputs/n18"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")
P0 = ROOT / "outputs/n9/p0_train"

CAL10 = json.loads((ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]["calibration10"]


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
    d = DanceTrackDataset(str(DT), sequences=[], split="train").load_gt(seq)
    return d


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
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        out[f0].append(np.asarray([x, y, x + w, y + h], dtype=float))
    return out


def find_events(seqs, max_per_seq=25, seed=18, horizons=(1, 3, 5, 10, 30, 60)):
    rng = np.random.default_rng(seed)
    events = []
    for seq in seqs:
        gt = load_gt(seq)
        p0 = load_p0(seq)
        frames = sorted(gt.keys())
        cands = []
        for t in frames:
            if t + max(horizons) > max(frames):
                continue
            for gid, box in zip(gt[t].gt_ids, gt[t].boxes):
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < 1200:
                    continue
                # AUTO lost at t: no P0 box matches GT
                if any(iou(b, box) >= 0.5 for b in p0.get(t, [])):
                    continue
                # sustained loss: also lost at t+1 and t+2 (AUTO did not
                # recover on its own immediately)
                sustained = True
                for dt in (1, 2):
                    f2 = t + dt
                    if f2 not in gt or gid not in gt[f2].gt_ids:
                        sustained = False
                        break
                    target2 = np.asarray(
                        gt[f2].boxes[gt[f2].gt_ids.index(gid)], dtype=float
                    )
                    if any(iou(b, target2) >= 0.5 for b in p0.get(f2, [])):
                        sustained = False
                        break
                if not sustained:
                    continue
                # enough GT in the future to measure retention
                future = sum(
                    1 for h in horizons
                    if t + h in gt and gid in gt[t + h].gt_ids
                )
                if future < len(horizons) - 2:
                    continue
                cands.append((t, gid, tuple(box)))
        rng.shuffle(cands)
        for t, gid, box in cands[:max_per_seq]:
            events.append({"sequence": seq, "t": t, "gid": gid,
                           "box": list(box)})
    return events


def run_reactivation(seq, t, gid, box, horizon, runner, backend, model):
    """add_prompt(GT box at t) + propagate; follow delivered trajectory."""
    from sam3_intermot.adaptation.cfa_backend_runner import parse_raw_outputs
    from sam3_intermot.detection_query.prompt_replay import (
        _best_delivery, invalidate_detector_prefetch, set_frame_geometric_prompt,
    )
    video = str(DT / "train" / seq / "img1")
    backend.start_video(video)
    model.use_batched_grounding = False
    iw, ih = backend._frame_w, backend._frame_h
    x1, y1, x2, y2 = box
    req = dict(
        type="add_prompt",
        session_id=backend._session_id,
        frame_index=t,
        text="person",
        bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]],
        bounding_box_labels=[1],
        clear_old_boxes=True,
    )
    backend._predictor.handle_request(req)
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"].clear()
    prev = np.asarray(box, dtype=float).copy()
    records = {t: prev.copy()}
    nf = t + 1
    set_frame_geometric_prompt(runner, nf, None)
    req = dict(
        type="propagate_in_video",
        session_id=backend._session_id,
        propagation_direction="forward",
        start_frame_index=t,
        max_frame_num_to_track=None,
    )
    for response in backend._predictor.handle_stream_request(request=req):
        f = int(response["frame_index"])
        cands = parse_raw_outputs(response, frame_size=(iw, ih))
        cand_boxes = [np.asarray(b, dtype=float) for _, b in cands]
        delivered = _best_delivery(prev, cand_boxes)
        if delivered is not None:
            prev = delivered.copy()
        records[f] = delivered
        if f >= t + horizon:
            break
        nf2 = f + 1
        set_frame_geometric_prompt(runner, nf2, None)
        invalidate_detector_prefetch(runner, f)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--max-per-seq", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    import torch
    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner

    seqs = args.seqs.split(",") if args.seqs else CAL10
    events = find_events(seqs, max_per_seq=args.max_per_seq)
    events = events[args.shard:: args.nshards]
    if args.max_events:
        events = events[: args.max_events]
    print(f"events={len(events)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for ev in events:
        seq, t, gid = ev["sequence"], ev["t"], ev["gid"]
        box = np.asarray(ev["box"], dtype=float)
        gt = load_gt(seq)
        p0 = load_p0(seq)
        try:
            rec = run_reactivation(seq, t, gid, box, args.horizon, runner, backend, model)
        finally:
            try:
                backend.close()
            except Exception:
                pass
        row = {"sequence": seq, "t": t, "gid": gid}
        for h in (1, 3, 5, 10, 30, 60):
            if h > args.horizon:
                continue
            f = t + h
            entry = gt.get(f)
            if entry is None or gid not in entry.gt_ids:
                row[f"gt_present_{h}"] = 0
                row[f"a0_{h}"] = ""
                row[f"react_{h}"] = ""
                continue
            target = np.asarray(entry.boxes[entry.gt_ids.index(gid)], dtype=float)
            row[f"gt_present_{h}"] = 1
            a0_hit = any(iou(b, target) >= 0.5 for b in p0.get(f, []))
            row[f"a0_{h}"] = int(a0_hit)
            rb = rec.get(f)
            row[f"react_{h}"] = int(rb is not None and iou(rb, target) >= 0.5)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    tag = f"_s{args.shard}" if args.nshards > 1 else ""
    with (OUT / f"oracle_reactivation{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / f"oracle_reactivation_events{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "t", "gid", "box"])
        w.writeheader()
        for ev in events:
            w.writerow(
                {"sequence": ev["sequence"], "t": ev["t"], "gid": ev["gid"],
                 "box": json.dumps(ev["box"])}
            )
    # aggregate retention
    agg = []
    for h in (1, 3, 5, 10, 30, 60):
        if h > args.horizon:
            continue
        sub = [r for r in rows if r[f"gt_present_{h}"] == 1]
        a0 = np.mean([r[f"a0_{h}"] for r in sub]) if sub else float("nan")
        rc = np.mean([r[f"react_{h}"] for r in sub]) if sub else float("nan")
        agg.append({"horizon": h, "n": len(sub), "A0_retention": a0,
                    "reactivation_retention": rc})
    with (OUT / f"reactivation_retention{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["horizon", "n", "A0_retention",
                                          "reactivation_retention"])
        w.writeheader()
        w.writerows(agg)
    print("AGG", json.dumps(agg, ensure_ascii=False), flush=True)
    runner.close()


if __name__ == "__main__":
    main()
