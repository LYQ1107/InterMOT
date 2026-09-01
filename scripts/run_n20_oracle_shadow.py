#!/usr/bin/env python
"""N20.2: Oracle Shadow Propagation.

For every real N19-learned FULL_LOOP recovery attempt whose correct target is
present in the GFN top-K candidate set, run one real SAM3 isolated shadow
tracklet from the correct candidate box (H_max frames forward). The shadow
never binds a public ID and never mutates identity memory; GT is only used
afterwards to label the tracklet (offline oracle).

Outputs one JSONL row per attempt (shard-split for parallelism):
  attempt fields + per-frame shadow trajectory + confirmation verdicts.
"""

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import iou  # noqa: E402

DT = Path("/path/to/dancetrack")
N19 = ROOT / "outputs/n19"
N20 = ROOT / "outputs/n20"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit-attempts", type=int, default=0)
    ap.add_argument("--out-tag", default="oracle_shadow")
    ap.add_argument("--attempts-csv", default="")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--reset-session", action="store_true",
                    help="reset the SAM3 session after every shadow to bound "
                         "feature-cache memory growth")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    avail = Path(args.attempts_csv) if args.attempts_csv else \
        N20 / "topk_recovery_availability.csv"
    if not avail.exists():
        raise SystemExit("topk_recovery_availability.csv missing; run "
                         "analyze_n20_topk_availability.py first")
    rows = []
    with avail.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rm = r.get("rank_mem")
            if r["target_present"] == "1" and rm not in (None, ""):
                if int(rm) <= args.k:
                    rows.append(r)
    rows.sort(key=lambda r: (r["sequence"], int(r["frame"]), int(r["gid"])))
    rows = rows[args.shard:: args.nshards]
    if args.limit_attempts:
        rows = rows[: args.limit_attempts]
    print(f"SHADOW_PLAN shard={args.shard} attempts={len(rows)} "
          f"k={args.k} horizon={args.horizon}", flush=True)

    # ---- SAM3 backend (identical to N19 reactivate wiring)
    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner, parse_raw_outputs)
    from sam3_intermot.detection_query.prompt_replay import (
        _best_delivery, invalidate_detector_prefetch,
        set_frame_geometric_prompt)

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    state_box = {"seq": None}

    def shadow_track(seq, f, box):
        if state_box["seq"] != seq:
            backend.start_video(str(DT / "train" / seq / "img1"))
            state_box["seq"] = seq
        iw, ih = backend._frame_w, backend._frame_h
        x1, y1, x2, y2 = box
        req = dict(type="add_prompt", session_id=backend._session_id,
                   frame_index=f, text="person",
                   bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw,
                                    (y2 - y1) / ih]],
                   bounding_box_labels=[1], clear_old_boxes=True)
        try:
            backend._predictor.handle_request(req)
        except Exception:
            return None
        state = backend._predictor._all_inference_states[
            backend._session_id]["state"]
        nf = state["num_frames"]
        prev = np.asarray(box, dtype=float).copy()
        records = {f: prev.copy()}
        if f + 1 < nf:
            set_frame_geometric_prompt(runner, f + 1, None)
        req2 = dict(type="propagate_in_video", session_id=backend._session_id,
                    propagation_direction="forward", start_frame_index=f,
                    max_frame_num_to_track=args.horizon)
        try:
            for response in backend._predictor.handle_stream_request(
                    request=req2):
                ff = int(response["frame_index"])
                cands = parse_raw_outputs(response, frame_size=(iw, ih))
                cand_boxes = [np.asarray(b, dtype=float) for _, b in cands]
                delivered = _best_delivery(prev, cand_boxes)
                if delivered is not None:
                    prev = delivered.copy()
                records[ff] = delivered
                if ff >= f + args.horizon:
                    break
                if ff + 1 < nf:
                    set_frame_geometric_prompt(runner, ff + 1, None)
                invalidate_detector_prefetch(runner, ff)
        except Exception:
            pass
        return records

    out_dir = N20 / "shadow_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_tag}_k{args.k}_s{args.shard}.jsonl"
    meta_path = out_dir / f"{args.out_tag}_k{args.k}_s{args.shard}.meta.json"
    meta = {"attempts_planned": len(rows), "k": args.k,
            "horizon": args.horizon, "gpu": args.gpu}
    done_keys = set()
    if args.skip_existing and out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r0 = json.loads(line)
            done_keys.add((r0["sequence"], int(r0["frame"]),
                           int(r0["gid"])))
        print(f"RESUME skip_existing={len(done_keys)}", flush=True)
    started = time.time()
    gt_cache = {}
    z_cache = {}

    def get_gt(seq):
        if seq not in gt_cache:
            gt_cache[seq] = load_gt(seq)
        return gt_cache[seq]

    def get_z(seq):
        if seq not in z_cache:
            z = np.load(CACHE / f"{seq}.npz")
            z_cache[seq] = {
                "frames": z["frames"], "offsets": z["offsets"],
                "boxes": z["boxes"]}
            z.close()
        return z_cache[seq]

    with out_path.open("a" if args.skip_existing and done_keys
                       else "w", encoding="utf-8") as fout:
        for n, r in enumerate(rows):
            seq, f0, gid = r["sequence"], int(r["frame"]), int(r["gid"])
            if (seq, f0, gid) in done_keys:
                continue
            gt = get_gt(seq)
            gf = gt.get(f0)
            if gf is None or gid not in gf.gt_ids:
                continue
            tgt = np.asarray(gf.boxes[gf.gt_ids.index(gid)], dtype=float)
            z = get_z(seq)
            frames, offsets, dets = z["frames"], z["offsets"], z["boxes"]
            o = int(np.searchsorted(frames, f0))
            lo = int(offsets[o - 1]) if o > 0 else 0
            hi = int(offsets[o])
            if hi <= lo:
                continue
            ious = np.asarray([iou(b, tgt) for b in dets[lo:hi]])
            bi = int(np.argmax(ious))
            if ious[bi] < 0.5:
                continue
            box = dets[lo + bi].astype(float).copy()
            t0 = time.time()
            traj = shadow_track(seq, f0, box)
            if args.reset_session:
                try:
                    backend._predictor.handle_request(dict(
                        type="reset_session",
                        session_id=backend._session_id))
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            dt = time.time() - t0
            if traj is None:
                traj = {}
            per_frame = []
            for ff in sorted(traj):
                b = traj[ff]
                if b is None:
                    per_frame.append({"frame": ff, "box": None,
                                      "iou": None, "correct": None})
                    continue
                b = np.asarray(b, dtype=float)
                g = gt.get(ff)
                ti = None
                if g is not None and gid in g.gt_ids:
                    ti = iou(b, np.asarray(
                        g.boxes[g.gt_ids.index(gid)], dtype=float))
                per_frame.append({"frame": ff,
                                  "box": [round(float(v), 2) for v in b],
                                  "iou": None if ti is None else round(ti, 4),
                                  "correct": None if ti is None
                                  else int(ti >= 0.5)})
            row = {
                "sequence": seq, "frame": f0, "gid": gid,
                "rank_mem": int(r["rank_mem"]),
                "start_box": [round(float(v), 2) for v in box],
                "runtime_s": round(dt, 2),
                "traj_len": len([x for x in per_frame if x["box"] is not None]),
                "frames": per_frame,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (n + 1) % 25 == 0 or n == len(rows) - 1:
                print(f"SHADOW_PROGRESS shard={args.shard} {n + 1}/{len(rows)} "
                      f"elapsed={time.time() - started:.0f}s", flush=True)
    meta["runtime_s"] = round(time.time() - started, 1)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"SHADOW_DONE shard={args.shard} rows={len(rows)} "
          f"runtime_s={meta['runtime_s']}", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
