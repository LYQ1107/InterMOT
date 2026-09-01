#!/usr/bin/env python
"""N20.5B/6: generate real SAM3 shadow tracklets for ALL top-K candidates.

For each recovery attempt (from a real no-commit FULL_LOOP distribution),
replay the learned Writer memory to obtain the query embedding, rank the
GFN gallery at the attempt frame, and run one isolated SAM3 shadow for each
top-K candidate (H_max frames). GT is only used offline to label
is_correct. No public ID / memory mutation.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from n19_writer_features import feature_names, to_feature_vec  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import iou  # noqa: E402
from train_n19_writer import WriterMLP  # noqa: E402

DT = Path("/path/to/dancetrack")
N19 = ROOT / "outputs/n19"
N20 = ROOT / "outputs/n20"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--attempts-csv", required=True)
    ap.add_argument("--out-tag", default="all_candidates")
    ap.add_argument("--out-dir", default="full_shadow_cache_cal10")
    ap.add_argument("--events-jsonl", default="")
    ap.add_argument("--limit-attempts", type=int, default=0)
    ap.add_argument("--writer-threshold", type=float, default=0.95)
    ap.add_argument("--memory-k", type=int, default=2)
    ap.add_argument(
        "--official-frame-fetch",
        action="store_true",
        help="N25-R repair: recover chronological short-window outputs from the official cached-frame API after propagation.",
    )
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    rows = list(csv.DictReader(Path(args.attempts_csv).open(
        newline="", encoding="utf-8")))
    rows = [r for r in rows if r.get("target_present") == "1"]
    rows.sort(key=lambda r: (r["sequence"], int(r["frame"]), int(r["gid"])))
    rows = rows[args.shard:: args.nshards]
    if args.limit_attempts:
        rows = rows[: args.limit_attempts]
    print(f"PLAN shard={args.shard} attempts={len(rows)} k={args.k} "
          f"horizon={args.horizon}", flush=True)

    # ---- learned writer + memory replay inputs
    wcfg = json.loads((N19 / "models/writer_v0/writer_config.json").read_text())
    writer = WriterMLP(len(feature_names()), hidden=wcfg["hidden"])
    writer.load_state_dict(torch.load(
        N19 / "models/writer_v0/writer_v0.pt", map_location="cpu"))
    writer.eval()
    mean = np.asarray(wcfg["scaler_mean"], dtype=np.float32)
    std = np.asarray(wcfg["scaler_std"], dtype=np.float32)

    gt_cache = {}
    z_cache = {}

    def get_gt(seq):
        if seq not in gt_cache:
            gt_cache[seq] = load_gt(seq)
        return gt_cache[seq]

    def get_z(seq):
        if seq not in z_cache:
            z = np.load(CACHE / f"{seq}.npz")
            qz = np.load(CACHE / f"{seq}_queries.npz")
            emb = z["emb"].astype(np.float32)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            qemb = qz["qemb"].astype(np.float32)
            qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-8)
            rz = np.load(N20 / "gfn_cache_r0" / f"{seq}.npz")
            z_cache[seq] = {
                "frames": z["frames"], "offsets": z["offsets"],
                "boxes": z["boxes"], "scores": z["scores"], "emb": emb,
                "r0g": rz["r0g"], "r0q": rz["r0q"],
                "qgids": [int(g) for g in qz["gids"]],
                "qemb": qemb,
            }
            z.close(); qz.close(); rz.close()
        return z_cache[seq]

    # ---- SAM3 backend (same wiring as N19 reactivate)
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
        if args.official_frame_fetch:
            # SAM3.1's 15-frame hot-start and 16-frame batched postprocessor can
            # delay short-window stream delivery.  The official fetch API reads
            # the same cached masks in deterministic frame order.  This opt-in
            # repair preserves the historical N20 default path and artifacts.
            fetched = {}
            prev_fetch = np.asarray(box, dtype=float).copy()
            state = backend._predictor._all_inference_states[
                backend._session_id]["state"]
            stop = min(f + args.horizon, nf - 1)
            for ff in range(f, stop + 1):
                try:
                    _, outputs = model.fetch_and_process_single_frame_results(
                        state, ff)
                    cands = parse_raw_outputs(
                        {"outputs": outputs}, frame_size=(iw, ih))
                    cand_boxes = [np.asarray(b, dtype=float) for _, b in cands]
                    delivered = _best_delivery(prev_fetch, cand_boxes)
                except (KeyError, RuntimeError, IndexError):
                    delivered = None
                if delivered is not None:
                    prev_fetch = delivered.copy()
                fetched[ff] = delivered
            records = fetched
        try:
            backend._predictor.handle_request(dict(
                type="reset_session", session_id=backend._session_id))
            torch.cuda.empty_cache()
        except Exception:
            pass
        return records

    out_dir = N20 / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_tag}_k{args.k}_s{args.shard}.jsonl"
    done_keys = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                r0 = json.loads(line)
                done_keys.add((r0["sequence"], int(r0["frame"]),
                               int(r0["gid"]), int(r0["candidate_rank"])))
        print(f"RESUME skip={len(done_keys)}", flush=True)
    started = time.time()
    with out_path.open("a" if done_keys else "w", encoding="utf-8") as fout:
        for n, r in enumerate(rows):
            seq, f0, gid = r["sequence"], int(r["frame"]), int(r["gid"])
            z = get_z(seq)
            gt = get_gt(seq)
            qidx = {g: i for i, g in enumerate(z["qgids"])}
            qi = qidx.get(gid)
            if qi is None:
                continue
            qe_h = z["qemb"][qi]
            # --- replay learned memory up to the attempt frame (causal)
            first_app = None
            for ff in sorted(gt):
                if gid in gt[ff].gt_ids:
                    first_app = ff
                    break
            if first_app is None:
                continue
            qe_r0 = z["r0q"][qi]
            slots = [(first_app, qe_h.copy(), qe_r0.copy())]
            # writer replay needs delivered events; use dump-only events if
            # available, otherwise fall back to Human Root only.
            ev_path = Path(args.events_jsonl) if args.events_jsonl else \
                N20 / "full_loop_oracle_shadow" / "events_dump_only.jsonl"
            hist = []
            if ev_path.exists():
                for line in ev_path.open(encoding="utf-8"):
                    if line.strip():
                        e = json.loads(line)
                        if e["sequence"] == seq and e["gid"] == gid:
                            hist.append(e)
            hist.sort(key=lambda e: int(e["frame"]))
            prev_box = None
            for e in hist:
                ff = int(e["frame"])
                if ff >= f0:
                    break
                if not e.get("delivered") or e.get("delivered_box") is None:
                    prev_box = None
                    continue
                box = np.asarray(e["delivered_box"], dtype=float)
                o = int(np.searchsorted(z["frames"], ff))
                lo = int(z["offsets"][o - 1]) if o > 0 else 0
                hi = int(z["offsets"][o])
                if hi == lo:
                    prev_box = box
                    continue
                ious = np.asarray([iou(b, box) for b in z["boxes"][lo:hi]])
                bi = int(np.argmax(ious))
                if ious[bi] < 0.5:
                    prev_box = box
                    continue
                ge = z["emb"][lo + bi]
                r0_e = z["r0g"][lo + bi]
                src = e.get("source", "")
                dscore = e.get("delivery_score")
                if src in ("p0_tid", "p0") and dscore is not None and \
                        float(dscore) >= 0.5 and prev_box is not None and \
                        iou(box, prev_box) >= 0.5:
                    pass
                feats = {
                    "gfn_sim_human_root": float(ge @ qe_h),
                    "r0_sim_human_root": float(r0_e @ qe_r0),
                    "gfn_sim_oracle_last": "",
                    "gfn_sim_oracle_max": "",
                    "r0_sim_oracle_max": "",
                    "gfn_sim_heur_last": "",
                    "gfn_sim_heur_max": "",
                    "gfn_margin_h": "",
                    "det_score": dscore if dscore is not None else 0.0,
                    "box_area": float((box[2] - box[0]) *
                                      (box[3] - box[1])),
                    "temporal_iou": float(iou(box, prev_box))
                    if prev_box is not None else 0.0,
                    "center_delta": "",
                    "consecutive_delivered": 0,
                    "missing_streak": 0,
                    "crowd": hi - lo,
                    "overlap_max": "",
                    "nearest_det_distance": "",
                    "heur_memory_age": "",
                    "oracle_memory_age": "",
                    "candidate_age": ff - first_app,
                    "slots_oracle_count": min(len(slots), 2),
                    "slots_heur_count": 0,
                    "source": src,
                }
                if len(slots) > 1:
                    gsims = [s[1] @ ge for s in slots[-args.memory_k:]]
                    rsims = [s[2] @ r0_e for s in slots[-args.memory_k:]]
                    feats["gfn_sim_oracle_last"] = float(gsims[-1])
                    feats["gfn_sim_oracle_max"] = float(max(gsims))
                    feats["r0_sim_oracle_max"] = float(max(rsims))
                    feats["oracle_memory_age"] = ff - slots[-1][0]
                sims_all = z["emb"][lo:hi] @ qe_h
                if len(sims_all) > 1:
                    order = np.argsort(-sims_all)
                    feats["gfn_margin_h"] = float(
                        sims_all[order[0]] - sims_all[order[1]])
                x = to_feature_vec(feats)
                x = (x - mean) / std
                with torch.inference_mode():
                    p = float(torch.sigmoid(
                        writer(torch.from_numpy(x[None]))).item())
                if p >= args.writer_threshold:
                    slots.append((ff, ge.copy(), r0_e.copy()))
                    if len(slots) > args.memory_k:
                        slots.pop(0)
                prev_box = box
            # --- rank gallery by max learned-memory similarity
            o = int(np.searchsorted(z["frames"], f0))
            lo = int(z["offsets"][o - 1]) if o > 0 else 0
            hi = int(z["offsets"][o])
            if hi == lo:
                continue
            G = z["emb"][lo:hi]
            sims = np.maximum.reduce([G @ s[1] for s in slots[-args.memory_k:]])
            order = np.argsort(-sims)
            gf = gt.get(f0)
            tgt = None
            if gf is not None and gid in gf.gt_ids:
                tgt = np.asarray(gf.boxes[gf.gt_ids.index(gid)], dtype=float)
            cand_boxes = []
            for rank in range(min(args.k, len(order))):
                idx = order[rank]
                box = z["boxes"][lo + idx].astype(float).copy()
                cand_boxes.append((rank + 1, box))
            for rank, box in cand_boxes:
                key = (seq, f0, gid, rank)
                if key in done_keys:
                    continue
                t0 = time.time()
                traj = shadow_track(seq, f0, box)
                dt = time.time() - t0
                if traj is None:
                    traj = {}
                is_correct = 0
                if tgt is not None:
                    is_correct = int(iou(box, tgt) >= 0.5)
                per_frame = []
                for ff in sorted(traj):
                    b = traj[ff]
                    if b is None:
                        per_frame.append({"frame": ff, "box": None,
                                          "iou": None, "correct": None})
                        continue
                    b = np.asarray(b, dtype=float)
                    g2 = gt.get(ff)
                    ti = None
                    if g2 is not None and gid in g2.gt_ids:
                        ti = iou(b, np.asarray(
                            g2.boxes[g2.gt_ids.index(gid)], dtype=float))
                    per_frame.append({"frame": ff,
                                      "box": [round(float(v), 2) for v in b],
                                      "iou": None if ti is None
                                      else round(ti, 4),
                                      "correct": None if ti is None
                                      else int(ti >= 0.5)})
                fout.write(json.dumps({
                    "sequence": seq, "frame": f0, "gid": gid,
                    "candidate_rank": rank, "is_correct": is_correct,
                    "start_box": [round(float(v), 2) for v in box],
                    "runtime_s": round(dt, 2),
                    "traj_len": len([x for x in per_frame
                                     if x["box"] is not None]),
                    "frames": per_frame,
                }, ensure_ascii=False) + "\n")
            if (n + 1) % 10 == 0 or n == len(rows) - 1:
                print(f"PROGRESS shard={args.shard} attempts={n + 1}/"
                      f"{len(rows)} elapsed={time.time() - started:.0f}s",
                      flush=True)
    print(f"DONE shard={args.shard} attempts={len(rows)} "
          f"runtime_s={time.time() - started:.0f}", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
