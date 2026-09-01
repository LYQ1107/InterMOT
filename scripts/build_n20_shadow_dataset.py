#!/usr/bin/env python
"""N20.5: build the shadow-tracklet verification dataset.

Each recovery attempt with a real SAM3 shadow tracklet produces one sample
per evidence frame (f0+1 .. f0+E). All features are causal (<= evidence
frame); the memory slots are frozen at the attempt frame. GT is used only
for the offline labels CORRECT_TARGET / WRONG_IDENTITY / SAFE_TO_COMMIT.
"""

import argparse
import csv
import json
import sys
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

N19 = ROOT / "outputs/n19"
N20 = ROOT / "outputs/n20"
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"
OUT = N20 / "features"

FEATURE_COLS = [
    "attempt", "sequence", "frame", "gid", "rank_mem", "evidence_step",
    "shadow_box", "gfn_sim_human_root", "r0_sim_human_root",
    "gfn_sim_mem_last", "gfn_sim_mem_max", "r0_sim_mem_last",
    "r0_sim_mem_max", "mem_age", "n_mem_slots",
    "temp_sim_prev", "temp_sim_first", "box_area", "area_change",
    "center_delta", "velocity", "temporal_iou", "consecutive_delivered",
    "shadow_delivered", "n_dets", "gfn_margin_h", "candidate_age",
    "memory_fresh", "label_correct", "label_wrong", "safe_to_commit",
    "future_correct_3", "future_correct_8",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-steps", type=int, default=8)
    ap.add_argument("--split", default="cal10", choices=["cal10", "train30"])
    ap.add_argument("--out", default="shadow_tracklets_cal10.csv")
    ap.add_argument("--writer-threshold", type=float, default=0.95)
    ap.add_argument("--memory-k", type=int, default=2)
    ap.add_argument("--limit-attempts", type=int, default=0)
    args = ap.parse_args()

    wcfg = json.loads((N19 / "models/writer_v0/writer_config.json").read_text())
    writer = WriterMLP(len(feature_names()), hidden=wcfg["hidden"])
    writer.load_state_dict(torch.load(
        N19 / "models/writer_v0/writer_v0.pt", map_location="cpu"))
    writer.eval()
    mean = np.asarray(wcfg["scaler_mean"], dtype=np.float32)
    std = np.asarray(wcfg["scaler_std"], dtype=np.float32)

    # ---- memory replay events (no-commit distribution)
    ev = []
    ep = OUT.parent / "full_loop_oracle_shadow" / "events_dump_only.jsonl"
    for line in ep.open(encoding="utf-8"):
        if line.strip():
            ev.append(json.loads(line))
    by_seq_ev = defaultdict(list)
    for e in ev:
        by_seq_ev[e["sequence"]].append(e)

    # ---- shadow cache
    shadow_rows = []
    for p in sorted((N20 / "shadow_cache").glob("oracle_shadow_k*_s*.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                shadow_rows.append(json.loads(line))
    if args.limit_attempts:
        shadow_rows = shadow_rows[: args.limit_attempts]
    print(f"shadow_rows={len(shadow_rows)}", flush=True)

    z_cache = {}
    q_cache = {}
    gt_cache = {}

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

    def det_emb(seq, f0, box):
        z = get_z(seq)
        o = int(np.searchsorted(z["frames"], f0))
        lo = int(z["offsets"][o - 1]) if o > 0 else 0
        hi = int(z["offsets"][o])
        if hi == lo:
            return None
        ious = np.asarray([iou(b, box) for b in z["boxes"][lo:hi]])
        bi = int(np.argmax(ious))
        if ious[bi] < 0.5:
            return None
        gi = lo + bi
        return z["emb"][gi], z["r0g"][gi]

    rows_out = []
    for r in shadow_rows:
        seq, f0, gid = r["sequence"], int(r["frame"]), int(r["gid"])
        z = get_z(seq)
        qidx = {g: i for i, g in enumerate(z["qgids"])}
        qi = qidx.get(gid)
        if qi is None:
            continue
        qe_h = z["qemb"][qi]
        qe_r0 = z["r0q"][qi]
        gt = get_gt(seq)
        first_app = None
        for ff in sorted(gt):
            if gid in gt[ff].gt_ids:
                first_app = (ff, np.asarray(
                    gt[ff].boxes[gt[ff].gt_ids.index(gid)], dtype=float))
                break
        if first_app is None:
            continue
        # human root embedding from the query cache
        # Human Root is the first memory slot (frozen initial authority)
        slots = [(first_app[0], qe_h.copy(), qe_r0.copy())]
        hist = sorted((e for e in by_seq_ev[seq] if e["gid"] == gid),
                      key=lambda x: int(x["frame"]))
        prev_box = None
        for e in hist:
            f = int(e["frame"])
            if f > f0:
                break
            if not e.get("delivered") or e.get("delivered_box") is None:
                prev_box = None
                continue
            box = np.asarray(e["delivered_box"], dtype=float)
            de = det_emb(seq, f, box)
            src = e.get("source", "")
            dscore = e.get("delivery_score")
            if de is not None and src in ("p0_tid", "p0") and \
                    dscore is not None and float(dscore) >= 0.5 and \
                    prev_box is not None and iou(box, prev_box) >= 0.5:
                pass  # heuristic slots not needed for writer features
            if de is not None:
                ge, re = de
                feats = {
                    "gfn_sim_human_root": float(ge @ qe_h),
                    "r0_sim_human_root": float(re @ qe_r0),
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
                    "crowd": 0,
                    "overlap_max": "",
                    "nearest_det_distance": "",
                    "heur_memory_age": "",
                    "oracle_memory_age": "",
                    "candidate_age": f - first_app[0],
                    "slots_oracle_count": min(len(slots), 2),
                    "slots_heur_count": 0,
                    "source": src,
                }
                if len(slots) > 1:
                    mem = slots[-args.memory_k:]
                    gsims = [s[1] @ ge for s in mem]
                    rsims = [s[2] @ re for s in mem]
                    feats["gfn_sim_oracle_last"] = float(gsims[-1])
                    feats["gfn_sim_oracle_max"] = float(max(gsims))
                    feats["r0_sim_oracle_max"] = float(max(rsims))
                    feats["oracle_memory_age"] = f - slots[-1][0]
                o2 = int(np.searchsorted(z["frames"], f))
                lo2 = int(z["offsets"][o2 - 1]) if o2 > 0 else 0
                hi2 = int(z["offsets"][o2])
                if hi2 > lo2:
                    sims_all = z["emb"][lo2:hi2] @ qe_h
                    order = np.argsort(-sims_all)
                    if len(order) > 1:
                        feats["gfn_margin_h"] = float(
                            sims_all[order[0]] - sims_all[order[1]])
                x = to_feature_vec(feats)
                x = (x - mean) / std
                with torch.inference_mode():
                    p = float(torch.sigmoid(
                        writer(torch.from_numpy(x[None]))).item())
                if p >= args.writer_threshold:
                    slots.append((f, ge.copy(), re.copy()))
                    if len(slots) > args.memory_k:
                        slots.pop(0)
            prev_box = box
        mem = slots[-args.memory_k:] if slots else []
        frames_map = {x["frame"]: x for x in r["frames"]}
        start_emb = det_emb(seq, f0, np.asarray(r["start_box"], dtype=float))
        # attempt-frame margin against the human root
        o0 = int(np.searchsorted(z["frames"], f0))
        lo0 = int(z["offsets"][o0 - 1]) if o0 > 0 else 0
        hi0 = int(z["offsets"][o0])
        margin_h = ""
        if hi0 > lo0:
            sims0 = z["emb"][lo0:hi0] @ qe_h
            order0 = np.argsort(-sims0)
            if len(order0) > 1:
                margin_h = float(sims0[order0[0]] - sims0[order0[1]])
        prev_emb = None
        prev_box2 = None
        prev_center = None
        prev_delta = None
        consecutive = 0
        for step in range(1, args.evidence_steps + 1):
            fe = f0 + step
            x = frames_map.get(fe)
            if x is None:
                continue
            box = x["box"]
            delivered = int(box is not None)
            de = det_emb(seq, fe, box) if box is not None else None
            gf = gt.get(fe)
            ti = None
            if gf is not None and gid in gf.gt_ids:
                ti = iou(np.asarray(box, dtype=float),
                         np.asarray(gf.boxes[gf.gt_ids.index(gid)],
                                    dtype=float)) if box is not None else 0.0
            correct = int(ti is not None and ti >= 0.5)
            if delivered:
                consecutive += 1
            else:
                consecutive = 0
            # future correctness (offline labels only)
            future3 = future8 = 0
            for hh, key in ((3, "future_correct_3"), (8, "future_correct_8")):
                hits = 0
                total = 0
                for ff in range(fe + 1, fe + hh + 1):
                    g2 = gt.get(ff)
                    if g2 is None or gid not in g2.gt_ids:
                        continue
                    total += 1
                    b2 = frames_map.get(ff)
                    if b2 and b2["box"] is not None:
                        if iou(np.asarray(b2["box"], dtype=float),
                               np.asarray(g2.boxes[g2.gt_ids.index(gid)],
                                          dtype=float)) >= 0.5:
                            hits += 1
                if key == "future_correct_3":
                    future3 = int(total > 0 and hits == total)
                else:
                    future8 = int(total > 0 and hits == total)
            # feature values
            sim_mem_last = sim_mem_max = rsim_mem_last = rsim_mem_max = ""
            if de is not None and mem:
                gsims = [s[1] @ de[0] for s in mem]
                rsims = [s[2] @ de[1] for s in mem]
                sim_mem_last = float(gsims[-1])
                sim_mem_max = float(max(gsims))
                rsim_mem_last = float(rsims[-1])
                rsim_mem_max = float(max(rsims))
            temp_sim_prev = temp_sim_first = ""
            if de is not None:
                if prev_emb is not None:
                    temp_sim_prev = float(de[0] @ prev_emb)
                if start_emb is not None:
                    temp_sim_first = float(de[0] @ start_emb[0])
            area = 0.0
            area_change = ""
            center_delta = 0.0
            velocity = ""
            temporal_iou = ""
            if box is not None:
                b = np.asarray(box, dtype=float)
                area = float((b[2] - b[0]) * (b[3] - b[1]))
                prev_for_iou = prev_box2
                center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                if prev_center is not None:
                    cd = float(np.hypot(center[0] - prev_center[0],
                                        center[1] - prev_center[1]))
                    center_delta = cd
                    if prev_delta is not None:
                        velocity = float(cd - prev_delta)
                    prev_delta = cd
                prev_center = center
                if prev_for_iou is not None:
                    area_change = float(area / max(
                        1e-6, (prev_for_iou[2] - prev_for_iou[0]) *
                        (prev_for_iou[3] - prev_for_iou[1])) - 1.0)
                    temporal_iou = float(iou(b, prev_for_iou))
                prev_box2 = b
            mem_age = f0 - mem[-1][0] if mem else -1
            rows_out.append({
                "attempt": f"{seq}:{f0}:{gid}",
                "sequence": seq, "frame": f0, "gid": gid,
                "rank_mem": r["rank_mem"], "evidence_step": step,
                "shadow_box": None if box is None else
                json.dumps([round(float(v), 2) for v in box]),
                "gfn_sim_human_root": float(de[0] @ qe_h) if de else "",
                "r0_sim_human_root": float(de[1] @ qe_r0) if de else "",
                "gfn_sim_mem_last": sim_mem_last,
                "gfn_sim_mem_max": sim_mem_max,
                "r0_sim_mem_last": rsim_mem_last,
                "r0_sim_mem_max": rsim_mem_max,
                "mem_age": mem_age, "n_mem_slots": len(mem),
                "temp_sim_prev": temp_sim_prev,
                "temp_sim_first": temp_sim_first,
                "box_area": round(area, 2),
                "area_change": "" if area_change == "" else
                round(float(area_change), 4),
                "center_delta": round(center_delta, 2),
                "velocity": "" if velocity == "" else round(float(velocity), 2),
                "temporal_iou": "" if temporal_iou == "" else
                round(float(temporal_iou), 4),
                "consecutive_delivered": consecutive,
                "shadow_delivered": delivered,
                "n_dets": hi0 - lo0,
                "gfn_margin_h": margin_h,
                "candidate_age": f0 - first_app[0],
                "memory_fresh": f0 - slots[-1][0] if slots else -1,
                "label_correct": correct,
                "label_wrong": int(correct == 0 and delivered == 1),
                "safe_to_commit": int(correct and future3),
                "future_correct_3": future3,
                "future_correct_8": future8,
            })
            prev_emb = de[0] if de is not None else prev_emb
        if len(rows_out) % 500 == 0 and rows_out:
            print(f"rows={len(rows_out)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / args.out
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"SHADOW_DATASET_DONE rows={len(rows_out)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
