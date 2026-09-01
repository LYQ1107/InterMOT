#!/usr/bin/env python
"""N20.8-10: build the K+1 set-level shadow dataset.

For every recovery attempt, every top-K hypothesis contributes one sample
per evidence step (1..H). Per-step features are causal (<= t+h); labels use
offline GT only. Competition features are computed across the K hypotheses
of the same attempt at the same step.
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

FEATURE_COLS = [
    "attempt", "sequence", "frame", "gid", "candidate_rank", "evidence_step",
    "gfn_sim_human_root", "r0_sim_human_root", "gfn_sim_mem_last",
    "gfn_sim_mem_max", "r0_sim_mem_last", "r0_sim_mem_max",
    "mem_age", "n_mem_slots", "temp_sim_prev", "temp_sim_first",
    "box_area", "area_change", "center_delta", "velocity", "temporal_iou",
    "consecutive_delivered", "shadow_delivered", "n_dets", "gfn_margin_h",
    "candidate_age", "memory_fresh",
    "rank_mem", "initial_correct", "init_rank_correct",
    "comp_delivered_ratio", "comp_mean_gfn_sim", "comp_max_gfn_sim",
    "comp_overlap_max", "comp_sim_margin",
    "label_correct", "label_wrong", "safe_to_commit",
]


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="full_shadow_cache_cal10")
    ap.add_argument("--events-jsonl", default="")
    ap.add_argument("--evidence-steps", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="shadow_kplus1_cal10.csv")
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

    # load all-candidate cache
    hyps = defaultdict(list)
    for p in sorted((N20 / args.cache_dir).glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                hyps[(r["sequence"], int(r["frame"]), int(r["gid"]))].append(r)
    attempts = sorted(hyps)
    if args.limit_attempts:
        attempts = attempts[: args.limit_attempts]
    print(f"attempts={len(attempts)} hypotheses={sum(len(v) for v in hyps.values())}",
          flush=True)

    ev_rows = []
    if args.events_jsonl:
        for line in Path(args.events_jsonl).open(encoding="utf-8"):
            if line.strip():
                ev_rows.append(json.loads(line))
    ev_by = defaultdict(list)
    for e in ev_rows:
        ev_by[(e["sequence"], e["gid"])].append(e)
    for k in ev_by:
        ev_by[k].sort(key=lambda e: int(e["frame"]))

    z_cache = {}
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

    out_rows = []
    for att in attempts:
        seq, f0, gid = att
        z = get_z(seq)
        gt = get_gt(seq)
        qidx = {g: i for i, g in enumerate(z["qgids"])}
        qi = qidx.get(gid)
        if qi is None:
            continue
        qe_h = z["qemb"][qi]
        qe_r0 = z["r0q"][qi]
        first_app = None
        for ff in sorted(gt):
            if gid in gt[ff].gt_ids:
                first_app = ff
                break
        if first_app is None:
            continue
        # memory replay (causal, <= attempt)
        slots = [(first_app, qe_h.copy(), qe_r0.copy())]
        prev_box = None
        for e in ev_by.get((seq, gid), []):
            ff = int(e["frame"])
            if ff >= f0:
                break
            if not e.get("delivered") or e.get("delivered_box") is None:
                prev_box = None
                continue
            box = np.asarray(e["delivered_box"], dtype=float)
            de = det_emb(seq, ff, box)
            src = e.get("source", "")
            dscore = e.get("delivery_score")
            if de is not None:
                ge, r0e = de
                feats = {
                    "gfn_sim_human_root": float(ge @ qe_h),
                    "r0_sim_human_root": float(r0e @ qe_r0),
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
                    "candidate_age": ff - first_app,
                    "slots_oracle_count": min(len(slots), 2),
                    "slots_heur_count": 0,
                    "source": src,
                }
                if len(slots) > 1:
                    gsims = [s[1] @ ge for s in slots[-args.memory_k:]]
                    rsims = [s[2] @ r0e for s in slots[-args.memory_k:]]
                    feats["gfn_sim_oracle_last"] = float(gsims[-1])
                    feats["gfn_sim_oracle_max"] = float(max(gsims))
                    feats["r0_sim_oracle_max"] = float(max(rsims))
                    feats["oracle_memory_age"] = ff - slots[-1][0]
                o2 = int(np.searchsorted(z["frames"], ff))
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
                    slots.append((ff, ge.copy(), r0e.copy()))
                    if len(slots) > args.memory_k:
                        slots.pop(0)
            prev_box = box
        mem = slots[-args.memory_k:] if slots else []
        # per-candidate per-step features
        cand_rows = {}
        for h in hyps[att]:
            rank = int(h["candidate_rank"])
            fm = {x["frame"]: x for x in h["frames"]}
            cand_rows[rank] = {"row": h, "frames": fm}
        o0 = int(np.searchsorted(z["frames"], f0))
        lo0 = int(z["offsets"][o0 - 1]) if o0 > 0 else 0
        hi0 = int(z["offsets"][o0])
        margin_h = ""
        if hi0 > lo0:
            sims0 = z["emb"][lo0:hi0] @ qe_h
            ord0 = np.argsort(-sims0)
            if len(ord0) > 1:
                margin_h = float(sims0[ord0[0]] - sims0[ord0[1]])
        for rank, cand in sorted(cand_rows.items()):
            fm = cand["frames"]
            start_box = np.asarray(cand["row"]["start_box"], dtype=float)
            start_emb = det_emb(seq, f0, start_box)
            prev_emb = None
            prev_box2 = None
            prev_center = None
            prev_delta = None
            consecutive = 0
            for step in range(1, args.evidence_steps + 1):
                fe = f0 + step
                x = fm.get(fe)
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
                # competition features from the other hypotheses at this step
                other_delivered = []
                other_sims = []
                other_boxes = []
                for r2, cd2 in cand_rows.items():
                    if r2 == rank:
                        continue
                    x2 = cd2["frames"].get(fe)
                    if x2 is None or x2["box"] is None:
                        other_delivered.append(0)
                        continue
                    other_delivered.append(1)
                    other_boxes.append(np.asarray(x2["box"], dtype=float))
                    de2 = det_emb(seq, fe, other_boxes[-1])
                    if de2 is not None and mem:
                        other_sims.append(max(s[1] @ de2[0] for s in mem))
                comp_overlap_max = ""
                if box is not None and other_boxes:
                    comp_overlap_max = float(max(
                        iou(np.asarray(box, dtype=float), b)
                        for b in other_boxes))
                comp_delivered_ratio = float(np.mean(other_delivered)) \
                    if other_delivered else 1.0
                comp_mean_gfn_sim = float(np.mean(other_sims)) \
                    if other_sims else ""
                comp_max_gfn_sim = float(np.max(other_sims)) \
                    if other_sims else ""
                comp_sim_margin = ""
                if de is not None and mem and other_sims:
                    my_sim = max(s[1] @ de[0] for s in mem)
                    comp_sim_margin = float(my_sim - max(other_sims))
                # per-step features
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
                        cdist = float(np.hypot(center[0] - prev_center[0],
                                               center[1] - prev_center[1]))
                        center_delta = cdist
                        if prev_delta is not None:
                            velocity = float(cdist - prev_delta)
                        prev_delta = cdist
                    prev_center = center
                    if prev_for_iou is not None:
                        area_change = float(area / max(
                            1e-6, (prev_for_iou[2] - prev_for_iou[0]) *
                            (prev_for_iou[3] - prev_for_iou[1])) - 1.0)
                        temporal_iou = float(iou(b, prev_for_iou))
                    prev_box2 = b
                out_rows.append({
                    "attempt": f"{seq}:{f0}:{gid}",
                    "sequence": seq, "frame": f0, "gid": gid,
                    "candidate_rank": rank, "evidence_step": step,
                    "gfn_sim_human_root": float(de[0] @ qe_h) if de else "",
                    "r0_sim_human_root": float(de[1] @ qe_r0) if de else "",
                    "gfn_sim_mem_last": sim_mem_last,
                    "gfn_sim_mem_max": sim_mem_max,
                    "r0_sim_mem_last": rsim_mem_last,
                    "r0_sim_mem_max": rsim_mem_max,
                    "mem_age": f0 - mem[-1][0] if mem else -1,
                    "n_mem_slots": len(mem),
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
                    "candidate_age": f0 - first_app,
                    "memory_fresh": f0 - slots[-1][0] if slots else -1,
                    "rank_mem": float(cand["row"]["candidate_rank"]),
                    "initial_correct": int(cand["row"]["is_correct"]),
                    "init_rank_correct": int(any(
                        int(cd2["row"]["is_correct"]) == 1
                        for cd2 in cand_rows.values())),
                    "comp_delivered_ratio": round(comp_delivered_ratio, 4),
                    "comp_mean_gfn_sim": "" if comp_mean_gfn_sim == "" else
                    round(float(comp_mean_gfn_sim), 4),
                    "comp_max_gfn_sim": "" if comp_max_gfn_sim == "" else
                    round(float(comp_max_gfn_sim), 4),
                    "comp_overlap_max": "" if comp_overlap_max == "" else
                    round(float(comp_overlap_max), 4),
                    "comp_sim_margin": "" if comp_sim_margin == "" else
                    round(float(comp_sim_margin), 4),
                    "label_correct": correct,
                    "label_wrong": int(correct == 0 and delivered == 1),
                    "safe_to_commit": int(correct),
                })
                prev_emb = de[0] if de is not None else prev_emb
        if len(out_rows) % 2000 == 0 and out_rows:
            print(f"rows={len(out_rows)}", flush=True)
    out_path = N20 / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"KPLUS1_DATASET_DONE rows={len(out_rows)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
