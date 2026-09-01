#!/usr/bin/env python
"""N19.5: Memory-write candidate dataset from the real causal tracker
distribution.

One row per delivered observation (ACTIVE/UNCERTAIN/REACTIVATED frames with a
delivered box). Features are causal (<= t). GT correctness is stored only as
an offline label; it is never read at inference time. R0 embeddings are
computed from the cached feat4/feat5 (R0 head-only, frozen backbone).
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

from eval_route_c_r0 import iou  # noqa: E402
from gfn_recovery_model import load_model  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402

OUT = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"
R0_CKPT = ROOT / "outputs/n18/route_c/models/r0_best.pt"


def head_embed(model, f4, f5):
    with torch.inference_mode():
        emb, _ = model.roi_heads.embedding_head(
            {"feat_res4": f4, "feat_res5": f5})
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)


def load_events(prefix):
    rows = []
    for p in sorted(OUT.glob(f"{prefix}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(a):
    a = np.asarray(a, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-prefix",
                    default="full_loop_v0_events_oracle_n19")
    ap.add_argument("--out-tag", default="cal10")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    N19.mkdir(parents=True, exist_ok=True)

    device = args.device
    model, _, _, _, _ = load_model(device)
    model.eval()
    if R0_CKPT.exists():
        model.roi_heads.embedding_head.load_state_dict(
            torch.load(R0_CKPT, map_location="cpu"))
        model.roi_heads.embedding_head.eval()
    else:
        print("WARN: r0_best.pt missing; R0 features will be empty",
              flush=True)

    events = load_events(args.events_prefix)
    by_seq = defaultdict(list)
    for e in events:
        by_seq[e["sequence"]].append(e)

    out_csv = N19 / f"write_dataset_{args.out_tag}.csv"
    fields = [
        "sequence", "gid", "public_id", "frame", "state", "source",
        "gt_present", "iou_gt", "correct", "safe_write",
        "gfn_sim_human_root", "r0_sim_human_root",
        "gfn_sim_oracle_last", "gfn_sim_oracle_max",
        "r0_sim_oracle_max",
        "gfn_sim_heur_last", "gfn_sim_heur_max",
        "gfn_margin_h", "det_score", "box_area", "temporal_iou",
        "center_delta", "consecutive_delivered", "missing_streak",
        "crowd", "overlap_max", "nearest_det_distance",
        "oracle_memory_age", "heur_memory_age", "candidate_age",
        "slots_oracle_count", "slots_heur_count",
        "future_utility_10", "future_utility_30", "future_utility_60",
        "future_utility_120", "future_utility_240", "future_utility_480",
    ]
    total_rows = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for seq in sorted(by_seq):
            gt = load_gt(seq)
            z = np.load(CACHE / f"{seq}.npz")
            qz = np.load(CACHE / f"{seq}_queries.npz")
            frames = z["frames"]
            offsets = z["offsets"]
            boxes_d = z["boxes"]
            scores_d = z["scores"]
            emb_frozen = z["emb"].astype(np.float32)
            f4 = torch.from_numpy(z["feat4"].astype(np.float32))
            f5 = torch.from_numpy(z["feat5"].astype(np.float32))
            qemb = qz["qemb"].astype(np.float32)
            qf4 = torch.from_numpy(qz["qfeat4"].astype(np.float32))
            qf5 = torch.from_numpy(qz["qfeat5"].astype(np.float32))
            gids = [int(g) for g in qz["gids"]]
            z.close()
            qz.close()
            emb_frozen = emb_frozen / (
                np.linalg.norm(emb_frozen, axis=1, keepdims=True) + 1e-8)
            qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-8)
            # R0 embeddings for all gallery dets and human-root queries
            r0_gallery = None
            r0_q = None
            if R0_CKPT.exists():
                with torch.inference_mode():
                    r0_gallery = head_embed(model, f4, f5).cpu().numpy()
                    r0_q = head_embed(model, qf4, qf5).cpu().numpy()
            qidx = {g: i for i, g in enumerate(gids)}
            first_app = {}
            for f0 in sorted(gt):
                for g0 in gt[f0].gt_ids:
                    first_app.setdefault(g0, f0)
            # per-identity causal history from trace
            hist = defaultdict(list)  # gid -> list of event dicts
            for e in by_seq[seq]:
                hist[e["gid"]].append(e)
            for gid, evs in hist.items():
                qi = qidx.get(gid)
                if qi is None:
                    continue
                qe_h = qemb[qi]
                qe_r0 = r0_q[qi] if r0_q is not None else None
                oracle_slots = [
                    (first_app.get(gid, 0), qe_h.copy(),
                     qe_r0.copy() if qe_r0 is not None else None)
                ]  # human root seeds the memory
                heur_slots = []    # (frame, emb)
                last_delivered = None
                consecutive = 0
                missing = 0
                for e in sorted(evs, key=lambda x: int(x["frame"])):
                    f0 = int(e["frame"])
                    if not e.get("delivered") or e.get("delivered_box") is None:
                        missing += 1
                        consecutive = 0
                        last_delivered = None
                        continue
                    if e.get("delivered_box") is not None:
                        box = np.asarray(e["delivered_box"], dtype=float)
                    else:
                        # old V0 events lack delivered_box; best-effort GT box
                        gf = gt.get(f0)
                        if gf is None or gid not in gf.gt_ids:
                            last_delivered = None
                            consecutive = 0
                            missing = 0
                            continue
                        box = np.asarray(
                            gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                    o = int(np.searchsorted(frames, f0))
                    lo = int(offsets[o - 1]) if o > 0 else 0
                    hi = int(offsets[o])
                    if hi == lo:
                        last_delivered = box
                        continue
                    dets = boxes_d[lo:hi]
                    ious = np.asarray([iou(b, box) for b in dets])
                    best = int(np.argmax(ious))
                    if ious[best] < 0.5:
                        # delivered box has no matching cached detection
                        last_delivered = box
                        consecutive += 1
                        missing = 0
                        continue
                    gi = lo + best
                    gfn_e = emb_frozen[gi]
                    r0_e = r0_gallery[gi] if r0_gallery is not None else None
                    # GT label at this frame
                    gf = gt.get(f0)
                    iou_gt = None
                    correct = 0
                    if gf is not None and gid in gf.gt_ids:
                        tgt = np.asarray(
                            gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                        iou_gt = float(iou(box, tgt))
                        correct = int(iou_gt >= 0.5)
                    row = {
                        "sequence": seq, "gid": gid,
                        "public_id": e.get("public_id"), "frame": f0,
                        "state": e.get("state"), "source": e.get("source"),
                        "gt_present": e.get("gt_present"),
                        "iou_gt": "" if iou_gt is None else round(iou_gt, 4),
                        "correct": correct,
                        "safe_write": correct,
                        "gfn_sim_human_root": round(float(gfn_e @ qe_h), 4),
                        "r0_sim_human_root":
                            "" if r0_e is None or qe_r0 is None
                            else round(float(r0_e @ qe_r0), 4),
                        "gfn_sim_oracle_last": "",
                        "gfn_sim_oracle_max": "",
                        "r0_sim_oracle_max": "",
                        "gfn_sim_heur_last": "",
                        "gfn_sim_heur_max": "",
                        "gfn_margin_h": "",
                        "det_score": e.get("delivery_score"),
                        "box_area": round(float(
                            (box[2] - box[0]) * (box[3] - box[1])), 1),
                        "temporal_iou": e.get("delivery_iou_prev"),
                        "center_delta": "",
                        "consecutive_delivered": consecutive,
                        "missing_streak": missing,
                        "crowd": hi - lo,
                        "overlap_max": "",
                        "nearest_det_distance": "",
                        "oracle_memory_age": "",
                        "heur_memory_age": "",
                        "candidate_age": f0 - first_app.get(gid, f0),
                        "slots_oracle_count": min(len(oracle_slots), 2),
                        "slots_heur_count": len(heur_slots),
                    }
                    # margin of gallery top1/top2 vs human root
                    if hi - lo > 0:
                        sims_all = emb_frozen[lo:hi] @ qe_h
                        if len(sims_all) > 1:
                            order = np.argsort(-sims_all)
                            row["gfn_margin_h"] = round(float(
                                sims_all[order[0]] - sims_all[order[1]]), 4)
                    if last_delivered is not None:
                        c0 = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        c1 = ((last_delivered[0] + last_delivered[2]) / 2,
                              (last_delivered[1] + last_delivered[3]) / 2)
                        row["center_delta"] = round(
                            float(np.hypot(c0[0] - c1[0], c0[1] - c1[1])), 1)
                    # det overlap / nearest det distance (observable)
                    if hi - lo > 1:
                        other = np.delete(dets, best, axis=0)
                        ov = [iou(b, box) for b in other]
                        row["overlap_max"] = round(float(max(ov)), 4)
                        cc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        dc = np.stack([
                            ((o2[0] + o2[2]) / 2, (o2[1] + o2[3]) / 2)
                            for o2 in other])
                        dd = np.hypot(dc[:, 0] - cc[0], dc[:, 1] - cc[1])
                        row["nearest_det_distance"] = round(
                            float(np.min(dd)), 1)
                    # memory sims
                    if oracle_slots:
                        embs = np.stack([s[1] for s in oracle_slots[-2:]])
                        sims_o = embs @ gfn_e
                        row["gfn_sim_oracle_last"] = round(
                            float(sims_o[-1]), 4)
                        row["gfn_sim_oracle_max"] = round(
                            float(sims_o.max()), 4)
                        row["oracle_memory_age"] = f0 - oracle_slots[-1][0]
                        if qe_r0 is not None:
                            r0embs = np.stack([s[2] for s in oracle_slots[-2:]])
                            row["r0_sim_oracle_max"] = round(
                                float((r0embs @ r0_e).max()), 4)
                    if heur_slots:
                        embs = np.stack([s[1] for s in heur_slots[-8:]])
                        sims_h = embs @ gfn_e
                        row["gfn_sim_heur_last"] = round(
                            float(sims_h[-1]), 4)
                        row["gfn_sim_heur_max"] = round(
                            float(sims_h.max()), 4)
                        row["heur_memory_age"] = f0 - heur_slots[-1][0]
                    for k in ["future_utility_10", "future_utility_30",
                              "future_utility_60", "future_utility_120",
                              "future_utility_240", "future_utility_480"]:
                        row[k] = ""
                    w.writerow(row)
                    total_rows += 1
                    # update causal memory state (after the observation)
                    if correct:
                        oracle_slots.append((f0, gfn_e, r0_e))
                    if e.get("source") in ("p0_tid", "p0") and \
                            e.get("delivery_score") is not None and \
                            float(e["delivery_score"]) >= 0.5 and \
                            e.get("delivery_iou_prev") is not None and \
                            float(e["delivery_iou_prev"]) >= 0.5:
                        heur_slots.append((f0, gfn_e))
                    last_delivered = box
                    consecutive += 1
                    missing = 0
            print(f"write-dataset {seq} rows={len(hist)}", flush=True)
    print(f"WRITE_DATASET_DONE rows={total_rows} file={out_csv}",
          flush=True)


if __name__ == "__main__":
    main()
