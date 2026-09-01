#!/usr/bin/env python
"""N19.11a: Verifier training dataset under the learned memory.

Replays the causal deliveries, maintains the learned K-slot memory, and for
every recovery attempt records the top candidate's GFN/R0/memory features.
Label = candidate GT-correct (offline). This is the input for the N19
verifier retrain (train30 -> fit, cal10 -> calibrate).
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
from n19_writer_features import feature_names, to_feature_vec  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402
from train_n19_writer import WriterMLP  # noqa: E402

OUT = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"


def head_embed(model, f4, f5):
    with torch.inference_mode():
        emb, _ = model.roi_heads.embedding_head(
            {"feat_res4": f4, "feat_res5": f5})
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="cal10")
    ap.add_argument("--writer",
                    default=str(ROOT / "outputs/n19/models/writer_v0/writer_v0.pt"))
    ap.add_argument("--writer-config",
                    default=str(ROOT / "outputs/n19/models/writer_v0/writer_config.json"))
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--memory-k", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    frozen = json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]
    seqs = frozen["calibration10"] if args.split == "cal10" \
        else frozen["train30"]
    ev_prefix = "full_loop_v0_events_oracle_n19" if args.split == "cal10" \
        else "full_loop_v0_events_train30_n19"
    tx_prefix = "reactivation_transactions_full" if args.split == "cal10" \
        else "reactivation_transactions_train30_n19"

    wcfg = json.loads(Path(args.writer_config).read_text())
    writer = WriterMLP(len(feature_names()), hidden=wcfg["hidden"])
    writer.load_state_dict(torch.load(args.writer, map_location="cpu"))
    writer.eval()
    mean = np.asarray(wcfg["scaler_mean"], dtype=np.float32)
    std = np.asarray(wcfg["scaler_std"], dtype=np.float32)
    gfn, _, _, _, _ = load_model(args.device)
    gfn.eval()
    if Path(ROOT / "outputs/n18/route_c/models/r0_best.pt").exists():
        gfn.roi_heads.embedding_head.load_state_dict(torch.load(
            ROOT / "outputs/n18/route_c/models/r0_best.pt",
            map_location="cpu"))
        gfn.roi_heads.embedding_head.eval()

    ev = []
    for p in sorted(OUT.glob(f"{ev_prefix}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                ev.append(json.loads(line))
    tx = []
    for p in sorted(OUT.glob(f"{tx_prefix}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                tx.append(json.loads(line))
    ev_by_seq = defaultdict(list)
    for e in ev:
        ev_by_seq[e["sequence"]].append(e)
    tx_by_seq = defaultdict(list)
    for t in tx:
        tx_by_seq[t["sequence"]].append(t)

    rows = []
    for seq in sorted(seqs):
        gt = load_gt(seq)
        z = np.load(CACHE / f"{seq}.npz")
        qz = np.load(CACHE / f"{seq}_queries.npz")
        frames = z["frames"]
        offsets = z["offsets"]
        dets = z["boxes"]
        scores = z["scores"]
        emb = z["emb"].astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        qgids = [int(g) for g in qz["gids"]]
        qemb = qz["qemb"].astype(np.float32)
        qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-8)
        qidx = {g: i for i, g in enumerate(qgids)}
        f4 = torch.from_numpy(z["feat4"].astype(np.float32))
        f5 = torch.from_numpy(z["feat5"].astype(np.float32))
        qf4 = torch.from_numpy(qz["qfeat4"].astype(np.float32))
        qf5 = torch.from_numpy(qz["qfeat5"].astype(np.float32))
        with torch.inference_mode():
            r0g = head_embed(gfn, f4, f5).numpy()
            r0q = head_embed(gfn, qf4, qf5).numpy()
        z.close()
        qz.close()
        hist = defaultdict(list)
        for e in ev_by_seq[seq]:
            hist[e["gid"]].append(e)
        first_app = {}
        for f0 in sorted(gt):
            for g0 in gt[f0].gt_ids:
                first_app.setdefault(g0, f0)

        def det_idx(f0, box):
            o = int(np.searchsorted(frames, f0))
            lo = int(offsets[o - 1]) if o > 0 else 0
            hi = int(offsets[o])
            if hi == lo:
                return None
            ious = np.asarray([iou(b, box) for b in dets[lo:hi]])
            b = int(np.argmax(ious))
            return (lo + b) if ious[b] >= 0.5 else None

        for gid, evs in hist.items():
            qi = qidx.get(gid)
            if qi is None:
                continue
            qe_h = qemb[qi]
            qe_r0 = r0q[qi]
            slots = [(first_app.get(gid, 0), qe_h.copy(), qe_r0.copy())]
            heur = []
            prev_box = None
            for e in sorted(evs, key=lambda x: int(x["frame"])):
                f0 = int(e["frame"])
                if not e.get("delivered") or e.get("delivered_box") is None:
                    prev_box = None
                    continue
                box = np.asarray(e["delivered_box"], dtype=float)
                gi = det_idx(f0, box)
                if gi is None:
                    prev_box = box
                    continue
                gfn_e = emb[gi]
                r0_e = r0g[gi]
                src = e.get("source", "")
                dscore = e.get("delivery_score")
                if src in ("p0_tid", "p0") and dscore is not None and \
                        float(dscore) >= 0.5 and prev_box is not None and \
                        iou(box, prev_box) >= 0.5:
                    heur.append((f0, gfn_e.copy()))
                    if len(heur) > 8:
                        heur.pop(0)
                feats = {
                    "gfn_sim_human_root": float(gfn_e @ qe_h),
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
                    "crowd": 0,
                    "overlap_max": "",
                    "nearest_det_distance": "",
                    "heur_memory_age": "",
                    "oracle_memory_age": "",
                    "candidate_age": f0 - first_app.get(gid, f0),
                    "slots_oracle_count": min(len(slots), 2),
                    "slots_heur_count": len(heur),
                    "source": src,
                }
                if slots:
                    mem = slots[-args.memory_k:]
                    gsims = [s[1] @ gfn_e for s in mem]
                    rsims = [s[2] @ r0_e for s in mem]
                    feats["gfn_sim_oracle_last"] = float(gsims[-1])
                    feats["gfn_sim_oracle_max"] = float(max(gsims))
                    feats["r0_sim_oracle_max"] = float(max(rsims))
                    feats["oracle_memory_age"] = f0 - slots[-1][0]
                if heur:
                    hs = np.stack([s[1] for s in heur[-8:]]) @ gfn_e
                    feats["gfn_sim_heur_last"] = float(hs[-1])
                    feats["gfn_sim_heur_max"] = float(hs.max())
                    feats["heur_memory_age"] = f0 - heur[-1][0]
                o2 = int(np.searchsorted(frames, f0))
                lo2 = int(offsets[o2 - 1]) if o2 > 0 else 0
                hi2 = int(offsets[o2])
                if hi2 > lo2:
                    sims_all = emb[lo2:hi2] @ qe_h
                    order = np.argsort(-sims_all)
                    if len(order) > 1:
                        feats["gfn_margin_h"] = float(
                            sims_all[order[0]] - sims_all[order[1]])
                    feats["crowd"] = hi2 - lo2
                if prev_box is not None:
                    c0 = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    c1 = ((prev_box[0] + prev_box[2]) / 2,
                          (prev_box[1] + prev_box[3]) / 2)
                    feats["center_delta"] = float(
                        np.hypot(c0[0] - c1[0], c0[1] - c1[1]))
                x = to_feature_vec(feats)
                x = (x - mean) / std
                with torch.inference_mode():
                    p = float(torch.sigmoid(
                        writer(torch.from_numpy(x[None]))).item())
                if p >= args.threshold:
                    slots.append((f0, gfn_e.copy(), r0_e.copy()))
                    if len(slots) > args.memory_k:
                        slots.pop(0)
                prev_box = box
            # attempts
            for t in tx_by_seq[seq]:
                if int(t["gid"]) != gid:
                    continue
                f0 = int(t["frame"])
                if not slots:
                    continue
                o = int(np.searchsorted(frames, f0))
                lo = int(offsets[o - 1]) if o > 0 else 0
                hi = int(offsets[o])
                if hi == lo:
                    continue
                G = emb[lo:hi]
                mem = slots[-args.memory_k:]
                sims = np.maximum.reduce([G @ s[1] for s in mem])
                order = np.argsort(-sims)
                top = int(order[0])
                s1 = float(sims[top])
                s2 = float(sims[order[1]]) if len(order) > 1 else 0.0
                cand_box = dets[lo + top]
                gf = gt.get(f0)
                label = 0
                if gf is not None and gid in gf.gt_ids:
                    tgt = np.asarray(
                        gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                    label = int(iou(cand_box, tgt) >= 0.5)
                n_agree = int(sum(float(s[1] @ G[top]) >= 0.5
                                  for s in mem))
                rows.append({
                    "sequence": seq, "frame": f0, "gid": gid,
                    "gfn_top1_sim": round(s1, 4),
                    "gfn_margin": round(s1 - s2, 4),
                    "gfn_top1_score": float(scores[lo + top]),
                    "n_dets": hi - lo,
                    "memory_age": f0 - slots[-1][0],
                    "n_agree_slots": n_agree,
                    "r0_top1_sim": round(float(
                        max(s[2] @ r0g[lo + top] for s in mem)), 4),
                    "top1_correct": label,
                })
        print(f"verifier-data {seq} rows={len(rows)}", flush=True)
    out = N19 / f"verifier_dataset_{args.split}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"VERIFIER_DATASET_DONE rows={len(rows)} file={out}", flush=True)


if __name__ == "__main__":
    main()
