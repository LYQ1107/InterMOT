#!/usr/bin/env python
"""N18 FULL_LOOP_V0 on calibration sequences.

Wires the pure loop core with the real GPU components:
  - GFN recovery (official ConvNeXt-B, H_i crop query)
  - deployed logistic verifier (outputs/n18/models/verifier_v0.joblib)
  - SAM3 reactivation in an isolated session (no other identities touched)
"""

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import (  # noqa: E402
    LoopConfig, P0Row, aggregate_metrics, iou, run_full_loop)

DT = Path("/path/to/dancetrack")
OUT = ROOT / "outputs/n18"
P0 = ROOT / "outputs/n9/p0_train"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
CAL10 = json.loads(
    (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]["calibration10"]


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
        if len(parts) < 7:
            continue
        f0 = int(float(parts[0])) - 1
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]),
                      float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        out[f0].append(P0Row(tid=int(parts[1]),
                             box=np.asarray([x, y, x + w, y + h], dtype=float),
                             score=float(parts[6])))
    return out


def crop_query(img, box, margin=0.2):
    W, H = img.size
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0.0, x1 - margin * w)
    y1 = max(0.0, y1 - margin * h)
    x2 = min(float(W), x2 + margin * w)
    y2 = min(float(H), y2 + margin * h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--split", default="",
                    choices=["", "cal10", "train30"])
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-recovery", action="store_true")
    ap.add_argument("--no-reactivation", action="store_true")
    ap.add_argument("--fresh-anchor", action="store_true")
    ap.add_argument("--oracle-anchor", action="store_true")
    ap.add_argument("--oracle-verifier", action="store_true")
    ap.add_argument("--anchor-score", type=float, default=0.5)
    ap.add_argument("--anchor-continuity", type=float, default=0.5)
    ap.add_argument("--verified-anchor", action="store_true")
    ap.add_argument("--memory-health-threshold", type=float, default=0.5)
    ap.add_argument("--memory-matched-iou", type=float, default=0.3)
    ap.add_argument("--two-frame", action="store_true")
    ap.add_argument("--accept-threshold", type=float, default=0.4)
    ap.add_argument("--confirm-threshold", type=float, default=0.3)
    ap.add_argument("--confirm-iou", type=float, default=0.5)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    frozen = json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]
    if args.seqs:
        seqs = args.seqs.split(",")
    elif args.split == "cal10":
        seqs = frozen["calibration10"]
    elif args.split == "train30":
        seqs = frozen["train30"]
    else:
        seqs = CAL10
    seqs = sorted(seqs)[args.shard:: args.nshards]
    if args.oracle_anchor:
        anchor_policy = "oracle"
    elif args.verified_anchor:
        anchor_policy = "verified"
    elif args.fresh_anchor:
        anchor_policy = "trusted"
    else:
        anchor_policy = "first"
    cfg = LoopConfig(reactivation_horizon=args.horizon,
                     enable_recovery=not args.no_recovery,
                     enable_reactivation=not args.no_reactivation,
                     anchor_policy=anchor_policy,
                     anchor_score=args.anchor_score,
                     anchor_continuity_iou=args.anchor_continuity,
                     memory_health_threshold=args.memory_health_threshold,
                     memory_matched_iou=args.memory_matched_iou,
                     use_two_frame=args.two_frame,
                     accept_threshold=args.accept_threshold,
                     confirm_threshold=args.confirm_threshold,
                     confirm_iou=args.confirm_iou)
    clf = None
    feats = []
    gfn = None
    oracle_mode = args.oracle_verifier
    if cfg.enable_recovery and not oracle_mode:
        bundle = joblib.load(OUT / "models/verifier_v0.joblib")
        clf = bundle["model"]
        feats = bundle["features"]
    if cfg.enable_recovery:
        gfn, _, _, _, _ = load_model(f"cuda:{args.gpu}")

    # ---- SAM3 reactivation: isolated session, state reused per sequence
    runner = None
    backend = None
    if cfg.enable_reactivation:
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

    def reactivate(seq, f, box):
        if not cfg.enable_reactivation:
            return {}
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
            return {}
        state = backend._predictor._all_inference_states[
            backend._session_id]["state"]
        nf = state["num_frames"]
        prev = np.asarray(box, dtype=float).copy()
        records = {f: prev.copy()}
        if f + 1 < nf:
            set_frame_geometric_prompt(runner, f + 1, None)
        req2 = dict(type="propagate_in_video", session_id=backend._session_id,
                    propagation_direction="forward", start_frame_index=f,
                    max_frame_num_to_track=None)
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

    # ---- GFN recovery + verifier features
    img_cache = OrderedDict()

    def frame_img(seq, f):
        key = (seq, f)
        if key not in img_cache:
            p = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
            img_cache[key] = Image.open(p).convert("RGB") \
                if p.exists() else None
            if len(img_cache) > 128:
                img_cache.popitem(last=False)
        return img_cache[key]

    gal_cache = {}
    hq_cache = {}
    hfirst_cache = {}

    def gallery(seq, f):
        key = (seq, f)
        if key not in gal_cache:
            gimg = frame_img(seq, f)
            if gimg is None:
                gal_cache[key] = None
            else:
                with torch.inference_mode():
                    out = gfn([F.to_tensor(gimg).cuda()], None,
                              inference_mode="det")[0]
                boxes = out["det_boxes"].float().cpu().numpy()
                scores = out["det_scores"].float().cpu().numpy()
                embs = out["det_emb"].float()
                if boxes.ndim == 1:
                    boxes = boxes.reshape(1, -1)
                    scores = np.atleast_1d(scores)
                    embs = embs.reshape(1, -1)
                if len(boxes) == 0:
                    gal_cache[key] = (boxes, scores, None)
                else:
                    ge = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
                    gal_cache[key] = (boxes, scores, ge)
        return gal_cache[key]

    def h_query(seq, gid):
        key = (seq, gid)
        if key not in hq_cache:
            hq_cache[key] = None
            gt = load_gt(seq)
            for f0 in sorted(gt):
                gf = gt[f0]
                if gid in gf.gt_ids:
                    box = np.asarray(
                        gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                    img = frame_img(seq, f0)
                    if img is None:
                        break
                    qcrop = crop_query(img, box)
                    with torch.inference_mode():
                        qout = gfn([F.to_tensor(qcrop).cuda()], None,
                                   inference_mode="det")[0]
                    if qout["det_emb"].shape[0] == 0:
                        break
                    qi = int(torch.argmax(qout["det_scores"]).item())
                    qe = qout["det_emb"].float()[qi].reshape(-1)
                    hq_cache[key] = qe / (qe.norm() + 1e-8)
                    hfirst_cache[key] = (f0, box)
                    break
        return hq_cache[key]

    def health_fn(seq, f, gid, box):
        gal = gallery(seq, f)
        if gal is None:
            return None
        boxes, scores, ge = gal
        if len(boxes) == 0 or ge is None:
            return None
        ious = np.asarray([iou(b, box) for b in boxes], dtype=float)
        best = int(np.argmax(ious))
        if ious[best] < cfg.memory_matched_iou:
            return None
        qe = h_query(seq, gid)
        if qe is None:
            return None
        sims = (ge @ qe).float().cpu().numpy()
        order = np.argsort(-sims)
        s1 = float(sims[order[0]])
        s2 = float(sims[order[1]]) if len(order) > 1 else 0.0
        rec = {"gfn_top1_sim": s1, "gfn_margin": s1 - s2,
               "gfn_top1_score": float(scores[best]), "n_dets": len(boxes)}
        return verify(rec)

    def recover(seq, f, anchor_box, anchor_frame, gid=None):
        state_seq["seq"] = seq
        state_seq["frame"] = f
        if gid is not None:
            state_seq["gid"] = gid
        if seq not in gt_cache:
            gt_cache[seq] = load_gt(seq)
        if not cfg.enable_recovery:
            return None
        gal = gallery(seq, f)
        if gal is None:
            return None
        boxes, scores, ge = gal
        aimg = frame_img(seq, anchor_frame)
        if aimg is None or len(boxes) == 0 or ge is None:
            return None
        qcrop = crop_query(aimg, anchor_box)
        with torch.inference_mode():
            qout = gfn([F.to_tensor(qcrop).cuda()], None,
                       inference_mode="det")[0]
        if qout["det_emb"].shape[0] == 0:
            qe = torch.zeros(ge.shape[1], device=ge.device, dtype=torch.float32)
        else:
            qi = int(torch.argmax(qout["det_scores"]).item())
            qe = qout["det_emb"].float()[qi].reshape(-1)
        qe = qe / (qe.norm() + 1e-8)
        sims = (ge @ qe).float().cpu().numpy()
        order = np.argsort(-sims)
        s1, s2 = float(sims[order[0]]), float(sims[order[1]]) \
            if len(order) > 1 else 0.0
        candidates = []
        for rank, idx in enumerate(order[:3]):
            si = float(sims[idx])
            margin = si - (s2 if rank == 0 else s1)
            candidates.append({
                "box": boxes[idx],
                "gfn_top1_sim": si,
                "gfn_margin": margin,
                "gfn_top1_score": float(scores[idx]),
                "n_dets": len(boxes),
            })
        rec = {
            "box": boxes[order[0]],
            "candidates": candidates,
            "gfn_top1_sim": s1,
            "gfn_margin": s1 - s2,
            "gfn_top1_score": float(scores[order[0]]),
            "n_dets": len(boxes),
        }
        return rec

    def verify(rec):
        if oracle_mode:
            # offline oracle accept: candidate box is GT-correct at its frame
            box = np.asarray(rec["box"], dtype=float)
            gf = gt_cache[state_seq["seq"]].get(state_seq["frame"])
            gid = state_seq["gid"]
            if gf is not None and gid in gf.gt_ids:
                tgt = np.asarray(gf.boxes[gf.gt_ids.index(gid)],
                                 dtype=float)
                return float(iou(box, tgt) >= 0.5)
            return 0.0
        x = np.asarray([[rec[k] for k in feats]], dtype=float)
        return float(clf.predict_proba(x)[0, 1])

    # ---- run
    gt_cache = {}
    state_seq = {"seq": None, "frame": -1, "gid": -1}
    for seq in seqs:
        gt = load_gt(seq)
        p0 = load_p0(seq)
        num_frames = max(gt.keys()) + 1 if gt else 0
        if args.limit_frames:
            num_frames = min(num_frames, args.limit_frames)
        t0 = time.time()
        result = run_full_loop(seq, gt, p0, num_frames, cfg,
                               recover, verify, reactivate, health_fn)
        elapsed = time.time() - t0
        metrics = aggregate_metrics(seq, result, cfg)
        metrics["runtime_s"] = round(elapsed, 1)
        mode = args.out_tag or ("human" if args.no_recovery else
                                ("gfn" if args.no_reactivation else "full"))
        tag = f"_{mode}_s{args.shard}" if args.nshards > 1 else f"_{mode}"
        with (OUT / f"full_loop_v0_events{tag}.jsonl").open(
                "a", encoding="utf-8") as f:
            for e in result["trace"]:
                e["sequence"] = seq
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with (OUT / f"reactivation_transactions{tag}.jsonl").open(
                "a", encoding="utf-8") as f:
            for t in result["transactions"]:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with (OUT / f"full_loop_v0_metrics{tag}.csv").open(
                "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    if runner is not None:
        runner.close()
    print("LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
