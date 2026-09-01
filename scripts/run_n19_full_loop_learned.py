#!/usr/bin/env python
"""N19.12: FULL_LOOP with the learned memory-write policy (Writer V0).

Same causal loop as N18 V0, but the recovery query anchor is maintained by
the learned writer: at every delivered frame a causal feature vector is
scored and, above the calibrated threshold, the observation is written into
the identity's K-slot memory. Recovery uses the most recent slot. The
verifier stays the deployed N18 logistic verifier unless --oracle-verifier.
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
from n19_writer_features import feature_names, to_feature_vec  # noqa: E402
from run_n18_full_loop_v0 import load_gt, load_p0  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import (  # noqa: E402
    LoopConfig, aggregate_metrics, iou, run_full_loop)
from train_n19_writer import WriterMLP  # noqa: E402

DT = Path("/path/to/dancetrack")
OUT = ROOT / "outputs/n18"
P0 = ROOT / "outputs/n9/p0_train"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"


def head_embed(model, f4, f5):
    with torch.inference_mode():
        emb, _ = model.roi_heads.embedding_head(
            {"feat_res4": f4, "feat_res5": f5})
    return emb / (emb.norm(dim=1, keepdim=True) + 1e-8)


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
    ap.add_argument("--split", default="cal10")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--writer",
                    default=str(ROOT / "outputs/n19/models/writer_v0/writer_v0.pt"))
    ap.add_argument("--writer-config",
                    default=str(ROOT / "outputs/n19/models/writer_v0/writer_config.json"))
    ap.add_argument("--writer-threshold", type=float, default=None)
    ap.add_argument("--memory-k", type=int, default=2)
    ap.add_argument("--oracle-verifier", action="store_true")
    ap.add_argument("--verifier", default="")
    ap.add_argument("--verifier-threshold", type=float, default=None)
    ap.add_argument("--dump-verify-csv", default="")
    ap.add_argument("--two-frame", action="store_true")
    ap.add_argument("--accept-threshold", type=float, default=0.4)
    ap.add_argument("--confirm-threshold", type=float, default=0.3)
    ap.add_argument("--confirm-iou", type=float, default=0.5)
    ap.add_argument("--out-tag", default="learned_n19")
    ap.add_argument("--no-recovery", action="store_true")
    ap.add_argument("--no-reactivation", action="store_true")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    frozen = json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]
    if args.seqs:
        seqs = args.seqs.split(",")
    elif args.split == "train30":
        seqs = frozen["train30"]
    else:
        seqs = frozen["calibration10"]
    seqs = sorted(seqs)[args.shard:: args.nshards]

    cfg = LoopConfig(reactivation_horizon=args.horizon,
                     enable_recovery=not args.no_recovery,
                     enable_reactivation=not args.no_reactivation,
                     anchor_policy="learned",
                     anchor_score=(args.writer_threshold
                                   if args.writer_threshold is not None
                                   else 0.5),
                     memory_k=args.memory_k,
                     use_two_frame=args.two_frame,
                     accept_threshold=args.accept_threshold,
                     confirm_threshold=args.confirm_threshold,
                     confirm_iou=args.confirm_iou)

    # ---- writer model + scaler
    wcfg = json.loads(Path(args.writer_config).read_text())
    writer = WriterMLP(len(feature_names()), hidden=wcfg["hidden"])
    writer.load_state_dict(torch.load(args.writer, map_location="cpu"))
    writer.eval()
    writer.to(f"cuda:{args.gpu}")
    mean = np.asarray(wcfg["scaler_mean"], dtype=np.float32)
    std = np.asarray(wcfg["scaler_std"], dtype=np.float32)

    # ---- GFN + R0 head
    gfn, _, _, _, _ = load_model(f"cuda:{args.gpu}")
    gfn.eval()
    if Path(ROOT / "outputs/n18/route_c/models/r0_best.pt").exists():
        gfn.roi_heads.embedding_head.load_state_dict(torch.load(
            ROOT / "outputs/n18/route_c/models/r0_best.pt",
            map_location="cpu"))
        gfn.roi_heads.embedding_head.eval()

    # ---- SAM3 reactivation (isolated session per sequence)
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

    # ---- GFN recovery + deployed verifier (or oracle)
    verifier_path = Path(args.verifier) if args.verifier else \
        ROOT / "outputs/n19/models/verifier_n19.joblib"
    if not verifier_path.exists():
        verifier_path = OUT / "models/verifier_v0.joblib"
    verifier_bundle = joblib.load(verifier_path)
    clf = verifier_bundle["model"]
    vfeats = verifier_bundle["features"]
    cfg.verifier_threshold = float(
        verifier_bundle.get("threshold", cfg.verifier_threshold))
    if args.verifier_threshold is not None:
        cfg.verifier_threshold = args.verifier_threshold
    print("VERIFIER_LOADED", verifier_path,
          "thr", cfg.verifier_threshold, flush=True)
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
                    break
        return hq_cache[key]

    def verify(rec):
        if args.oracle_verifier:
            box = np.asarray(rec["box"], dtype=float)
            gf = gt_cache[state_seq["seq"]].get(state_seq["frame"])
            gid = state_seq["gid"]
            if gf is not None and gid in gf.gt_ids:
                tgt = np.asarray(gf.boxes[gf.gt_ids.index(gid)],
                                 dtype=float)
                return float(iou(box, tgt) >= 0.5)
            return 0.0
        x = {k: rec[k] for k in vfeats if k in rec}
        if any(k in vfeats for k in
               ("memory_age", "n_agree_slots", "r0_top1_sim")):
            seq = state_seq["seq"]
            gid = state_seq["gid"]
            f = state_seq["frame"]
            prepare_seq_cache(seq)
            z = cache_store[seq]
            o = int(np.searchsorted(z["frames"], f))
            lo = int(z["offsets"][o - 1]) if o > 0 else 0
            hi = int(z["offsets"][o])
            cand_g = cand_r0 = None
            if hi > lo:
                ious = np.asarray(
                    [iou(b, rec["box"]) for b in z["boxes"][lo:hi]])
                bi = int(np.argmax(ious))
                if ious[bi] >= 0.5:
                    cand_g = z["emb"][lo + bi]
                    cand_r0 = r0_store[seq][0][lo + bi]
            ms = mem_state[(seq, gid)]["slots"]
            if ms:
                x["memory_age"] = f - ms[-1][0]
            else:
                x["memory_age"] = 0.0
            if cand_g is not None and ms:
                x["n_agree_slots"] = float(sum(
                    1 for _, ge, _ in ms if float(ge @ cand_g) >= 0.5))
                if cand_r0 is not None:
                    x["r0_top1_sim"] = float(max(
                        re @ cand_r0 for _, _, re in ms))
            else:
                x["n_agree_slots"] = 0.0
                x["r0_top1_sim"] = 0.0
        arr = np.asarray([[x.get(k, 0.0) for k in vfeats]], dtype=float)
        p = float(clf.predict_proba(arr)[0, 1])
        if args.dump_verify_csv or len(verify_debug) < 30:
            verify_debug.append({"seq": state_seq["seq"],
                                 "frame": state_seq["frame"],
                                 "gid": state_seq["gid"],
                                 "box": [round(float(v), 2)
                                         for v in rec["box"]],
                                 "x": {k: round(float(x.get(k, 0.0)), 4)
                                       for k in vfeats},
                                 "p": round(p, 4)})
        return p

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
        qe = None
        for sf, ge_slot, _ in mem_state.get((seq, gid), {}).get(
                "slots", []):
            if sf == anchor_frame:
                qe = torch.from_numpy(ge_slot).float().to(
                    ge.device)
                break
        if qe is None:
            qcrop = crop_query(aimg, anchor_box)
            with torch.inference_mode():
                qout = gfn([F.to_tensor(qcrop).cuda()], None,
                           inference_mode="det")[0]
            if qout["det_emb"].shape[0] == 0:
                qe = torch.zeros(ge.shape[1], device=ge.device,
                                 dtype=torch.float32)
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

    # ---- learned writer: causal features from the frozen GFN cache
    cache_store = {}
    r0_store = {}
    q_store = {}
    write_stats = {"calls": 0, "writes": 0, "writes_correct": 0}
    slot_emb_cache = defaultdict(dict)
    mem_state = defaultdict(lambda: {"slots": [], "last_frame": -1})
    verify_debug = []

    def prepare_seq_cache(seq):
        if seq in cache_store:
            return
        z = np.load(CACHE / f"{seq}.npz")
        qz = np.load(CACHE / f"{seq}_queries.npz")
        zmat = {
            "frames": z["frames"], "offsets": z["offsets"],
            "boxes": z["boxes"], "scores": z["scores"],
            "emb": z["emb"].astype(np.float32),
            "feat4": z["feat4"].astype(np.float32),
            "feat5": z["feat5"].astype(np.float32),
        }
        z.close()
        qmat = {"gids": [int(g) for g in qz["gids"]],
                "qemb": qz["qemb"].astype(np.float32),
                "qfeat4": qz["qfeat4"].astype(np.float32),
                "qfeat5": qz["qfeat5"].astype(np.float32)}
        qz.close()
        emb = zmat["emb"]
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        zmat["emb"] = emb
        qemb = qmat["qemb"]
        qemb = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-8)
        qmat["qemb"] = qemb
        cache_store[seq] = zmat
        q_store[seq] = qmat
        # R0 gallery + query embeddings
        f4 = torch.from_numpy(zmat["feat4"]).cuda()
        f5 = torch.from_numpy(zmat["feat5"]).cuda()
        with torch.inference_mode():
            r0g = head_embed(gfn, f4, f5).cpu().numpy()
            r0q = head_embed(
                gfn,
                torch.from_numpy(qmat["qfeat4"]).cuda(),
                torch.from_numpy(qmat["qfeat5"]).cuda()).cpu().numpy()
        r0_store[seq] = (r0g, r0q)

    def write_fn(seq, f, gid, delivered, source, delivery_score, ident):
        prepare_seq_cache(seq)
        z = cache_store[seq]
        frames, offsets = z["frames"], z["offsets"]
        o = int(np.searchsorted(frames, f))
        lo = int(offsets[o - 1]) if o > 0 else 0
        hi = int(offsets[o])
        if hi == lo:
            return 0.0
        dets = z["boxes"][lo:hi]
        ious = np.asarray([iou(b, delivered) for b in dets])
        best = int(np.argmax(ious))
        if ious[best] < 0.5:
            return 0.0
        gi = lo + best
        gfn_e = z["emb"][gi]
        r0g, r0q = r0_store[seq]
        r0_e = r0g[gi]
        qidx = {g: i for i, g in enumerate(q_store[seq]["gids"])}
        qi = qidx.get(gid)
        if qi is None:
            return 0.0
        qe_h = q_store[seq]["qemb"][qi]
        qe_r0 = r0q[qi]

        def slot_emb(sf, sbox):
            hit = slot_emb_cache[(seq, gid)].get(sf)
            if hit is not None:
                return hit
            if sf == ident.anchor_frame and \
                    np.allclose(np.asarray(sbox, dtype=float),
                                np.asarray(ident.anchor_box, dtype=float)):
                return qe_h.copy(), qe_r0.copy()
            o2 = int(np.searchsorted(frames, sf))
            lo2 = int(offsets[o2 - 1]) if o2 > 0 else 0
            hi2 = int(offsets[o2])
            if hi2 == lo2:
                return None
            ious2 = np.asarray([iou(b, sbox) for b in z["boxes"][lo2:hi2]])
            b2 = int(np.argmax(ious2))
            if ious2[b2] < 0.5:
                return None
            gi2 = lo2 + b2
            return z["emb"][gi2], r0g[gi2]

        slot_emb_cache[(seq, gid)][f] = (gfn_e.copy(), r0_e.copy())
        keep_frames = {sf for sf, _ in ident.memory_slots[-2:]} | {f}
        slot_emb_cache[(seq, gid)] = {
            sf: e for sf, e in slot_emb_cache[(seq, gid)].items()
            if sf in keep_frames}

        # heuristic memory slots (causal, score/IoU gated)
        prev_box = ident.prev_delivered_box
        if source in ("p0_tid", "p0") and delivery_score is not None and \
                float(delivery_score) >= 0.5 and prev_box is not None \
                and iou(delivered, prev_box) >= 0.5:
            ident.heur_slots.append((f, gfn_e.copy()))
            if len(ident.heur_slots) > 8:
                ident.heur_slots.pop(0)
        feats = {
            "gfn_sim_human_root": float(gfn_e @ qe_h),
            "r0_sim_human_root": float(r0_e @ qe_r0),
            "gfn_sim_oracle_last": "",
            "gfn_sim_oracle_max": "",
            "r0_sim_oracle_max": "",
            "gfn_sim_heur_last": "",
            "gfn_sim_heur_max": "",
            "gfn_margin_h": "",
            "det_score": delivery_score if delivery_score is not None else 0.0,
            "box_area": float((delivered[2] - delivered[0]) *
                              (delivered[3] - delivered[1])),
            "temporal_iou": (float(iou(delivered, prev_box))
                             if prev_box is not None else 0.0),
            "center_delta": "",
            "consecutive_delivered": max(0, ident.delivered_streak - 1),
            "missing_streak": ident.missing_streak,
            "crowd": hi - lo,
            "overlap_max": "",
            "nearest_det_distance": "",
            "heur_memory_age": "",
            "oracle_memory_age": "",
            "candidate_age": f - ident.anchor_frame,
            "slots_oracle_count": min(len(ident.memory_slots), 2),
            "slots_heur_count": len(ident.heur_slots),
            "source": source,
        }
        mem_embs = []
        for sf, sbox in ident.memory_slots[-2:]:
            e = slot_emb(sf, sbox)
            if e is not None:
                mem_embs.append(e)
        if mem_embs:
            gsims = [e[0] @ gfn_e for e in mem_embs]
            rsims = [e[1] @ r0_e for e in mem_embs]
            feats["gfn_sim_oracle_last"] = float(gsims[-1])
            feats["gfn_sim_oracle_max"] = float(max(gsims))
            feats["r0_sim_oracle_max"] = float(max(rsims))
            feats["oracle_memory_age"] = f - ident.memory_slots[-1][0]
        if len(ident.heur_slots):
            hembs = np.stack([s[1] for s in ident.heur_slots[-8:]])
            hs = hembs @ gfn_e
            feats["gfn_sim_heur_last"] = float(hs[-1])
            feats["gfn_sim_heur_max"] = float(hs.max())
            feats["heur_memory_age"] = f - ident.heur_slots[-1][0]
        sims_all = z["emb"][lo:hi] @ qe_h
        if len(sims_all) > 1:
            order = np.argsort(-sims_all)
            feats["gfn_margin_h"] = float(
                sims_all[order[0]] - sims_all[order[1]])
        if hi - lo > 1:
            other = np.delete(dets, best, axis=0)
            ov = [iou(b, delivered) for b in other]
            feats["overlap_max"] = float(max(ov))
            cc = ((delivered[0] + delivered[2]) / 2,
                  (delivered[1] + delivered[3]) / 2)
            dc = np.stack([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                           for b in other])
            dd = np.hypot(dc[:, 0] - cc[0], dc[:, 1] - cc[1])
            feats["nearest_det_distance"] = float(np.min(dd))
        if prev_box is not None:
            c0 = ((delivered[0] + delivered[2]) / 2,
                  (delivered[1] + delivered[3]) / 2)
            c1 = ((prev_box[0] + prev_box[2]) / 2,
                  (prev_box[1] + prev_box[3]) / 2)
        feats["center_delta"] = float(
            np.hypot(c0[0] - c1[0], c0[1] - c1[1]))
        ms = mem_state[(seq, gid)]
        if not ms["slots"]:
            ms["slots"].append(
                (ident.anchor_frame, qe_h.copy(), qe_r0.copy()))
        x = to_feature_vec(feats)
        x = (x - mean) / std
        with torch.inference_mode():
            p = float(torch.sigmoid(
                writer(torch.from_numpy(x[None]).cuda())).item())
        write_stats["calls"] += 1
        if p >= cfg.anchor_score:
            write_stats["writes"] += 1
            if seq not in gt_cache:
                gt_cache[seq] = load_gt(seq)
            gf = gt_cache.get(seq, {}).get(f)
            if gf is not None and gid in gf.gt_ids:
                tgt = np.asarray(
                    gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                if iou(delivered, tgt) >= 0.5:
                    write_stats["writes_correct"] += 1
            ms["slots"].append((f, gfn_e.copy(), r0_e.copy()))
            ms["slots"] = ms["slots"][-cfg.memory_k:]
            ms["last_frame"] = f
        return p

    cfg.write_fn = write_fn

    # ---- run
    gt_cache = {}
    state_seq = {"seq": None, "frame": -1, "gid": -1}
    for seq in seqs:
        gt = load_gt(seq)
        p0 = load_p0(seq)
        num_frames = max(gt.keys()) + 1 if gt else 0
        if args.limit_frames:
            num_frames = min(num_frames, args.limit_frames)
        prepare_seq_cache(seq)
        first_app = {}
        for f0 in sorted(gt):
            for g0 in gt[f0].gt_ids:
                first_app.setdefault(g0, f0)
        qmat = q_store[seq]
        r0q = r0_store[seq][1]
        for gi, gid in enumerate(qmat["gids"]):
            mem_state[(seq, gid)]["slots"] = [(
                first_app.get(gid, 0),
                qmat["qemb"][gi].copy(),
                r0q[gi].copy())]
        t0 = time.time()
        result = run_full_loop(seq, gt, p0, num_frames, cfg,
                               recover, verify, reactivate, None)
        elapsed = time.time() - t0
        metrics = aggregate_metrics(seq, result, cfg)
        metrics["runtime_s"] = round(elapsed, 1)
        tag = f"_{args.out_tag}_s{args.shard}" if args.nshards > 1 \
            else f"_{args.out_tag}"
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
    print(json.dumps({"write_stats": write_stats}), flush=True)
    if verify_debug:
        print("VERIFY_DEBUG", json.dumps(verify_debug), flush=True)
    if args.dump_verify_csv:
        out_path = Path(args.dump_verify_csv)
        keys = ["seq", "frame", "gid", "box"] + vfeats + ["p"]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for d in verify_debug:
                row = {"seq": d["seq"], "frame": d["frame"],
                       "gid": d["gid"], "box": d["box"], "p": d["p"]}
                row.update(d["x"])
                w.writerow(row)
        print("VERIFY_CSV_DONE", out_path, flush=True)
    if runner is not None:
        runner.close()
    print("LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
