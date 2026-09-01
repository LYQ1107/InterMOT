#!/usr/bin/env python
"""N21 Phase-III true-live cal10 FINAL GATE.

True LIVE FULL_LOOP on cal10 with the frozen N20 GRU (L0), the
true-live-retrained CATIL frozen (L1), CATIL C1 LoRA online (L2), CATIL C2
partial-FT online (L3), or N22 correction-driven R0 prototype memory
(N22_PROTO). Human corrections (needs_correction)
trigger causal supervised online updates that affect only future frames.
GT is used only to generate the simulated human correction input and for
post-hoc metrics/labels.
"""

import argparse
import csv
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from n19_writer_features import feature_names, to_feature_vec  # noqa: E402
from run_n18_full_loop_v0 import load_gt, load_p0  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import (  # noqa: E402
    LoopConfig, aggregate_metrics, iou, run_full_loop)
from train_n19_writer import WriterMLP  # noqa: E402
from train_n20_kplus1 import SharedGRUSet  # noqa: E402
from n21_tracklet_identity_experiment import (  # noqa: E402
    TrackletIdentityModel, set_mode)
from sam3_intermot.recovery.n23_query_discovery import (  # noqa: E402
    NoneGate, PairRanker, gate_features, generate_windows, pair_geometry)

DT = Path("/path/to/dancetrack")
N19 = ROOT / "outputs/n19"
N20 = ROOT / "outputs/n20"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"

FEAT_COLS = [
    "candidate_rank", "gfn_sim_human_root", "r0_sim_human_root",
    "gfn_sim_mem_last", "gfn_sim_mem_max", "r0_sim_mem_last",
    "r0_sim_mem_max", "mem_age", "n_mem_slots", "temp_sim_prev",
    "temp_sim_first", "box_area", "area_change", "center_delta",
    "velocity", "temporal_iou", "consecutive_delivered",
    "shadow_delivered", "n_dets", "gfn_margin_h", "candidate_age",
    "memory_fresh", "rank_mem", "initial_correct", "init_rank_correct",
    "comp_delivered_ratio", "comp_mean_gfn_sim", "comp_max_gfn_sim",
    "comp_overlap_max", "comp_sim_margin",
]


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--split", default="cal10", choices=["cal10", "train30"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--kplus1-model", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out-tag", default="onpolicy_v1")
    ap.add_argument("--dump-trajectories", action="store_true",
                    help="persist per-attempt shadow trajectories (JSONL)")
    ap.add_argument("--out-dir", default="",
                    help="override output directory")
    ap.add_argument("--variant", default="L0",
                    choices=["L0", "L1", "L2", "L3", "N22_PROTO", "N23"])
    ap.add_argument("--catil-model", default="",
                    help="CATIL checkpoint for L1/L2/L3")
    ap.add_argument("--n23-ranker", default=str(
        ROOT / "outputs/n23/models/query_ranker.pt"))
    ap.add_argument("--n23-gate", default=str(
        ROOT / "outputs/n23/models/none_gate.pt"))
    ap.add_argument("--n23-raw-gate", default=str(
        ROOT / "outputs/n23/models/raw_none_gate.pt"))
    ap.add_argument("--n23-score-mode", choices=["raw", "adapter"],
                    default="raw")
    ap.add_argument("--n23-window-max", type=int, default=600)
    ap.add_argument("--n23-rank-batch", type=int, default=256)
    ap.add_argument("--online-epochs", type=int, default=10)
    ap.add_argument("--online-replay", type=int, default=32)
    ap.add_argument("--kl-lambda", type=float, default=2.0)
    ap.add_argument("--online-lr", type=float, default=1e-4)
    ap.add_argument("--proto-alpha", type=float, default=0.05)
    ap.add_argument("--proto-beta", type=float, default=0.05)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    frozen = json.loads(
        (ROOT / "outputs/n15/n15_frozen.json").read_text())["split"]
    if args.seqs:
        seqs = sorted(args.seqs.split(","))
    else:
        seqs = sorted(frozen["train30" if args.split == "train30"
                                 else "calibration10"])

    cfg = LoopConfig(reactivation_horizon=120, enable_recovery=True,
                     enable_reactivation=True, anchor_policy="learned",
                     anchor_score=0.95, memory_k=2, shadow_mode=True,
                     shadow_horizon=args.h, shadow_timeout=args.h + 2)
    if args.variant == "N23":
        # N23 is a direct causal recovery branch: a correction-conditioned
        # whole-frame bank is scored and gated, then the accepted box is
        # reactivated by the same official SAM3 path.
        cfg.shadow_mode = False
        cfg.verifier_threshold = 0.5
    cfg.on_correction = None  # set below once the online learner exists

    # ---- K+1 verifier
    bundle = torch.load(args.kplus1_model, map_location="cpu")
    feats = bundle["feature_cols"]
    mu = bundle["mu"]
    sd = bundle["sd"]
    verifier = SharedGRUSet(len(feats))
    verifier.load_state_dict(bundle["model"])
    verifier.eval()
    print(f"KPLUS1 loaded h={bundle['h']} feats={len(feats)}", flush=True)

    # ---- CATIL verifier (L1/L2/L3) ----
    catil = None
    catil_online = None
    catil_mode = None
    if args.variant not in ("L0", "N22_PROTO", "N23"):
        if not args.catil_model:
            raise SystemExit("--catil-model required for L1/L2/L3")
        cb = torch.load(args.catil_model, map_location="cpu")
        catil = TrackletIdentityModel().to(f"cuda:{args.gpu}")
        catil.load_state_dict(cb["model"])
        catil.eval()
        catil_mode = {"L1": "none", "L2": "C1_lora",
                      "L3": "C2_partial_ft"}[args.variant]
        if catil_mode != "none":
            set_mode(catil, catil_mode)
            import copy as _copy
            catil_online = _copy.deepcopy(catil)
            catil_online.eval()
        print(f"CATIL loaded variant={args.variant} mode={catil_mode} "
              f"params={catil.n_params()}", flush=True)

    # ---- GFN + R0 head (GPU) + Writer (CPU)
    gfn, _, _, _, _ = load_model(f"cuda:{args.gpu}")
    gfn.eval()
    r0_path = ROOT / "outputs/n18/route_c/models/r0_best.pt"
    if r0_path.exists():
        gfn.roi_heads.embedding_head.load_state_dict(
            torch.load(r0_path, map_location="cpu"))
        gfn.roi_heads.embedding_head.eval()
    wcfg = json.loads((N19 / "models/writer_v0/writer_config.json").read_text())
    writer = WriterMLP(len(feature_names()), hidden=wcfg["hidden"])
    writer.load_state_dict(torch.load(
        N19 / "models/writer_v0/writer_v0.pt", map_location="cpu"))
    writer.eval()
    wmean = np.asarray(wcfg["scaler_mean"], dtype=np.float32)
    wstd = np.asarray(wcfg["scaler_std"], dtype=np.float32)

    # ---- SAM3 backend with lazy frame loading
    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner, parse_raw_outputs)
    from sam3_intermot.detection_query.prompt_replay import (
        _best_delivery, invalidate_detector_prefetch,
        set_frame_geometric_prompt)
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend.async_loading_frames = True
    backend._ensure_model()
    model = backend._predictor.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n23_ranker = None
    n23_gate = None
    n23_gate_threshold = 0.5
    n23_clip = None
    n23_query_cache = {}
    if args.variant == "N23":
        from scripts.build_n23_window_cache import (
            expand_box, image_tensor, roi_batch)
        from scripts.run_n15_extract_features import build_clipreid

        dev = f"cuda:{args.gpu}"
        rank_payload = torch.load(args.n23_ranker, map_location="cpu",
                                  weights_only=False)
        n23_ranker = PairRanker(**rank_payload.get("arch", {})).to(dev)
        n23_ranker.load_state_dict(rank_payload["state"])
        n23_ranker.eval()
        gate_path = (args.n23_raw_gate if args.n23_score_mode == "raw"
                     else args.n23_gate)
        gate_payload = torch.load(gate_path, map_location="cpu",
                                  weights_only=False)
        n23_gate = NoneGate(**gate_payload.get("arch", {})).to(dev)
        n23_gate.load_state_dict(gate_payload["state"])
        n23_gate.eval()
        n23_gate_threshold = float(gate_payload.get("threshold", 0.5))
        n23_clip = build_clipreid(
            str(ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"),
            dev,
        )
        print(f"N23 loaded score_mode={args.n23_score_mode} ranker/gate/CLIP "
              f"window_max={args.n23_window_max} "
              f"gate_threshold={n23_gate_threshold}", flush=True)
    iw = ih = None
    state_seq = {"seq": None}

    def ensure_video(seq):
        nonlocal iw, ih
        if state_seq["seq"] != seq:
            backend.start_video(str(DT / "train" / seq / "img1"))
            iw, ih = backend._frame_w, backend._frame_h
            state_seq["seq"] = seq

    def propagate_track(seq, f, box, horizon):
        """Run one isolated hypothesis in the sequence base session.
        reset_session isolates the hypothesis (clears prompts/features) and
        keeps the video frames, so host memory stays bounded (one session)."""
        sid = backend._session_id
        try:
            backend._predictor.handle_request(dict(
                type="reset_session", session_id=sid))
        except Exception as e:
            print(f"RESET_FAIL {e}", flush=True)
            return {}
        x1, y1, x2, y2 = box
        req = dict(type="add_prompt", session_id=sid, frame_index=f,
                   text="person",
                   bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw,
                                    (y2 - y1) / ih]],
                   bounding_box_labels=[1], clear_old_boxes=True)
        try:
            backend._predictor.handle_request(req)
        except Exception as e:
            print(f"PROMPT_FAIL {e}", flush=True)
            return {}
        state = backend._predictor._all_inference_states[sid]["state"]
        nf = state["num_frames"]
        if f + 1 < nf:
            set_frame_geometric_prompt(runner, f + 1, None)
        req2 = dict(type="propagate_in_video", session_id=sid,
                    propagation_direction="forward", start_frame_index=f,
                    max_frame_num_to_track=None)
        prev = np.asarray(box, dtype=float).copy()
        records = {f: prev.copy()}
        try:
            for response in backend._predictor.handle_stream_request(
                    request=req2):
                ff = int(response["frame_index"])
                print(f"PROP_FRAME ff={ff} out_keys="
                      f"{list(response['outputs'].keys()) if isinstance(response.get('outputs'), dict) else type(response.get('outputs'))}",
                      flush=True)
                cands = parse_raw_outputs(response, frame_size=(iw, ih))
                if len(cands) == 0:
                    print(f"PROP_EMPTY ff={ff}", flush=True)
                cand_boxes = [np.asarray(b, dtype=float) for _, b in cands]
                delivered = _best_delivery(prev, cand_boxes)
                if delivered is None and cand_boxes:
                    delivered = cand_boxes[0]
                if delivered is not None:
                    prev = delivered.copy()
                records[ff] = delivered
                if ff >= f + horizon:
                    break
                if ff + 1 < nf:
                    set_frame_geometric_prompt(runner, ff + 1, None)
                invalidate_detector_prefetch(runner, ff)
        except Exception as e:
            print(f"PROP_EXC {e}", flush=True)
        print(f"PROP_TRACK f={f} horizon={horizon} keys="
              f"{sorted(records.keys())}", flush=True)
        return records

    # ---- GFN gallery + memory replay (live)
    img_cache = {}
    z_cache = {}
    gt_cache = {}
    mem_state = defaultdict(lambda: {"slots": [], "last_frame": -1})
    shadow_groups = {}  # (gid, attempt_frame) -> state
    traj_dump = []
    last_attempts = {}          # (seq, gid) -> stashed CATIL attempt
    corr_replay = []            # list of (vis, vm, aux, root, target, negs)
    corr_total = 0
    proto_state = {}            # (seq, gid) -> causal base-R0 prototypes
    proto_updates = 0
    proto_negative_updates = 0
    proto_repeated_wrong = 0
    n23_correction_queries = {}
    catil_base = None
    if catil_online is not None:
        import copy
        catil_base = copy.deepcopy(catil_online)
        catil_base.eval()
        for p in catil_base.parameters():
            p.requires_grad_(False)
    corr_opt = None
    corr_lossf = None
    if catil_online is not None:
        corr_opt = torch.optim.Adam(
            [p for p in catil_online.parameters() if p.requires_grad],
            lr=args.online_lr)
        corr_lossf = torch.nn.CrossEntropyLoss()

    def on_correction(seq, f, gid, pid, box):
        nonlocal corr_total, proto_updates, proto_negative_updates
        nonlocal proto_repeated_wrong
        if args.variant == "N23":
            # The correction becomes a query only after the current frame has
            # been evaluated.  recovery_fn reads this state on later frames,
            # preserving the FULL_LOOP causal boundary.
            if box is not None:
                n23_correction_queries[(seq, gid)] = (
                    int(f), np.asarray(box, dtype=float).copy())
                corr_total += 1
            return
        la = last_attempts.get((seq, gid))
        if la is None or box is None or f - la["frame"] > 120:
            return
        pos = 0
        if la["candidates"]:
            ious = [iou(b, box) for b in la["candidates"]]
            bi = int(np.argmax(ious))
            if ious[bi] >= 0.5:
                pos = bi + 1
        negs = [la["decision"]] if la["decision"] >= 1 else []
        target = pos if pos >= 1 else 0
        if args.variant == "N22_PROTO":
            state_key = (seq, gid)
            st = proto_state.setdefault(
                state_key, {"pos": la["root_r0"].copy(), "negs": [],
                            "had_correction": False})
            if st["had_correction"]:
                proto_repeated_wrong += 1
            st["had_correction"] = True
            if target >= 1 and target <= len(la["r0_candidates"]):
                vec = la["r0_candidates"][target - 1]
                if vec is not None:
                    st["pos"] = st["pos"] * (1.0 - args.proto_alpha) \
                        + vec * args.proto_alpha
                    st["pos"] /= max(1e-8, np.linalg.norm(st["pos"]))
                    proto_updates += 1
            if la["decision"] >= 1 and la["decision"] != target and \
                    la["decision"] <= len(la["r0_candidates"]):
                vec = la["r0_candidates"][la["decision"] - 1]
                if vec is not None:
                    st["negs"].append(vec.copy())
                    st["negs"] = st["negs"][-2:]
                    proto_negative_updates += 1
            corr_total += 1
            last_attempts.pop((seq, gid), None)
            return
        corr_replay.append({
            "vis": la["vis"], "vm": la["vm"], "aux": la["aux"],
            "root": la["root"], "target": target, "negs": negs,
        })
        corr_total += 1
        if len(corr_replay) > args.online_replay:
            corr_replay.pop(0)
        if catil_mode in ("C1_lora", "C2_partial_ft"):
            dev = f"cuda:{args.gpu}"
            vis = torch.tensor(corr_replay[0]["vis"], device=dev) \
                if len(corr_replay) == 1 \
                else torch.cat([torch.tensor(s["vis"], device=dev)
                                for s in corr_replay], 0)
            vm = torch.tensor(corr_replay[0]["vm"], device=dev) \
                if len(corr_replay) == 1 \
                else torch.cat([torch.tensor(s["vm"], device=dev)
                                for s in corr_replay], 0)
            aux = torch.tensor(corr_replay[0]["aux"], device=dev) \
                if len(corr_replay) == 1 \
                else torch.cat([torch.tensor(s["aux"], device=dev)
                                for s in corr_replay], 0)
            root = torch.tensor(corr_replay[0]["root"], device=dev) \
                if len(corr_replay) == 1 \
                else torch.cat([torch.tensor(s["root"], device=dev)
                                for s in corr_replay], 0)
            targets = torch.tensor(
                [s["target"] for s in corr_replay],
                dtype=torch.long, device=dev)
            for p in catil_online.parameters():
                if p.requires_grad:
                    p.data = p.data.clone()
            catil_online.train()
            for _ in range(args.online_epochs):
                corr_opt.zero_grad()
                tok, root_tok, raw_k, raw_root = catil_online.encode(
                    vis, vm, root)
                logits, _, _ = catil_online.forward_tokens(
                    tok, vm, root_tok, raw_k, raw_root, aux)
                loss = corr_lossf(logits, targets)
                margins = []
                for j, s in enumerate(corr_replay):
                    if s["target"] <= 0:
                        continue
                    for rn in s["negs"]:
                        margins.append(torch.clamp(
                            0.2 - (logits[j, s["target"]] -
                                   logits[j, rn]), min=0.0))
                if margins:
                    loss = loss + 0.5 * torch.stack(margins).mean()
                if catil_base is not None:
                    with torch.inference_mode():
                        blogits, _, _ = catil_base(vis, vm, root, aux)
                    loss = loss + args.kl_lambda * torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(logits, 1),
                        torch.nn.functional.softmax(blogits, 1),
                        reduction="batchmean")
                loss.backward()
                corr_opt.step()
            catil_online.eval()
            catil.load_state_dict(catil_online.state_dict())
            catil.eval()
        last_attempts.pop((seq, gid), None)

    cfg.on_correction = on_correction

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

    def write_fn(seq, f, gid, delivered, source, delivery_score, ident):
        z = get_z(seq)
        o = int(np.searchsorted(z["frames"], f))
        lo = int(z["offsets"][o - 1]) if o > 0 else 0
        hi = int(z["offsets"][o])
        if hi == lo:
            return 0.0
        ious = np.asarray([iou(b, delivered) for b in z["boxes"][lo:hi]])
        best = int(np.argmax(ious))
        if ious[best] < 0.5:
            return 0.0
        ge = z["emb"][lo + best]
        r0e = z["r0g"][lo + best]
        qidx = {g: i for i, g in enumerate(z["qgids"])}
        qi = qidx.get(gid)
        if qi is None:
            return 0.0
        qe_h = z["qemb"][qi]
        qe_r0 = z["r0q"][qi]
        prev_box = ident.prev_delivered_box
        feats = {
            "gfn_sim_human_root": float(ge @ qe_h),
            "r0_sim_human_root": float(r0e @ qe_r0),
            "gfn_sim_oracle_last": "",
            "gfn_sim_oracle_max": "",
            "r0_sim_oracle_max": "",
            "gfn_sim_heur_last": "",
            "gfn_sim_heur_max": "",
            "gfn_margin_h": "",
            "det_score": delivery_score if delivery_score is not None else 0.0,
            "box_area": float((delivered[2] - delivered[0]) *
                              (delivered[3] - delivered[1])),
            "temporal_iou": float(iou(delivered, prev_box))
            if prev_box is not None else 0.0,
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
        mem = ident.memory_slots[-2:]
        if mem:
            gsims = []
            for sf, sbox in mem:
                o2 = int(np.searchsorted(z["frames"], sf))
                lo2 = int(z["offsets"][o2 - 1]) if o2 > 0 else 0
                hi2 = int(z["offsets"][o2])
                if hi2 == lo2:
                    continue
                ious2 = np.asarray([iou(b, sbox)
                                    for b in z["boxes"][lo2:hi2]])
                b2 = int(np.argmax(ious2))
                if ious2[b2] >= 0.5:
                    gsims.append(z["emb"][lo2 + b2] @ ge)
            if gsims:
                feats["gfn_sim_oracle_last"] = float(gsims[-1])
                feats["gfn_sim_oracle_max"] = float(max(gsims))
                feats["oracle_memory_age"] = f - mem[-1][0]
        x = to_feature_vec(feats)
        x = (x - wmean) / wstd
        with torch.inference_mode():
            p = float(torch.sigmoid(
                writer(torch.from_numpy(x[None]))).item())
        return p

    cfg.write_fn = write_fn

    def n23_recovery(seq, f, qbox, qframe, gid):
        """Causal correction-query proposal and NONE decision.

        This is intentionally evaluated on the live recovery frame rather
        than reading the offline window cache: the query is the latest human
        correction available strictly before ``f`` and the future image is
        loaded only at the recovery attempt.
        """
        if n23_clip is None or n23_ranker is None or n23_gate is None:
            return None
        dev = torch.device(f"cuda:{args.gpu}")
        qarr = np.asarray(qbox, dtype=np.float32)
        qkey = (seq, int(qframe), tuple(np.rint(qarr).astype(int).tolist()))
        if qkey not in n23_query_cache:
            qpath = DT / "train" / seq / "img1" / f"{int(qframe) + 1:08d}.jpg"
            qimage = image_tensor(qpath)
            qcrop = expand_box(qarr, qimage.shape[-1], qimage.shape[-2])
            with torch.inference_mode():
                qx = roi_batch(qimage, qcrop.reshape(1, 4), dev)
                _, q12, qproj = n23_clip(qx)
                qfeat = torch.nn.functional.normalize(
                    torch.cat([q12[:, 0], qproj[:, 0]], dim=1), dim=-1
                )[0].detach()
            n23_query_cache[qkey] = qfeat.cpu()
        qfeat = n23_query_cache[qkey].to(dev)

        fpath = DT / "train" / seq / "img1" / f"{int(f) + 1:08d}.jpg"
        fimage = image_tensor(fpath)
        boxes_np = generate_windows(
            qarr,
            image_w=float(fimage.shape[-1]),
            image_h=float(fimage.shape[-2]),
            max_windows=args.n23_window_max,
        )
        with torch.inference_mode():
            emb_parts = []
            for start in range(0, len(boxes_np), args.n23_rank_batch):
                bpart = boxes_np[start:start + args.n23_rank_batch]
                x = roi_batch(fimage, bpart, dev)
                _, x12, xproj = n23_clip(x)
                emb_parts.append(torch.nn.functional.normalize(
                    torch.cat([x12[:, 0], xproj[:, 0]], dim=1), dim=-1
                ))
            emb = torch.cat(emb_parts, dim=0)
            boxes = torch.from_numpy(boxes_np).to(dev)
            raw = emb @ qfeat
            geom = pair_geometry(
                torch.from_numpy(qarr).to(dev), boxes, float(f - qframe))
            adapter_logits = n23_ranker(qfeat.expand(len(boxes), -1), emb, geom)
            score_logits = (raw if args.n23_score_mode == "raw"
                            else adapter_logits)
            gfeat = gate_features(
                score_logits, raw, boxes, torch.from_numpy(qarr).to(dev),
                float(f - qframe))
            gate_prob = float(torch.sigmoid(n23_gate(gfeat)).item())
        order = torch.argsort(score_logits, descending=True)
        best = int(order[0].item())
        best_box = boxes_np[best].astype(float)
        print(f"N23_DECIDE {seq} f={f} gid={gid} qframe={qframe} "
              f"windows={len(boxes_np)} gate={gate_prob:.4f} "
              f"score={float(score_logits[best]):.4f} accept="
              f"{int(gate_prob >= n23_gate_threshold)}", flush=True)
        if gate_prob < n23_gate_threshold:
            return None
        return {
            "candidates": [{"box": best_box,
                            "n23_gate_prob": gate_prob}],
            "box": best_box,
            "n23_gate_prob": gate_prob,
            "n23_windows": len(boxes_np),
        }

    def recovery_fn(seq, f, qbox, qframe, gid=None):
        if args.variant == "N23":
            correction = n23_correction_queries.get((seq, gid))
            if correction is not None and int(correction[0]) < int(f):
                qframe, qbox = correction
            return n23_recovery(seq, f, qbox, qframe, gid)
        # return top-K candidate boxes (live GFN ranking with learned memory)
        z = get_z(seq)
        o = int(np.searchsorted(z["frames"], f))
        lo = int(z["offsets"][o - 1]) if o > 0 else 0
        hi = int(z["offsets"][o])
        if hi == lo:
            return None
        qidx = {g: i for i, g in enumerate(z["qgids"])}
        qi = qidx.get(gid)
        if qi is None:
            return None
        qe_h = z["qemb"][qi]
        ms = mem_state[(seq, gid)]["slots"]
        if not ms:
            ms.append((qframe, qe_h.copy()))
        sims = np.maximum.reduce([z["emb"][lo:hi] @ s[1]
                                  for s in ms[-2:]])
        order = np.argsort(-sims)
        boxes = [z["boxes"][lo + idx].astype(float).copy()
                 for idx in order[:args.k]]
        return {"candidates": [{"box": b} for b in boxes],
                "box": boxes[0], "gfn_top1_sim": float(sims[order[0]]),
                "n_dets": hi - lo}

    def shadow_start_fn(seq, f, gid, qbox, qframe):
        rec = recovery_fn(seq, f, qbox, qframe, gid)
        if rec is None or not rec["candidates"]:
            return None
        z = get_z(seq)
        qidx = {g: i for i, g in enumerate(z["qgids"])}
        qi = qidx.get(gid)
        qe_h = z["qemb"][qi]
        qe_r0 = z["r0q"][qi]
        gt = get_gt(seq)
        first_app = None
        for ff in sorted(gt):
            if gid in gt[ff].gt_ids:
                first_app = ff
                break
        ms = mem_state[(seq, gid)]["slots"]
        mem = ms[-2:] if ms else []
        gid_s = str(uuid.uuid4().hex)
        shadow_groups[gid_s] = {
            "seq": seq, "f0": f, "gid": gid, "sessions": [],
            "qe_h": qe_h, "qe_r0": qe_r0, "mem": mem,
            "first_app": first_app, "n_dets": rec["n_dets"],
            "elapsed": 0,
            "candidates": [np.asarray(c["box"], dtype=float)
                           for c in rec["candidates"]],
            "trajectories": None,
        }
        return gid_s

    def build_kplus1_features(rows, h):
        feats_rows = []
        for steps in rows:
            vecs = []
            for r in steps[:h]:
                vecs.append([to_float(r.get(c, 0.0)) for c in feats])
            while len(vecs) < h:
                vecs.append(vecs[-1])
            feats_rows.append(np.nan_to_num(
                np.asarray(vecs, dtype=np.float32), nan=0.0))
        if len(feats_rows) != args.k:
            return None
        arr = torch.from_numpy(np.stack(feats_rows))[None]
        arr = (arr - torch.as_tensor(mu)) / torch.as_tensor(sd)
        raw_max = np.max(np.abs(np.stack(feats_rows)), axis=(0, 1))
        big = [(feats[j], float(raw_max[j])) for j in
               range(len(feats)) if raw_max[j] > 100]
        print(f"ARR_DEBUG shape={tuple(arr.shape)} mean={float(arr.mean()):.4f} "
              f"first={[round(float(v), 3) for v in arr[0, 0, 0, :6].tolist()]} "
              f"big={big[:6]}",
              flush=True)
        mask = torch.ones(1, args.k)
        with torch.inference_mode():
            probs = torch.softmax(verifier(arr, mask), 1)[0].numpy()
        return probs

    def shadow_step_fn(seq, f, gid, sid, elapsed):
        grp = shadow_groups.get(sid)
        if grp is None:
            return {"verdict": "REJECT"}
        grp["elapsed"] = elapsed
        if grp["elapsed"] < args.h:
            return {"verdict": "PENDING"}
        # on-policy: generate H-frame trajectories for all candidates
        # sequentially in isolated sessions (logical causality: decisions
        # below use only features from steps <= H).
        if grp["trajectories"] is None:
            trajs = []
            for rank, box in enumerate(grp["candidates"], start=1):
                t0 = time.time()
                traj = propagate_track(grp["seq"], grp["f0"], box,
                                       args.h)
                trajs.append({"rank": rank, "traj": traj,
                              "gen_s": round(time.time() - t0, 2)})
            grp["trajectories"] = trajs
            if args.dump_trajectories:
                traj_dump.append({
                    "sequence": grp["seq"], "frame": grp["f0"],
                    "gid": grp["gid"],
                    "attempt": f"{grp['seq']}:{grp['f0']}:{grp['gid']}",
                    "trajectories": [
                        {"rank": t["rank"],
                         "frames": {str(k): (v.tolist() if v is not None
                                             else None)
                                    for k, v in t["traj"].items()}}
                        for t in trajs],
                })
        # build per-step causal features for each candidate
        rows = []
        z = get_z(seq)
        qe_h = grp["qe_h"]
        qe_r0 = grp["qe_r0"]
        mem = grp["mem"]
        # attempt-frame margin and start embeddings (causal, <= f0)
        o0 = int(np.searchsorted(z["frames"], grp["f0"]))
        lo0 = int(z["offsets"][o0 - 1]) if o0 > 0 else 0
        hi0 = int(z["offsets"][o0])
        margin_h = ""
        if hi0 > lo0:
            sims0 = z["emb"][lo0:hi0] @ qe_h
            ord0 = np.argsort(-sims0)
            if len(ord0) > 1:
                margin_h = float(sims0[ord0[0]] - sims0[ord0[1]])
        start_embs = []
        for tr in grp["trajectories"]:
            de0 = det_emb(seq, grp["f0"], grp["candidates"][tr["rank"] - 1])
            start_embs.append(de0[0] if de0 is not None else None)
        total_hits = 0
        total_misses = 0
        vis_rows = []
        for tr in grp["trajectories"]:
            traj = tr["traj"]
            steps = []
            vis_seq = []
            vis_mask = []
            de_hits = 0
            de_misses = 0
            prev_box = None
            prev_emb = None
            prev_center = None
            prev_delta = None
            consec = 0
            for hh in range(1, args.h + 1):
                ff = grp["f0"] + hh
                nb = traj.get(ff)
                de = det_emb(seq, ff, nb) if nb is not None else None
                if de is not None:
                    vis_seq.append(np.concatenate([de[0], de[1]]))
                    vis_mask.append(1.0)
                else:
                    vis_seq.append(np.zeros(4096, dtype=np.float32))
                    vis_mask.append(0.0)
                if nb is not None:
                    if de is not None:
                        de_hits += 1
                    else:
                        de_misses += 1
                sim_mem_last = sim_mem_max = ""
                rsim_mem_last = rsim_mem_max = ""
                if de is not None and mem:
                    gsims = [m[1] @ de[0] for m in mem]
                    rsims = [m[2] @ de[1] for m in mem]
                    sim_mem_last = float(gsims[-1])
                    sim_mem_max = float(max(gsims))
                    rsim_mem_last = float(rsims[-1])
                    rsim_mem_max = float(max(rsims))
                temp_prev = ""
                temp_first = ""
                if de is not None and prev_emb is not None:
                    temp_prev = float(de[0] @ prev_emb)
                if de is not None and start_embs[tr["rank"] - 1] is not None:
                    temp_first = float(de[0] @ start_embs[tr["rank"] - 1])
                area = 0.0
                area_change = ""
                center_delta = 0.0
                velocity = ""
                temporal_iou = ""
                if nb is not None:
                    area = float((nb[2] - nb[0]) * (nb[3] - nb[1]))
                    prev_for_iou = prev_box
                    center = ((nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2)
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
                        temporal_iou = float(iou(nb, prev_for_iou))
                    prev_box = nb
                if nb is not None:
                    consec += 1
                else:
                    consec = 0
                # competition across candidates at this frame
                comp_delivered = []
                comp_sims = []
                comp_boxes = []
                for tr2 in grp["trajectories"]:
                    if tr2["rank"] == tr["rank"]:
                        continue
                    nb2 = tr2["traj"].get(ff)
                    if nb2 is None:
                        comp_delivered.append(0)
                        continue
                    comp_delivered.append(1)
                    comp_boxes.append(np.asarray(nb2, dtype=float))
                    de2 = det_emb(seq, ff, nb2)
                    if de2 is not None and mem:
                        comp_sims.append(max(m[1] @ de2[0] for m in mem))
                comp_overlap_max = ""
                if nb is not None and comp_boxes:
                    comp_overlap_max = float(max(
                        iou(np.asarray(nb, dtype=float), b)
                        for b in comp_boxes))
                comp_delivered_ratio = float(np.mean(comp_delivered)) \
                    if comp_delivered else 1.0
                comp_mean_gfn_sim = float(np.mean(comp_sims)) \
                    if comp_sims else ""
                comp_max_gfn_sim = float(np.max(comp_sims)) \
                    if comp_sims else ""
                comp_sim_margin = ""
                if de is not None and mem and comp_sims:
                    my_sim = max(m[1] @ de[0] for m in mem)
                    comp_sim_margin = float(my_sim - max(comp_sims))
                steps.append({
                    "candidate_rank": tr["rank"],
                    "gfn_sim_human_root": float(de[0] @ qe_h) if de else 0.0,
                    "r0_sim_human_root": float(de[1] @ qe_r0) if de else 0.0,
                    "gfn_sim_mem_last": sim_mem_last,
                    "gfn_sim_mem_max": sim_mem_max,
                    "r0_sim_mem_last": rsim_mem_last,
                    "r0_sim_mem_max": rsim_mem_max,
                    "mem_age": float(f - mem[-1][0]) if mem else -1.0,
                    "n_mem_slots": float(len(mem)),
                    "temp_sim_prev": temp_prev,
                    "temp_sim_first": temp_first,
                    "box_area": area,
                    "area_change": area_change,
                    "center_delta": center_delta,
                    "velocity": velocity,
                    "temporal_iou": temporal_iou,
                    "consecutive_delivered": float(consec),
                    "shadow_delivered": float(nb is not None),
                    "n_dets": float(grp["n_dets"]),
                    "gfn_margin_h": margin_h,
                    "candidate_age": float(
                        f - grp["first_app"]) if grp["first_app"] else 0.0,
                    "memory_fresh": float(f - mem[-1][0]) if mem else -1.0,
                    "rank_mem": float(tr["rank"]),
                    "initial_correct": 0.0,
                    "init_rank_correct": 0.0,
                    "comp_delivered_ratio": comp_delivered_ratio,
                    "comp_mean_gfn_sim": comp_mean_gfn_sim,
                    "comp_max_gfn_sim": comp_max_gfn_sim,
                    "comp_overlap_max": comp_overlap_max,
                    "comp_sim_margin": comp_sim_margin,
                })
                if de is not None:
                    prev_emb = de[0]
            while len(vis_seq) < args.h:
                vis_seq.append(np.zeros(4096, dtype=np.float32))
                vis_mask.append(0.0)
            vis_rows.append({
                "rank": tr["rank"],
                "vis": np.stack(vis_seq).astype(np.float32),
                "mask": np.asarray(vis_mask, dtype=np.float32),
            })
            rows.append(steps)
            total_hits += de_hits
            total_misses += de_misses
        grp["de_hits"] = total_hits
        grp["de_misses"] = total_misses
        probs = build_kplus1_features(rows, args.h)
        if probs is None:
            return {"verdict": "REJECT"}
        proto_r0 = []
        if args.variant == "N22_PROTO":
            # CDCIA prototype memory reads the current shadow tracklet in
            # the frozen R0 coordinate.  Missing all-frame evidence is an
            # unusable candidate and receives a strict reject score.
            for vi in vis_rows:
                vv = vi["vis"][:, 2048:]
                mm = vi["mask"]
                if float(mm.sum()) <= 0:
                    proto_r0.append(None)
                    continue
                vec = (vv * mm[:, None]).sum(0) / max(1.0, float(mm.sum()))
                norm = float(np.linalg.norm(vec))
                proto_r0.append(vec / max(1e-8, norm))
            st = proto_state.get((seq, gid))
            pos_proto = (st["pos"] if st is not None else qe_r0).copy()
            scores = np.asarray([
                -1.0 if v is None else float(v @ pos_proto)
                for v in proto_r0
            ], dtype=np.float32)
            if st is not None and st["negs"]:
                neg_sim = np.asarray([
                    -1.0 if v is None else max(float(v @ n)
                                               for n in st["negs"])
                    for v in proto_r0
                ], dtype=np.float32)
                scores -= args.proto_beta * np.maximum(neg_sim, 0.0)
            probs = np.concatenate([np.asarray([0.0], dtype=np.float32),
                                    scores])
        if catil is not None:
            # ---- CATIL: visual residual on top of the frozen GRU prior ----
            aux_list = []
            for vi, steps in zip(vis_rows, rows):
                vecs = []
                for r in steps[:args.h]:
                    vecs.append([to_float(r.get(c, 0.0)) for c in feats])
                mvec = np.mean(np.nan_to_num(
                    np.asarray(vecs, dtype=np.float32), nan=0.0), axis=0)
                aux_list.append(np.concatenate([
                    (mvec - mu.numpy()) / sd.numpy(),
                    np.asarray([probs[0], probs[vi["rank"]]],
                               dtype=np.float32)]))
            vis_arr = np.stack([v["vis"] for v in vis_rows])[None]
            vm_arr = np.stack([v["mask"] for v in vis_rows])[None]
            aux_arr = np.stack(aux_list)[None]
            root_arr = np.concatenate([qe_h, qe_r0])[None]
            dev = f"cuda:{args.gpu}"
            tvis = torch.from_numpy(vis_arr).to(dev)
            tvm = torch.from_numpy(vm_arr).to(dev)
            taux = torch.from_numpy(aux_arr).to(dev)
            troot = torch.from_numpy(root_arr).to(dev)
            with torch.inference_mode():
                cprobs = torch.softmax(
                    catil(tvis, tvm, troot, taux)[0], 1)[0].cpu().numpy()
            probs = cprobs
        grp["last_probs"] = [round(float(v), 4) for v in probs]
        best = int(np.argmax(probs))
        decision = 0
        if best >= 1 and probs[best] >= args.threshold:
            others = np.delete(probs, best)
            if probs[best] - others.max() >= args.margin:
                decision = best
        if catil is not None:
            last_attempts[(seq, gid)] = {
                "frame": f,
                "vis": vis_arr.copy(),
                "vm": vm_arr.copy(),
                "aux": aux_arr.copy(),
                "root": root_arr.copy(),
                "decision": decision,
                "candidates": [np.asarray(b, dtype=float)
                               for b in grp["candidates"]],
            }
        elif args.variant == "N22_PROTO":
            last_attempts[(seq, gid)] = {
                "frame": f,
                "root_r0": qe_r0.copy(),
                "r0_candidates": proto_r0,
                "decision": decision,
                "candidates": [np.asarray(b, dtype=float)
                               for b in grp["candidates"]],
            }
        print(f"ONPOLICY_DECIDE {seq} f={f} gid={gid} de_hits="
              f"{de_hits} de_misses={de_misses} probs="
              f"{[round(float(v), 4) for v in probs]}", flush=True)
        if rows:
            r0 = rows[0][0] if rows[0] else {}
            print("FEAT_DEBUG", {k: r0.get(k) for k in (
                "gfn_sim_human_root", "r0_sim_human_root",
                "gfn_sim_mem_max", "temp_sim_first", "shadow_delivered",
                "mem_age", "candidate_rank")}, flush=True)
        best = int(np.argmax(probs))
        if best >= 1 and probs[best] >= args.threshold:
            others = np.delete(probs, best)
            if probs[best] - others.max() >= args.margin:
                chosen = grp["trajectories"][best - 1]
                chosen_box = chosen["traj"].get(f)
                if chosen_box is None:
                    chosen_box = grp["candidates"][best - 1]
                # official SAM3 reactivation from the commit frame
                traj = propagate_track(seq, f, chosen_box, 120)
                return {"verdict": "ACCEPT", "traj": traj,
                        "box": chosen_box, "commit_frame": f,
                        "kplus1_best": best,
                        "kplus1_probs": [round(float(v), 4)
                                         for v in probs]}
        return {"verdict": "REJECT"}

    if args.out_dir:
        out_dir = ROOT / args.out_dir
    elif args.variant == "N23":
        out_dir = ROOT / "outputs/n23/live_full_loop"
    else:
        out_dir = ROOT / "outputs/n21/train30_true_onpolicy"
    out_dir.mkdir(parents=True, exist_ok=True)
    catil_init = None
    if catil is not None and catil_mode != "none":
        import copy as _copy
        catil_init = _copy.deepcopy(catil.state_dict())
    for seq in seqs:
        done_marker = out_dir / f"{seq}.done"
        if done_marker.exists():
            print(f"SKIP_EXISTING {seq}", flush=True)
            continue
        # episodic reset for online variants
        if catil is not None and catil_mode != "none":
            catil.load_state_dict(catil_init)
            catil.eval()
            if catil_online is not None:
                catil_online.load_state_dict(catil_init)
                catil_online.eval()
                import copy as _copy2
                catil_base = _copy2.deepcopy(catil_online)
                catil_base.eval()
                for p in catil_base.parameters():
                    p.requires_grad_(False)
                corr_opt = torch.optim.Adam(
                    [p for p in catil_online.parameters()
                     if p.requires_grad], lr=args.online_lr)
            corr_replay.clear()
            last_attempts.clear()
            corr_total = 0
            print(f"EPISODIC_RESET {seq} mode={catil_mode}", flush=True)
        if args.variant == "N22_PROTO":
            corr_total = 0
            proto_updates = 0
            proto_negative_updates = 0
            proto_repeated_wrong = 0
            last_attempts.clear()
            for key in list(proto_state):
                if key[0] == seq:
                    del proto_state[key]
            print(f"EPISODIC_RESET {seq} mode=N22_PROTO", flush=True)
        if args.variant == "N23":
            corr_total = 0
            n23_query_cache.clear()
            for key in list(n23_correction_queries):
                if key[0] == seq:
                    del n23_correction_queries[key]
            print(f"EPISODIC_RESET {seq} mode=N23", flush=True)
        backend.start_video(str(DT / "train" / seq / "img1"))
        iw, ih = backend._frame_w, backend._frame_h
        gt = get_gt(seq)
        p0 = load_p0(seq)
        num_frames = max(gt.keys()) + 1 if gt else 0
        if args.limit_frames:
            num_frames = min(num_frames, args.limit_frames)
        # seed live memory with Human Root
        z = get_z(seq)
        qidx = {g: i for i, g in enumerate(z["qgids"])}
        for ff in sorted(gt):
            for g0 in gt[ff].gt_ids:
                if g0 in qidx and (seq, g0) not in mem_state:
                    mem_state[(seq, g0)]["slots"] = [
                        (ff, z["qemb"][qidx[g0]].copy(),
                         z["r0q"][qidx[g0]].copy())]
        t0 = time.time()
        verifier_fn = (
            (lambda cand: float(cand.get("n23_gate_prob", 0.0)))
            if args.variant == "N23" else None)
        reactivate_fn = (
            (lambda s, ff, b: propagate_track(s, ff, b, 120))
            if args.variant == "N23" else None)
        result = run_full_loop(seq, gt, p0, num_frames, cfg,
                               recovery_fn, verifier_fn, reactivate_fn, None,
                               shadow_start_fn if args.variant != "N23" else None,
                               shadow_step_fn if args.variant != "N23" else None)
        elapsed = time.time() - t0
        metrics = aggregate_metrics(seq, result, cfg)
        metrics["runtime_s"] = round(elapsed, 1)
        metrics["online_updates"] = corr_total
        if args.variant == "N22_PROTO":
            metrics["prototype_positive_updates"] = proto_updates
            metrics["prototype_negative_updates"] = proto_negative_updates
            metrics["prototype_repeated_wrong"] = proto_repeated_wrong
        with (out_dir / f"events_{seq}.jsonl").open(
                "w", encoding="utf-8") as f:
            for e in result["trace"]:
                e["sequence"] = seq
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        with (out_dir / f"transactions_{seq}.jsonl").open(
                "w", encoding="utf-8") as f:
            for t in result["transactions"]:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with (out_dir / f"metrics_{seq}.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            w.writeheader()
            w.writerow(metrics)
        if args.dump_trajectories:
            with (out_dir / f"trajectories_{seq}.jsonl").open(
                    "w", encoding="utf-8") as f:
                for rec in traj_dump:
                    if rec["sequence"] == seq:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done_marker.write_text("done\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        # close any leftover sessions
        shadow_groups.clear()
        try:
            backend._predictor.handle_request(dict(
                type="close_session", session_id=backend._session_id))
        except Exception:
            pass
        backend._predictor._all_inference_states.pop(
            backend._session_id, None)
        torch.cuda.empty_cache()
        traj_dump = [r for r in traj_dump if r["sequence"] != seq]
    if catil is not None:
        import gc
        del catil
        if catil_online is not None:
            del catil_online
        if catil_base is not None:
            del catil_base
        torch.cuda.empty_cache()
        gc.collect()
    runner.close()
    print("ONPOLICY_DONE", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
