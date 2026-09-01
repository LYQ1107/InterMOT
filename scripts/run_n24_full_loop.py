#!/usr/bin/env python3
"""N24 strict-causal FULL_LOOP for a trained temporal reasoner.

This runner keeps the N18--N23 public-ID protocol and changes only the
recovery verifier. At a LOST event it ranks the current GFN gallery, starts
K isolated SAM3 shadow hypotheses, and waits until H observations are
available. The reasoner sees only shadow boxes/features through the current
frame. A successful decision commits the selected shadow at that frame and
then reactivates the official SAM3 trajectory. Human corrections update the
target's causal GFN/R0 memory only after the current frame has been scored.

GT is passed to ``run_full_loop`` only for simulated human input and
post-hoc metrics. It is not read by recovery, shadow construction, reasoner
scoring, or memory ranking.
"""

from __future__ import annotations

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

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from n24_temporal_reasoner import (  # noqa: E402
    C0MeanPrototype, C1Temporal, C2MultiPrototype, HMAX, K,
)
from run_n18_full_loop_v0 import load_gt, load_p0  # noqa: E402
from sam3_intermot.evaluation.full_loop_v0 import (  # noqa: E402
    LoopConfig, aggregate_metrics, iou, run_full_loop,
)

DT = Path("/path/to/dancetrack")
if not DT.exists():
    DT = Path("/path/to/dancetrack")
N15 = ROOT / "outputs/n15/n15_frozen.json"
GFN_CACHE = ROOT / "outputs/n18/route_c/gfn_cache"
R0_CACHE = ROOT / "outputs/n20/gfn_cache_r0"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")


def load_reasoner(name: str, path: Path, device):
    classes = {"C0": C0MeanPrototype, "C1": C1Temporal,
               "C2": C2MultiPrototype}
    model = classes[name]().to(device)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model


class GalleryCache:
    def __init__(self, seq: str):
        z = np.load(GFN_CACHE / f"{seq}.npz")
        qz = np.load(GFN_CACHE / f"{seq}_queries.npz")
        rz = np.load(R0_CACHE / f"{seq}.npz")
        self.frames = z["frames"].astype(np.int64)
        self.offsets = z["offsets"].astype(np.int64)
        self.boxes = z["boxes"].astype(np.float32)
        self.gfn = z["emb"].astype(np.float32)
        self.gfn /= np.linalg.norm(self.gfn, axis=1, keepdims=True) + 1e-8
        self.r0 = rz["r0g"].astype(np.float32)
        self.r0 /= np.linalg.norm(self.r0, axis=1, keepdims=True) + 1e-8
        self.qgids = [int(x) for x in qz["gids"]]
        self.qindex = {gid: i for i, gid in enumerate(self.qgids)}
        self.qgfn = qz["qemb"].astype(np.float32)
        self.qgfn /= np.linalg.norm(self.qgfn, axis=1, keepdims=True) + 1e-8
        self.qr0 = rz["r0q"].astype(np.float32)
        self.qr0 /= np.linalg.norm(self.qr0, axis=1, keepdims=True) + 1e-8
        z.close(); qz.close(); rz.close()

    def query(self, gid: int):
        i = self.qindex.get(int(gid))
        return None if i is None else (self.qgfn[i], self.qr0[i])

    def frame_slice(self, frame: int):
        pos = int(np.searchsorted(self.frames, int(frame)))
        if pos >= len(self.frames) or int(self.frames[pos]) != int(frame):
            return None
        lo = int(self.offsets[pos - 1]) if pos else 0
        hi = int(self.offsets[pos])
        return lo, hi

    def detection(self, frame: int, box):
        if box is None:
            return None
        sl = self.frame_slice(frame)
        if sl is None or sl[1] <= sl[0]:
            return None
        lo, hi = sl
        vals = np.asarray([iou(b, box) for b in self.boxes[lo:hi]])
        best = int(np.argmax(vals))
        if float(vals[best]) < 0.5:
            return None
        idx = lo + best
        return self.gfn[idx], self.r0[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--model", choices=["C0", "C1", "C2"], default="C1")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--split", choices=["cal10", "train30"], default="cal10")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--reactivation-horizon", type=int, default=120)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/n24/full_loop")
    ap.add_argument("--out-tag", default="n24_C1_h5")
    ap.add_argument("--shadow-timeout", type=int, default=0)
    args = ap.parse_args()
    if args.k != K:
        raise ValueError("N24 reasoner checkpoints use K=5")
    if args.h < 1 or args.h > HMAX:
        raise ValueError(f"h must be in [1,{HMAX}]")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model = load_reasoner(args.model, Path(args.model_path), device)

    frozen = json.loads(N15.read_text(encoding="utf-8"))["split"]
    if args.seqs:
        seqs = sorted(x for x in args.seqs.split(",") if x)
    else:
        seqs = sorted(frozen["calibration10" if args.split == "cal10" else "train30"])

    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner, parse_raw_outputs,
    )
    from sam3_intermot.detection_query.prompt_replay import (
        _best_delivery, invalidate_detector_prefetch,
        set_frame_geometric_prompt,
    )

    gfn, _, _, _, _ = load_model(f"cuda:{args.gpu}")
    gfn.eval()
    r0_head = ROOT / "outputs/n18/route_c/models/r0_best.pt"
    if r0_head.exists():
        gfn.roi_heads.embedding_head.load_state_dict(
            torch.load(r0_head, map_location="cpu", weights_only=False))
        gfn.roi_heads.embedding_head.eval()

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend.async_loading_frames = True
    backend._ensure_model()
    backend._predictor.model.eval()
    for p in backend._predictor.model.parameters():
        p.requires_grad_(False)
    state_seq = {"seq": None, "w": None, "h": None}
    zcache = {}
    mem_state = defaultdict(lambda: {"slots": []})
    shadow_groups = {}
    gt_cache = {}

    def get_z(seq):
        if seq not in zcache:
            zcache[seq] = GalleryCache(seq)
        return zcache[seq]

    def get_gt_cached(seq):
        if seq not in gt_cache:
            gt_cache[seq] = load_gt(seq)
        return gt_cache[seq]

    def ensure_video(seq):
        if state_seq["seq"] != seq:
            backend.start_video(str(DT / "train" / seq / "img1"))
            state_seq["seq"] = seq
            state_seq["w"] = backend._frame_w
            state_seq["h"] = backend._frame_h

    def propagate_track(seq, frame, box, horizon):
        ensure_video(seq)
        sid = backend._session_id
        try:
            backend._predictor.handle_request(dict(
                type="reset_session", session_id=sid))
            iw, ih = state_seq["w"], state_seq["h"]
            x1, y1, x2, y2 = [float(v) for v in box]
            backend._predictor.handle_request(dict(
                type="add_prompt", session_id=sid, frame_index=int(frame),
                text="person",
                bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw,
                                 (y2 - y1) / ih]],
                bounding_box_labels=[1], clear_old_boxes=True))
            state = backend._predictor._all_inference_states[sid]["state"]
            nf = int(state["num_frames"])
            prev = np.asarray(box, dtype=float).copy()
            records = {int(frame): prev.copy()}
            if frame + 1 < nf:
                set_frame_geometric_prompt(runner, frame + 1, None)
            req = dict(type="propagate_in_video", session_id=sid,
                       propagation_direction="forward",
                       start_frame_index=int(frame),
                       # Let the predictor stream the sequence and stop at
                       # the causal horizon below.  The multiplex backend's
                       # bounded max-frame path can produce an empty-token
                       # tensor after reset_session.
                       max_frame_num_to_track=None)
            for response in backend._predictor.handle_stream_request(request=req):
                ff = int(response["frame_index"])
                cands = parse_raw_outputs(response, frame_size=(iw, ih))
                boxes = [np.asarray(b, dtype=float) for _, b in cands]
                delivered = _best_delivery(prev, boxes)
                if delivered is None and boxes:
                    delivered = boxes[0]
                if delivered is not None:
                    prev = delivered.copy()
                records[ff] = delivered
                if ff >= frame + horizon:
                    break
                if ff + 1 < nf:
                    set_frame_geometric_prompt(runner, ff + 1, None)
                invalidate_detector_prefetch(runner, ff)
            return records
        except Exception as exc:
            print(f"N24_PROP_FAIL seq={seq} frame={frame} err={exc}", flush=True)
            return {int(frame): np.asarray(box, dtype=float).copy()}

    def recovery_fn(seq, frame, qbox, qframe, gid=None):
        z = get_z(seq)
        sl = z.frame_slice(frame)
        if sl is None or sl[1] <= sl[0] or gid is None:
            return None
        root = z.query(gid)
        if root is None:
            return None
        st = mem_state[(seq, int(gid))]
        if not st["slots"]:
            st["slots"].append((int(qframe), root[0].copy(), root[1].copy()))
        lo, hi = sl
        sims = np.maximum.reduce([z.gfn[lo:hi] @ slot[1]
                                   for slot in st["slots"][-3:]])
        order = np.argsort(-sims)
        candidates = []
        for idx in order[:K]:
            candidates.append({"box": z.boxes[lo + int(idx)].astype(float).copy(),
                               "rank": len(candidates) + 1})
        if not candidates:
            return None
        return {"candidates": candidates, "box": candidates[0]["box"],
                "n_dets": hi - lo}

    def shadow_start_fn(seq, frame, gid, qbox, qframe):
        rec = recovery_fn(seq, frame, qbox, qframe, gid)
        if rec is None or len(rec["candidates"]) != K:
            return None
        z = get_z(seq)
        root = z.query(gid)
        if root is None:
            return None
        sid = uuid.uuid4().hex
        shadow_groups[sid] = {
            "seq": seq, "f0": int(frame), "gid": int(gid),
            "root": np.concatenate([root[0], root[1]]).astype(np.float32),
            "candidates": [np.asarray(c["box"], dtype=float)
                           for c in rec["candidates"]],
            "trajectories": None,
        }
        return sid

    def shadow_step_fn(seq, frame, gid, sid, elapsed):
        grp = shadow_groups.get(sid)
        if grp is None:
            return {"verdict": "REJECT"}
        # At elapsed=h the current frame is f0+h-1; using horizon h-1 makes
        # the input exactly f0..current and never reads a later frame.
        if int(elapsed) < args.h:
            return {"verdict": "PENDING"}
        if grp["trajectories"] is None:
            horizon = max(0, int(frame) - int(grp["f0"]))
            grp["trajectories"] = [
                propagate_track(grp["seq"], grp["f0"], box, horizon)
                for box in grp["candidates"]
            ]
        vis = np.zeros((K, HMAX, 4096), dtype=np.float32)
        mask = np.zeros((K, HMAX), dtype=np.float32)
        z = get_z(seq)
        for k, traj in enumerate(grp["trajectories"]):
            for ff, box in sorted(traj.items()):
                step = int(ff) - int(grp["f0"])
                if step < 0 or step >= HMAX or ff > int(frame):
                    continue
                de = z.detection(int(ff), box)
                if de is None:
                    continue
                vis[k, step] = np.concatenate([de[0], de[1]])
                mask[k, step] = 1.0
        x = torch.from_numpy(vis[None]).to(device)
        m = torch.from_numpy(mask[None]).to(device)
        root = torch.from_numpy(grp["root"][None]).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model.forward_with_root(x, m, root), dim=1)[0].cpu().numpy()
        best = int(np.argmax(probs))
        print(f"N24_DECIDE seq={seq} frame={frame} gid={gid} model={args.model} "
              f"probs={[round(float(v),4) for v in probs]} mask={mask.sum(1).astype(int).tolist()}",
              flush=True)
        shadow_groups.pop(sid, None)
        if best < 1 or float(probs[best]) < args.threshold:
            return {"verdict": "REJECT", "probs": probs.tolist()}
        others = np.delete(probs, best)
        if float(probs[best] - others.max()) < args.margin:
            return {"verdict": "REJECT", "probs": probs.tolist()}
        chosen = grp["trajectories"][best - 1]
        chosen_box = chosen.get(int(frame))
        if chosen_box is None:
            chosen_box = grp["candidates"][best - 1]
        # The official track starts at the causal commit frame. It may read
        # later frames only for post-commit delivery, never for this verdict.
        official = propagate_track(seq, int(frame), chosen_box,
                                   args.reactivation_horizon)
        return {"verdict": "ACCEPT", "traj": official,
                "box": chosen_box, "commit_frame": int(frame),
                "probs": probs.tolist(), "best": best}

    def on_correction(seq, frame, gid, public_id, corrected_box):
        # This callback is invoked by run_full_loop only after the current
        # frame's decision/output has been evaluated.
        z = get_z(seq)
        de = z.detection(frame, corrected_box)
        if de is None:
            return
        st = mem_state[(seq, int(gid))]
        st["slots"].append((int(frame), de[0].copy(), de[1].copy()))
        if len(st["slots"]) > 3:
            st["slots"] = st["slots"][-3:]

    timeout = args.shadow_timeout or (args.h + 2)
    cfg = LoopConfig(
        lost_streak=3, retry_interval=5,
        reactivation_horizon=args.reactivation_horizon,
        anchor_policy="first", shadow_mode=True, shadow_horizon=args.h,
        shadow_timeout=timeout, enable_recovery=True,
        enable_reactivation=True, on_correction=on_correction,
    )

    # N24 operates exclusively through the K-way shadow verifier. The legacy
    # core has a fallback path for a recovery result that cannot start a
    # complete shadow group; provide an explicit reject-only callable so a
    # short gallery never reaches a None verifier or commits unscreened.
    def reject_legacy_recovery(_candidate):
        return 0.0

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{args.out_tag}.csv"
    all_metrics = []
    for seq in seqs:
        ensure_video(seq)
        gt = get_gt_cached(seq)
        p0 = load_p0(seq)
        z = get_z(seq)
        qidx = {g: i for i, g in enumerate(z.qgids)}
        # Human-root initialization is the legal first labelled observation.
        for ff in sorted(gt):
            for gid0 in gt[ff].gt_ids:
                if gid0 in qidx and (seq, int(gid0)) not in mem_state:
                    qi = qidx[gid0]
                    mem_state[(seq, int(gid0))]["slots"] = [
                        (int(ff), z.qgfn[qi].copy(), z.qr0[qi].copy())]
        num_frames = max(gt.keys()) + 1 if gt else 0
        if args.limit_frames:
            num_frames = min(num_frames, args.limit_frames)
        start = time.time()
        result = run_full_loop(
            seq, gt, p0, num_frames, cfg,
            recovery_fn, reject_legacy_recovery, None, None,
            shadow_start_fn, shadow_step_fn,
        )
        metrics = aggregate_metrics(seq, result, cfg)
        metrics.update({"model": args.model, "h": args.h,
                        "threshold": args.threshold, "margin": args.margin,
                        "runtime_s": round(time.time() - start, 1),
                        "gpu": args.gpu})
        all_metrics.append(metrics)
        event_path = out_dir / f"events_{seq}.jsonl"
        with event_path.open("w", encoding="utf-8") as handle:
            for e in result["trace"]:
                e["sequence"] = seq
                handle.write(json.dumps(e, ensure_ascii=False) + "\n")
        tx_path = out_dir / f"transactions_{seq}.jsonl"
        with tx_path.open("w", encoding="utf-8") as handle:
            for t in result["transactions"]:
                handle.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        shadow_groups.clear()
        try:
            backend._predictor.handle_request(dict(
                type="close_session", session_id=backend._session_id))
        except Exception:
            pass
        backend._predictor._all_inference_states.pop(
            backend._session_id, None)
        torch.cuda.empty_cache()
        state_seq["seq"] = None
    if all_metrics:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            fields = list(all_metrics[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(all_metrics)
    (out_dir / f"summary_{args.out_tag}.json").write_text(
        json.dumps({"model": args.model, "h": args.h, "seqs": seqs,
                    "metrics": all_metrics}, indent=2), encoding="utf-8")
    runner.close()
    print(json.dumps({"model": args.model, "h": args.h,
                      "seqs": seqs, "metrics_path": str(metrics_path)},
                     indent=2), flush=True)
    print("N24_FULL_LOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
