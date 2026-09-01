#!/usr/bin/env python
"""N18 RouteC.1: cache per-frame GFN detections + identity features.

For every frame we store the final post-NMS detections and, for each
detection, (a) the frozen L2-normalized 2048-dim GFN embedding and (b) the
raw input of ``roi_heads.embedding_head`` (feat_res4=512d + feat_res5=1024d)
captured with a forward pre-hook.  The raw features let R0 train only the
identity projection head while the backbone/RPN/detection heads stay frozen.

For every identity we also cache the first-appearance human-like query crop's
features (same code path as FULL_LOOP_V0's H_i query).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gfn_recovery_model import load_model  # noqa: E402
from run_n18_full_loop_v0 import crop_query, load_gt  # noqa: E402

DT = Path("/path/to/dancetrack")
OUT = ROOT / "outputs/n18/route_c/gfn_cache"


def align_pre_features(pre_emb, final_emb, pre4, pre5, device):
    """Match captured pre-head feature rows to the final post-NMS rows.

    ``postprocess_boxes`` reorders/subsets the embeddings; the head is
    deterministic, so exact embedding matching recovers the alignment.
    """
    if final_emb.numel() == 0 or pre_emb.numel() == 0:
        return torch.zeros(0, 512), torch.zeros(0, 1024)
    with torch.inference_mode():
        # chunk to bound memory
        idx = []
        for i in range(final_emb.shape[0]):
            d = torch.norm(pre_emb - final_emb[i], dim=1)
            j = int(torch.argmin(d))
            if d[j] > 5e-2:
                raise RuntimeError(f"feature alignment failed d={float(d[j])}")
            idx.append(j)
    idx = torch.tensor(idx, dtype=torch.long)
    return pre4[idx], pre5[idx]


def run_gallery(model, hook_state, img, device):
    with torch.inference_mode():
        out = model([F.to_tensor(img).to(device)], None,
                    inference_mode="det")[0]
    boxes = out["det_boxes"].float().cpu().numpy().reshape(-1, 4)
    scores = out["det_scores"].float().cpu().numpy().reshape(-1)
    emb = out["det_emb"].float()
    pre4, pre5 = hook_state.pop("capture", (None, None))
    if pre4 is None or pre5 is None:
        raise RuntimeError("embedding_head hook did not fire")
    ge = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    pre_emb = None
    with torch.inference_mode():
        head = model.roi_heads.embedding_head
        pre_emb, _ = head({
            "feat_res4": pre4.float().to(device),
            "feat_res5": pre5.float().to(device)})
    f4, f5 = align_pre_features(pre_emb, ge, pre4, pre5, device)
    return (boxes.astype(np.float32), scores.astype(np.float32),
            ge.half().cpu().numpy(), f4.half().cpu().numpy(),
            f5.half().cpu().numpy())


def run_query(model, hook_state, img, device):
    with torch.inference_mode():
        out = model([F.to_tensor(img).to(device)], None,
                    inference_mode="det")[0]
    if out["det_emb"].shape[0] == 0:
        return None
    scores = out["det_scores"].float()
    qi = int(torch.argmax(scores))
    qe = out["det_emb"].float()[qi].reshape(1, -1)
    qe = qe / (qe.norm(dim=1, keepdim=True) + 1e-8)
    pre4, pre5 = hook_state.pop("capture", (None, None))
    if pre4 is None or pre5 is None:
        raise RuntimeError("embedding_head hook did not fire")
    with torch.inference_mode():
        head = model.roi_heads.embedding_head
        pre_emb, _ = head({
            "feat_res4": pre4.float().to(device),
            "feat_res5": pre5.float().to(device)})
    f4, f5 = align_pre_features(pre_emb, qe, pre4, pre5, device)
    return (qe.half().cpu().numpy()[0], f4.half().cpu().numpy()[0],
            f5.half().cpu().numpy()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--limit-frames", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    model, _, _, _, _ = load_model(device)
    model.eval()
    hook_state = {}

    def pre_hook(module, inputs):
        fd = inputs[0]
        f4 = fd["feat_res4"]
        f5 = fd["feat_res5"]
        if f4.dim() == 4:
            f4 = f4.flatten(start_dim=1)
        if f5.dim() == 4:
            f5 = f5.flatten(start_dim=1)
        hook_state["capture"] = (f4.float().cpu(), f5.float().cpu())

    handle = model.roi_heads.embedding_head.register_forward_pre_hook(
        pre_hook)
    try:
        seqs = sorted(args.seqs.split(",")) if args.seqs else []
        for si, seq in enumerate(seqs):
            OUT.mkdir(parents=True, exist_ok=True)
            gt = load_gt(seq)
            num_frames = max(gt.keys()) + 1 if gt else 0
            if args.limit_frames:
                num_frames = min(num_frames, args.limit_frames)
            frames, offsets = [], []
            boxes_l, scores_l, emb_l, f4_l, f5_l = [], [], [], [], []
            for f in range(num_frames):
                p = DT / "train" / seq / "img1" / f"{f + 1:08d}.jpg"
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                boxes, scores, emb, f4, f5 = run_gallery(
                    model, hook_state, img, device)
                frames.append(f)
                offsets.append(offsets[-1] + len(boxes) if offsets
                               else len(boxes))
                boxes_l.append(boxes)
                scores_l.append(scores)
                emb_l.append(emb)
                f4_l.append(f4)
                f5_l.append(f5)
                if (f + 1) % 200 == 0:
                    print(f"{seq} {f+1}/{num_frames} "
                          f"dets={len(boxes)}", flush=True)
            frames_arr = np.asarray(frames, dtype=np.int32)
            off_arr = np.asarray(offsets, dtype=np.int64)
            np.savez_compressed(
                OUT / f"{seq}.npz",
                frames=frames_arr, offsets=off_arr,
                boxes=np.concatenate(boxes_l) if boxes_l else
                np.zeros((0, 4), np.float32),
                scores=np.concatenate(scores_l) if scores_l else
                np.zeros((0,), np.float32),
                emb=np.concatenate(emb_l) if emb_l else
                np.zeros((0, 2048), np.float16),
                feat4=np.concatenate(f4_l) if f4_l else
                np.zeros((0, 512), np.float16),
                feat5=np.concatenate(f5_l) if f5_l else
                np.zeros((0, 1024), np.float16))

            # query cache: first GT appearance, human-like crop
            first = {}
            for f in sorted(gt):
                gf = gt[f]
                for gid, box in zip(gf.gt_ids, gf.boxes):
                    if gid not in first:
                        first[gid] = (f, np.asarray(box, dtype=float))
            gids, qf, qb, qe, qf4, qf5 = [], [], [], [], [], []
            for gid, (af, abox) in sorted(first.items()):
                p = DT / "train" / seq / "img1" / f"{af + 1:08d}.jpg"
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                qcrop = crop_query(img, abox)
                got = run_query(model, hook_state, qcrop, device)
                if got is None:
                    continue
                e, x4, x5 = got
                gids.append(gid)
                qf.append(af)
                qb.append(abox.astype(np.float32))
                qe.append(e)
                qf4.append(x4)
                qf5.append(x5)
            np.savez_compressed(
                OUT / f"{seq}_queries.npz",
                gids=np.asarray(gids, dtype=np.int32),
                qframe=np.asarray(qf, dtype=np.int32),
                qbox=np.asarray(qb, dtype=np.float32).reshape(-1, 4),
                qemb=np.asarray(qe, dtype=np.float16).reshape(-1, 2048),
                qfeat4=np.asarray(qf4, dtype=np.float16).reshape(-1, 512),
                qfeat5=np.asarray(qf5, dtype=np.float16).reshape(-1, 1024))
            print(f"CACHE_DONE seq={seq} frames={len(frames_arr)} "
                  f"dets={off_arr[-1]} queries={len(gids)}", flush=True)
    finally:
        handle.remove()
    print("ALL_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
