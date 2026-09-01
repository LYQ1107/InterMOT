#!/usr/bin/env python
"""Build the frozen SAM3 encoder-memory cache for N17 HTD training/eval."""

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "outputs/n17"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")


def clone_find_input(fin, img_id: int):
    out = copy.copy(fin)
    for field in fin.__dataclass_fields__:
        v = getattr(out, field)
        if isinstance(v, torch.Tensor):
            setattr(out, field, v.clone())
        elif isinstance(v, list) and v and all(isinstance(x, torch.Tensor) for x in v):
            setattr(out, field, [x.clone() for x in v])
    out.img_ids = torch.tensor([img_id], dtype=torch.long, device="cuda")
    return out


def clear_model_caches(model):
    for m in model.modules():
        for k in list(vars(m)):
            v = getattr(m, k, None)
            if k == "cache" and isinstance(v, dict):
                setattr(m, k, {})
            elif k == "coord_cache" and isinstance(v, dict):
                setattr(m, k, {})
            elif k == "compilable_cord_cache":
                setattr(m, k, None)
            if isinstance(v, dict):
                for kk, vv in list(v.items()):
                    if isinstance(vv, torch.Tensor) and torch.is_inference(vv):
                        v[kk] = vv.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3.model.geometry_encoders import Prompt

    with (OUT / "unique_frames.csv").open(encoding="utf-8") as f:
        frames = [(r["sequence"], int(r["frame"])) for r in csv.DictReader(f)]
    frames = sorted(set(frames))
    if args.limit:
        frames = frames[: args.limit]
    shard = frames[args.shard:: args.nshards]
    print(f"shard {args.shard}/{args.nshards} frames={len(shard)}", flush=True)

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.use_batched_grounding = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    image = model.detector
    cache_dir = OUT / "enc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_out = None
    empty_geo = None
    cur_seq = None

    def text_features():
        nonlocal text_out
        if text_out is None:
            with torch.no_grad():
                tx = model.detector.backbone.forward_text(["person"], device="cuda")
                text_out = {
                    "language_features": tx["language_features"].clone(),
                    "language_mask": tx["language_mask"].clone(),
                }
        return text_out

    t0 = time.time()
    done = 0
    for seq, f in shard:
        out_path = cache_dir / f"{seq}_{f}.npy"
        if out_path.exists():
            done += 1
            continue
        if cur_seq != seq:
            if cur_seq is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            backend.start_video(str(DT / "train" / seq / "img1"))
            cur_seq = seq
            clear_model_caches(model)
        state = backend._predictor._all_inference_states[backend._session_id]["state"]
        ib = state["input_batch"]
        fin = clone_find_input(ib.find_inputs[f], img_id=0)
        img_t = ib.img_batch.tensors[f].unsqueeze(0).clone().to("cuda")
        tx = text_features()
        bo = {
            "img_batch_all_stages": img_t,
            "language_features": tx["language_features"],
            "language_mask": tx["language_mask"],
        }
        if empty_geo is None:
            empty_geo = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device="cuda"),
                box_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                box_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
                point_embeddings=torch.zeros(0, 1, 2, device="cuda"),
                point_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                point_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
            )
        with torch.no_grad():
            prompt, pmask, bo2 = model.detector._encode_prompt(bo, fin, empty_geo)
            bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)
        mem = enc["encoder_hidden_states"].detach().cpu().to(torch.float16).numpy()
        np.save(out_path, mem)
        done += 1
        if done % 200 == 0:
            print(f"done={done}/{len(shard)} elapsed={time.time()-t0:.0f}s", flush=True)
    try:
        runner.close()
    except Exception:
        pass
    print(f"SHARD_DONE {args.shard} frames={done} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
