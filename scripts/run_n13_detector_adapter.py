#!/usr/bin/env python
"""N13 detector-adapter fallback: lightweight online LoRA on the official
SAM3.1 detector, trained on the human frame, evaluated on future frames."""

import argparse
import csv
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(".")
OUT = ROOT / "outputs/n13"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DEFAULT_SEQS = (
    "dancetrack0074 dancetrack0075 dancetrack0080 dancetrack0082 "
    "dancetrack0083 dancetrack0086 dancetrack0087 dancetrack0096"
)


def load_events(seq: str):
    path = ROOT / "outputs/n10/real/human_b8" / seq / "interaction_events.jsonl"
    evs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("accepted") and e.get("event_type") == "TRUE_MISS_NEW":
            evs.append(e)
    return evs


def cxcywh_norm(box, iw, ih):
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.asarray(
        [(x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, (x2 - x1) / iw, (y2 - y1) / ih],
        dtype=float,
    )


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def cxcywh_to_xyxy(cx, cy, w, h):
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def clone_find_input(fin, img_id: int):
    """Return a normal-tensor copy of a FindStage for autograd."""
    out = copy.copy(fin)
    for field in fin.__dataclass_fields__:
        v = getattr(out, field)
        if isinstance(v, torch.Tensor):
            setattr(out, field, v.clone())
        elif isinstance(v, list) and v and all(isinstance(x, torch.Tensor) for x in v):
            setattr(out, field, [x.clone() for x in v])
    out.img_ids = torch.tensor([img_id], dtype=torch.long, device="cuda")
    return out


def clear_model_caches(model) -> int:
    """Drop module dict caches filled under inference mode (position encoding,
    decoder boxRPB coords).  These inference tensors break autograd later."""
    n = 0
    for m in model.modules():
        for k in list(vars(m)):
            v = getattr(m, k, None)
            if k == "cache" and isinstance(v, dict):
                setattr(m, k, {})
                n += 1
            elif k == "coord_cache" and isinstance(v, dict):
                setattr(m, k, {})
                n += 1
            elif k == "compilable_cord_cache":
                setattr(m, k, None)
                n += 1
            if isinstance(v, dict):
                for kk, vv in list(v.items()):
                    if isinstance(vv, torch.Tensor) and torch.is_inference(vv):
                        v[kk] = vv.clone()
                        n += 1
    return n


def scan_inference_tensors(model, path="", hits=None, depth=0):
    if hits is None:
        hits = []
    if depth > 8:
        return hits
    if isinstance(model, torch.Tensor):
        if torch.is_inference(model):
            hits.append(path or "tensor")
        return hits
    if isinstance(model, torch.nn.Module):
        for k, v in vars(model).items():
            if k in ("_parameters", "_buffers", "_modules", "_backward_hooks",
                     "_forward_hooks", "_forward_pre_hooks", "_state_dict_hooks",
                     "_load_state_dict_pre_hooks", "_non_persistent_buffers_set",
                     "_forward_hooks_with_kwargs", "_forward_pre_hooks_with_kwargs"):
                continue
            scan_inference_tensors(v, f"{path}.{k}" if path else k, hits, depth + 1)
        return hits
    if isinstance(model, dict):
        for k, v in model.items():
            scan_inference_tensors(v, f"{path}[{k}]", hits, depth + 1)
        return hits
    if isinstance(model, (list, tuple)):
        for i, v in enumerate(model):
            scan_inference_tensors(v, f"{path}[{i}]", hits, depth + 1)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--event-idx", type=int, default=0)
    ap.add_argument("--seqs", default=DEFAULT_SEQS)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--target-mode", default="target",
                    choices=["target", "other"])
    ap.add_argument("--preserve-weight", type=float, default=0.0)
    ap.add_argument("--out", default="detector_adapter")
    args = ap.parse_args()

    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else args.gpu)

    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.adaptation.lora import inject_lora, lora_parameter_count
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.detection_query.prompt_replay import run_pdr_episode, recall_at, false_capture

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    ds = DanceTrackDataset(
        "/path/to/dancetrack", sequences=[], split="train"
    )

    # Build model once.
    backend = runner._ensure_backend()
    backend.start_video(str(
        Path("/path/to/dancetrack/train")
        / DEFAULT_SEQS.split()[0] / "img1"
    ))
    backend.close()
    model = backend._predictor.model
    model.use_batched_grounding = False

    prefixes = (
        "detector.transformer.decoder.layers",
        "detector.transformer.decoder.bbox_embed",
        "detector.dot_prod_scoring",
    )
    modified, lora_params = inject_lora(model, prefixes, r=args.rank, alpha=args.alpha)
    print(json.dumps({
        "n_modified": len(modified),
        "n_lora_params": lora_parameter_count(lora_params),
        "surfaces": sorted(set(m.split(".")[0] + "." + m.split(".")[1]
                               for m in modified))[:8],
    }), flush=True)

    init_lora = [p.detach().clone() for p in lora_params]

    def reset_lora():
        with torch.no_grad():
            for p, v in zip(lora_params, init_lora):
                p.copy_(v)

    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)
    seqs = args.seqs.split()
    rows = []
    for seq in seqs:
        evs = load_events(seq)
        if not evs:
            continue
        ev = evs[args.event_idx % len(evs)]
        gid = int(ev["dataset_gt_id"])
        human_box = np.asarray(ev["gt_box"], dtype=float)
        gt = ds.load_gt(seq)
        video = str(Path("/path/to/dancetrack/train") / seq / "img1")

        # A0 baseline (zero LoRA).
        reset_lora()
        ep0 = None
        if not os.environ.get("N13_SKIP_A0"):
            ep0 = run_pdr_episode(
                runner, seq, int(ev["frame"]), ev["event_type"], gid,
                human_box, "one_shot", gt, horizon=args.horizon,
            )

        # Train adapter on the human frame.
        reset_lora()
        backend.start_video(video)
        iw, ih = backend._frame_w, backend._frame_h
        state = backend._predictor._all_inference_states[backend._session_id]["state"]
        t = int(ev["frame"])
        fin = clone_find_input(state["input_batch"].find_inputs[t], img_id=0)
        img_batch = state["input_batch"].img_batch
        if hasattr(img_batch, "tensors"):
            img_t = img_batch.tensors[t].unsqueeze(0).clone().to("cuda")
        elif isinstance(img_batch, torch.Tensor):
            img_t = img_batch[t].unsqueeze(0).clone().to("cuda")
        else:
            img_t = torch.as_tensor(img_batch[t]).unsqueeze(0).clone().to("cuda")
        with torch.no_grad():
            text_out = model.detector.backbone.forward_text(["person"], device="cuda")
        backbone_out = {
            "img_batch_all_stages": img_t,
            "language_features": text_out["language_features"].clone(),
            "language_mask": text_out["language_mask"].clone(),
        }
        from sam3.model.geometry_encoders import Prompt
        empty_geo = Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device="cuda"),
            box_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
            box_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
            point_embeddings=torch.zeros(0, 1, 2, device="cuda"),
            point_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
        )
        gt_cxcywh = torch.as_tensor(
            cxcywh_norm(human_box, iw, ih), dtype=torch.float32, device="cuda"
        )
        train_t0 = time.time()
        shapes = None
        n_cleared = clear_model_caches(model)
        if os.environ.get("N13_DEBUG"):
            print("cleared_caches", n_cleared, flush=True)
            hits = scan_inference_tensors(model)
            print("inf_tensor_hits", len(hits), hits[:20], flush=True)
        with torch.no_grad():
            prompt, prompt_mask, bo2 = model.detector._encode_prompt(
                backbone_out, fin, empty_geo
            )
            bo2, encoder_out, _ = model.detector._run_encoder(
                bo2, fin, prompt, prompt_mask
            )

        def cl(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            if isinstance(x, (list, tuple)):
                return [cl(v) for v in x]
            if isinstance(x, dict):
                return {k: cl(v) for k, v in x.items()}
            return x

        enc2 = cl(encoder_out)
        prompt2 = prompt.clone()
        pmask2 = prompt_mask.clone()
        mem2 = enc2["encoder_hidden_states"]
        if os.environ.get("N13_DEBUG"):
            print("cache_clear", n_cleared, flush=True)
            for _k in ("encoder_hidden_states", "pos_embed", "padding_mask",
                       "valid_ratios", "level_start_index", "spatial_shapes"):
                _v = enc2[_k]
                if isinstance(_v, torch.Tensor):
                    print("ENC2", _k, torch.is_inference(_v), tuple(_v.shape), flush=True)
            print("REFPT", torch.is_inference(
                model.detector.transformer.decoder.reference_points.weight
            ), flush=True)
        # The decoder caches boxRPB coordinate tensors on its first forward.
        # Since no inference-mode decoder forward has run in this fresh
        # session, they will be created as normal tensors under enable_grad.
        _dec = model.detector.transformer.decoder
        for _k in list(vars(_dec)):
            if "cache" in _k.lower():
                try:
                    setattr(_dec, _k, None if not isinstance(
                        getattr(_dec, _k), dict) else {})
                except Exception:
                    pass
        if hasattr(_dec, "compilable_cord_cache"):
            _dec.compilable_cord_cache = None
        if hasattr(_dec, "coord_cache"):
            _dec.coord_cache = {}
        if os.environ.get("N13_DEBUG"):
            for k, v in enc2.items():
                if isinstance(v, torch.Tensor) and torch.is_inference(v):
                    print("ENC_INF", k, flush=True)
            for nm, tv in [("mem", mem2), ("pos", enc2["pos_embed"]),
                           ("pad", enc2["padding_mask"]),
                           ("prompt", prompt2), ("pmask", pmask2)]:
                if isinstance(tv, torch.Tensor):
                    print("CLONE", nm, torch.is_inference(tv), flush=True)
            print("query_embed inf?",
                  torch.is_inference(model.detector.transformer.decoder.query_embed.weight),
                  "ref_points inf?",
                  torch.is_inference(
                      model.detector.transformer.decoder.reference_points.weight
                  ), flush=True)
        first_logits = None
        init_scores = None
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                out2 = {"encoder_hidden_states": mem2}
                out2, hs = model.detector._run_decoder(
                    pos_embed=enc2["pos_embed"],
                    memory=mem2,
                    src_mask=enc2["padding_mask"],
                    out=out2,
                    prompt=prompt2,
                    prompt_mask=pmask2,
                    encoder_out=enc2,
                )
                if shapes is None:
                    shapes = {k: tuple(v.shape) for k, v in out2.items()}
                boxes = out2["pred_boxes"][0].float()
                scores = out2["pred_logits"][0].float()
                if scores.dim() == 2:
                    scores = scores[:, 0]
                # target query: max IoU with human box
                gt_xyxy = cxcywh_to_xyxy(*gt_cxcywh.tolist())
                bx1, by1 = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
                bx2, by2 = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
                gx1, gy1, gx2, gy2 = gt_xyxy
                ix1 = torch.maximum(bx1, torch.as_tensor(gx1, device="cuda"))
                iy1 = torch.maximum(by1, torch.as_tensor(gy1, device="cuda"))
                ix2 = torch.minimum(bx2, torch.as_tensor(gx2, device="cuda"))
                iy2 = torch.minimum(by2, torch.as_tensor(gy2, device="cuda"))
                inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                ua = (
                    (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
                    + (gx2 - gx1) * (gy2 - gy1)
                    - inter
                    + 1e-9
                )
                ious = inter / ua
                target_idx = int(ious.argmax().item())
                loss_score = F.binary_cross_entropy_with_logits(
                    scores[target_idx].unsqueeze(0),
                    torch.ones(1, device="cuda"),
                )
                loss_box = F.l1_loss(boxes[target_idx], gt_cxcywh)
                loss = loss_score + 0.1 * loss_box
                if step == 0:
                    init_scores = scores.detach().clone()
                if args.preserve_weight > 0:
                    other_mask = torch.ones_like(scores, dtype=torch.bool)
                    other_mask[target_idx] = False
                    keep = other_mask & (init_scores > 0.3)
                    if keep.any():
                        loss = loss + args.preserve_weight * F.mse_loss(
                            scores[keep], init_scores[keep].detach()
                        )
                if os.environ.get("N13_DEBUG"):
                    if step == 0:
                        first_logits = scores.detach().clone()
                    if step % 2 == 0 or step == args.steps - 1:
                        print("STEP", step, "loss", round(float(loss.item()), 5),
                              "target", target_idx, flush=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
        train_sec = time.time() - train_t0
        if os.environ.get("N13_DEBUG") and first_logits is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out2 = {"encoder_hidden_states": mem2}
                out2, _ = model.detector._run_decoder(
                    pos_embed=enc2["pos_embed"],
                    memory=mem2,
                    src_mask=enc2["padding_mask"],
                    out=out2,
                    prompt=prompt2,
                    prompt_mask=pmask2,
                    encoder_out=enc2,
                )
                last_logits = out2["pred_logits"][0].float()
            print("LOGITS_MAX_DIFF", round(float(
                (last_logits - first_logits.float()).abs().max()), 6),
                "lora_l2", round(float(sum(
                    (p.detach() ** 2).sum().item() for p in lora_params
                ) ** 0.5), 6), flush=True)
        backend.close()

        # Adapted evaluation (same event, fresh session).
        ep1 = run_pdr_episode(
            runner, seq, t, ev["event_type"], gid,
            human_box, "one_shot", gt, horizon=args.horizon,
        )

        row = {
            "sequence": seq, "frame": t, "gid": gid,
            "target_mode": args.target_mode,
            "steps": args.steps, "lr": args.lr, "rank": args.rank,
            "train_seconds": round(train_sec, 2),
            "shapes": json.dumps(shapes),
            "baseline_delivered_1": 0.0 if ep0 is None else round(recall_at(ep0, gt, 1, "delivered"), 3),
            "baseline_delivered_3": 0.0 if ep0 is None else round(recall_at(ep0, gt, 3, "delivered"), 3),
            "baseline_delivered_5": 0.0 if ep0 is None else round(recall_at(ep0, gt, 5, "delivered"), 3),
            "baseline_delivered_10": 0.0 if ep0 is None else round(recall_at(ep0, gt, 10, "delivered"), 3),
            "baseline_delivered_30": 0.0 if ep0 is None else round(recall_at(ep0, gt, 30, "delivered"), 3),
            "adapted_delivered_1": round(recall_at(ep1, gt, 1, "delivered"), 3),
            "adapted_delivered_3": round(recall_at(ep1, gt, 3, "delivered"), 3),
            "adapted_delivered_5": round(recall_at(ep1, gt, 5, "delivered"), 3),
            "adapted_delivered_10": round(recall_at(ep1, gt, 10, "delivered"), 3),
            "adapted_delivered_30": round(recall_at(ep1, gt, 30, "delivered"), 3),
            "baseline_admission_30": 0.0 if ep0 is None else round(recall_at(ep0, gt, 30, "admission"), 3),
            "adapted_admission_30": round(recall_at(ep1, gt, 30, "admission"), 3),
            "baseline_false_capture": None if ep0 is None else round(float(np.mean([
                false_capture(ep0, gt, f) for f in range(t + 1, t + args.horizon + 1)
                if gt.get(f) is not None and gid in gt[f].gt_ids
            ])), 3) if any(
                gt.get(f) is not None and gid in gt[f].gt_ids
                for f in range(t + 1, t + args.horizon + 1)
            ) else None,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    runner.close()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.out}_events.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "n_events": len(rows),
        "n_lora_params": lora_parameter_count(lora_params),
        "mean_delta_30": round(float(np.mean([
            r["adapted_delivered_30"] - r["baseline_delivered_30"] for r in rows
        ])), 4),
        "mean_baseline_30": round(float(np.mean(
            [r["baseline_delivered_30"] for r in rows])), 4),
        "mean_adapted_30": round(float(np.mean(
            [r["adapted_delivered_30"] for r in rows])), 4),
    }
    with (OUT / f"{args.out}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
