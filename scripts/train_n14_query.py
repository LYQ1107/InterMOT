"""Train HumanWriteEncoder (F0: detector decoder frozen).

One human correction at frame t -> ROI feature -> Q_i^det; Q_i^det is injected
into a reserved detector query slot at future frames; loss supervises the slot
output (score + box) with GT at the future frames.  No future GT at
inference; GT is training labels only.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(".")
OUT = ROOT / "outputs/n14"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")


def clone_find_input(fin, img_id: int):
    import copy

    out = copy.copy(fin)
    for field in fin.__dataclass_fields__:
        v = getattr(out, field)
        if isinstance(v, torch.Tensor):
            setattr(out, field, v.clone())
        elif isinstance(v, list) and v and all(
            isinstance(x, torch.Tensor) for x in v
        ):
            setattr(out, field, [x.clone() for x in v])
    out.img_ids = torch.tensor([img_id], dtype=torch.long, device="cuda")
    return out


def clear_model_caches(model) -> int:
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


def deep_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return [deep_clone(v) for v in x]
    if isinstance(x, dict):
        return {k: deep_clone(v) for k, v in x.items()}
    return x


def cxcywh_norm(box, iw, ih):
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.asarray(
        [(x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih, (x2 - x1) / iw, (y2 - y1) / ih],
        dtype=float,
    )


def cxcywh_to_xyxy(cx, cy, w, h):
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--manifest", default="outputs/n14/episode_manifest.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--box-weight", type=float, default=5.0)
    ap.add_argument("--score-weight", type=float, default=1.0)
    ap.add_argument("--contrastive-weight", type=float, default=1.0)
    ap.add_argument("--n-neg", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--slot", type=int, default=199)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-delta", type=int, default=0,
                    help="keep only future_frame-human_frame <= max-delta (0=all)")
    ap.add_argument("--out", default="human_write_encoder_f0")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.persistent_identity import (
        HumanWriteEncoder,
        SlotHeadAdapter,
        build_ref_boxes_with_queries,
        build_tgt_with_queries,
        roi_pool_feature,
        run_decoder_with_tgt,
    )
    from sam3.model.geometry_encoders import Prompt

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.use_batched_grounding = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    image = model.detector
    d_model = image.transformer.decoder.query_embed.weight.shape[1]
    encoder = HumanWriteEncoder(d_model=d_model, hidden=args.hidden).cuda()
    adapter = SlotHeadAdapter(d_model=d_model, hidden=args.hidden // 4).cuda()
    trainable = list(encoder.parameters()) + list(adapter.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    # Load manifest grouped by sequence.
    rows = []
    with open(ROOT / args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if args.max_delta and (
                int(r["future_frame"]) - int(r["human_frame"]) > args.max_delta
            ):
                continue
            rows.append(r)
    if args.max_samples:
        rows = rows[: args.max_samples]
    by_seq = {}
    for r in rows:
        by_seq.setdefault(r["sequence"], []).append(r)
    print(
        f"samples={len(rows)} seqs={sorted(by_seq)} d_model={d_model}",
        flush=True,
    )

    ds = DanceTrackDataset(str(DT), sequences=[], split="train")
    text_out = None
    frame_cache = {}
    seq_video = {}
    empty_geo = None

    def text_features():
        nonlocal text_out
        if text_out is None:
            with torch.no_grad():
                text_out = model.detector.backbone.forward_text(
                    ["person"], device="cuda"
                )
                text_out = {
                    "language_features": text_out["language_features"].clone(),
                    "language_mask": text_out["language_mask"].clone(),
                }
        return text_out

    def encoder_features(seq, f):
        key = (seq, f)
        if key in frame_cache:
            return frame_cache[key]
        video = seq_video.get(seq)
        if video is None:
            if seq_video and backend._session_id is not None:
                prev = next(iter(seq_video))
                if prev != seq:
                    backend.close()
            backend.start_video(
                str(DT / "train" / seq / "img1")
            )
            seq_video[seq] = True
            clear_model_caches(model)
        state = backend._predictor._all_inference_states[
            backend._session_id
        ]["state"]
        ib = state["input_batch"]
        fin = clone_find_input(ib.find_inputs[f], img_id=0)
        if hasattr(ib.img_batch, "tensors"):
            img_t = ib.img_batch.tensors[f].unsqueeze(0).clone().to("cuda")
        elif isinstance(ib.img_batch, torch.Tensor):
            img_t = ib.img_batch[f].unsqueeze(0).clone().to("cuda")
        else:
            img_t = torch.as_tensor(ib.img_batch[f]).unsqueeze(0).clone().to("cuda")
        tx = text_features()
        backbone_out = {
            "img_batch_all_stages": img_t,
            "language_features": tx["language_features"],
            "language_mask": tx["language_mask"],
        }
        nonlocal empty_geo
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
            prompt, prompt_mask, bo2 = model.detector._encode_prompt(
                backbone_out, fin, empty_geo
            )
            bo2, enc, _ = model.detector._run_encoder(
                bo2, fin, prompt, prompt_mask
            )
        feat = {
            "enc": deep_clone(enc),
            "prompt": prompt.clone(),
            "pmask": prompt_mask.clone(),
        }
        feat = {k: (v.cpu() if isinstance(v, torch.Tensor) else deep_clone(v))
                for k, v in feat.items()}
        frame_cache[key] = feat
        return feat

    def to_cuda(feat):
        out = {}
        for k, v in feat.items():
            if isinstance(v, dict):
                out[k] = {
                    kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, torch.Tensor):
                out[k] = v.cuda()
            else:
                out[k] = v
        return out

    history = []
    iw, ih = 1920, 1080
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    clear_model_caches(model)
    dec = image.transformer.decoder
    for k in list(vars(dec)):
        if "cache" in k.lower():
            try:
                setattr(dec, k, None if not isinstance(
                    getattr(dec, k), dict) else {})
            except Exception:
                pass

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = []
        pos_ious = []
        for r in rows:
            seq = r["sequence"]
            t = int(r["human_frame"])
            f = int(r["future_frame"])
            vis = int(r["target_visible"]) == 1
            hb = np.asarray(
                [r["human_box_x1"], r["human_box_y1"],
                 r["human_box_x2"], r["human_box_y2"]],
                dtype=float,
            )
            ref_box = hb
            if not vis and r.get("neg_box_x1"):
                ref_box = np.asarray(
                    [r["neg_box_x1"], r["neg_box_y1"],
                     r["neg_box_x2"], r["neg_box_y2"]],
                    dtype=float,
                )
            fb = np.asarray(
                [r["future_box_x1"], r["future_box_y1"],
                 r["future_box_x2"], r["future_box_y2"]],
                dtype=float,
            )
            ft = encoder_features(seq, t)
            ff = encoder_features(seq, f)
            enc_t = to_cuda(ft["enc"])
            enc_f = to_cuda(ff["enc"])
            prompt_f = ff["prompt"].cuda()
            pmask_f = ff["pmask"].cuda()

            box_norm = np.asarray(
                [hb[0] / iw, hb[1] / ih, hb[2] / iw, hb[3] / ih], dtype=float
            )
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                roi = roi_pool_feature(
                    enc_t["encoder_hidden_states"],
                    enc_t,
                    box_norm,
                )
                q = encoder(roi.float()).to(torch.float32)
                ref = torch.as_tensor(
                    cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda",
                )
                neg_score_mean = 0.0
                if vis:
                    roi_pos = roi_pool_feature(
                        enc_f["encoder_hidden_states"], enc_f, box_norm
                    )
                    gt_box_norm = np.asarray(
                        [fb[0] / iw, fb[1] / ih, fb[2] / iw, fb[3] / ih],
                        dtype=float,
                    )
                    roi_cand = roi_pool_feature(
                        enc_f["encoder_hidden_states"], enc_f, gt_box_norm
                    )
                    dbox, s_pos = adapter(
                        q.unsqueeze(0), roi_cand.unsqueeze(0),
                        roi_pos.unsqueeze(0),
                        ref.unsqueeze(0),
                    )
                    b = (ref.unsqueeze(0) + dbox)[0]
                    s = s_pos[0, 0]
                    loss = args.score_weight * (
                        F.binary_cross_entropy_with_logits(
                            s.unsqueeze(0),
                            torch.ones(1, device="cuda"),
                        )
                    )
                    gt_cxcywh = torch.as_tensor(
                        cxcywh_norm(fb, iw, ih), dtype=torch.float32,
                        device="cuda",
                    )
                    loss = loss + args.box_weight * F.l1_loss(
                        b, gt_cxcywh
                    )
                    slot_box = cxcywh_to_xyxy(
                        *b.detach().cpu().tolist()
                    ) * np.asarray([iw, ih, iw, ih])
                    pos_ious.append(iou_xyxy(slot_box, fb))
                    neg_boxes = json.loads(r.get("neg_boxes") or "[]")[
                        : args.n_neg
                    ]
                    neg_scores = []
                    qn = F.normalize(q.unsqueeze(0), dim=-1)
                    fp = F.normalize(
                        adapter.match_proj(
                            F.normalize(roi_cand.unsqueeze(0), dim=-1)
                        ),
                        dim=-1,
                    )
                    cos_pos = (qn * fp).sum(-1)
                    cos_negs = []
                    for nb in neg_boxes:
                        nb_arr = np.asarray(nb, dtype=float)
                        nb_norm = np.asarray(
                            [nb_arr[0] / iw, nb_arr[1] / ih,
                             nb_arr[2] / iw, nb_arr[3] / ih],
                            dtype=float,
                        )
                        roi_neg = roi_pool_feature(
                            enc_f["encoder_hidden_states"], enc_f, nb_norm
                        )
                        _, sn = adapter(
                            q.unsqueeze(0), roi_neg.unsqueeze(0),
                            roi_pos.unsqueeze(0),
                            ref.unsqueeze(0),
                        )
                        loss = loss + args.score_weight * (
                            F.binary_cross_entropy_with_logits(
                                sn.reshape(1),
                                torch.zeros(1, device="cuda"),
                            )
                        )
                        neg_scores.append(float(torch.sigmoid(sn).item()))
                        fn = F.normalize(
                            adapter.match_proj(
                                F.normalize(roi_neg.unsqueeze(0), dim=-1)
                            ),
                            dim=-1,
                        )
                        cos_negs.append((qn * fn).sum(-1))
                    if cos_negs:
                        cos_negs = torch.cat(cos_negs, dim=0)
                        logits_c = torch.cat(
                            [
                                cos_pos.reshape(1) / args.tau,
                                cos_negs / args.tau,
                            ]
                        )
                        info = -logits_c[0] + torch.logsumexp(logits_c, dim=0)
                        loss = loss + args.contrastive_weight * info
                    neg_score_mean = (
                        float(np.mean(neg_scores)) if neg_scores else 0.0
                    )
                else:
                    roi_pos = roi_pool_feature(
                        enc_f["encoder_hidden_states"], enc_f, box_norm
                    )
                    _, s_neg = adapter(
                        q.unsqueeze(0), roi_pos.unsqueeze(0),
                        roi_pos.unsqueeze(0),
                        ref.unsqueeze(0),
                    )
                    s = s_neg[0, 0]
                    loss = args.score_weight * (
                        F.binary_cross_entropy_with_logits(
                            s.unsqueeze(0),
                            torch.zeros(1, device="cuda"),
                        )
                    )
                    neg_score_mean = float(torch.sigmoid(s).item())
                loss.backward()
                if os.environ.get("N14_DEBUG_GRAD") and step < 3:
                    q_req = q.requires_grad
                    gnorm = sum(
                        float(
                            (
                                p.grad if p.grad is not None
                                else torch.zeros_like(p)
                            ).abs().sum().item()
                        )
                        for p in trainable
                    )
                    enc_g = sum(
                        float(
                            (
                                p.grad if p.grad is not None
                                else torch.zeros_like(p)
                            ).abs().sum().item()
                        )
                        for p in encoder.parameters()
                    )
                    adp_g = sum(
                        float(
                            (
                                p.grad if p.grad is not None
                                else torch.zeros_like(p)
                            ).abs().sum().item()
                        )
                        for p in adapter.parameters()
                    )
                    wnorm = float(
                        adapter.match_proj[-1].weight.detach().abs().sum().item()
                    )
                    sc_g = sum(
                        float(
                            (
                                p.grad if p.grad is not None
                                else torch.zeros_like(p)
                            ).abs().sum().item()
                        )
                        for p in [
                            *adapter.match_proj.parameters(),
                            adapter.score_scale,
                            adapter.score_bias,
                        ]
                    )
                    q_grad = (
                        float(q.grad.abs().sum().item())
                        if q.grad is not None else 0.0
                    )
                    print(
                        "STEP", step, "loss", round(float(loss.item()), 4),
                        "slot_score", round(float(torch.sigmoid(s).item()), 4),
                        "slot_box", [round(float(x), 4) for x in b[0].detach().cpu().tolist()],
                        "grad_abs_sum", round(gnorm, 6),
                        "enc_g", round(enc_g, 4), "adp_g", round(adp_g, 4),
                        "boxw", round(wnorm, 8), "sc_g", round(sc_g, 4),
                        "q_req", q_req,
                        "q_grad", round(q_grad, 4),
                        flush=True,
                    )
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step()
            losses.append(float(loss.item()))
            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "sequence": seq,
                    "gid": r["gid"],
                    "human_frame": t,
                    "future_frame": f,
                    "target_visible": int(vis),
                    "loss": round(float(loss.item()), 5),
                    "slot_score": round(float(torch.sigmoid(s).item()), 4),
                    "neg_score": round(neg_score_mean, 4),
                    "slot_iou": (
                        round(pos_ious[-1], 4) if vis and pos_ious else ""
                    ),
                }
            )
            step += 1
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_iou = float(np.mean(pos_ious)) if pos_ious else float("nan")
        print(
            f"epoch={epoch} loss={mean_loss:.4f} pos_iou={mean_iou:.4f} "
            f"steps={step} sec={time.time() - t0:.1f}",
            flush=True,
        )
        torch.save(
            {
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "adapter_state": adapter.state_dict(),
                "d_model": d_model,
                "slot": args.slot,
                "args": vars(args),
            },
            str(OUT / "models" / f"{args.out}_ep{epoch}.pt"),
        )

    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "adapter_state": adapter.state_dict(),
            "d_model": d_model,
            "slot": args.slot,
            "args": vars(args),
            "train_samples": len(rows),
        },
        str(OUT / "models" / f"{args.out}.pt"),
    )
    hist_path = OUT / "training_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(
        f"WROTE {OUT / 'models' / (args.out + '.pt')} "
        f"history={hist_path} total_sec={time.time() - t0:.1f}",
        flush=True,
    )
    runner.close()


if __name__ == "__main__":
    main()
