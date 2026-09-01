#!/usr/bin/env python
"""Evaluate HCRD-v0 on the HCC benchmark (candidate creation)."""

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "outputs/n16"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")
CLIP_CKPT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"
P0 = ROOT / "outputs/n9/p0_train"


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


def deep_clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return [deep_clone(v) for v in x]
    if isinstance(x, dict):
        return {k: deep_clone(v) for k, v in x.items()}
    return x


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


def load_p0(seq):
    p = P0 / f"{seq}.txt"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame = int(float(parts[0])) - 1
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        if w <= 0 or h <= 0:
            continue
        out.setdefault(frame, []).append(np.asarray([x, y, x + w, y + h], float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="outputs/n16/models/hcrd_v0_overfit.pt")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--tgt-tau", type=float, default=0.5)
    ap.add_argument("--pres-tau", type=float, default=0.5)
    ap.add_argument("--arch", default="correlation", choices=["decoder", "correlation"])
    ap.add_argument("--manifest", default="hcred_manifest.csv")
    ap.add_argument("--finetune", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner
    from sam3_intermot.persistent_identity import roi_pool_feature
    from sam3_intermot.recovery.recovery_decoder import RecoveryDecoder
    from sam3.model.geometry_encoders import Prompt
    from scripts.run_n15_extract_features import build_clipreid
    import torchvision.transforms as T

    ck = torch.load(ROOT / args.model, map_location="cpu", weights_only=False)
    d_model = ck["d_model"]
    if args.arch == "correlation":
        from sam3_intermot.recovery.correlation_decoder import CorrelationRecoveryDecoder
        model_rec = CorrelationRecoveryDecoder(
            anchor_dim=1280, d_model=d_model, template_hw=8, grid_hw=36
        ).cuda().eval()
    else:
        model_rec = RecoveryDecoder(
            anchor_dim=1280, roi_dim=d_model, d_model=d_model,
            n_queries=ck["args"]["n_queries"], n_layers=ck["args"]["n_layers"],
        ).cuda().eval()
    model_rec.load_state_dict(ck["state"])
    clip = build_clipreid(str(CLIP_CKPT), "cuda")
    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    if args.finetune and ck.get("encoder_state"):
        model.detector.transformer.encoder.load_state_dict(ck["encoder_state"])
        print("loaded fine-tuned encoder", flush=True)
    model.use_batched_grounding = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    image = model.detector

    rows = []
    with (OUT / args.manifest).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != args.split:
                continue
            if args.seqs and r["sequence"] not in args.seqs.split(","):
                continue
            rows.append(r)
    if args.max_samples:
        rng = np.random.default_rng(11)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
    from collections import defaultdict
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["sequence"]].append(r)
    rows = []
    for seq in sorted(by_seq):
        rows.extend(by_seq[seq])
    print(f"eval samples={len(rows)}", flush=True)

    def anchor_feat(seq, frame, box):
        img = Image.open(DT / "train" / seq / "img1" / f"{frame + 1:08d}.jpg").convert("RGB")
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        crop = img.crop((x1, y1, x2, y2))
        x = clip_tf(crop).unsqueeze(0).cuda()
        with torch.no_grad():
            _, x12, xproj = clip(x)
            fv = torch.cat([x12[:, 0], xproj[:, 0]], dim=1)
        return F.normalize(fv, dim=-1)

    ds_video, frame_cache, text_out, empty_geo = {}, {}, None, None

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

    def encoder_features(seq, f):
        key = (seq, f)
        if key in frame_cache:
            return frame_cache[key]
        cache_dir = OUT / "enc_cache"
        cache_path = cache_dir / f"{seq}_{f}.npz"
        if cache_path.exists() and not args.finetune:
            z = np.load(cache_path)
            enc = {
                "encoder_hidden_states": torch.from_numpy(z["mem"].astype(np.float32)),
                "pos_embed": torch.from_numpy(z["pos"].astype(np.float32)),
                "spatial_shapes": torch.from_numpy(z["shapes"]),
                "level_start_index": torch.from_numpy(z["starts"]),
            }
            feat = {"enc": enc}
            frame_cache[key] = feat
            return feat
        if ds_video.get("seq") != seq:
            if ds_video.get("seq") is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            backend.start_video(str(DT / "train" / seq / "img1"))
            ds_video["seq"] = seq
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
            prompt, pmask, bo2 = model.detector._encode_prompt(bo, fin, empty_geo)
            bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)
        feat = {
            "enc": deep_clone(enc),
            "prompt": prompt.clone(),
            "pmask": pmask.clone(),
        }
        feat = {
            k: ({kk: (vv.cpu() if isinstance(vv, torch.Tensor) else vv)
                 for kk, vv in v.items()} if isinstance(v, dict)
                else (v.cpu() if isinstance(v, torch.Tensor) else v))
            for k, v in feat.items()
        }
        if not args.finetune:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                mem=enc["encoder_hidden_states"].detach().cpu().to(torch.float16).numpy(),
                pos=enc["pos_embed"].detach().cpu().to(torch.float16).numpy(),
                shapes=enc["spatial_shapes"].detach().cpu().numpy(),
                starts=enc["level_start_index"].detach().cpu().numpy(),
            )
        frame_cache[key] = feat
        if len(frame_cache) > 96:
            frame_cache.pop(next(iter(frame_cache)))
        return feat

    def to_cuda(feat):
        out = {}
        for k, v in feat.items():
            if isinstance(v, dict):
                out[k] = {kk: (vv.cuda() if isinstance(vv, torch.Tensor) else vv)
                          for kk, vv in v.items()}
            elif isinstance(v, torch.Tensor):
                out[k] = v.cuda()
            else:
                out[k] = v
        return out

    iw, ih = 1920, 1080
    detail, per_seq = [], defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for r in rows:
        seq = r["sequence"]
        t, f = int(r["t"]), int(r["f"])
        hb = np.asarray(json.loads(r["human_box"]), dtype=float)
        present = int(r["target_present"]) == 1
        fb = np.asarray(json.loads(r["target_box"]), dtype=float) if present else None
        generic_miss = int(r["generic_miss"]) == 1 if r["generic_miss"] != "" else False
        enc_t = to_cuda(encoder_features(seq, t))
        enc_f = to_cuda(encoder_features(seq, f))
        box_norm = torch.as_tensor(
            cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda"
        )
        with torch.no_grad():
            roi = roi_pool_feature(
                enc_t["enc"]["encoder_hidden_states"], enc_t["enc"], box_norm.cpu().numpy()
            ).float().to("cuda")
            h = anchor_feat(seq, t, hb)
            if args.arch == "correlation":
                pred = model_rec(h, roi.unsqueeze(0), box_norm.unsqueeze(0),
                                 enc_t["enc"], enc_f["enc"])
                pboxes = model_rec.decode_boxes(
                    pred["obj"], pred["box"], ref_box=box_norm.unsqueeze(0)
                )
                boxes = pboxes[0].detach().cpu().numpy()
                tgt = torch.sigmoid(pred["obj"].flatten(1).max(dim=1).values).float().detach().cpu().numpy()
                presence = float(torch.sigmoid(pred["presence"][0]).item())
            else:
                pred = model_rec(h, roi.unsqueeze(0), box_norm.unsqueeze(0), enc_f["enc"])
                boxes = pred["boxes"][0].detach().cpu().numpy()
                tgt = torch.sigmoid(pred["targetness"][0]).detach().cpu().numpy()
                presence = float(torch.sigmoid(pred["presence"][0]).item())
        if args.arch == "correlation":
            prop_xyxy = boxes * np.asarray([iw, ih, iw, ih])
            order = np.arange(len(boxes))
        else:
            prop_xyxy = np.stack(
                [cxcywh_to_xyxy(*b) for b in boxes]
            ) * np.asarray([iw, ih, iw, ih])
            order = np.argsort(-tgt)
        best_iou = 0.0
        if present:
            for pxy in prop_xyxy:
                best_iou = max(best_iou, iou_xyxy(pxy, fb))
            recall_03 = best_iou >= 0.3
            recall_05 = best_iou >= 0.5
            recall_07 = best_iou >= 0.7
            top1 = iou_xyxy(prop_xyxy[order[0]], fb) >= 0.5
            top3 = any(iou_xyxy(pxy, fb) >= 0.5 for pxy in prop_xyxy[order[:3]])
        else:
            recall_03 = recall_05 = recall_07 = top1 = top3 = False
        ghost = (not present) and presence >= args.pres_tau
        # hard distractor false capture: proposal matches another GT person
        fc = 0
        if present:
            gt_all = load_gt_boxes(seq)
            for pxy in prop_xyxy:
                for ogid, obox in gt_all.get(f, []):
                    if ogid == int(r["gid"]):
                        continue
                    if iou_xyxy(pxy, obox) >= 0.5:
                        fc = 1
        detail.append(
            {
                "sequence": seq, "t": t, "gid": r["gid"], "f": f, "delta": r["delta"],
                "present": int(present), "generic_miss": int(generic_miss),
                "best_iou": round(best_iou, 3), "recall_03": int(recall_03),
                "recall_05": int(recall_05), "recall_07": int(recall_07),
                "top1": int(top1), "top3": int(top3),
                "presence": round(presence, 3), "ghost": int(ghost),
                "false_capture": fc, "crowd": r["crowd"],
            }
        )
        s = per_seq[seq]
        s[0] += 1
        s[1] += int(present)
        s[2] += int(generic_miss)
        s[3] += int(recall_05) if present else 0
        s[4] += int(ghost)
        s[5] += fc
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidate_creation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)
    with (OUT / "candidate_creation_per_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "n", "present", "generic_miss", "recall05_present",
                    "ghost", "false_capture"])
        for seq, s in sorted(per_seq.items()):
            w.writerow([seq, *s])
    if detail:
        pres = [d for d in detail if d["present"]]
        miss = [d for d in detail if d["generic_miss"]]
        abs_ = [d for d in detail if not d["present"]]
        print(
            f"n={len(detail)} present={len(pres)} miss={len(miss)} absent={len(abs_)} "
            f"recall05_present={np.mean([d['recall_05'] for d in pres]) if pres else 0:.4f} "
            f"CCR_miss={np.mean([d['recall_05'] for d in miss]) if miss else 0:.4f} "
            f"top1={np.mean([d['top1'] for d in pres]) if pres else 0:.4f} "
            f"ghost_rate={np.mean([d['ghost'] for d in abs_]) if abs_ else 0:.4f} "
            f"fc_rate={np.mean([d['false_capture'] for d in pres]) if pres else 0:.4f}"
        )
    runner.close()


def load_gt_boxes(seq):
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    d = DanceTrackDataset(str(DT), sequences=[], split="train").load_gt(seq)
    return {f: list(zip(g.gt_ids, g.boxes)) for f, g in d.items()}


if __name__ == "__main__":
    main()
