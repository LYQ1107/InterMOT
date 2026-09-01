#!/usr/bin/env python
"""N15 Shadow Query + Oracle Selective Commit on the 14 calibration events.

S1: official output with shadow never committed must equal A0 (raw-hash).
Oracle: commit only when AUTO is wrong and the shadow candidate is correct.
The shadow candidate comes from the frozen decoder slot with the I2Q query
injected; identity verification (G_id) is measured separately in the
pretrained feature space.
"""

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(".")
OUT = ROOT / "outputs/n15"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")
CLIP_CKPT = ROOT / "outputs/n15/checkpoints/clip_reid_hf/person_vit_clip_reid.pth"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="outputs/n15/models/i2q_linear_v1.pt")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--slot", type=int, default=199)
    ap.add_argument("--events", default="all")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner,
        parse_raw_outputs,
    )
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.detection_query.prompt_replay import (
        FrameRecord,
        PDREpisode,
        _best_delivery,
        invalidate_detector_prefetch,
        recall_at,
        set_frame_geometric_prompt,
    )
    from sam3_intermot.identity_anchor.identity_to_query import LinearI2Q
    from sam3_intermot.persistent_identity import install_query_patch
    from sam3.model.geometry_encoders import Prompt
    from scripts.run_n15_extract_features import build_clipreid

    ck = torch.load(ROOT / args.model, map_location="cpu", weights_only=False)
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
    i2q = LinearI2Q(in_dim=1280, d_model=d_model, hidden=1024).cuda().eval()
    i2q.load_state_dict(ck["state"])
    clip = build_clipreid(str(CLIP_CKPT), "cuda")
    import torchvision.transforms as T

    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def clip_feat(img_path, box):
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        x = clip_tf(img.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
        with torch.no_grad():
            _, x12, xproj = clip(x)
            fv = torch.cat([x12[:, 0], xproj[:, 0]], dim=1)
        return F.normalize(fv, dim=-1)

    ds = DanceTrackDataset(str(DT), sequences=None, split="train")
    evs = []
    for path in ("outputs/n13/pdr_idx0_events.csv", "outputs/n13/pdr_idx1_events.csv"):
        with open(ROOT / path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["policy"] != "one_shot":
                    continue
                key = (r["sequence"], int(r["frame"]), int(r["gid"]))
                if key not in evs:
                    evs.append(key)
    if args.events != "all":
        evs = [evs[int(i)] for i in args.events.split(",")]
    print(f"events: {len(evs)}", flush=True)

    def run_episode(seq, t, gid, human_box, shadow_q, shadow_ref, slot_records,
                    raw_hashes, gid_scores, with_shadow):
        video = str(DT / "train" / seq / "img1")
        backend.start_video(video)
        model.use_batched_grounding = False
        iw, ih = backend._frame_w, backend._frame_h
        ep = PDREpisode(
            sequence=seq, frame=t, event_type="TRUE_MISS_NEW", gid=gid,
            human_box=np.asarray(human_box, dtype=float), policy="n15_shadow",
        )
        x1, y1, x2, y2 = ep.human_box
        req_prompt = dict(
            type="add_prompt",
            session_id=backend._session_id,
            frame_index=t,
            text="person",
            bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]],
            bounding_box_labels=[1],
            clear_old_boxes=True,
        )
        backend._predictor.handle_request(req_prompt)
        state = backend._predictor._all_inference_states[backend._session_id]["state"]
        state["action_history"].clear()
        prev = np.asarray(human_box, dtype=float).copy()
        ep.records[t] = FrameRecord(
            frame=t, cand_boxes=[], prompt_box=ep.human_box.copy(),
            delivered_box=prev.copy(),
        )
        orig_rbd = model.run_backbone_and_detection
        enc_capture = {}
        orig_run_encoder = image._run_encoder

        def hook_encoder(bo2, fin, prompt, pmask, **kwargs):
            out = orig_run_encoder(bo2, fin, prompt, pmask, **kwargs)
            bo2, enc, _ = out
            enc_capture["enc"] = enc
            enc_capture["prompt"] = prompt
            enc_capture["pmask"] = pmask
            return out

        def wrap(frame_idx, num_frames, input_batch, geometric_prompt,
                 feature_cache, reverse, use_batched_grounding=False,
                 batched_grounding_batch_size=16):
            enc_capture.clear()
            det_out, pos = orig_rbd(
                frame_idx, num_frames, input_batch, geometric_prompt,
                feature_cache, reverse, use_batched_grounding,
                batched_grounding_batch_size,
            )
            if det_out is not None:
                h = hashlib.sha256()
                h.update(det_out["bbox"].detach().float().cpu().numpy().astype("<f4").tobytes())
                h.update(det_out["scores"].detach().float().cpu().numpy().astype("<f4").tobytes())
                raw_hashes[int(frame_idx)] = h.hexdigest()[:16]
            if with_shadow and frame_idx > t and enc_capture.get("enc") is not None:
                enc = enc_capture["enc"]
                bank = lambda: ([shadow_q], [shadow_ref])
                uninstall = install_query_patch(image, bank, [args.slot])
                try:
                    out = {"encoder_hidden_states": enc["encoder_hidden_states"]}
                    with torch.no_grad():
                        out, _ = image._run_decoder(
                            pos_embed=enc["pos_embed"],
                            memory=enc["encoder_hidden_states"],
                            src_mask=enc["padding_mask"],
                            out=out,
                            prompt=enc_capture["prompt"],
                            prompt_mask=enc_capture["pmask"],
                            encoder_out=enc,
                        )
                    slot_box_norm = out["pred_boxes"][0, args.slot].detach().float().cpu()
                    slot_logit = out["pred_logits"][0, args.slot].detach().float().cpu()
                    sb = cxcywh_to_xyxy(*slot_box_norm.tolist())
                    sb_px = sb * np.asarray([iw, ih, iw, ih])
                    sb_px = sb_px.clip(0, iw - 1)
                    score = float(torch.sigmoid(slot_logit).item())
                    gscore = None
                    if gid_scores is not None:
                        fv = clip_feat(
                            f"{video}/{frame_idx + 1:08d}.jpg", sb_px
                        )
                        if fv is not None:
                            gscore = float((fv[0] * gid_scores[0]).sum().item())
                    slot_records.append(
                        {
                            "frame": int(frame_idx), "score": round(score, 4),
                            "gid_cosine": round(gscore, 4) if gscore is not None else "",
                            "box": [round(float(v), 1) for v in sb_px],
                        }
                    )
                finally:
                    uninstall()
            return det_out, pos

        image._run_encoder = hook_encoder
        model.run_backbone_and_detection = wrap
        nf = t + 1
        set_frame_geometric_prompt(runner, nf, None)
        req = dict(
            type="propagate_in_video",
            session_id=backend._session_id,
            propagation_direction="forward",
            start_frame_index=t,
            max_frame_num_to_track=None,
        )
        try:
            for response in backend._predictor.handle_stream_request(request=req):
                f = int(response["frame_index"])
                cands = parse_raw_outputs(response, frame_size=(iw, ih))
                cand_boxes = [np.asarray(b, dtype=float) for _, b in cands]
                delivered = _best_delivery(prev, cand_boxes)
                if delivered is not None:
                    prev = delivered.copy()
                ep.records[f] = FrameRecord(
                    frame=f, cand_boxes=cand_boxes, prompt_box=None,
                    delivered_box=delivered,
                )
                if f >= t + args.horizon:
                    break
                nf2 = f + 1
                set_frame_geometric_prompt(runner, nf2, None)
                invalidate_detector_prefetch(runner, f)
        finally:
            model.run_backbone_and_detection = orig_rbd
            image._run_encoder = orig_run_encoder
        return ep

    eq_rows, commit_rows, per_event_rows = [], [], []
    inter_util = {}
    for seq, t, gid in evs:
        gt = ds.load_gt(seq)
        hb = np.asarray(gt[t].boxes[gt[t].gt_ids.index(gid)], dtype=float)
        video = str(DT / "train" / seq / "img1")
        iw, ih = 1920, 1080
        hf = clip_feat(f"{video}/{t + 1:08d}.jpg", hb)
        if hf is None:
            print("SKIP no anchor", seq, t, gid, flush=True)
            continue
        with torch.no_grad():
            q = i2q(hf)[0]
        ref = torch.as_tensor(cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda")
        slot_records_a = []
        hashes_a = {}
        ep_a = run_episode(seq, t, gid, hb, None, None, slot_records_a,
                           hashes_a, None, with_shadow=False)
        try:
            backend.close()
        except Exception:
            pass
        slot_records_s = []
        hashes_s = {}
        ep_s = run_episode(seq, t, gid, hb, q, ref, slot_records_s,
                           hashes_s, hf, with_shadow=True)
        try:
            backend.close()
        except Exception:
            pass
        common = sorted(set(hashes_a) & set(hashes_s))
        eq_rows.append(
            {
                "sequence": seq, "event_frame": t, "gid": gid,
                "frames_compared": len(common),
                "identical_raw_hash": sum(1 for f in common if hashes_a[f] == hashes_s[f]),
                "shadow_events": len(slot_records_s),
            }
        )
        for sh in slot_records_s:
            f = sh["frame"]
            entry = gt.get(f)
            rec_a = ep_a.records.get(f)
            auto_box = None if rec_a is None else rec_a.delivered_box
            target_box = None
            if entry is not None and gid in entry.gt_ids:
                target_box = np.asarray(entry.boxes[entry.gt_ids.index(gid)], dtype=float)
            auto_correct = (
                auto_box is not None and target_box is not None
                and iou_xyxy(auto_box, target_box) >= 0.5
            )
            shadow_correct = (
                target_box is not None
                and iou_xyxy(sh["box"], target_box) >= 0.5
            )
            commit = (not auto_correct) and shadow_correct
            committed_box = sh["box"] if commit else auto_box
            committed_correct = (
                committed_box is not None and target_box is not None
                and iou_xyxy(committed_box, target_box) >= 0.5
            )
            commit_rows.append(
                {
                    "sequence": seq, "event_frame": t, "gid": gid, "frame": f,
                    "auto_correct": int(auto_correct), "shadow_score": sh["score"],
                    "gid_cosine": sh["gid_cosine"], "shadow_correct": int(shadow_correct),
                    "commit": int(commit), "committed_correct": int(committed_correct),
                    "auto_iou_gt": round(
                        iou_xyxy(auto_box, target_box), 3
                    ) if auto_box is not None and target_box is not None else "",
                    "shadow_iou_gt": round(
                        iou_xyxy(sh["box"], target_box), 3
                    ) if target_box is not None else "",
                }
            )
            if auto_correct:
                k = "UNNECESSARY_INTERVENTION" if commit else "CORRECT_ABSTENTION"
            else:
                if commit:
                    k = "BENEFICIAL_COMMIT"
                elif shadow_correct:
                    k = "MISSED_OPPORTUNITY"
                else:
                    k = "NEUTRAL_ABSTENTION"
            inter_util[k] = inter_util.get(k, 0) + 1

        def oracle_recall(h):
            hits = n = 0
            for f in range(t + 1, t + h + 1):
                entry = gt.get(f)
                if entry is None or gid not in entry.gt_ids:
                    continue
                n += 1
                rec_a = ep_a.records.get(f)
                auto_box = None if rec_a is None else rec_a.delivered_box
                target_box = np.asarray(entry.boxes[entry.gt_ids.index(gid)], dtype=float)
                sh = next((s for s in slot_records_s if s["frame"] == f), None)
                if auto_box is not None and iou_xyxy(auto_box, target_box) >= 0.5:
                    hits += 1
                    continue
                if sh is not None and iou_xyxy(sh["box"], target_box) >= 0.5:
                    hits += 1
            return hits / max(1, n)

        per_event_rows.append(
            {
                "sequence": seq, "event_frame": t, "gid": gid,
                "oracle_delivered_1": round(oracle_recall(1), 3),
                "oracle_delivered_3": round(oracle_recall(3), 3),
                "oracle_delivered_5": round(oracle_recall(5), 3),
                "oracle_delivered_10": round(oracle_recall(10), 3),
                "oracle_delivered_30": round(oracle_recall(30), 3),
                "a0_delivered_1": round(recall_at(ep_a, gt, 1, "delivered"), 3),
                "a0_delivered_3": round(recall_at(ep_a, gt, 3, "delivered"), 3),
                "a0_delivered_5": round(recall_at(ep_a, gt, 5, "delivered"), 3),
                "a0_delivered_10": round(recall_at(ep_a, gt, 10, "delivered"), 3),
                "a0_delivered_30": round(recall_at(ep_a, gt, 30, "delivered"), 3),
            }
        )
        print(json.dumps(per_event_rows[-1], ensure_ascii=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("shadow_equivalence.csv", eq_rows),
        ("commit_dataset.csv", commit_rows),
        ("oracle_commit.csv", per_event_rows),
    ):
        if not rows:
            continue
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (OUT / "intervention_utility.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for k in sorted(inter_util):
            w.writerow([k, inter_util[k]])
    easy = [r for r in commit_rows if r["auto_correct"]]
    preserved = sum(1 for r in easy if r["committed_correct"])
    with (OUT / "easy_frame_preservation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["easy_frames", "preserved", "preservation_rate"])
        w.writerow([len(easy), preserved, round(preserved / max(1, len(easy)), 4)])
    print("WROTE n15 shadow/oracle CSVs", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
