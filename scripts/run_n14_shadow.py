"""N14.6 Shadow Identity Query + Selective Commit.

S1: v6 shadow candidate computed separately, NEVER committed -> official
output must equal A0 (hard gate).
S2: oracle selective commit (GT-only, TRAIN/CAL diagnostic): commit only when
AUTO is wrong and the shadow candidate is correct.  Measures the upper bound
of selective intervention and the do-no-harm rates.
"""

import argparse
import copy
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(".")
OUT = ROOT / "outputs/n14"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")


def clone_find_input(fin, img_id: int):
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
    ap.add_argument("--model", default="outputs/n14/models/human_write_encoder_f0_v6.pt")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--events", default="all")
    ap.add_argument("--slot", type=int, default=199)
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
        set_frame_geometric_prompt,
    )
    from sam3_intermot.persistent_identity import (
        HumanWriteEncoder,
        SlotHeadAdapter,
        roi_pool_feature,
    )
    from sam3.model.geometry_encoders import Prompt

    ck = torch.load(ROOT / args.model, map_location="cuda", weights_only=False)
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
    hidden = int(ck["args"].get("hidden", 512))
    encoder = HumanWriteEncoder(d_model=d_model, hidden=hidden).cuda().eval()
    adapter = SlotHeadAdapter(d_model=d_model, hidden=hidden // 4).cuda().eval()
    encoder.load_state_dict(ck["encoder_state"])
    adapter.load_state_dict(ck["adapter_state"])

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

    enc_cache = {}

    def enc_for(seq, f):
        key = (seq, f)
        if key in enc_cache:
            return enc_cache[key]
        state = backend._predictor._all_inference_states[
            backend._session_id
        ]["state"]
        ib = state["input_batch"]
        fin = clone_find_input(ib.find_inputs[f], img_id=0)
        img_t = ib.img_batch.tensors[f].unsqueeze(0).clone().to("cuda")
        with torch.no_grad():
            text = model.detector.backbone.forward_text(["person"], device="cuda")
            bo = {
                "img_batch_all_stages": img_t,
                "language_features": text["language_features"].clone(),
                "language_mask": text["language_mask"].clone(),
            }
            geo = Prompt(
                box_embeddings=torch.zeros(0, 1, 4, device="cuda"),
                box_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                box_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
                point_embeddings=torch.zeros(0, 1, 2, device="cuda"),
                point_mask=torch.zeros(1, 0, device="cuda", dtype=torch.bool),
                point_labels=torch.zeros(0, 1, device="cuda", dtype=torch.long),
            )
            prompt, pmask, bo2 = model.detector._encode_prompt(bo, fin, geo)
            bo2, enc, _ = model.detector._run_encoder(bo2, fin, prompt, pmask)
        enc_cache[key] = enc
        return enc

    def run_official(seq, t, gid, hb, with_shadow):
        video = str(DT / "train" / seq / "img1")
        backend.start_video(video)
        model.use_batched_grounding = False
        iw, ih = backend._frame_w, backend._frame_h
        ep = PDREpisode(
            sequence=seq, frame=t, event_type="TRUE_MISS_NEW", gid=gid,
            human_box=np.asarray(hb, dtype=float), policy="A0",
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
        state = backend._predictor._all_inference_states[
            backend._session_id
        ]["state"]
        state["action_history"].clear()
        prev = np.asarray(hb, dtype=float).copy()
        ep.records[t] = FrameRecord(
            frame=t, cand_boxes=[], prompt_box=hb.copy(),
            delivered_box=prev.copy(),
        )
        raw_hashes = {}
        orig_rbd = model.run_backbone_and_detection

        def wrap(frame_idx, num_frames, input_batch, geometric_prompt,
                 feature_cache, reverse, use_batched_grounding=False,
                 batched_grounding_batch_size=16):
            det_out, pos = orig_rbd(
                frame_idx, num_frames, input_batch, geometric_prompt,
                feature_cache, reverse, use_batched_grounding,
                batched_grounding_batch_size,
            )
            if det_out is not None:
                h = hashlib.sha256()
                h.update(
                    det_out["bbox"][0].detach().float().cpu().numpy()
                    .astype("<f4").tobytes()
                )
                h.update(
                    det_out["scores"][0].detach().float().cpu().numpy()
                    .astype("<f4").tobytes()
                )
                raw_hashes[int(frame_idx)] = h.hexdigest()[:16]
            return det_out, pos

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

        shadow = None
        if with_shadow:
            # Shadow computed AFTER the official branch (non-destructive).
            shadow = []
            box_norm = np.asarray(
                [x1 / iw, y1 / ih, x2 / iw, y2 / ih], dtype=float
            )
            enc_t = enc_for(seq, t)
            roi_t = roi_pool_feature(
                enc_t["encoder_hidden_states"], enc_t, box_norm
            )
            with torch.no_grad():
                q = encoder(roi_t.float()).to(torch.float32)
                ref = torch.as_tensor(
                    cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda",
                )
                for f in range(t + 1, t + args.horizon + 1):
                    enc_f = enc_for(seq, f)
                    roi_f = roi_pool_feature(
                        enc_f["encoder_hidden_states"], enc_f, box_norm
                    )
                    dbox, _ = adapter(
                        q.unsqueeze(0), roi_f.unsqueeze(0),
                        roi_f.unsqueeze(0), ref.unsqueeze(0),
                    )
                    box_n = (ref + dbox[0]).clamp(0.01, 1.0)
                    cx, cy, w, h = box_n.detach().cpu().tolist()
                    cand_box_norm = np.asarray(
                        [
                            (cx - w / 2), (cy - h / 2),
                            (cx + w / 2), (cy + h / 2),
                        ]
                    ).clip(0.0, 1.0)
                    roi_cand = roi_pool_feature(
                        enc_f["encoder_hidden_states"], enc_f, cand_box_norm
                    )
                    _, dscore = adapter(
                        q.unsqueeze(0), roi_cand.unsqueeze(0),
                        roi_f.unsqueeze(0), ref.unsqueeze(0),
                    )
                    sb = cxcywh_to_xyxy(*box_n.detach().cpu().tolist())
                    sb = sb * np.asarray([iw, ih, iw, ih])
                    shadow.append(
                        {
                            "frame": f,
                            "score": float(torch.sigmoid(dscore).item()),
                            "box": sb,
                        }
                    )
        return ep, raw_hashes, shadow

    eq_rows = []
    commit_rows = []
    per_event_rows = []
    inter_util = defaultdict(int)
    for seq, t, gid in evs:
        gt = ds.load_gt(seq)
        hb = np.asarray(gt[t].boxes[gt[t].gt_ids.index(gid)], dtype=float)
        enc_cache.clear()
        ep_a, hashes_a, _ = run_official(seq, t, gid, hb, with_shadow=False)
        enc_cache.clear()
        ep_s, hashes_s, shadow = run_official(seq, t, gid, hb, with_shadow=True)
        # S1 equivalence.
        common = sorted(set(hashes_a) & set(hashes_s))
        eq_rows.append(
            {
                "sequence": seq, "event_frame": t, "gid": gid,
                "frames_compared": len(common),
                "identical_raw_hash": sum(
                    1 for f in common if hashes_a[f] == hashes_s[f]
                ),
                "shadow_events": len(shadow) if shadow else 0,
            }
        )
        # Oracle commit simulation.
        for sh in shadow or []:
            f = sh["frame"]
            entry = gt.get(f)
            rec_a = ep_a.records.get(f)
            auto_box = None if rec_a is None else rec_a.delivered_box
            target_box = None
            if entry is not None and gid in entry.gt_ids:
                target_box = np.asarray(
                    entry.boxes[entry.gt_ids.index(gid)], dtype=float
                )
            auto_correct = (
                auto_box is not None and target_box is not None
                and iou_xyxy(auto_box, target_box) >= 0.5
            )
            shadow_correct = (
                target_box is not None
                and iou_xyxy(sh["box"], target_box) >= 0.5
            )
            commit = (not auto_correct) and shadow_correct
            if commit:
                committed_box = sh["box"]
            else:
                committed_box = auto_box
            commit_rows.append(
                {
                    "sequence": seq, "event_frame": t, "gid": gid,
                    "frame": f,
                    "auto_correct": int(auto_correct),
                    "shadow_score": round(sh["score"], 4),
                    "shadow_correct": int(shadow_correct),
                    "commit": int(commit),
                    "auto_iou_gt": round(
                        iou_xyxy(auto_box, target_box), 3
                    ) if auto_box is not None and target_box is not None else "",
                    "shadow_iou_gt": round(
                        iou_xyxy(sh["box"], target_box), 3
                    ) if target_box is not None else "",
                    "committed_correct": int(
                        committed_box is not None and target_box is not None
                        and iou_xyxy(committed_box, target_box) >= 0.5
                    ),
                }
            )
            if auto_correct:
                if commit:
                    inter_util["UNNECESSARY_INTERVENTION"] += 1
                else:
                    inter_util["CORRECT_ABSTENTION"] += 1
            else:
                if commit:
                    inter_util["BENEFICIAL_COMMIT"] += 1
                elif shadow_correct:
                    inter_util["MISSED_OPPORTUNITY"] += 1
                else:
                    inter_util["NEUTRAL_ABSTENTION"] += 1

        # Frame-level oracle recall.
        def oracle_recall(h):
            hits = n = 0
            for f in range(t + 1, t + h + 1):
                entry = gt.get(f)
                if entry is None or gid not in entry.gt_ids:
                    continue
                n += 1
                rec_a = ep_a.records.get(f)
                auto_box = None if rec_a is None else rec_a.delivered_box
                target_box = np.asarray(
                    entry.boxes[entry.gt_ids.index(gid)], dtype=float
                )
                sh = next(
                    (s for s in shadow if s["frame"] == f), None
                )
                if auto_box is not None and iou_xyxy(auto_box, target_box) >= 0.5:
                    hits += 1
                    continue
                if sh is not None and iou_xyxy(sh["box"], target_box) >= 0.5:
                    hits += 1
            return hits / max(1, n)

        row = {
            "sequence": seq, "event_frame": t, "gid": gid,
            "oracle_delivered_1": round(oracle_recall(1), 3),
            "oracle_delivered_3": round(oracle_recall(3), 3),
            "oracle_delivered_10": round(oracle_recall(10), 3),
            "oracle_delivered_30": round(oracle_recall(30), 3),
        }
        per_event_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        try:
            backend.close()
        except Exception:
            pass

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("shadow_equivalence.csv", eq_rows),
        ("commit_dataset.csv", commit_rows),
        ("oracle_commit.csv", per_event_rows),
    ):
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (OUT / "intervention_utility.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for k in sorted(inter_util):
            w.writerow([k, inter_util[k]])
    # Easy-frame preservation summary.
    easy = []
    for r in commit_rows:
        if r["auto_correct"]:
            easy.append(r)
    preserved = sum(1 for r in easy if r["committed_correct"])
    with (OUT / "easy_frame_preservation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["easy_frames", "preserved", "preservation_rate"])
        w.writerow([len(easy), preserved,
                    round(preserved / max(1, len(easy)), 4)])
    print("WROTE shadow/oracle CSVs", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
