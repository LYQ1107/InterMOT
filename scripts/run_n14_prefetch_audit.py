"""N14.4A Prefetch Execution Audit.

Decompose the prefetch/execution-path effect from the renewal failure:
  E0 = query patch installed AFTER add_prompt (current/historical path)
  E1 = query state registered BEFORE add_prompt (corrected causal path)
No renewal in either branch.  A0/v4/v5 x E0/E1 on representative events.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(".")
OUT = ROOT / "outputs/n14"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DT = Path("/path/to/dancetrack")
EVENTS = [
    ("dancetrack0074", 6, 1),
    ("dancetrack0074", 7, 4),
    ("dancetrack0075", 1, 4),
    ("dancetrack0080", 1, 3),
    ("dancetrack0080", 1, 5),
    ("dancetrack0082", 1, 11),
    ("dancetrack0082", 1, 16),
    ("dancetrack0083", 1, 1),
    ("dancetrack0083", 1, 2),
    ("dancetrack0086", 1, 1),
    ("dancetrack0086", 1, 3),
    ("dancetrack0087", 1, 7),
    ("dancetrack0096", 1, 2),
    ("dancetrack0096", 1, 5),
]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--slot", type=int, default=199)
    ap.add_argument("--variants", default="A0,v4,v5")
    ap.add_argument("--timings", default="E0,E1")
    ap.add_argument("--events", default="all")
    ap.add_argument("--gate-score", type=float, default=None)
    ap.add_argument("--gate-iou", type=float, default=None)
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
        false_capture,
        invalidate_detector_prefetch,
        recall_at,
        set_frame_geometric_prompt,
    )
    from sam3_intermot.persistent_identity import (
        HumanWriteEncoder,
        SlotHeadAdapter,
        install_query_patch,
        roi_pool_feature,
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

    models = {}
    for v in ("v6",):
        if not (ROOT / f"outputs/n14/models/human_write_encoder_f0_{v}.pt").exists():
            continue
        ck = torch.load(
            ROOT / f"outputs/n14/models/human_write_encoder_f0_{v}.pt",
            map_location="cuda", weights_only=False,
        )
        hidden = int(ck["args"].get("hidden", 512))
        enc = HumanWriteEncoder(d_model=d_model, hidden=hidden).cuda().eval()
        ada = SlotHeadAdapter(d_model=d_model, hidden=hidden // 4).cuda().eval()
        enc.load_state_dict(ck["encoder_state"])
        ada.load_state_dict(ck["adapter_state"])
        models[v] = (enc, ada)

    ds = DanceTrackDataset(str(DT), sequences=None, split="train")
    evs = EVENTS if args.events == "all" else [
        EVENTS[int(i)] for i in args.events.split(",")
    ]
    variants = args.variants.split(",")
    timings = args.timings.split(",")

    def compute_q(seq, t, hb, iw, ih):
        state = backend._predictor._all_inference_states[
            backend._session_id
        ]["state"]
        ib = state["input_batch"]
        fin = clone_find_input(ib.find_inputs[t], img_id=0)
        img_t = ib.img_batch.tensors[t].unsqueeze(0).clone().to("cuda")
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
        x1, y1, x2, y2 = (float(v) for v in hb)
        box_norm = np.asarray([x1 / iw, y1 / ih, x2 / iw, y2 / ih])
        roi = roi_pool_feature(enc["encoder_hidden_states"], enc, box_norm)
        enc_model, ada = models[variant]
        q = enc_model(roi.float()).to(torch.float32)
        ref = torch.as_tensor(
            cxcywh_norm(hb, iw, ih), dtype=torch.float32, device="cuda"
        )
        return q, ref

    per_event = []
    trace_rows = []
    for seq, t, gid in evs:
        gt = ds.load_gt(seq)
        hb = np.asarray(gt[t].boxes[gt[t].gt_ids.index(gid)], dtype=float)
        for variant in variants:
            for timing in timings:
                video = str(DT / "train" / seq / "img1")
                backend.start_video(video)
                model.use_batched_grounding = False
                iw, ih = backend._frame_w, backend._frame_h
                ep = PDREpisode(
                    sequence=seq, frame=t, event_type="TRUE_MISS_NEW",
                    gid=gid, human_box=hb.copy(), policy=f"{variant}_{timing}",
                )
                patch_active = {"v": False}
                local_trace = []
                orig_rbd = model.run_backbone_and_detection

                def wrap(frame_idx, num_frames, input_batch, geometric_prompt,
                         feature_cache, reverse, use_batched_grounding=False,
                         batched_grounding_batch_size=16):
                    det_out, pos = orig_rbd(
                        frame_idx, num_frames, input_batch, geometric_prompt,
                        feature_cache, reverse, use_batched_grounding,
                        batched_grounding_batch_size,
                    )
                    rec = {
                        "sequence": seq, "event_frame": t, "gid": gid,
                        "variant": variant, "timing": timing,
                        "frame": int(frame_idx),
                        "patch_active": bool(patch_active["v"]),
                        "order": len(local_trace),
                    }
                    if det_out is not None:
                        rec["score"] = float(
                            det_out["scores"][0, args.slot]
                            .detach().float().cpu().item()
                        )
                        rec["box"] = [
                            float(x)
                            for x in det_out["bbox"][0, args.slot]
                            .detach().float().cpu().tolist()
                        ]
                        h = hashlib.sha256()
                        h.update(
                            det_out["bbox"][0].detach().float().cpu()
                            .numpy().astype("<f4").tobytes()
                        )
                        h.update(
                            det_out["scores"][0].detach().float().cpu()
                            .numpy().astype("<f4").tobytes()
                        )
                        rec["raw_hash"] = h.hexdigest()[:16]
                    local_trace.append(rec)
                    trace_rows.append(rec)
                    return det_out, pos

                model.run_backbone_and_detection = wrap

                def install_empty():
                    install_query_patch(
                        image, lambda: ([None], [None]), [args.slot], None
                    )

                use_query = variant != "A0"
                if use_query:
                    enc_model, ada = models[variant]
                q = ref = None
                if timing == "E1":
                    if use_query:
                        q, ref = compute_q(seq, t, hb, iw, ih)
                        install_query_patch(
                            image,
                            (lambda qq, rr: lambda: ([qq], [rr]))(q, ref),
                            [args.slot], ada,
                            gate_score=args.gate_score,
                            gate_iou=args.gate_iou,
                        )
                        patch_active["v"] = True
                    else:
                        install_empty()

                x1, y1, x2, y2 = hb
                req_prompt = dict(
                    type="add_prompt",
                    session_id=backend._session_id,
                    frame_index=t,
                    text="person",
                    bounding_boxes=[
                        [x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]
                    ],
                    bounding_box_labels=[1],
                    clear_old_boxes=True,
                )
                backend._predictor.handle_request(req_prompt)
                state = backend._predictor._all_inference_states[
                    backend._session_id
                ]["state"]
                state["action_history"].clear()

                if timing == "E0":
                    if use_query:
                        q, ref = compute_q(seq, t, hb, iw, ih)
                        install_query_patch(
                            image,
                            (lambda qq, rr: lambda: ([qq], [rr]))(q, ref),
                            [args.slot], ada,
                            gate_score=args.gate_score,
                            gate_iou=args.gate_iou,
                        )
                        patch_active["v"] = True
                    else:
                        install_empty()

                prev = None
                if ep.human_box is not None:
                    prev = ep.human_box.copy()
                ep.records[t] = FrameRecord(
                    frame=t, cand_boxes=[], prompt_box=hb.copy(),
                    delivered_box=prev.copy(),
                )
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
                    for response in backend._predictor.handle_stream_request(
                        request=req
                    ):
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
                    try:
                        backend.close()
                    except Exception:
                        pass
                    model.run_backbone_and_detection = orig_rbd

                row = {
                    "sequence": seq, "event_frame": t, "gid": gid,
                    "variant": variant, "timing": timing,
                }
                for h in (1, 3, 10, 30):
                    row[f"delivered_{h}"] = round(
                        recall_at(ep, gt, h, "delivered"), 3
                    )
                fc = [
                    false_capture(ep, gt, f)
                    for f in range(t + 1, t + args.horizon + 1)
                    if gt.get(f) is not None and gid in gt[f].gt_ids
                ]
                row["fc"] = round(float(np.mean(fc)), 3) if fc else ""
                t1 = [r for r in local_trace if r["frame"] == t + 1]
                row["t1_computed_with_patch"] = (
                    any(r["patch_active"] for r in t1) if t1 else "NOT_CALLED"
                )
                row["t1_score"] = (
                    round(t1[-1]["score"], 4) if t1 else ""
                )
                per_event.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "prefetch_execution_per_event.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(per_event[0].keys()))
        w.writeheader()
        w.writerows(per_event)
    with (OUT / "prefetch_state_trace.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for r in trace_rows:
            f.write(json.dumps(r) + "\n")
    # Summary audit CSV.
    summary = []
    for variant in variants:
        for timing in timings:
            rr = [r for r in per_event if r["variant"] == variant
                  and r["timing"] == timing]
            d30 = [float(r["delivered_30"]) for r in rr]
            fcs = [float(r["fc"]) for r in rr if r["fc"] != ""]
            t1p = sum(1 for r in rr if r["t1_computed_with_patch"] is True)
            summary.append(
                {
                    "variant": variant, "timing": timing,
                    "events": len(rr),
                    "delivered_30_mean": round(sum(d30) / len(d30), 4),
                    "fc_mean": round(sum(fcs) / len(fcs), 4) if fcs else "",
                    "t1_with_patch_count": t1p,
                }
            )
    with (OUT / "prefetch_execution_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print("WROTE prefetch audit CSVs + trace", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
