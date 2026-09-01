"""N14 PIR benchmark: one human write -> persistent detector query.

Compares official one-shot A0 (no query) vs N14 query injection on the N13
calibration TRUE_MISS_NEW events (8 sequences x 2 events).  GT is used only
for evaluation; the query is written once from the human frame.
"""

import argparse
import copy
import csv
import json
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="outputs/n14/models/human_write_encoder_f0_v3.pt")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--slot", type=int, default=199)
    ap.add_argument("--events", default="all")
    ap.add_argument("--branches", default="A0,N14")
    ap.add_argument("--slot-log", default="")
    ap.add_argument("--timing", default="E0", choices=["E0", "E1"])
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner,
        parse_raw_outputs,
    )
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.detection_query.prompt_replay import (
        _best_delivery,
        FrameRecord,
        PDREpisode,
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
    adapter = SlotHeadAdapter(
        d_model=d_model, hidden=hidden // 4
    ).cuda().eval()
    encoder.load_state_dict(ck["encoder_state"])
    adapter.load_state_dict(ck["adapter_state"])

    ds = DanceTrackDataset(str(DT), sequences=None, split="train")

    # Distinct N13 calibration events (one_shot rows).
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

    def run_episode(seq, t, gid, human_box, use_query, slot_records):
        video = str(DT / "train" / seq / "img1")
        backend.start_video(video)
        model.use_batched_grounding = False
        iw, ih = backend._frame_w, backend._frame_h
        ep = PDREpisode(
            sequence=seq, frame=t, event_type="TRUE_MISS_NEW", gid=gid,
            human_box=np.asarray(human_box, dtype=float), policy="n14_query",
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
        prompt_resp = backend._predictor.handle_request(req_prompt)
        cands0 = parse_raw_outputs(prompt_resp, frame_size=(iw, ih))
        state = backend._predictor._all_inference_states[
            backend._session_id
        ]["state"]
        state["action_history"].clear()
        prev = None
        if cands0:
            best_box = max(cands0, key=lambda c: _iou_xyxy(c[1], ep.human_box))
            if _iou_xyxy(best_box[1], ep.human_box) >= 0.3:
                prev = np.asarray(best_box[1], dtype=float).copy()
        if prev is None:
            prev = ep.human_box.copy()
        ep.records[t] = FrameRecord(
            frame=t, cand_boxes=[np.asarray(b, dtype=float) for _, b in cands0],
            prompt_box=ep.human_box.copy(), delivered_box=prev.copy(),
        )
        t0 = time.time()
        slot_out = {}
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
                try:
                    sl = det_out["scores"][0, args.slot].detach().float().cpu().item()
                    sb = det_out["bbox"][0, args.slot].detach().float().cpu().numpy()
                    slot_records.append(
                        {
                            "frame": int(frame_idx),
                            "score": float(sl),
                            "box": [float(x) for x in sb],
                        }
                    )
                    slot_out[int(frame_idx)] = (float(sl), sb)
                except Exception:
                    pass
            return det_out, pos

        model.run_backbone_and_detection = wrap

        ref_state = {"box": None}

        def compute_and_install():
            ib = state["input_batch"]
            fin = clone_find_input(ib.find_inputs[t], img_id=0)
            if hasattr(ib.img_batch, "tensors"):
                img_t = ib.img_batch.tensors[t].unsqueeze(0).clone().to("cuda")
            else:
                img_t = ib.img_batch[t].unsqueeze(0).clone().to("cuda")
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
            box_norm = np.asarray(
                [x1 / iw, y1 / ih, x2 / iw, y2 / ih], dtype=float
            )
            roi = roi_pool_feature(
                enc["encoder_hidden_states"], enc, box_norm
            )
            q = encoder(roi.float()).to(torch.float32)
            ref = torch.as_tensor(
                cxcywh_norm(ep.human_box, iw, ih),
                dtype=torch.float32, device="cuda",
            )
            ref_state["box"] = ref

            def bank():
                return ([q], [ref_state["box"]])

            install_query_patch(image, bank, [args.slot], adapter)

        if use_query and args.timing == "E1":
            compute_and_install()
        if not use_query:
            install_query_patch(
                image, lambda: ([None], [None]), [args.slot], None
            )
        if use_query and args.timing == "E0":
            compute_and_install()

        nf = t + 1
        set_frame_geometric_prompt(runner, nf, None)
        if use_query:
            invalidate_detector_prefetch(runner, t)
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
                if use_query and f > t:
                    so = slot_out.get(f)
                    if so is not None and so[0] > 0.5 and delivered is not None:
                        if _iou_xyxy(so[1], delivered) >= 0.5:
                            ref_state["box"] = torch.as_tensor(
                                cxcywh_norm(delivered, iw, ih),
                                dtype=torch.float32, device="cuda",
                            )
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
        ep.seconds = time.time() - t0
        return ep

    rows = []
    for seq, t, gid in evs:
        gt = ds.load_gt(seq)
        hb = None
        for path in ("outputs/n13/pdr_idx0_events.csv", "outputs/n13/pdr_idx1_events.csv"):
            with open(ROOT / path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if (r["sequence"], int(r["frame"]), int(r["gid"])) == (seq, t, gid):
                        hb = np.asarray(
                            [
                                float(gt[t].boxes[gt[t].gt_ids.index(gid)][0]),
                                float(gt[t].boxes[gt[t].gt_ids.index(gid)][1]),
                                float(gt[t].boxes[gt[t].gt_ids.index(gid)][2]),
                                float(gt[t].boxes[gt[t].gt_ids.index(gid)][3]),
                            ],
                            dtype=float,
                        )
                        break
                if hb is not None:
                    break
        if hb is None:
            hb = np.asarray(
                gt[t].boxes[gt[t].gt_ids.index(gid)], dtype=float
            )
        for branch in args.branches.split(","):
            slot_records = []
            ep = run_episode(
                seq, t, gid, hb, use_query=(branch == "N14"), slot_records=slot_records,
            )
            row = {
                "sequence": seq, "frame": t, "gid": gid, "branch": branch,
                "seconds": round(ep.seconds, 2),
            }
            for h in (1, 3, 5, 10, 30):
                row[f"admission_{h}"] = round(
                    recall_at(ep, gt, h, "admission"), 3
                )
                row[f"delivered_{h}"] = round(
                    recall_at(ep, gt, h, "delivered"), 3
                )
            fc = [
                false_capture(ep, gt, f)
                for f in range(t + 1, t + args.horizon + 1)
                if gt.get(f) is not None and gid in gt[f].gt_ids
            ]
            row["false_capture_rate"] = (
                round(float(np.mean(fc)), 3) if fc else ""
            )
            if args.slot_log and branch == "N14":
                with open(ROOT / args.slot_log, "a", encoding="utf-8") as f:
                    for rec in slot_records:
                        f.write(
                            json.dumps(
                                {
                                    "sequence": seq, "frame": t, "gid": gid,
                                    **rec,
                                }
                            )
                            + "\n"
                        )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pir_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"WROTE {path}", flush=True)
    runner.close()


def _iou_xyxy(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


if __name__ == "__main__":
    main()
