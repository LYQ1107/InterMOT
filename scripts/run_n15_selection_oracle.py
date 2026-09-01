#!/usr/bin/env python
"""N15 identity-conditioned shadow selection + oracle selective commit.

The injected-decoder-query slot branch (I2Q -> frozen slot output) was
diagnosed as degenerate (see debug_n15_slot.py / N15 report).  This runner
tests the alternative shadow state: rank the official detector candidates by
pretrained identity cosine with the human anchor H_i, keep the top candidate
as the shadow state, and simulate oracle selective commit (GT-only).
"""

import argparse
import csv
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
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--events", default="all")
    ap.add_argument("--gid-min-cos", type=float, default=0.70)
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
    from scripts.run_n15_extract_features import build_clipreid
    import torchvision.transforms as T

    clip = build_clipreid(str(CLIP_CKPT), "cuda")
    clip_tf = T.Compose(
        [
            T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def clip_feat(img, box):
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

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    backend._ensure_model()
    model = backend._predictor.model
    model.use_batched_grounding = False
    model.eval()
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

    commit_rows = []
    per_event_rows = []
    inter_util = {}
    for seq, t, gid in evs:
        video = str(DT / "train" / seq / "img1")
        backend.start_video(video)
        iw, ih = backend._frame_w, backend._frame_h
        gt = ds.load_gt(seq)
        hb = np.asarray(gt[t].boxes[gt[t].gt_ids.index(gid)], dtype=float)
        img_t = Image.open(f"{video}/{t + 1:08d}.jpg").convert("RGB")
        hf = clip_feat(img_t, hb)
        ep = PDREpisode(
            sequence=seq, frame=t, event_type="TRUE_MISS_NEW", gid=gid,
            human_box=hb.copy(), policy="n15_selection",
        )
        x1, y1, x2, y2 = hb
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
        prev = hb.copy()
        ep.records[t] = FrameRecord(
            frame=t, cand_boxes=[], prompt_box=hb.copy(), delivered_box=prev.copy()
        )
        cand_log = []
        nf = t + 1
        set_frame_geometric_prompt(runner, nf, None)
        req = dict(
            type="propagate_in_video",
            session_id=backend._session_id,
            propagation_direction="forward",
            start_frame_index=t,
            max_frame_num_to_track=None,
        )
        t0 = time.time()
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
            # shadow selection over this frame's candidates
            img_f = Image.open(f"{video}/{f + 1:08d}.jpg").convert("RGB")
            gscores = []
            for _, cb in cands:
                cb = np.asarray(cb, dtype=float)
                fv = clip_feat(img_f, cb)
                g = float((fv[0] * hf[0]).sum().item()) if fv is not None else -1.0
                gscores.append(g)
            best_i = int(np.argmax(gscores)) if gscores else -1
            shadow_box = (
                np.asarray(cands[best_i][1], dtype=float)
                if best_i >= 0 and gscores[best_i] >= args.gid_min_cos
                else None
            )
            shadow_score = gscores[best_i] if best_i >= 0 else -1.0
            entry = gt.get(f)
            target_box = (
                np.asarray(entry.boxes[entry.gt_ids.index(gid)], dtype=float)
                if entry is not None and gid in entry.gt_ids else None
            )
            auto_correct = (
                delivered is not None and target_box is not None
                and iou_xyxy(delivered, target_box) >= 0.5
            )
            shadow_correct = (
                shadow_box is not None and target_box is not None
                and iou_xyxy(shadow_box, target_box) >= 0.5
            )
            commit = (not auto_correct) and shadow_correct
            committed = shadow_box if commit else delivered
            committed_correct = (
                committed is not None and target_box is not None
                and iou_xyxy(committed, target_box) >= 0.5
            )
            commit_rows.append(
                {
                    "sequence": seq, "event_frame": t, "gid": gid, "frame": f,
                    "n_cands": len(cands), "best_gid_cos": round(shadow_score, 4),
                    "auto_correct": int(auto_correct), "shadow_correct": int(shadow_correct),
                    "commit": int(commit), "committed_correct": int(committed_correct),
                    "shadow_box": json.dumps(
                        [round(float(v), 1) for v in shadow_box]
                    ) if shadow_box is not None else "",
                }
            )
            if auto_correct:
                k = "UNNECESSARY_INTERVENTION" if commit else "CORRECT_ABSTENTION"
            elif commit:
                k = "BENEFICIAL_COMMIT"
            elif shadow_correct:
                k = "MISSED_OPPORTUNITY"
            else:
                k = "NEUTRAL_ABSTENTION"
            inter_util[k] = inter_util.get(k, 0) + 1
            if f >= t + args.horizon:
                break
            nf2 = f + 1
            set_frame_geometric_prompt(runner, nf2, None)
            invalidate_detector_prefetch(runner, f)
        try:
            backend.close()
        except Exception:
            pass

        def oracle_recall(h):
            hits = n = 0
            for f in range(t + 1, t + h + 1):
                entry = gt.get(f)
                if entry is None or gid not in entry.gt_ids:
                    continue
                n += 1
                target = np.asarray(entry.boxes[entry.gt_ids.index(gid)], dtype=float)
                rec = ep.records.get(f)
                auto_box = None if rec is None else rec.delivered_box
                if auto_box is not None and iou_xyxy(auto_box, target) >= 0.5:
                    hits += 1
                    continue
                shadow = None
                for r in commit_rows:
                    if (r["sequence"], r["event_frame"], r["gid"], r["frame"]) == (seq, t, gid, f):
                        if r["commit"]:
                            shadow = np.asarray(json.loads(r["shadow_box"]), dtype=float)
                        break
                if shadow is not None and iou_xyxy(shadow, target) >= 0.5:
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
                "a0_delivered_1": round(recall_at(ep, gt, 1, "delivered"), 3),
                "a0_delivered_3": round(recall_at(ep, gt, 3, "delivered"), 3),
                "a0_delivered_5": round(recall_at(ep, gt, 5, "delivered"), 3),
                "a0_delivered_10": round(recall_at(ep, gt, 10, "delivered"), 3),
                "a0_delivered_30": round(recall_at(ep, gt, 30, "delivered"), 3),
            }
        )
        print(json.dumps(per_event_rows[-1], ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "selection_commit_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(commit_rows[0].keys()))
        w.writeheader()
        w.writerows(commit_rows)
    with (OUT / "selection_oracle_commit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_event_rows[0].keys()))
        w.writeheader()
        w.writerows(per_event_rows)
    with (OUT / "selection_intervention_utility.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for k in sorted(inter_util):
            w.writerow([k, inter_util[k]])
    print("WROTE selection oracle CSVs", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
