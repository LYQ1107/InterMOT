#!/usr/bin/env python
"""N15 online rule commit: identity-conditioned shadow selection + G_commit.

At each future frame: official AUTO runs normally; shadow state selects the
detector candidate with the highest pretrained identity cosine >= tau; the
rule commits the shadow box only when AUTO is uncertain (delivered box far
from the shadow candidate) and the shadow candidate's detector score >= s0.
Commits update the delivered trajectory online (prev box), so the evaluation
is a true causal intervention.
"""

import argparse
import csv
import json
import sys
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
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--s0", type=float, default=0.30)
    ap.add_argument("--max-iou-auto-shadow", type=float, default=0.5)
    ap.add_argument("--out-tag", default="rule_commit")
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
    print(f"events: {len(evs)} tau={args.tau} s0={args.s0}", flush=True)

    commit_rows = []
    per_event_rows = []
    inter_util = {}
    a0_ref = {}
    if (OUT / "selection_oracle_commit.csv").exists():
        with (OUT / "selection_oracle_commit.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                a0_ref[(r["sequence"], int(r["event_frame"]), int(r["gid"]))] = r
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
            human_box=hb.copy(), policy="n15_rule_commit",
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
        nf = t + 1
        set_frame_geometric_prompt(runner, nf, None)
        req = dict(
            type="propagate_in_video",
            session_id=backend._session_id,
            propagation_direction="forward",
            start_frame_index=t,
            max_frame_num_to_track=None,
        )
        for response in backend._predictor.handle_stream_request(request=req):
            f = int(response["frame_index"])
            cands = parse_raw_outputs(response, frame_size=(iw, ih))
            cand_boxes = [np.asarray(b, dtype=float) for _, b in cands]
            delivered = _best_delivery(prev, cand_boxes)
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
                if best_i >= 0 and gscores[best_i] >= args.tau else None
            )
            shadow_det_score = cands[best_i][0] if best_i >= 0 else 0.0
            shadow_score = gscores[best_i] if best_i >= 0 else -1.0
            uncertain = (
                delivered is None or shadow_box is None
                or iou_xyxy(delivered, shadow_box) < args.max_iou_auto_shadow
            )
            commit = (
                shadow_box is not None and uncertain
                and shadow_det_score >= args.s0
            )
            if commit:
                delivered = shadow_box.copy()
                prev = delivered.copy()
            ep.records[f] = FrameRecord(
                frame=f, cand_boxes=cand_boxes, prompt_box=None,
                delivered_box=delivered,
            )
            entry = gt.get(f)
            target_box = (
                np.asarray(entry.boxes[entry.gt_ids.index(gid)], dtype=float)
                if entry is not None and gid in entry.gt_ids else None
            )
            shadow_correct = (
                shadow_box is not None and target_box is not None
                and iou_xyxy(shadow_box, target_box) >= 0.5
            )
            delivered_correct = (
                delivered is not None and target_box is not None
                and iou_xyxy(delivered, target_box) >= 0.5
            )
            commit_rows.append(
                {
                    "sequence": seq, "event_frame": t, "gid": gid, "frame": f,
                    "n_cands": len(cands), "gid_cos": round(shadow_score, 4),
                    "shadow_det_score": round(shadow_det_score, 4),
                    "commit": int(commit), "shadow_correct": int(shadow_correct),
                    "delivered_correct": int(delivered_correct),
                }
            )
            if commit:
                k = "BENEFICIAL_COMMIT" if delivered_correct else "HARMFUL_COMMIT"
            else:
                k = "NO_COMMIT"
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
        row = {"sequence": seq, "event_frame": t, "gid": gid}
        for h in (1, 3, 5, 10, 30):
            row[f"rule_delivered_{h}"] = round(recall_at(ep, gt, h, "delivered"), 3)
            ref = a0_ref.get((seq, t, gid))
            row[f"a0_delivered_{h}"] = (
                float(ref[f"a0_delivered_{h}"]) if ref is not None else ""
            )
        per_event_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag
    with (OUT / f"{tag}_commit_frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(commit_rows[0].keys()))
        w.writeheader()
        w.writerows(commit_rows)
    with (OUT / f"{tag}_events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_event_rows[0].keys()))
        w.writeheader()
        w.writerows(per_event_rows)
    with (OUT / f"{tag}_utility.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for k in sorted(inter_util):
            w.writerow([k, inter_util[k]])
    print("WROTE rule-commit CSVs", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
