#!/usr/bin/env python
"""N12 single-event online-LoRA feasibility: no-update vs update branch."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/n12"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")


def _iou(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def branch_delta(ep_a, ep_b, frames):
    both = [f for f in frames if ep_a.frames.get(f) is not None and ep_b.frames.get(f) is not None]
    if not both:
        return {"mean_iou_between_branches": None, "n_frames_changed": None, "n_frames_both": 0}
    ious = [_iou(ep_a.frames[f], ep_b.frames[f]) for f in both]
    return {
        "mean_iou_between_branches": round(float(np.mean(ious)), 4),
        "n_frames_changed": int(sum(1 for v in ious if v < 0.99)),
        "n_frames_both": len(both),
    }


def load_events(seq: str, event_type: str = "TRUE_MISS_NEW", budget: str = "b8"):
    path = ROOT / "outputs/n10/real" / f"human_{budget}" / seq / "interaction_events.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("accepted") and e.get("event_type") == event_type:
            events.append(e)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="dancetrack0074")
    ap.add_argument("--event-idx", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--surface", default="decoder")
    ap.add_argument("--out", default="cfa_lora.csv")
    args = ap.parse_args()

    from sam3_intermot.adaptation.cfa_backend_runner import (
        CFABackendRunner,
        parse_raw_outputs,
        recall_at,
    )
    from sam3_intermot.adaptation.lora import inject_lora
    from sam3_intermot.adaptation.online_update import (
        copy_lora_params,
        inner_update_shadow,
        load_frame_tensor,
    )
    from sam3_intermot.adaptation.sam3_loader import build_tracker_model
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset

    import torch
    torch.cuda.set_device(args.gpu)

    events = load_events(args.seq)
    if not events:
        raise SystemExit(f"no accepted TRUE_MISS_NEW events in {args.seq}")
    ev = events[args.event_idx % len(events)]
    box = np.asarray(ev["gt_box"], dtype=float)
    gt = DanceTrackDataset(
        "/path/to/dancetrack", sequences=[], split="train"
    ).load_gt(args.seq)

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    backend = runner._ensure_backend()
    video = str(
        Path("/path/to/dancetrack/train") / args.seq / "img1"
    )
    backend.start_video(video)
    iw, ih = backend._frame_w, backend._frame_h
    x1, y1, x2, y2 = box
    def make_prompt_req():
        return dict(
            type="add_prompt",
            session_id=backend._session_id,
            frame_index=int(ev["frame"]),
            text="person",
            bounding_boxes=[[x1 / iw, y1 / ih, (x2 - x1) / iw, (y2 - y1) / ih]],
            bounding_box_labels=[1],
            clear_old_boxes=True,
        )

    prompt_req = make_prompt_req()
    prompt_resp = backend._predictor.handle_request(prompt_req)
    cands0 = parse_raw_outputs(prompt_resp, frame_size=(iw, ih))
    if not cands0:
        print(json.dumps({"event": int(ev["frame"]), "prompt_failed": True}))
        runner.close()
        return

    # Branch A: no-update
    ep_a = runner.run_episode(
        sequence=args.seq,
        frame_idx=int(ev["frame"]),
        event_type=ev["event_type"],
        gid=int(ev["dataset_gt_id"]),
        human_box=box,
        horizon=args.horizon,
    )

    # Branch B: train LoRA on a shadow standalone tracker, copy back to the
    # official full pipeline, then re-prompt + propagate with updated weights.
    demo = backend._predictor.model
    tracker_model = demo.tracker.model
    if args.surface == "decoder":
        targets = (
            "interactive_sam_mask_decoder",
            "interactive_sam_prompt_encoder",
            "maskmem_backbone",
            "sam_mask_decoder",
            "transformer",
        )
    elif args.surface == "memory":
        targets = ("maskmem_backbone",)
    else:
        raise SystemExit(f"unknown surface: {args.surface}")
    shadow, _rep = build_tracker_model(device=f"cuda:{args.gpu}")
    modified_shadow, shadow_params = inject_lora(
        shadow, targets, r=args.rank, alpha=1.0
    )
    modified, params = inject_lora(tracker_model, targets, r=args.rank, alpha=1.0)
    frame_tensor, orig_w, orig_h = load_frame_tensor(
        video, int(ev["frame"]), image_size=1008
    )
    frame_tensor = frame_tensor.cuda()
    loss, upd_s = inner_update_shadow(
        shadow,
        frame_tensor,
        box,
        shadow_params,
        orig_h,
        orig_w,
        steps=args.steps,
        lr=args.lr,
    )
    n_copied = copy_lora_params(shadow, tracker_model)
    del shadow
    ep_b = runner.run_episode(
        sequence=args.seq,
        frame_idx=int(ev["frame"]),
        event_type=ev["event_type"],
        gid=int(ev["dataset_gt_id"]),
        human_box=box,
        horizon=args.horizon,
    )
    runner.close()

    row = {
        "sequence": args.seq,
        "frame": int(ev["frame"]),
        "event_type": ev["event_type"],
        "gid": int(ev["dataset_gt_id"]),
        "surface": args.surface,
        "rank": args.rank,
        "steps": args.steps,
        "lr": args.lr,
        "update_seconds": round(upd_s, 3),
        "inner_loss": round(loss, 5),
        "n_lora_params": sum(p.numel() for p in params),
        "n_lora_copied": n_copied,
        **{f"no_update_recall_{h}": round(recall_at(ep_a, gt, h), 3) for h in (1, 3, 5, 10, 30)},
        **{f"update_recall_{h}": round(recall_at(ep_b, gt, h), 3) for h in (1, 3, 5, 10, 30)},
        **{f"delta_recall_{h}": round(recall_at(ep_b, gt, h) - recall_at(ep_a, gt, h), 3) for h in (1, 3, 5, 10, 30)},
        **branch_delta(
            ep_a, ep_b, range(int(ev["frame"]) + 1, int(ev["frame"]) + args.horizon + 1)
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / args.out
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
