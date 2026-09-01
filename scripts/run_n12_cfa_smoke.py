#!/usr/bin/env python
"""N12 CFA feasibility smoke: no-update vs online-LoRA update on visual events."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(".")


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
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    import torch

    from sam3_intermot.adaptation.cfa_runner import CFARunner, iou, target_gt_boxes
    from sam3_intermot.adaptation.lora import inject_lora, lora_parameter_count
    from sam3_intermot.adaptation.sam3_loader import build_tracker_model
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset

    torch.cuda.set_device(args.gpu)
    model, report = build_tracker_model(device=f"cuda:{args.gpu}")
    print("load:", json.dumps(report, ensure_ascii=False))

    if args.surface == "decoder":
        targets = ("maskmem_backbone", "sam_mask_decoder", "transformer")
    elif args.surface == "memory":
        targets = ("maskmem_backbone",)
    elif args.surface == "backbone":
        targets = ("backbone",)
    else:
        raise SystemExit(f"unknown surface: {args.surface}")
    modified, params = inject_lora(model, targets, r=args.rank, alpha=1.0)
    print(f"LoRA targets: {len(modified)} modules, {lora_parameter_count(params)} params")
    print("sample:", modified[:8])

    events = load_events(args.seq)
    if not events:
        raise SystemExit(f"no accepted TRUE_MISS_NEW events in {args.seq}")
    ev = events[args.event_idx % len(events)]
    box = np.asarray(ev["gt_box"], dtype=float)
    print("event:", ev["frame"], ev["dataset_gt_id"], box.tolist())

    runner = CFARunner(model, split="train")
    res = runner.run_episode(
        sequence=args.seq,
        frame_idx=int(ev["frame"]),
        event_type=ev["event_type"],
        gid=int(ev["dataset_gt_id"]),
        human_box=box,
        horizon=args.horizon,
        lora_params=params,
        update_steps=args.steps,
        lr=args.lr,
    )
    ds = DanceTrackDataset(
        "/path/to/dancetrack", sequences=[], split="train"
    )
    gt = ds.load_gt(args.seq)
    frames = list(range(int(ev["frame"]) + 1, int(ev["frame"]) + args.horizon + 1))
    gt_boxes = target_gt_boxes(gt, int(ev["dataset_gt_id"]), frames)

    horizons = [1, 3, 5, 10, 30]
    rows = []
    for h in horizons:
        fs = [f for f in frames if f <= int(ev["frame"]) + h]
        if not fs:
            continue
        r0 = res.recall(res.no_update, gt_boxes)
        r1 = res.recall(res.update, gt_boxes)
        iou0 = np.mean(
            [
                (0.0 if res.no_update[f] is None or f not in gt_boxes else iou(res.no_update[f], gt_boxes[f]))
                for f in fs
            ]
        )
        iou1 = np.mean(
            [
                (0.0 if res.update[f] is None or f not in gt_boxes else iou(res.update[f], gt_boxes[f]))
                for f in fs
            ]
        )
        rows.append(
            {
                "horizon": h,
                "n_frames_with_gt": len([f for f in fs if f in gt_boxes]),
                "no_update_recall": round(r0, 3),
                "update_recall": round(r1, 3),
                "no_update_mean_iou": round(float(iou0), 3),
                "update_mean_iou": round(float(iou1), 3),
            }
        )
    print(json.dumps({"episode": res.sequence, "frame": res.frame,
                      "update_seconds": round(res.update_seconds, 2),
                      "horizons": rows}, indent=2, ensure_ascii=False))
    if args.debug:
        print(
            "init_frame_t box:",
            None if res.no_update.get(int(ev["frame"])) is None
            else np.round(res.no_update[int(ev["frame"])], 1).tolist(),
            "human_box:", box.tolist(),
        )
        for f in sorted(gt_boxes)[:5]:
            print(
                "frame", f, "gt", np.round(gt_boxes[f], 1).tolist(),
                "no_upd", None if res.no_update.get(f) is None else np.round(res.no_update[f], 1).tolist(),
                "upd", None if res.update.get(f) is None else np.round(res.update[f], 1).tolist(),
            )


if __name__ == "__main__":
    main()
