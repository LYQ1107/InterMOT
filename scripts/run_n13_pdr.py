#!/usr/bin/env python
"""N13 PDR benchmark: one human seed -> A0/A1/A2/A3/Oracle prompt policies.

Each episode starts with one text+box prompt at the human frame, then the
policy generates per-frame detector prompts for t+1..t+H (oracle uses future
GT, diagnostic only).  Writes event-level metrics, frame-level records, and
the detector causal audit JSON.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/n13"
CKPT = str(ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
DEFAULT_SEQS = (
    "dancetrack0074 dancetrack0075 dancetrack0080 dancetrack0082 "
    "dancetrack0083 dancetrack0086 dancetrack0087 dancetrack0096"
)


def load_events(seq: str, event_type: str, budget: str = "b8"):
    path = ROOT / "outputs/n10/real" / f"human_{budget}" / seq / "interaction_events.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("accepted") and (event_type == "ALL" or e.get("event_type") == event_type):
            events.append(e)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--event-idx", type=int, default=0)
    ap.add_argument("--event-type", default="TRUE_MISS_NEW")
    ap.add_argument("--policies", default="one_shot,last_box,motion,gated,oracle")
    ap.add_argument("--seqs", default=DEFAULT_SEQS)
    ap.add_argument("--out", default="pdr")
    args = ap.parse_args()

    import torch
    torch.cuda.set_device(0 if __import__("os").environ.get("CUDA_VISIBLE_DEVICES") else args.gpu)

    from sam3_intermot.adaptation.cfa_backend_runner import CFABackendRunner, _iou
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from sam3_intermot.detection_query.prompt_replay import (
        admission_hit,
        delivered_hit,
        false_capture,
        recall_at,
        run_pdr_episode,
    )

    runner = CFABackendRunner(checkpoint_path=CKPT, split="train")
    ds = DanceTrackDataset(
        "/path/to/dancetrack", sequences=[], split="train"
    )
    policies = [p for p in args.policies.split(",") if p]
    seqs = args.seqs.split()

    events: List[dict] = []
    for seq in seqs:
        evs = load_events(seq, args.event_type)
        if not evs:
            print(f"no {args.event_type} events: {seq}", flush=True)
            continue
        events.append({"seq": seq, "event": evs[args.event_idx % len(evs)]})
    print(f"PDR events: {len(events)}  policies: {policies}  horizon: {args.horizon}",
          flush=True)

    rows = []
    frame_rows = []
    audit_rows = []
    for evd in events:
        seq, ev = evd["seq"], evd["event"]
        gid = int(ev["dataset_gt_id"])
        box = np.asarray(ev["gt_box"], dtype=float)
        gt = ds.load_gt(seq)
        base = None
        for policy in policies:
            ep = run_pdr_episode(
                runner=runner,
                sequence=seq,
                frame_idx=int(ev["frame"]),
                event_type=ev["event_type"],
                gid=gid,
                human_box=box,
                policy=policy,
                gt=gt,
                horizon=args.horizon,
            )
            row = {
                "sequence": seq,
                "frame": ep.frame,
                "event_type": ep.event_type,
                "gid": gid,
                "policy": policy,
                "prompt_had_output": int(ep.prompt_had_output),
                "seconds": round(ep.seconds, 2),
                "admission_recall_1": round(recall_at(ep, gt, 1, "admission"), 3),
                "admission_recall_3": round(recall_at(ep, gt, 3, "admission"), 3),
                "admission_recall_5": round(recall_at(ep, gt, 5, "admission"), 3),
                "admission_recall_10": round(recall_at(ep, gt, 10, "admission"), 3),
                "admission_recall_30": round(recall_at(ep, gt, 30, "admission"), 3),
                "delivered_recall_1": round(recall_at(ep, gt, 1, "delivered"), 3),
                "delivered_recall_3": round(recall_at(ep, gt, 3, "delivered"), 3),
                "delivered_recall_5": round(recall_at(ep, gt, 5, "delivered"), 3),
                "delivered_recall_10": round(recall_at(ep, gt, 10, "delivered"), 3),
                "delivered_recall_30": round(recall_at(ep, gt, 30, "delivered"), 3),
            }
            # false-capture / identity over evaluated frames with GT
            fc = n = 0
            for f in range(ep.frame + 1, ep.frame + args.horizon + 1):
                if gt.get(f) is None or gid not in gt[f].gt_ids:
                    continue
                n += 1
                fc += int(false_capture(ep, gt, f))
            row["false_capture_rate"] = round(fc / max(1, n), 3)
            row["auto_query_frames"] = sum(
                1 for f, r in ep.records.items()
                if f > ep.frame and r.prompt_box is not None
            )
            rows.append(row)
            for f, rec in ep.records.items():
                if f <= ep.frame:
                    continue
                entry = gt.get(f)
                gt_box = None
                if entry is not None and gid in entry.gt_ids:
                    gt_box = entry.boxes[entry.gt_ids.index(gid)]
                frame_rows.append({
                    "sequence": seq, "frame": ep.frame, "gid": gid,
                    "policy": policy, "f": f,
                    "n_cands": len(rec.cand_boxes),
                    "prompt_box": json.dumps(None if rec.prompt_box is None
                                             else list(rec.prompt_box)),
                    "delivered_box": json.dumps(None if rec.delivered_box is None
                                                else list(rec.delivered_box)),
                    "gt_box": json.dumps(None if gt_box is None else list(gt_box)),
                    "admission_hit": int(admission_hit(ep, gt, f)),
                    "delivered_hit": int(delivered_hit(ep, gt, f)),
                    "false_capture": int(false_capture(ep, gt, f)),
                })
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if policy == "one_shot":
                base = ep
            elif base is not None:
                n_eval = 0
                adm_diff = deliv_diff = box_diff = 0
                for f in range(ep.frame + 1, ep.frame + args.horizon + 1):
                    if gt.get(f) is None or gid not in gt[f].gt_ids:
                        continue
                    n_eval += 1
                    a0 = admission_hit(base, gt, f)
                    ap = admission_hit(ep, gt, f)
                    d0 = delivered_hit(base, gt, f)
                    dp = delivered_hit(ep, gt, f)
                    adm_diff += int(a0 != ap)
                    deliv_diff += int(d0 != dp)
                    b0 = base.records.get(f).delivered_box if base.records.get(f) else None
                    bp = ep.records.get(f).delivered_box if ep.records.get(f) else None
                    if (b0 is None) != (bp is None):
                        box_diff += 1
                    elif b0 is not None and _iou(b0, bp) < 0.5:
                        box_diff += 1
                audit_rows.append({
                    "sequence": seq, "frame": ep.frame, "gid": gid,
                    "event_type": ep.event_type,
                    "baseline": "one_shot", "policy": policy,
                    "n_eval_frames": n_eval,
                    "admission_changed_frames": adm_diff,
                    "delivered_changed_frames": deliv_diff,
                    "box_changed_frames": box_diff,
                })
    runner.close()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out
    with (OUT / f"{tag}_events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / f"{tag}_frame_level.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        w.writeheader()
        w.writerows(frame_rows)
    if audit_rows:
        with (OUT / f"{tag}_causal_audit.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            w.writeheader()
            w.writerows(audit_rows)

    summary = {}
    for policy in policies:
        pr = [r for r in rows if r["policy"] == policy]
        summary[policy] = {
            "n_events": len(pr),
            "mean_admission_recall": {
                h: round(float(np.mean([r[f"admission_recall_{h}"] for r in pr])), 3)
                for h in (1, 3, 5, 10, 30)
            },
            "mean_delivered_recall": {
                h: round(float(np.mean([r[f"delivered_recall_{h}"] for r in pr])), 3)
                for h in (1, 3, 5, 10, 30)
            },
            "mean_false_capture_rate": round(
                float(np.mean([r["false_capture_rate"] for r in pr])), 3
            ),
            "total_auto_query_frames": int(sum(r["auto_query_frames"] for r in pr)),
            "total_seconds": round(float(sum(r["seconds"] for r in pr)), 1),
        }
    summary["causal_audit"] = {
        "n_pairs": len(audit_rows),
        "admission_changed_frames_total": sum(
            r["admission_changed_frames"] for r in audit_rows
        ),
        "delivered_changed_frames_total": sum(
            r["delivered_changed_frames"] for r in audit_rows
        ),
        "box_changed_frames_total": sum(r["box_changed_frames"] for r in audit_rows),
    }
    with (OUT / f"{tag}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
