#!/usr/bin/env python
"""Re-audit N10 TTE/retention: same-identity vs global error density."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset


ROOT = Path(".")
DT = Path("/path/to/dancetrack")
CAL = [
    "dancetrack0074",
    "dancetrack0075",
    "dancetrack0080",
    "dancetrack0082",
    "dancetrack0083",
    "dancetrack0086",
    "dancetrack0087",
    "dancetrack0096",
    "dancetrack0098",
    "dancetrack0099",
]
VARIANTS = ["human_b1", "human_b2", "human_b4", "human_b8", "human_auto_b4", "human_auto_b8"]
OUT = ROOT / "outputs/n11"


def load_jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def load_mot(p):
    out = defaultdict(list)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        out[int(float(parts[0])) - 1].append(
            (
                int(float(parts[1])),
                [float(parts[2]), float(parts[3]), float(parts[2]) + float(parts[4]), float(parts[3]) + float(parts[5])],
            )
        )
    return out


def bestpid(gb, cand):
    best, bi = None, 0.5
    gx1, gy1, gx2, gy2 = gb
    for pid, b in cand:
        x1, y1, x2, y2 = b
        ix1, iy1 = max(gx1, x1), max(gy1, y1)
        ix2, iy2 = min(gx2, x2), min(gy2, y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (gx2 - gx1) * (gy2 - gy1) + (x2 - x1) * (y2 - y1) - inter
        iou = inter / union if union > 0 else 0
        if iou > bi:
            bi = iou
            best = pid
    return best


def reaudit(variant):
    ds = DanceTrackDataset(str(DT), split="train")
    same_ret = defaultdict(list)
    same_tte, global_tte = [], []
    target_future_err, unrelated_future_err = [], []
    per_event = []
    for s in CAL:
        d = ROOT / "outputs/n10/real" / variant / s
        events = load_jsonl(d / "interaction_events.jsonl")
        errs = load_jsonl(d / "verified_errors.jsonl")
        post = load_mot(d / "post_mot" / f"{s}.txt")
        gt = ds.load_gt(s)
        errs_by_frame = defaultdict(list)
        for er in errs:
            if er["event_type"] in ("TEMPORAL_ID_BREAK", "RECOVERABLE_MISS", "TEMPORAL_ID_SWAP", "TRUE_MISS_NEW"):
                errs_by_frame[er["frame"]].append(er)
        err_frames_by_gid = defaultdict(list)
        for er in errs:
            if er.get("dataset_gt_id") is not None and er["event_type"] in (
                "TEMPORAL_ID_BREAK",
                "RECOVERABLE_MISS",
                "TEMPORAL_ID_SWAP",
            ):
                err_frames_by_gid[er["dataset_gt_id"]].append(er["frame"])
        for e in events:
            gid = e.get("dataset_gt_id")
            canon = e.get("canonical_public_id") or e.get("public_mot_id")
            if gid is None or canon is None:
                continue
            t = e["frame"]
            # same-ID retention / correct frames
            ret = {}
            for off in (1, 3, 5, 10, 30):
                f0 = t - 1 + off
                gtf = gt.get(f0)
                if gtf is None or gid not in gtf.gt_ids:
                    ret[off] = None
                    continue
                gi = gtf.gt_ids.index(gid)
                pid = bestpid(gtf.boxes[gi], post.get(f0, []))
                ret[off] = int(pid is not None and int(pid) == int(canon))
                same_ret[off].append(ret[off])
            nxt_same = [f for f in err_frames_by_gid.get(gid, []) if f > t]
            nxt_global = [f for f in sorted(errs_by_frame) if f > t]
            same_tte.append(min(nxt_same) - t if nxt_same else None)
            global_tte.append(min(nxt_global) - t if nxt_global else None)
            window = range(t + 1, t + 31)
            target_err = sum(1 for f in window for er in errs_by_frame.get(f, []) if er.get("dataset_gt_id") == gid)
            unrelated_err = sum(
                1 for f in window for er in errs_by_frame.get(f, []) if er.get("dataset_gt_id") != gid
            )
            target_future_err.append(target_err)
            unrelated_future_err.append(unrelated_err)
            per_event.append(
                {
                    "variant": variant,
                    "sequence": s,
                    "frame": t,
                    "event_type": e["event_type"],
                    "gid": gid,
                    "canonical_pid": canon,
                    "ret_t1": ret[1],
                    "ret_t3": ret[3],
                    "ret_t5": ret[5],
                    "ret_t10": ret[10],
                    "ret_t30": ret[30],
                    "same_tte": min(nxt_same) - t if nxt_same else None,
                    "global_tte": min(nxt_global) - t if nxt_global else None,
                    "target_future_err_30": target_err,
                    "unrelated_future_err_30": unrelated_err,
                }
            )
    def med(vals):
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 2) if vals else None
    return {
        "variant": variant,
        "n": len(per_event),
        "same_id_retention": {f"t+{o}": round(100.0 * sum(v) / len(v), 2) for o, v in sorted(same_ret.items()) if v},
        "same_id_tte": {"median": med(same_tte), "mean": mean(same_tte), "n": len([v for v in same_tte if v is not None])},
        "global_tte": {"median": med(global_tte), "mean": mean(global_tte), "n": len([v for v in global_tte if v is not None])},
        "target_future_err_30_mean": mean(target_future_err),
        "unrelated_future_err_30_mean": mean(unrelated_future_err),
        "per_event": per_event,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    all_events = []
    for v in VARIANTS:
        r = reaudit(v)
        rows.append({k: vv for k, vv in r.items() if k != "per_event"})
        all_events.extend(r["per_event"])
    with (OUT / "n10_metric_reaudit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "sequence",
                "frame",
                "event_type",
                "gid",
                "canonical_pid",
                "ret_t1",
                "ret_t3",
                "ret_t5",
                "ret_t10",
                "ret_t30",
                "same_tte",
                "global_tte",
                "target_future_err_30",
                "unrelated_future_err_30",
            ],
        )
        w.writeheader()
        w.writerows(all_events)
    (OUT / "n10_metric_reaudit_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
