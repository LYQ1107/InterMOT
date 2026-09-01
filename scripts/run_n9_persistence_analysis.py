"""Retention / time-to-next-error / anchor-usage analysis for N9 variants."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset


ROOT = Path(".")
SPLIT = "val"
SEQS = sorted(
    p.name
    for p in (Path("/path/to/dancetrack") / "val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
OUT = ROOT / "outputs/n9/tables"


def load_jsonl(p):
    out = []
    if not p.exists():
        return out
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
                [
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[2]) + float(parts[4]),
                    float(parts[3]) + float(parts[5]),
                ],
            )
        )
    return out


def bestpid(gb, cand):
    best = None
    bi = 0.5
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


def analyze_variant(variant, budget, seqs):
    if variant == "n8":
        root = ROOT / "outputs/n8/real" / f"route_a_b{budget}"
        sub = "post_mot"
    else:
        root = ROOT / "outputs/n9/real" / f"{variant}_b{budget}"
        sub = "post_mot"
    ds = DanceTrackDataset("/path/to/dancetrack", split=SPLIT)
    ret = defaultdict(list)
    tte = []
    anchor = {"usage": 0, "relinks": 0, "via_anchor": 0}
    n_events = 0
    for s in seqs:
        d = root / s
        events = load_jsonl(d / "interaction_events.jsonl")
        post = load_mot(d / sub / f"{s}.txt")
        gt = ds.load_gt(s)
        n_events += len(events)
        if variant != "n8":
            relinks = load_jsonl(d / "relink_events.jsonl")
            anchor["relinks"] += len(relinks)
            anchor["via_anchor"] += sum(1 for r in relinks if r.get("via_anchor"))
        for e in events:
            gid = e.get("dataset_gt_id")
            canon = e.get("canonical_public_id") or e.get("public_mot_id")
            if gid is None or canon is None:
                continue
            for off in (0, 1, 3, 5, 10, 30):
                f0 = e["frame"] - 1 + off
                gtf = gt.get(f0)
                if gtf is None or gid not in gtf.gt_ids:
                    continue
                gi = gtf.gt_ids.index(gid)
                pid = bestpid(gtf.boxes[gi], post.get(f0, []))
                ret[off].append(pid is not None and int(pid) == int(canon))
        # time to next error for accepted events (same gid)
        errs = load_jsonl(d / "verified_errors.jsonl")
        by_gid = defaultdict(list)
        for er in errs:
            if er.get("dataset_gt_id") is not None and er["event_type"] in (
                "TEMPORAL_ID_BREAK",
                "RECOVERABLE_MISS",
                "TEMPORAL_ID_SWAP",
            ):
                by_gid[er["dataset_gt_id"]].append(er["frame"])
        for e in events:
            nxt = [f for f in by_gid.get(e.get("dataset_gt_id"), []) if f > e["frame"]]
            tte.append(min(nxt) - e["frame"] if nxt else None)
    tte_vals = [x for x in tte if x is not None]
    return {
        "variant": variant,
        "budget": budget,
        "accepted_events": n_events,
        "retention": {
            f"t+{off}": round(100.0 * sum(v) / len(v), 2) if v else None
            for off, v in sorted(ret.items())
        },
        "time_to_next_error": {
            "median": float(np.median(tte_vals)) if tte_vals else None,
            "mean": round(float(np.mean(tte_vals)), 2) if tte_vals else None,
            "n": len(tte_vals),
            "reached_end": sum(1 for x in tte if x is None),
        },
        "anchor_usage": anchor,
    }


def main():
    import os

    global SPLIT
    SPLIT = os.environ.get("N9_SPLIT", "val")
    seqs = sorted(os.environ.get("N9_SEQS", "").split()) or SEQS
    budgets = [int(x) for x in os.environ.get("N9_BUDGETS", "1 2 4 8").split()]
    variants = os.environ.get("N9_VARIANTS", "n8 reid pairwise auto proposed").split()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in variants:
        for b in budgets:
            rows.append(analyze_variant(v, b, seqs))
    with (OUT / "persistence_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "budget",
                "accepted_events",
                "retention_t1",
                "retention_t3",
                "retention_t5",
                "retention_t10",
                "retention_t30",
                "tte_median",
                "tte_mean",
                "relinks",
                "anchor_relinks",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["variant"],
                    r["budget"],
                    r["accepted_events"],
                    r["retention"].get("t+1"),
                    r["retention"].get("t+3"),
                    r["retention"].get("t+5"),
                    r["retention"].get("t+10"),
                    r["retention"].get("t+30"),
                    r["time_to_next_error"]["median"],
                    r["time_to_next_error"]["mean"],
                    r["anchor_usage"]["relinks"],
                    r["anchor_usage"]["via_anchor"],
                ]
            )
    (OUT / "persistence_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
