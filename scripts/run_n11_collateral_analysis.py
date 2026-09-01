#!/usr/bin/env python
"""N11 collateral / persistence analysis across variants."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import read_mot_rows


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
VARIANTS = {
    "global_b8": ROOT / "outputs/n10/real/human_b8",
    "auto_b8": ROOT / "outputs/n10/real/human_auto_b8",
    "local_perm_b8": ROOT / "outputs/n11/real/local_perm_b8",
    "local_decay_b8": ROOT / "outputs/n11/real/local_native0_decay_b8",
    "local_decay_b4": ROOT / "outputs/n11/real/local_native0_decay_b4",
    "local_decay_b2": ROOT / "outputs/n11/real/local_native0_decay_b2",
    "local_decay_b1": ROOT / "outputs/n11/real/local_native0_decay_b1",
}
OUT = ROOT / "outputs/n11"


def load_jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def pid_at(mot, f, box):
    best, bi = None, 0.5
    for pid, b in mot.get(f, []):
        x1, y1 = max(box[0], b[0]), max(box[1], b[1])
        x2, y2 = min(box[2], b[2]), min(box[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = (box[2] - box[0]) * (box[3] - box[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        iou = inter / union if union > 0 else 0
        if iou > bi:
            bi = iou
            best = pid
    return best


def analyze(variant, root):
    ds = DanceTrackDataset(str(DT), split="train")
    same_ret = defaultdict(list)
    same_tte, global_tte = [], []
    changes = defaultdict(int)
    target_err = []
    unrelated_err = []
    per_event = []
    for s in CAL:
        d = root / s
        events = load_jsonl(d / "interaction_events.jsonl")
        errs = load_jsonl(d / "verified_errors.jsonl")
        post = read_mot_rows(d / "post_mot" / f"{s}.txt")
        auto = read_mot_rows((ROOT / "outputs/n10/real/human_auto_b8") / s / "post_mot" / f"{s}.txt")
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
            t = e["frame"] - 1
            for off in (1, 3, 5, 10, 30):
                f0 = t + off
                gtf = gt.get(f0)
                if gtf is None or gid not in gtf.gt_ids:
                    continue
                gi = gtf.gt_ids.index(gid)
                pid = pid_at(post, f0, gtf.boxes[gi])
                same_ret[off].append(int(pid is not None and int(pid) == int(canon)))
            nxt_same = [f for f in err_frames_by_gid.get(gid, []) if f > e["frame"]]
            nxt_global = [f for f in sorted(errs_by_frame) if f > e["frame"]]
            same_tte.append(min(nxt_same) - e["frame"] if nxt_same else None)
            global_tte.append(min(nxt_global) - e["frame"] if nxt_global else None)
            target_err.append(
                sum(
                    1
                    for f in range(e["frame"] + 1, e["frame"] + 31)
                    for er in errs_by_frame.get(f, [])
                    if er.get("dataset_gt_id") == gid
                )
            )
            unrelated_err.append(
                sum(
                    1
                    for f in range(e["frame"] + 1, e["frame"] + 31)
                    for er in errs_by_frame.get(f, [])
                    if er.get("dataset_gt_id") != gid
                )
            )
            self_ch = other_ch = 0
            for off in range(1, 31):
                f0 = t + off
                gtf = gt.get(f0)
                if gtf is None or not gtf.boxes:
                    continue
                for gi, bb in enumerate(gtf.boxes):
                    g2 = gtf.gt_ids[gi]
                    pa = pid_at(auto, f0, bb)
                    ph = pid_at(post, f0, bb)
                    if pa is not None and pa != ph:
                        if g2 == gid:
                            self_ch += 1
                            changes["self"] += 1
                        else:
                            other_ch += 1
                            changes["other"] += 1
            per_event.append(
                {
                    "variant": variant,
                    "sequence": s,
                    "frame": e["frame"],
                    "event_type": e["event_type"],
                    "gid": gid,
                    "canonical_pid": canon,
                    "same_tte": min(nxt_same) - e["frame"] if nxt_same else None,
                    "global_tte": min(nxt_global) - e["frame"] if nxt_global else None,
                    "self_changes_30": self_ch,
                    "other_changes_30": other_ch,
                    "target_err_30": target_err[-1],
                    "unrelated_err_30": unrelated_err[-1],
                }
            )
    def med(v):
        v = [x for x in v if x is not None]
        return float(np.median(v)) if v else None
    def mean(v):
        v = [x for x in v if x is not None]
        return round(float(np.mean(v)), 2) if v else None
    n = len(per_event)
    return {
        "variant": variant,
        "n": n,
        "same_id_retention": {
            f"t+{o}": round(100.0 * sum(v) / len(v), 2) for o, v in sorted(same_ret.items()) if v
        },
        "same_id_tte": {"median": med(same_tte), "mean": mean(same_tte)},
        "global_tte": {"median": med(global_tte), "mean": mean(global_tte)},
        "target_err_30_mean": mean(target_err),
        "unrelated_err_30_mean": mean(unrelated_err),
        "self_changes_30_total": changes["self"],
        "other_changes_30_total": changes["other"],
        "self_changes_per_event": round(changes["self"] / max(1, n), 2),
        "other_changes_per_event": round(changes["other"] / max(1, n), 2),
        "per_event": per_event,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    events = []
    for v, root in VARIANTS.items():
        r = analyze(v, root)
        rows.append({k: vv for k, vv in r.items() if k != "per_event"})
        events.extend(r["per_event"])
    with (OUT / "collateral_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "sequence",
                "frame",
                "event_type",
                "gid",
                "canonical_pid",
                "same_tte",
                "global_tte",
                "self_changes_30",
                "other_changes_30",
                "target_err_30",
                "unrelated_err_30",
            ],
        )
        w.writeheader()
        w.writerows(events)
    (OUT / "collateral_analysis_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
