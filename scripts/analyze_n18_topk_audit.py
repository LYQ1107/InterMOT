#!/usr/bin/env python
"""Merge the per-shard GFN top-K audit and split F3 (detector miss) from
F4 (ranking miss), both at attempt level and at lost-episode level."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
OUT = ROOT / "outputs/n18/tables"


def main():
    rows = []
    for p in sorted(OUT.glob("gfn_full_loop_topk_audit_s*.csv")):
        with p.open(encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    merged = OUT / "gfn_full_loop_topk_audit.csv"
    with merged.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    present = [r for r in rows if r["gt_present"] == "1"]
    absent = [r for r in rows if r["gt_present"] != "1"]

    def rate(rows_, key):
        return np.mean([float(r[key]) for r in rows_]) if rows_ else None

    attempt = {
        "level": "attempt", "n_attempts": len(rows),
        "n_present": len(present), "n_absent": len(absent),
        "any_det_correct": rate(present, "any_det_correct"),
        "top1_correct": rate(present, "top1_correct"),
        "top3_correct": rate(present, "top3_correct"),
        "top5_correct": rate(present, "top5_correct"),
        "top10_correct": rate(present, "top10_correct"),
        "F3_detector_miss": rate(present, "any_det_correct") is not None
        and float(np.mean([1 - float(r["any_det_correct"])
                           for r in present])),
        "F4_ranking_miss": float(np.mean([
            (1 - float(r["any_det_correct"])) * 0
            + (1 - float(r["top1_correct"])) * float(r["any_det_correct"])
            for r in present])) if present else None,
    }

    # ---- episode level: consecutive attempts with frame gap <= 10
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["sequence"], r["gid"])].append(r)
    eps = []
    for (seq, gid), rs in by_key.items():
        rs = sorted(rs, key=lambda r: int(r["frame"]))
        cur = [rs[0]]
        for r in rs[1:]:
            if int(r["frame"]) - int(cur[-1]["frame"]) <= 10:
                cur.append(r)
            else:
                eps.append((seq, gid, cur))
                cur = [r]
        eps.append((seq, gid, cur))
    ep_rows = []
    for seq, gid, rs in eps:
        pres = [r for r in rs if r["gt_present"] == "1"]
        ep_rows.append({
            "sequence": seq, "gid": gid, "n_attempts": len(rs),
            "n_present_attempts": len(pres),
            "any_top1": int(any(float(r["top1_correct"]) for r in pres)),
            "any_top3": int(any(float(r["top3_correct"]) for r in pres)),
            "any_top10": int(any(float(r["top10_correct"]) for r in pres)),
            "any_det": int(any(float(r["any_det_correct"]) for r in pres)),
            "first_top3": int(bool(pres)
                              and float(pres[0]["top3_correct"])),
        })
    ep_present = [r for r in ep_rows if r["n_present_attempts"] > 0]
    episode = {
        "level": "episode", "n_episodes": len(ep_rows),
        "n_episodes_with_present": len(ep_present),
        "n_absent_only_episodes": len(ep_rows) - len(ep_present),
        "any_det_correct": np.mean([r["any_det"] for r in ep_present])
        if ep_present else None,
        "any_top1_correct": np.mean([r["any_top1"] for r in ep_present])
        if ep_present else None,
        "any_top3_correct": np.mean([r["any_top3"] for r in ep_present])
        if ep_present else None,
        "any_top10_correct": np.mean([r["any_top10"] for r in ep_present])
        if ep_present else None,
        "first_top3_correct": np.mean([r["first_top3"] for r in ep_present])
        if ep_present else None,
    }

    with (OUT / "gfn_full_loop_topk_summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        fields = list(attempt.keys()) + [
            k for k in episode.keys() if k not in attempt]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerow(attempt)
        w.writerow(episode)
    with (OUT / "gfn_full_loop_episode_audit.csv").open(
            "w", newline="", encoding="utf-8") as f:
        fields = list(ep_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ep_rows)
    print("attempt:", attempt)
    print("episode:", episode)
    print(f"wrote {merged}")


if __name__ == "__main__":
    main()
