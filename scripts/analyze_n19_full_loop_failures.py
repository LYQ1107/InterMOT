#!/usr/bin/env python
"""N19.13: FULL_LOOP_N19 failure accounting on cal10.

For every recovery transaction, mark whether the accept was GT-correct and
whether the target was present in the GFN gallery (offline GT audit). This
separates detection-side misses (F3) from ranking/verifier-side misses (F4)
and verifier false accepts (F6).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_route_c_r0 import iou  # noqa: E402
from run_n18_full_loop_v0 import load_gt  # noqa: E402

OUT = ROOT / "outputs/n18"
N19 = ROOT / "outputs/n19"
CACHE = ROOT / "outputs/n18/route_c/gfn_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="learned_n19")
    ap.add_argument("--events-prefix", default=None)
    ap.add_argument("--transactions-prefix", default=None)
    ap.add_argument("--metrics-prefix", default=None)
    args = ap.parse_args()
    ep = args.events_prefix or f"full_loop_v0_events_{args.tag}"
    tp = args.transactions_prefix or f"reactivation_transactions_{args.tag}"
    mp = args.metrics_prefix or f"full_loop_v0_metrics_{args.tag}"

    tx = []
    for p in sorted(OUT.glob(f"{tp}_s[0-3].jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                tx.append(json.loads(line))
    rows = []
    for p in sorted(OUT.glob(f"{mp}_s[0-3].csv")):
        with p.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    by_seq = defaultdict(list)
    for t in tx:
        by_seq[t["sequence"]].append(t)
    counts = defaultdict(int)
    per_seq = defaultdict(lambda: defaultdict(int))
    details = []
    for seq, ts in sorted(by_seq.items()):
        gt = load_gt(seq)
        z = np.load(CACHE / f"{seq}.npz")
        frames = z["frames"]
        offsets = z["offsets"]
        dets = z["boxes"]
        z.close()
        for t in ts:
            f0 = int(t["frame"])
            gid = int(t["gid"])
            accepted = bool(t.get("accepted"))
            gf = gt.get(f0)
            target = None
            if gf is not None and gid in gf.gt_ids:
                target = np.asarray(
                    gf.boxes[gf.gt_ids.index(gid)], dtype=float)
            target_present = False
            o = int(np.searchsorted(frames, f0))
            lo = int(offsets[o - 1]) if o > 0 else 0
            hi = int(offsets[o])
            if target is not None and hi > lo:
                ious = np.asarray([iou(b, target) for b in dets[lo:hi]])
                target_present = bool(ious.max() >= 0.5)
            correct_accept = False
            if accepted and t.get("recovery_box") is not None and \
                    target is not None:
                correct_accept = iou(t["recovery_box"], target) >= 0.5
            if accepted and correct_accept:
                cat = "ACCEPT_CORRECT"
            elif accepted:
                cat = "ACCEPT_WRONG_VERIFIER"
            elif not target_present:
                cat = "REJECT_TARGET_ABSENT"
            else:
                cat = "REJECT_TARGET_PRESENT"
            counts[cat] += 1
            per_seq[seq][cat] += 1
            details.append({
                "sequence": seq, "frame": f0, "gid": gid,
                "accepted": int(accepted), "target_present": int(target_present),
                "correct_accept": int(correct_accept), "category": cat,
            })
    # aggregate metrics
    agg = {"attempts": 0, "accepts": 0, "lost_episodes": 0,
           "mean_recorrection": [], "runtime": 0.0}
    for r in rows:
        agg["attempts"] += float(r.get("recovery_attempts", 0))
        agg["accepts"] += float(r.get("accepted_recoveries", 0))
        agg["lost_episodes"] += float(r.get("lost_episodes", 0))
        agg["runtime"] += float(r.get("runtime_s", 0))
        if r.get("mean_recorrection_prob"):
            agg["mean_recorrection"].append(
                float(r["mean_recorrection_prob"]))
    agg["mean_recorrection"] = (
        sum(agg["mean_recorrection"]) / len(agg["mean_recorrection"])
        if agg["mean_recorrection"] else None)

    with (N19 / "full_loop_failure_accounting.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "count", "fraction"])
        tot = max(sum(counts.values()), 1)
        for k, v in sorted(counts.items()):
            w.writerow([k, v, round(v / tot, 4)])
    with (N19 / "full_loop_failure_accounting_per_seq.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "category", "count"])
        for s in sorted(per_seq):
            for k, v in sorted(per_seq[s].items()):
                w.writerow([s, k, v])
    with (N19 / "full_loop_failure_details.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        w.writeheader()
        w.writerows(details)
    print(json.dumps({"counts": dict(counts), "aggregate": agg},
                     indent=2, ensure_ascii=False), flush=True)
    print("FAILURE_ACCOUNTING_DONE", flush=True)


if __name__ == "__main__":
    main()
