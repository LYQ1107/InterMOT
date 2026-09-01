#!/usr/bin/env python
"""Offline verified-renewal and multi-query competition analysis.

Renewal: for a (t -> t+30) retrieval, compare fixed anchor H(t) vs verified
renewal H(t+10) (used only when cosine(H(t), crop(t+10)) >= tau) vs naive
renewal (always renew).  Multi-query: conflict rate when two anchors rank the
same candidate set.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
BENCH = ROOT / "outputs/n15/identity_benchmark/benchmark.json"
FEATS = ROOT / "outputs/n15/features"


def main() -> None:
    payload = json.loads(BENCH.read_text(encoding="utf-8"))
    queries = payload["queries"]
    by_id = {q["query_id"]: q for q in queries}
    pos_by_key = {}
    for q in queries:
        pos_by_key[(q["seq"], q["query_frame"], q["query_gid"], q["delta"])] = q["positive_crop_ids"][0]
    feats = {b: np.load(FEATS / f"{b}.npy") for b in ("osnet", "clipreid", "dinov2")}

    renewal_rows = []
    multi_rows = []
    for backbone, F in feats.items():
        for q in queries:
            if q["delta"] != 30 or q["split"] != "calibration":
                continue
            key = (q["seq"], q["query_frame"], q["query_gid"], 10)
            if key not in pos_by_key:
                continue
            anchor0 = F[q["query_crop_id"]]
            anchor10 = F[pos_by_key[key]]
            pos30 = F[q["positive_crop_ids"][0]]
            neg_ids = q["negative_crop_ids"]
            gal = np.stack([pos30] + [F[i] for i in neg_ids])
            labels = np.zeros(len(gal), dtype=int)
            labels[0] = 1

            def r1(anchor):
                sims = gal @ anchor
                return int(labels[np.argmax(sims)] == 1)

            cos010 = float(anchor0 @ anchor10)
            r_fixed = r1(anchor0)
            r_renew = r1(anchor10)
            r_avg = r1((anchor0 + anchor10) / np.linalg.norm(anchor0 + anchor10))
            verified = int(cos010 >= 0.90)
            renewal_rows.append(
                {
                    "backbone": backbone, "seq": q["seq"], "gid": q["query_gid"],
                    "cos010": round(cos010, 4), "r_fixed": r_fixed,
                    "r_verified_renew": r_renew if verified else r_fixed,
                    "r_naive_renew": r_renew, "r_avg": r_avg,
                }
            )
        # multi-query conflict at delta=5 on calibration
        groups = defaultdict(list)
        for q in queries:
            if q["delta"] != 5 or q["split"] != "calibration":
                continue
            groups[(q["seq"], q["query_frame"])].append(q)
        for (seq, frame), items in groups.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if a["query_gid"] == b["query_gid"]:
                        continue
                    ga = F[a["positive_crop_ids"][0]]
                    gb = F[b["positive_crop_ids"][0]]
                    fa = F[a["query_crop_id"]]
                    fb = F[b["query_crop_id"]]
                    sa = float(fa @ ga) - float(fa @ gb)
                    sb = float(fb @ gb) - float(fb @ ga)
                    conflict = int((sa > 0) != (sb > 0))
                    multi_rows.append(
                        {
                            "backbone": backbone, "seq": seq, "frame": frame,
                            "gid_a": a["query_gid"], "gid_b": b["query_gid"],
                            "swap": 1 if (sa > 0 and sb > 0) else 0,
                            "conflict": conflict,
                        }
                    )
    out = ROOT / "outputs/n15"
    with (out / "renewal_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(renewal_rows[0].keys()))
        w.writeheader()
        w.writerows(renewal_rows)
    with (out / "multi_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(multi_rows[0].keys()))
        w.writeheader()
        w.writerows(multi_rows)
    for backbone in ("osnet", "clipreid", "dinov2"):
        rr = [r for r in renewal_rows if r["backbone"] == backbone]
        mr = [r for r in multi_rows if r["backbone"] == backbone]
        if not rr:
            continue
        print(
            f"{backbone}: renewal n={len(rr)} "
            f"fixed={np.mean([r['r_fixed'] for r in rr]):.4f} "
            f"verified={np.mean([r['r_verified_renew'] for r in rr]):.4f} "
            f"naive={np.mean([r['r_naive_renew'] for r in rr]):.4f} "
            f"avg={np.mean([r['r_avg'] for r in rr]):.4f}"
        )
        if mr:
            print(
                f"  multi-query d5: pairs={len(mr)} swap_acc="
                f"{np.mean([r['swap'] for r in mr]):.4f} "
                f"conflict={np.mean([r['conflict'] for r in mr]):.4f}"
            )


if __name__ == "__main__":
    main()

