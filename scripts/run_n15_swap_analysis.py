#!/usr/bin/env python
"""Query-swap analysis on the identity benchmark (selection-based mechanism).

For two human seeds A and B at the same frame t and the same future delta,
the pretrained anchor must rank its own identity's future crop above the
other's.  This is the selection-space analogue of the decoder Query Swap Test.
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
    feats = {b: np.load(FEATS / f"{b}.npy") for b in ("osnet", "clipreid", "dinov2")}
    crop_box = {c["crop_id"]: c["box"] for c in payload["crops"]}
    for backbone, F in feats.items():
        groups = defaultdict(list)
        for q in queries:
            pos = q["positive_crop_ids"][0]
            groups[(q["seq"], q["query_frame"], q["delta"])].append((q, pos))
        rows = []
        n_pairs = n_swap = 0
        by_delta = defaultdict(lambda: [0, 0])
        by_split = defaultdict(lambda: [0, 0])
        for key, items in groups.items():
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    (qa, pos_a), (qb, pos_b) = items[i], items[j]
                    if qa["query_gid"] == qb["query_gid"]:
                        continue
                    fa, fb = F[qa["query_crop_id"]], F[qb["query_crop_id"]]
                    ga, gb = F[pos_a], F[pos_b]
                    sim_aa = float(fa @ ga)
                    sim_ab = float(fa @ gb)
                    sim_bb = float(fb @ gb)
                    sim_ba = float(fb @ ga)
                    ok = int(sim_aa > sim_ab and sim_bb > sim_ba)
                    n_pairs += 1
                    n_swap += ok
                    by_delta[qa["delta"]][0] += ok
                    by_delta[qa["delta"]][1] += 1
                    by_split[qa["split"]][0] += ok
                    by_split[qa["split"]][1] += 1
                    rows.append(
                        {
                            "backbone": backbone, "seq": qa["seq"],
                            "split": qa["split"], "frame": qa["query_frame"],
                            "delta": qa["delta"],
                            "gid_a": qa["query_gid"], "gid_b": qb["query_gid"],
                            "sim_aa": round(sim_aa, 4), "sim_ab": round(sim_ab, 4),
                            "sim_bb": round(sim_bb, 4), "sim_ba": round(sim_ba, 4),
                            "swap": ok,
                        }
                    )
        with (ROOT / "outputs/n15" / f"swap_test_{backbone}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{backbone}: pairs={n_pairs} swap_acc={n_swap/max(1,n_pairs):.4f}")
        for d in (1, 3, 5, 10, 30):
            h, t = by_delta[d]
            print(f"  d{d}: {h}/{t} = {h/max(1,t):.4f}")
        for s in ("train", "calibration"):
            h, t = by_split[s]
            print(f"  {s}: {h}/{t} = {h/max(1,t):.4f}")


if __name__ == "__main__":
    main()

