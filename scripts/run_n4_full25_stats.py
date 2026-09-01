#!/usr/bin/env python
"""Compute full-25 per-sequence deltas and statistical tests."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sam3_intermot.utils.io import atomic_write_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs/n4/full25/winner/trackeval_full25.log"


def parse_log():
    sections = defaultdict(dict)  # tracker -> {seq: {metric: val}}
    current_tracker = None
    current_metric = None
    with LOG.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Evaluating "):
                current_tracker = line.split()[1]
            elif line.startswith("HOTA:"):
                current_metric = "HOTA"
            elif line.startswith("CLEAR:"):
                current_metric = "CLEAR"
            elif line.startswith("Identity:"):
                current_metric = "Identity"
            elif line.startswith("Count:"):
                current_metric = None
            elif current_metric and line.startswith("dancetrack"):
                parts = line.split()
                seq = parts[0]
                vals = [float(x) for x in parts[1:]]
                if current_metric == "HOTA" and len(vals) >= 3:
                    sections[current_tracker].setdefault(seq, {})["HOTA"] = vals[0]
                    sections[current_tracker][seq]["DetA"] = vals[1]
                    sections[current_tracker][seq]["AssA"] = vals[2]
                elif current_metric == "CLEAR" and len(vals) >= 17:
                    sections[current_tracker].setdefault(seq, {})["MOTA"] = vals[0]
                    sections[current_tracker][seq]["IDSW"] = vals[12]
                    sections[current_tracker][seq]["Frag"] = vals[16]
                elif current_metric == "Identity" and len(vals) >= 3:
                    sections[current_tracker][seq]["IDF1"] = vals[0]
    return sections


def paired_stats(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(diff), size=len(diff))
        boots.append(diff[idx].mean())
    boots = np.sort(boots)
    ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))])
    # Wilcoxon signed-rank approximation
    d = diff[diff != 0]
    if len(d) == 0:
        p = 1.0
    else:
        ranks = np.argsort(np.argsort(np.abs(d))) + 1
        w = np.sum(ranks[d > 0])
        n = len(d)
        mu = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w - mu) / sigma if sigma > 0 else 0.0
        p = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(z)))
    return {
        "mean_delta": float(diff.mean()),
        "median_delta": float(np.median(diff)),
        "ci95": [float(ci[0]), float(ci[1])],
        "p": float(p),
        "improved": int(np.sum(diff > 1e-6)),
        "degraded": int(np.sum(diff < -1e-6)),
        "unchanged": int(np.sum(np.abs(diff) <= 1e-6)),
    }


def main():
    sections = parse_log()
    seqs = sorted(sections["b0"].keys())
    rows = []
    for seq in seqs:
        base = sections["b0"][seq]
        row = {"sequence": seq}
        for b in ["b1", "b2", "b5"]:
            cur = sections[b][seq]
            for m in ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]:
                row[f"{b}_{m}"] = cur.get(m)
                row[f"{b}_delta_{m}"] = cur.get(m, 0) - base.get(m, 0)
        rows.append(row)
    write_csv(ROOT / "outputs/n4/full25/per_sequence_metrics.csv", rows)

    combined = {
        "b0": {"HOTA": 49.83, "DetA": 55.335, "AssA": 45.291, "MOTA": 45.688, "IDF1": 52.818, "IDSW": 2462, "Frag": 13688},
        "b1": {"HOTA": 50.089, "DetA": 55.543, "AssA": 45.591, "MOTA": 45.945, "IDF1": 52.993, "IDSW": 2409, "Frag": 13687},
        "b2": {"HOTA": 50.043, "DetA": 55.645, "AssA": 45.422, "MOTA": 45.968, "IDF1": 52.837, "IDSW": 2410, "Frag": 13696},
        "b5": {"HOTA": 49.957, "DetA": 55.728, "AssA": 45.201, "MOTA": 46.002, "IDF1": 52.645, "IDSW": 2455, "Frag": 13706},
    }
    write_csv(
        ROOT / "outputs/n4/full25/combined_metrics.csv",
        [{"budget": k, **v} for k, v in combined.items()],
    )

    stats = {}
    for b in ["b1", "b2", "b5"]:
        stats[b] = {}
        for m in ["HOTA", "AssA", "IDF1", "MOTA", "IDSW", "Frag"]:
            a = [sections["b0"][s].get(m, 0) for s in seqs]
            c = [sections[b][s].get(m, 0) for s in seqs]
            stats[b][m] = paired_stats(a, c)
    atomic_write_json(ROOT / "outputs/n4/full25/statistical_tests.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
