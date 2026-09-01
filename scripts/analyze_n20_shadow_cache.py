#!/usr/bin/env python
"""N20.2 diagnostics: per-H shadow confirmation availability."""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
N20 = ROOT / "outputs/n20"


def main():
    rows = []
    for p in sorted((N20 / "shadow_cache").glob("oracle_shadow_k*_s*.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    print(f"shadow_rows={len(rows)}", flush=True)
    hs = [1, 3, 5, 8, 10, 30, 60, 120]
    stats = {"n": len(rows)}
    for h in hs:
        n = correct = delivered = 0
        for r in rows:
            f0 = int(r["frame"])
            x = next((fr for fr in r["frames"] if fr["frame"] == f0 + h),
                     None)
            if x is None:
                continue
            n += 1
            if x["box"] is not None:
                delivered += 1
            if x["correct"] == 1:
                correct += 1
        stats[f"h{h}"] = {
            "n": n,
            "delivered": delivered,
            "correct": correct,
            "correct_rate": correct / max(1, n),
            "delivered_rate": delivered / max(1, n),
        }
    # first drift / loss frame distribution
    first_drift = defaultdict(int)
    for r in rows:
        f0 = int(r["frame"])
        drift = None
        for fr in sorted(r["frames"], key=lambda x: x["frame"]):
            if fr["frame"] <= f0:
                continue
            if fr["box"] is None or fr["correct"] != 1:
                drift = fr["frame"] - f0
                break
        first_drift[drift if drift is not None else "never"] += 1
    stats["first_drift_or_loss"] = dict(sorted(
        first_drift.items(), key=lambda kv: (kv[0] == "never", kv[0])))
    out = N20 / "shadow_cache_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print("SHADOW_CACHE_STATS_DONE", flush=True)


if __name__ == "__main__":
    main()
