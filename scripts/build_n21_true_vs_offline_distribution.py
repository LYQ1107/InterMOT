#!/usr/bin/env python
"""Compare the true-live train30 attempt distribution with the offline
shadow-cache proxy (CPU-only, uses completed .done sequences)."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_n18_full_loop_v0 import load_gt  # noqa: E402

OUT = ROOT / "outputs/n21/train30_true_onpolicy"
N20 = ROOT / "outputs/n20"


def main():
    done = sorted(p.stem for p in OUT.glob("*.done")
                  if p.stem != "STAGE")
    rows = []
    live_total = proxy_total = 0
    for seq in done:
        att = []
        with (OUT / f"transactions_{seq}.jsonl").open(
                encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if t.get("shadow_event") == "START":
                    att.append((int(t["frame"]), int(t["gid"])))
        gt = load_gt(seq)
        tp = sum(any(gid in gt[ff].gt_ids
                     for ff in range(fr, min(fr + 8, max(gt) + 1)))
                 for fr, gid in att)
        proxy = set()
        for fp in (N20 / "full_shadow_cache_train30").glob("*.jsonl"):
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["sequence"] == seq:
                        proxy.add(f"{r['sequence']}:{r['frame']}:{r['gid']}")
        rows.append({
            "sequence": seq,
            "live_attempts": len(att),
            "live_target_present": tp,
            "live_target_absent": len(att) - tp,
            "offline_proxy_attempts": len(proxy),
        })
        live_total += len(att)
        proxy_total += len(proxy)
    with (ROOT / "outputs/n21/true_vs_offline_distribution.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "status": "PARTIAL" if len(done) < 30 else "COMPLETE",
        "sequences": len(done),
        "live_attempts": live_total,
        "offline_proxy_attempts": proxy_total,
        "coverage_ratio": round(live_total / max(1, proxy_total), 4),
    }
    (ROOT / "outputs/n21/true_vs_offline_distribution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("DISTRIBUTION_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
