#!/usr/bin/env python
"""Aggregate the N21 train30 true on-policy rollout (per-seq .done files).

CPU-only. Works on any subset of completed sequences; output is explicitly
marked PARTIAL until 30/30 `.done` files exist. No scientific conclusion
should be drawn from a partial aggregate.

Outputs:
  outputs/n21/train30_true_onpolicy_aggregate.csv
  outputs/n21/train30_true_onpolicy_aggregate_partial.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_n18_full_loop_v0 import load_gt  # noqa: E402

OUT = ROOT / "outputs/n21/train30_true_onpolicy"


def main():
    done = sorted(p.stem for p in OUT.glob("*.done")
                  if p.stem != "STAGE")
    total = 30
    rows = []
    totals = {"attempts": 0, "commits": 0, "correct_commits": 0,
              "false_commits": 0, "timeouts": 0, "target_present": 0,
              "target_absent": 0, "repeat_attempt_gids": 0}
    for seq in done:
        att_frames = []
        commits = []
        timeouts = 0
        with (OUT / f"transactions_{seq}.jsonl").open(
                encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if t.get("shadow_event") == "START":
                    att_frames.append((int(t["frame"]), int(t["gid"])))
                elif t.get("shadow_event") == "VERDICT":
                    if t.get("shadow_commit"):
                        commits.append((int(t.get("commit_frame") or
                                           t["frame"]), int(t["gid"])))
                    else:
                        timeouts += 1
        correct = 0
        false = 0
        with (OUT / f"events_{seq}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e.get("source") == "shadow_commit":
                    if e.get("correct"):
                        correct += 1
                    else:
                        false += 1
        gt = load_gt(seq)
        tp = 0
        ta = 0
        gid_attempts = Counter(g for _, g in att_frames)
        for fr, gid in att_frames:
            present = any(gid in gt[ff].gt_ids
                          for ff in range(fr, min(fr + 8, max(gt) + 1)))
            tp += int(present)
            ta += int(not present)
        repeat = sum(1 for g, c in gid_attempts.items() if c > 1)
        rows.append({
            "sequence": seq,
            "attempts": len(att_frames),
            "commits": len(commits),
            "correct_commits": correct,
            "false_commits": false,
            "commit_precision": round(
                correct / max(1, len(commits)), 4),
            "timeouts": timeouts,
            "target_present_attempts": tp,
            "target_absent_attempts": ta,
            "repeat_attempt_gids": repeat,
        })
        for k in ("attempts", "commits", "correct_commits", "false_commits",
                  "timeouts", "target_present", "target_absent",
                  "repeat_attempt_gids"):
            if k == "target_present":
                totals[k] += tp
            elif k == "target_absent":
                totals[k] += ta
            elif k == "repeat_attempt_gids":
                totals[k] += repeat
            else:
                totals[k] += rows[-1][k]

    with (ROOT / "outputs/n21/train30_true_onpolicy_aggregate.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    status = "PARTIAL" if len(done) < total else "COMPLETE"
    totals.update({
        "status": status,
        "sequences_done": len(done),
        "sequences_total": total,
        "overall_commit_precision": round(
            totals["correct_commits"] / max(1, totals["commits"]), 4),
        "note": ("COMPLETE aggregate (30/30) may be used; "
                 "PARTIAL aggregates must not be used for final conclusions"
                 if status == "COMPLETE" else
                 "PARTIAL aggregates must not be used for final conclusions"),
    })
    (ROOT / "outputs/n21/train30_true_onpolicy_aggregate_partial.json").write_text(
        json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2), flush=True)
    print("AGGREGATE_DONE", flush=True)


if __name__ == "__main__":
    main()
