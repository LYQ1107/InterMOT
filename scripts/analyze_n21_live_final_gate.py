#!/usr/bin/env python
"""Aggregate the N21 Phase-III true-live cal10 FINAL GATE (L0/L1/L2/L3)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")
GATE = ROOT / "outputs/n21/live_final_gate"
VARIANTS = ["L0", "L1", "L2", "L3"]


def load_rows(variant):
    d = GATE / variant
    metrics = []
    for p in sorted(d.glob("metrics_*.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                metrics.append(r)
    events = []
    for p in sorted(d.glob("events_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    return metrics, events


def analyze(variant):
    metrics, events = load_rows(variant)
    attempts = sum(int(m.get("recovery_attempts") or 0) for m in metrics)
    commits = sum(int(m.get("shadow_commits") or 0) for m in metrics)
    frames = sum(int(m.get("frames") or 0) for m in metrics)
    runtime = sum(float(m.get("runtime_s") or 0) for m in metrics)
    updates = sum(int(m.get("online_updates") or 0) for m in metrics)
    # The runner's historical per-sequence field ``online_updates`` stores
    # correction-supervision events.  L1 deliberately has no trainable
    # online model, so reporting that field as parameter updates is wrong.
    # L2/L3 perform one online training call (10 optimizer epochs) per event.
    parameter_update_events = updates if variant in ("L2", "L3") else 0
    correct = false = 0
    corr_events = []
    for e in events:
        if e.get("source") == "shadow_commit":
            if e.get("correct"):
                correct += 1
            else:
                false += 1
        if e.get("needs_correction"):
            corr_events.append(e)
    by_gid = defaultdict(list)
    for e in corr_events:
        by_gid[(e["sequence"], e["gid"])].append(e["frame"])
    for v in by_gid.values():
        v.sort()
    repeated = sum(max(0, len(v) - 1) for v in by_gid.values())
    # same-ID TTE (time between consecutive corrections of the same gid)
    ttes = [b - a for v in by_gid.values()
            for a, b in zip(v, v[1:])]
    mean_tte = round(sum(ttes) / len(ttes), 1) if ttes else None
    # correction persistence: correction followed by another <=120 frames
    persist = []
    for v in by_gid.values():
        for i, fr in enumerate(v[:-1]):
            persist.append(int(v[i + 1] - fr <= 120))
    persistence = round(sum(persist) / len(persist), 4) if persist else None
    recorr_probs = [float(m["mean_recorrection_prob"])
                    for m in metrics
                    if m.get("mean_recorrection_prob")]
    mean_recorr = round(sum(recorr_probs) / len(recorr_probs), 4) \
        if recorr_probs else None
    return {
        "variant": variant,
        "sequences": len(metrics),
        "frames": frames,
        "recovery_attempts": attempts,
        "shadow_timeouts": sum(int(m.get("shadow_timeouts") or 0)
                               for m in metrics),
        "shadow_lost_frames": sum(int(m.get("shadow_lost_frames") or 0)
                                  for m in metrics),
        "commits": commits,
        "correct_commits": correct,
        "false_commits": false,
        "commit_precision": round(correct / max(1, commits), 4),
        "human_corrections": len(corr_events),
        "repeated_corrections": repeated,
        "mean_recorrection_prob": mean_recorr,
        "same_id_mean_tte_frames": mean_tte,
        "correction_persistence_120": persistence,
        "correction_supervision_events": updates,
        "parameter_update_events": parameter_update_events,
        "optimizer_steps": parameter_update_events * 10,
        "runtime_s": round(runtime, 1),
    }


def main():
    rows = [analyze(v) for v in VARIANTS]
    with (GATE / "live_final_gate_metrics.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # failures CSV
    with (GATE / "live_final_gate_failures.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "sequence", "frame", "gid", "public_id",
                    "failure_type"])
        for variant in VARIANTS:
            _, events = load_rows(variant)
            for e in events:
                ft = None
                if e.get("source") == "shadow_commit" and \
                        not e.get("correct"):
                    ft = "FALSE_COMMIT"
                elif e.get("needs_correction"):
                    ft = ("MISS" if not e.get("delivered") else "ID_WRONG")
                if ft:
                    w.writerow([variant, e["sequence"], e["frame"],
                                e["gid"], e["public_id"], ft])
    by = {r["variant"]: r for r in rows}
    l0 = by["L0"]
    verdicts = {}
    for v in ("L1", "L2", "L3"):
        r = by[v]
        d_corr = r["human_corrections"] - l0["human_corrections"]
        d_false = r["false_commits"] - l0["false_commits"]
        rel = d_corr / max(1, l0["human_corrections"])
        pass_ = (rel <= -0.10 and d_false <= max(2, 0.05 * l0["false_commits"])
                 and r["repeated_corrections"] <= l0["repeated_corrections"])
        verdicts[v] = {
            "delta_corrections_vs_L0": d_corr,
            "relative_change_vs_L0": round(rel, 4),
            "delta_false_commits_vs_L0": d_false,
            "pass_criteria": pass_,
        }
    final_pass = any(verdicts[v]["pass_criteria"] for v in ("L2", "L3"))
    verdict = {
        "verdict": ("PASS_CATIL_LIVE" if final_pass else
                    "FAIL_CATIL_REPRESENTATION_ADAPTATION"),
        "basis": "true-live cal10 FULL_LOOP (L0/L1/L2/L3)",
        "details": verdicts,
        "rows": rows,
        "note": "PASS requires >=10% fewer human corrections vs L0, no "
                "material false-commit increase, and no more repeated "
                "corrections.",
    }
    (GATE / "live_final_gate_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print("ANALYZE_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
