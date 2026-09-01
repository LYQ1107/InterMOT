#!/usr/bin/env python
"""Aggregate the N22_PROTO true-live cal10 FULL_LOOP outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(".")
OUT = ROOT / "outputs/n22/live_cal10_proto"
BASELINE = ROOT / "outputs/n21/live_final_gate/live_final_gate_verdict.json"


def load_outputs():
    metrics = []
    for path in sorted(OUT.glob("metrics_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            metrics.extend(csv.DictReader(handle))
    events = []
    for path in sorted(OUT.glob("events_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            events.extend(json.loads(line) for line in handle if line.strip())
    return metrics, events


def as_int(row, key):
    return int(float(row.get(key) or 0))


def as_float(row, key):
    value = row.get(key)
    return None if value in (None, "", "None") else float(value)


def analyze(metrics, events):
    correction_frames = defaultdict(list)
    correct_commits = 0
    false_commits = 0
    corrections = []
    for event in events:
        if event.get("source") == "shadow_commit":
            if event.get("correct"):
                correct_commits += 1
            else:
                false_commits += 1
        if event.get("needs_correction"):
            corrections.append(event)
            correction_frames[(event["sequence"], event["gid"])].append(
                event["frame"])
    for frames in correction_frames.values():
        frames.sort()
    repeated = sum(max(0, len(frames) - 1)
                   for frames in correction_frames.values())
    ttes = [later - earlier for frames in correction_frames.values()
            for earlier, later in zip(frames, frames[1:])]
    persistence = [
        int(frames[i + 1] - frames[i] <= 120)
        for frames in correction_frames.values()
        for i in range(len(frames) - 1)
    ]
    rec_probs = [value for row in metrics
                 if (value := as_float(row, "mean_recorrection_prob"))
                 is not None]
    attempts = sum(as_int(row, "recovery_attempts") for row in metrics)
    commits = sum(as_int(row, "shadow_commits") for row in metrics)
    return {
        "variant": "N22_PROTO",
        "sequences": len(metrics),
        "frames": sum(as_int(row, "frames") for row in metrics),
        "recovery_attempts": attempts,
        "shadow_timeouts": sum(as_int(row, "shadow_timeouts")
                                for row in metrics),
        "shadow_lost_frames": sum(as_int(row, "shadow_lost_frames")
                                   for row in metrics),
        "commits": commits,
        "correct_commits": correct_commits,
        "false_commits": false_commits,
        "commit_precision": round(correct_commits / max(1, commits), 4),
        "human_corrections": len(corrections),
        "repeated_corrections": repeated,
        "same_id_mean_tte_frames": round(sum(ttes) / len(ttes), 1)
        if ttes else None,
        "correction_persistence_120": round(sum(persistence) /
                                             len(persistence), 4)
        if persistence else None,
        "mean_recorrection_prob": round(sum(rec_probs) / len(rec_probs), 4)
        if rec_probs else None,
        "correction_supervision_events": sum(
            as_int(row, "online_updates") for row in metrics),
        "prototype_positive_updates": sum(
            as_int(row, "prototype_positive_updates") for row in metrics),
        "prototype_negative_updates": sum(
            as_int(row, "prototype_negative_updates") for row in metrics),
        "prototype_repeated_wrong": sum(
            as_int(row, "prototype_repeated_wrong") for row in metrics),
        "runtime_s": round(sum(float(row["runtime_s"]) for row in metrics),
                            1),
    }


def main():
    metrics, events = load_outputs()
    aggregate = analyze(metrics, events)
    comparison = None
    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        l0 = next(row for row in baseline["rows"] if row["variant"] == "L0")
        delta = aggregate["human_corrections"] - l0["human_corrections"]
        comparison = {
            "baseline": "N21_L0",
            "baseline_human_corrections": l0["human_corrections"],
            "delta_human_corrections": delta,
            "relative_change_human_corrections": round(
                delta / max(1, l0["human_corrections"]), 4),
            "delta_false_commits": (aggregate["false_commits"]
                                    - l0["false_commits"]),
            "delta_repeated_corrections": (aggregate["repeated_corrections"]
                                            - l0["repeated_corrections"]),
            "pass_criteria": False,
        }
    per_sequence = []
    for row in metrics:
        per_sequence.append({
            key: (as_int(row, key) if key in {
                "n_identities", "frames", "recovery_attempts",
                "accepted_recoveries", "shadow_commits", "shadow_timeouts",
                "shadow_lost_frames", "online_updates",
                "prototype_positive_updates", "prototype_negative_updates",
                "prototype_repeated_wrong"}
                else as_float(row, key) if key in {
                    "accept_rate", "mean_recorrection_prob", "retention_1",
                    "retention_3", "retention_5", "retention_10",
                    "retention_30", "retention_60", "retention_120",
                    "runtime_s"}
                else row.get(key))
            for key in row
        })
    summary = {
        "aggregate": aggregate,
        "comparison_vs_n21_l0": comparison,
        "per_sequence": per_sequence,
    }
    (OUT / "live_cal10_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "live_cal10_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("N22_LIVE_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
