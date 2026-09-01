#!/usr/bin/env python
"""Aggregate the N23 true-live calibration FULL_LOOP outputs.

The N23 branch accepts a direct query-bank recovery, so an accepted
transaction is scored by the first delivered ``react`` event strictly after
the transaction frame.  This keeps recovery precision separate from the
ordinary frame-level correction count.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def as_int(row, key):
    return int(float(row.get(key) or 0))


def as_float(row, key):
    value = row.get(key)
    return None if value in (None, "", "None") else float(value)


def load_outputs(out_dir: Path):
    metrics = []
    for path in sorted(out_dir.glob("metrics_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            metrics.extend(csv.DictReader(handle))
    events = []
    for path in sorted(out_dir.glob("events_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            events.extend(json.loads(line) for line in handle if line.strip())
    transactions = []
    for path in sorted(out_dir.glob("transactions_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            transactions.extend(json.loads(line) for line in handle if line.strip())
    return metrics, events, transactions


def analyze(metrics, events, transactions):
    correction_frames = defaultdict(list)
    corrections = []
    for event in events:
        if event.get("needs_correction"):
            corrections.append(event)
            correction_frames[(event["sequence"], event["gid"])].append(
                int(event["frame"])
            )
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

    by_track = defaultdict(list)
    for event in events:
        by_track[(event["sequence"], int(event["gid"]))].append(event)
    for rows in by_track.values():
        rows.sort(key=lambda row: int(row["frame"]))

    accepted = [tx for tx in transactions if tx.get("accepted")]
    accepted_correct = 0
    accepted_false = 0
    accepted_no_delivery = 0
    accepted_details = []
    for tx in accepted:
        key = (tx["sequence"], int(tx["gid"]))
        next_rows = [row for row in by_track[key]
                     if int(row["frame"]) > int(tx["frame"])
                     and row.get("source") == "react"
                     and row.get("delivered")]
        first = next_rows[0] if next_rows else None
        if first is None:
            accepted_no_delivery += 1
            status = "no_post_accept_delivery"
        elif first.get("correct"):
            accepted_correct += 1
            status = "correct"
        else:
            accepted_false += 1
            status = "false"
        accepted_details.append({
            "sequence": tx["sequence"],
            "frame": int(tx["frame"]),
            "gid": int(tx["gid"]),
            "verifier_score": tx.get("verifier_score"),
            "status": status,
            "first_delivery_frame": None if first is None
            else int(first["frame"]),
            "first_delivery_correct": None if first is None
            else bool(first.get("correct")),
        })

    rec_probs = [value for row in metrics
                 if (value := as_float(row, "mean_recorrection_prob"))
                 is not None]
    attempts = sum(as_int(row, "recovery_attempts") for row in metrics)
    accepted_metric = sum(as_int(row, "accepted_recoveries") for row in metrics)
    frames = sum(as_int(row, "frames") for row in metrics)
    aggregate = {
        "variant": "N23",
        "sequences": len(metrics),
        "frames": frames,
        "recovery_attempts": attempts,
        "accepted_recoveries_metrics": accepted_metric,
        "accepted_recoveries_transactions": len(accepted),
        "accept_rate": round(len(accepted) / max(1, attempts), 4),
        "correct_recoveries": accepted_correct,
        "false_recoveries": accepted_false,
        "accepted_without_post_delivery": accepted_no_delivery,
        "recovery_precision": round(
            accepted_correct / max(1, accepted_correct + accepted_false), 4
        ),
        "human_corrections": len(corrections),
        "repeated_corrections": repeated,
        "same_id_mean_tte_frames": round(sum(ttes) / len(ttes), 1)
        if ttes else None,
        "correction_persistence_120": round(sum(persistence) / len(persistence), 4)
        if persistence else None,
        "mean_recorrection_prob": round(sum(rec_probs) / len(rec_probs), 4)
        if rec_probs else None,
        "correction_supervision_events": sum(
            as_int(row, "online_updates") for row in metrics
        ),
        "react_delivered_frames": sum(
            int(event.get("delivered", 0))
            for event in events if event.get("source") == "react"
        ),
        "react_correct_frames": sum(
            int(event.get("correct", 0))
            for event in events if event.get("source") == "react"
        ),
        "runtime_s": round(sum(float(row["runtime_s"]) for row in metrics), 1),
    }
    return aggregate, accepted_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics, events, transactions = load_outputs(args.out_dir)
    aggregate, accepted_details = analyze(metrics, events, transactions)

    baseline_path = args.out_dir.parents[1] / "n21/live_final_gate/live_final_gate_verdict.json"
    comparison = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        l0 = next(row for row in baseline["rows"] if row["variant"] == "L0")
        delta = aggregate["human_corrections"] - l0["human_corrections"]
        comparison = {
            "baseline": "N21_L0",
            "baseline_human_corrections": l0["human_corrections"],
            "delta_human_corrections": delta,
            "relative_change_human_corrections": round(
                delta / max(1, l0["human_corrections"]), 4
            ),
            "delta_false_recoveries_vs_l0_false_commits": (
                aggregate["false_recoveries"] - l0["false_commits"]
            ),
            "pass_criteria": (
                delta <= -0.10 * l0["human_corrections"]
                and aggregate["false_recoveries"] <= l0["false_commits"]
            ),
        }

    per_sequence = []
    for row in metrics:
        per_sequence.append({
            key: (
                as_int(row, key) if key in {
                    "n_identities", "frames", "recovery_attempts",
                    "accepted_recoveries", "verifier_accepts",
                    "shadow_commits", "shadow_timeouts", "shadow_lost_frames",
                    "online_updates"
                }
                else as_float(row, key) if key in {
                    "accept_rate", "mean_recorrection_prob", "retention_1",
                    "retention_3", "retention_5", "retention_10",
                    "retention_30", "retention_60", "retention_120", "runtime_s"
                }
                else row.get(key)
            )
            for key in row
        })

    summary = {
        "aggregate": aggregate,
        "comparison_vs_n21_l0": comparison,
        "per_sequence": per_sequence,
        "accepted_recovery_details": accepted_details,
    }
    out_json = args.out_dir / "live_cal10_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    with (args.out_dir / "live_cal10_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("N23_LIVE_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
