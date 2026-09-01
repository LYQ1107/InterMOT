#!/usr/bin/env python3
"""Aggregate the N24 C0 strict-causal FULL_LOOP shards.

The runner writes one metrics CSV per shard and one event/transaction JSONL
pair per sequence.  This utility keeps the per-sequence runner metrics as the
source of truth for attempts/accepts and independently audits corrections and
accepted commits from the causal traces.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(".")
OUT = ROOT / "outputs/n24/full_loop/C0_h5_cal10"
EXPECTED = {
    "dancetrack0074", "dancetrack0075", "dancetrack0080",
    "dancetrack0082", "dancetrack0083", "dancetrack0086",
    "dancetrack0087", "dancetrack0096", "dancetrack0098",
    "dancetrack0099",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    metric_rows: dict[str, dict] = {}
    metric_sources: dict[str, Path] = {}
    metric_files = sorted(OUT.glob("metrics_n24_C0_h5_s*.csv"))
    for path in metric_files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                seq = row["sequence"]
                if seq in metric_rows:
                    # The recovery run for shard s1 was resumed while an
                    # idle-GPU single-sequence accelerator was also running
                    # dancetrack0086.  Prefer the primary resume shard, whose
                    # event file is the final canonical writer for that seq.
                    old = metric_sources[seq]
                    old_priority = int("resume" in old.stem)
                    new_priority = int("resume" in path.stem)
                    if new_priority <= old_priority:
                        continue
                for key in (
                    "n_identities", "frames", "recovery_attempts",
                    "accepted_recoveries", "verifier_accepts", "lost_episodes",
                    "shadow_commits", "shadow_timeouts", "shadow_lost_frames",
                ):
                    row[key] = int(row[key])
                for key in (
                    "accept_rate", "mean_recorrection_prob", "retention_1",
                    "retention_3", "retention_5", "retention_10",
                    "retention_30", "retention_60", "retention_120",
                    "threshold", "margin", "runtime_s",
                ):
                    row[key] = None if row[key] in ("", "None") else float(row[key])
                metric_rows[seq] = row
                metric_sources[seq] = path

    event_rows: dict[str, list[dict]] = {}
    tx_rows: dict[str, list[dict]] = {}
    for path in sorted(OUT.glob("events_*.jsonl")):
        seq = path.stem.removeprefix("events_")
        event_rows[seq] = read_jsonl(path)
    for path in sorted(OUT.glob("transactions_*.jsonl")):
        seq = path.stem.removeprefix("transactions_")
        tx_rows[seq] = read_jsonl(path)

    missing = EXPECTED - set(metric_rows)
    if missing:
        raise RuntimeError(f"missing metrics for: {sorted(missing)}")
    if EXPECTED - set(event_rows) or EXPECTED - set(tx_rows):
        raise RuntimeError("all expected event and transaction files are required")

    per_sequence = []
    global_corrected = 0
    global_present_after_anchor = 0
    global_identity_rates = []
    total_correct_accepts = 0
    total_false_accepts = 0
    total_attempt_target_present = 0
    total_attempt_target_absent = 0
    for seq in sorted(EXPECTED):
        metrics = metric_rows[seq]
        events = event_rows[seq]
        transactions = tx_rows[seq]
        event_by_key = {(int(e["frame"]), int(e["gid"])): e for e in events}

        anchors = {}
        for tx in transactions:
            anchors.setdefault(int(tx["gid"]), int(tx.get("anchor_frame", 0)))
        by_gid = {}
        for event in events:
            by_gid.setdefault(int(event["gid"]), []).append(event)
        identity_rates = []
        for gid, rows in by_gid.items():
            anchor = anchors.get(gid, 0)
            after = [e for e in rows if int(e["frame"]) > anchor and e["gt_present"]]
            if after:
                identity_rates.append(
                    sum(int(e["needs_correction"]) for e in after) / len(after)
                )
            global_present_after_anchor += len(after)
            global_corrected += sum(int(e["needs_correction"]) for e in after)
        global_identity_rates.extend(identity_rates)

        accepted = [t for t in transactions if t.get("reactivated")]
        correct_accepts = 0
        for tx in accepted:
            frame = int(tx.get("commit_frame", tx["frame"]))
            event = event_by_key.get((frame, int(tx["gid"])))
            if event is not None and int(event.get("correct", 0)):
                correct_accepts += 1
        false_accepts = len(accepted) - correct_accepts
        total_correct_accepts += correct_accepts
        total_false_accepts += false_accepts

        for tx in transactions:
            if tx.get("shadow_event") == "VERDICT":
                continue
            event = event_by_key.get((int(tx["frame"]), int(tx["gid"])))
            if event is not None:
                if int(event.get("gt_present", 0)):
                    total_attempt_target_present += 1
                else:
                    total_attempt_target_absent += 1

        per_sequence.append({
            "sequence": seq,
            "frames": metrics["frames"],
            "n_identities": metrics["n_identities"],
            "recovery_attempts": metrics["recovery_attempts"],
            "accepted_recoveries": metrics["accepted_recoveries"],
            "correct_accepted": correct_accepts,
            "false_accepted": false_accepts,
            "commit_precision": correct_accepts / len(accepted) if accepted else None,
            "shadow_timeouts": metrics["shadow_timeouts"],
            "mean_recorrection_prob": metrics["mean_recorrection_prob"],
            "runtime_s": metrics["runtime_s"],
        })

    total_attempts = sum(r["recovery_attempts"] for r in per_sequence)
    total_accepts = sum(r["accepted_recoveries"] for r in per_sequence)
    summary = {
        "model": "C0",
        "h": 5,
        "threshold": 0.18,
        "margin": 0.05,
        "sequences": sorted(EXPECTED),
        "n_sequences": len(per_sequence),
        "total_frames": sum(r["frames"] for r in per_sequence),
        "total_identities": sum(r["n_identities"] for r in per_sequence),
        "total_recovery_attempts": total_attempts,
        "total_accepted_recoveries": total_accepts,
        "correct_accepted_recoveries": total_correct_accepts,
        "false_accepted_recoveries": total_false_accepts,
        "commit_precision": total_correct_accepts / total_accepts if total_accepts else None,
        "target_present_attempts": total_attempt_target_present,
        "target_absent_attempts": total_attempt_target_absent,
        "total_corrections_after_anchor": global_corrected,
        "total_present_frames_after_anchor": global_present_after_anchor,
        "mean_recorrection_prob_global_identity": (
            sum(global_identity_rates) / len(global_identity_rates)
            if global_identity_rates else None
        ),
        "mean_recorrection_prob_per_sequence": (
            sum(r["mean_recorrection_prob"] for r in per_sequence
                if r["mean_recorrection_prob"] is not None) / len(per_sequence)
        ),
        "total_runtime_s": sum(r["runtime_s"] for r in per_sequence),
        "per_sequence": per_sequence,
        "metric_files": [str(p) for p in sorted(set(metric_sources.values()))],
    }
    (OUT / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(per_sequence[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sequence)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
