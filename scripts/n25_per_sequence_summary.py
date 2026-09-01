#!/usr/bin/env python3
"""Summarize frozen N25 episode availability by sequence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/n25/dataset")
    parser.add_argument("--out", default="outputs/n25/dataset/n25_per_sequence_summary.csv")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    records = []
    for split in ("train30", "cal10"):
        with (dataset / f"episodes_{split}.jsonl").open(encoding="utf-8") as handle:
            records.extend((split, json.loads(line)) for line in handle if line.strip())
    groups = defaultdict(list)
    for split, row in records:
        groups[(split, row["sequence"], row["group_key"])].append(row)
    seqs = defaultdict(list)
    for (split, sequence, _), members in groups.items():
        seqs[(split, sequence)].append(members)
    rows = []
    for (split, sequence), grouped in sorted(seqs.items()):
        all_members = [row for members in grouped for row in members]
        present = sum(any(row["positive"] for row in members) for members in grouped)
        absent = len(grouped) - present
        valid_steps = sum(len(row["raw_visual_feature"]["valid_steps"]) for row in all_members)
        requested_steps = sum(len(row["candidate_shadow_tracklet"]) for row in all_members)
        lost_steps = sum(
            sum(not step["valid"] for step in row["candidate_shadow_tracklet"])
            for row in all_members
        )
        rank_counts = {str(rank): sum(row["candidate_rank"] == rank for row in all_members) for rank in range(1, 6)}
        rows.append(
            {
                "split": split,
                "sequence": sequence,
                "groups": len(grouped),
                "rows": len(all_members),
                "target_present_groups": present,
                "target_absent_groups": absent,
                "candidate_availability": present / max(1, len(grouped)),
                "mean_candidates_per_group": len(all_members) / max(1, len(grouped)),
                "hard_negative_rows_in_present_groups": sum(
                    not row["positive"] for members in grouped if any(x["positive"] for x in members) for row in members
                ),
                "raw_crop_valid_fraction": valid_steps / max(1, requested_steps),
                "shadow_lost_step_fraction": lost_steps / max(1, requested_steps),
                **{f"rank_{rank}_rows": rank_counts[str(rank)] for rank in range(1, 6)},
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"N25_PER_SEQUENCE_SUMMARY_DONE {out}")


if __name__ == "__main__":
    main()
