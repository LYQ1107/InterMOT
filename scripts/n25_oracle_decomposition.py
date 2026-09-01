#!/usr/bin/env python3
"""Write the pre-training N25 oracle decomposition from frozen episodes.

This is an evaluation-only calculation.  It uses the post-hoc labels already
isolated in each episode and never writes a feature, memory state, score, or
decision used by the information gate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_oracles(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)
    present = [members for members in groups.values() if any(r["positive"] for r in members)]
    absent = [members for members in groups.values() if not any(r["positive"] for r in members)]
    total = len(groups)
    present_count = len(present)
    absent_count = len(absent)
    availability = present_count / max(1, total)
    return {
        "groups": total,
        "target_present_groups": present_count,
        "target_absent_groups": absent_count,
        "current_candidate_set_target_availability": availability,
        "candidate_oracle": {
            "definition": "current GFN-top5 set; target can be committed only when a posthoc-positive row exists",
            "max_correct_commit_coverage": availability,
            "missing_target_fraction": 1.0 - availability,
        },
        "identity_oracle_current_set": {
            "target_present_top1": 1.0 if present_count else None,
            "target_present_top5": 1.0 if present_count else None,
            "hardest_negative_margin": None,
            "target_absent_false_acceptance": 0.0 if absent_count else None,
            "commit_precision": 1.0 if present_count else None,
            "commit_coverage_all_groups": availability,
        },
        "none_oracle": {
            "target_absent_rejection": 1.0 if absent_count else None,
            "exact_candidate_or_NONE": 1.0 if total else None,
            "definition": "posthoc target-present label is available to the oracle only; not a deployable score",
        },
        "commit_oracle": {
            "safe_commit_precision": 1.0 if present_count else None,
            "false_commits": 0,
            "accepted_groups": present_count,
            "coverage_all_groups": availability,
        },
        "sam3_propagation_oracle": {
            "status": "INHERITED_REFERENCE_NOT_A_NEW_N25_RUN",
            "source": "outputs/n20/oracle_shadow_gate.csv",
            "k5_h0_mean_recorrection": 0.654981554587933,
            "k5_h1_mean_recorrection": 0.6613622455246627,
            "k5_h5_mean_recorrection": 0.6731152196635627,
            "k5_h0_correct_commits": 170,
            "k5_h0_false_commits": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/n25/dataset")
    parser.add_argument("--out", default="outputs/n25/n25_oracle_decomposition.json")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    result = {
        "protocol": {
            "evaluation_only": True,
            "gt_used_for_features_or_decisions": False,
            "posthoc_labels_used": True,
            "canonical_val25_read": False,
            "note": "N25 stopped before CCRIM training and new FULL_LOOP; propagation/full-loop entries are inherited oracle references.",
        },
        "splits": {},
        "full_oracle_inherited_reference": {
            "status": "INHERITED_REFERENCE_NOT_A_NEW_N25_RUN",
            "source": "outputs/n19/oracle_gate_summary.json",
            "mean_recorrection": 0.6291904104850248,
            "correct_commits": 261,
            "false_commits": 0,
            "retention_1": 0.839553872053872,
            "retention_5": 0.7070462986756186,
            "retention_30": 0.5192957042957043,
            "retention_120": 0.31098023005234013,
        },
    }
    for split in ("train30", "cal10"):
        result["splits"][split] = split_oracles(
            load(dataset / f"episodes_{split}.jsonl")
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"N25_ORACLE_DECOMPOSITION_DONE {out}")


if __name__ == "__main__":
    main()
