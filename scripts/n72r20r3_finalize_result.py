#!/usr/bin/env python3
"""Assemble the compact, machine-readable N72R20R3 final result."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.n72r20r3_common import ROOT, SEQUENCES


FINAL_DECISION = "FAIL_OPEN_SET_CROSS_SEQUENCE_GENERALIZATION"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_metric(record: dict[str, Any]) -> dict[str, Any]:
    pooled = record["pooled"]
    return {
        "condition": record["condition"],
        "policy_id": record["policy_id"],
        "trainable_parameters": record["trainable_parameters"],
        "pooled": {
            "negative_fpr": pooled.get("false_present_rate"),
            "p0_fpr": pooled.get("p0_false_present_rate"),
            "p1_fpr": pooled.get("p1_false_present_rate"),
            "presence_recall": pooled.get("presence_recall"),
            "open_set_correct_identification_recall": pooled.get("open_set_correct_identification_recall"),
            "conditional_candidate_accuracy": pooled.get("conditional_candidate_accuracy_given_present"),
            "wrong_write_rate_diagnostic": pooled.get("wrong_write_rate"),
            "correct_write_retention_diagnostic": pooled.get("correct_write_retention_vs_r2_c0"),
            "presence_gate_pass": pooled.get("presence_gate_pass"),
            "cross_sequence_gate_pass": pooled.get("cross_sequence_gate_pass"),
        },
        "macro": {
            "negative_fpr": record["macro"].get("false_present_rate"),
            "open_set_correct_identification_recall": record["macro"].get("open_set_correct_identification_recall"),
        },
        "per_sequence": {
            sequence: {
                "negative_count": value.get("candidate_set_negative_count"),
                "positive_count": value.get("candidate_set_positive_count"),
                "fpr": value.get("false_present_rate"),
                "open_set_correct_identification_recall": value.get("open_set_correct_identification_recall"),
                "wrong_write_rate_diagnostic": value.get("wrong_write_rate"),
                "correct_write_retention_diagnostic": value.get("correct_write_retention_vs_r2_c0"),
            }
            for sequence, value in record["per_sequence"].items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R3/FINAL_RESULT.json")
    args = parser.parse_args()
    taxonomy = read_json(ROOT / "outputs/N72R20R3/failure_taxonomy.json")
    reclassification = read_json(ROOT / "outputs/N72R20R3/r2_wrong_write_reclassification.json")
    loso = read_json(ROOT / "outputs/N72R20R3/presence/loso_results.json")
    audit = read_json(ROOT / "outputs/N72R20R3/presence/presence_feature_audit.json")
    authorization = read_json(ROOT / "outputs/N72R20R3/B4_AUTHORIZATION.json")
    immutability = read_json(ROOT / "outputs/N72R20R3/r2_immutability_audit.json")
    headroom = read_json(ROOT / "outputs/N72R20R3/future_association_headroom.json")
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    max_features = []
    for condition, features in audit.get("features", {}).items():
        for name, record in features.items():
            if record.get("auroc_present_vs_absent") is not None:
                max_features.append({"condition": condition, "feature": name, "auroc": record["auroc_present_vs_absent"]})
    max_features.sort(key=lambda item: (-float(item["auroc"]), item["condition"], item["feature"]))
    compact_aggregates = [compact_metric(record) for record in loso["aggregates"]]
    result = {
        "stage": "N72R20R3",
        "goal_reference": "outputs/N72R20R3/FINAL_GOAL.json",
        "goal": "Open-Set Persistent Identity Recognition",
        "central_question": "Can a human-initialized persistent identity memory determine whether the target identity is represented in the current candidate set, abstain when it is not, and identify the correct candidate when it is present?",
        "final_decision": FINAL_DECISION,
        "answer": "No reliable cross-sequence open-set runtime policy was established: raw presence-related signal exists, but no LOSO policy met both the required safety and correct-identification gates on all eight held-out sequences.",
        "source_stage": "N72R20R2",
        "source_commit": "b2abc90fd90bc5eb36b31f753561ea0aa3da900b",
        "branch": branch,
        "artifact_generation_commit": head,
        "sequences": list(SEQUENCES),
        "frames": taxonomy["future_frames"],
        "candidates": 49202,
        "taxonomy_summary": {
            "future_frames": taxonomy["future_frames"],
            "p0": taxonomy["p0_count"],
            "p1": taxonomy["p1_count"],
            "p1a": taxonomy["p1a_count"],
            "p1b": taxonomy["p1b_count"],
            "p2": taxonomy["p2_count"],
            "p3": taxonomy["p3_count"],
            "candidate_set_present": taxonomy["candidate_set_present_count"],
            "candidate_set_absent": taxonomy["candidate_set_absent_count"],
        },
        "historical_r2_wrong_write_reclassification": reclassification["classification_counts"],
        "historical_r2_wrong_write_count": reclassification["historical_r2_wrong_write_count"],
        "historical_r2_c0_correct_write_count": reclassification["historical_r2_c0_correct_write_count"],
        "best_presence_method": None,
        "selected_static_policy": None,
        "presence_features": {
            "max_pooled_audit_auroc": authorization["max_pooled_audit_auroc"],
            "top_diagnostic_features": max_features[:10],
            "runtime_conditions": ["S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"],
            "S2_ORACLE_CLEAN": "posthoc_oracle_only_not_eligible_for_final_runtime",
        },
        "trainable_parameters": {"B0_B3": 0, "B4_LOGISTIC_PRESENCE": 11, "selected_runtime_policy": 0},
        "b4_authorization": {
            "authorized": authorization["authorized"],
            "usable_separation": authorization["usable_separation"],
            "simple_baselines_limited": authorization["simple_baselines_limited"],
        },
        "loso_protocol": {
            "fold_count": loso["fold_count"],
            "heldout_sequences": loso["heldout_sequences"],
            "frame_level_random_split": False,
            "sequence_cluster_bootstrap": "outputs/N72R20R3/presence/bootstrap_results.json",
        },
        "loso_pooled_metrics": compact_aggregates,
        "loso_macro_metrics": [
            {"condition": item["condition"], "policy_id": item["policy_id"], **item["macro"]}
            for item in compact_aggregates
        ],
        "loso_per_sequence_metrics": "embedded in each item of loso_pooled_metrics.per_sequence",
        "primary_gate": {
            "negative_fpr_max": 0.02,
            "open_set_correct_identification_recall_min": 0.60,
            "any_pooled_and_all_sequence_policy_passed": False,
            "static_gate_passed": False,
        },
        "false_present_rate": None,
        "p0_false_present_rate": None,
        "p1_false_present_rate": None,
        "open_set_correct_identification_recall": None,
        "causal_wrong_write_rate": None,
        "causal_correct_write_retention": None,
        "causal_replay": "NOT_RUN_STATIC_PRESENCE_GATE_FAILED",
        "future_association_headroom": {
            "p2_frames": headroom["p2_frames"],
            "learned_top1_correct_count": headroom["learned_top1_correct_count"],
            "artifact": "outputs/N72R20R3/future_association_headroom.json",
            "assignment_changed": False,
        },
        "runtime_future_gt_used": False,
        "human_anchor_immutable": True,
        "public_id_immutable": True,
        "candidate_creation_forbidden": True,
        "candidate_axis_unchanged": True,
        "r2_artifacts_unchanged": immutability["all_unchanged"],
        "sam3_rerun": False,
        "val_accessed": False,
        "test_accessed": False,
        "association_rescue": False,
        "training_started": False,
        "next_association_authority_stage_authorized": False,
        "artifact_paths": {
            "runtime_features": "outputs/N72R20R3/presence/frame_runtime_features.jsonl.zst",
            "posthoc_labels": "outputs/N72R20R3/presence/frame_posthoc_labels.jsonl.zst",
            "loso": "outputs/N72R20R3/presence/loso_results.json",
            "causal_stop": "outputs/N72R20R3/causal/causal_replay.json",
        },
        "tests": {
            "r3_targeted": {"passed": 42, "failed": 0},
            "full_suite": {"passed": 324, "failed": 4, "known_trackeval_cli_failures": 4},
            "known_failure": "Pinned TrackEval CLI receives SEQMAP_FILE as a list and passes it to os.path.isfile; no R3 code is exercised.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "final_decision": FINAL_DECISION, "artifact_generation_commit": head}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
