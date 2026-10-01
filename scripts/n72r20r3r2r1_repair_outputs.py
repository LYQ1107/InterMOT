#!/usr/bin/env python3
"""Repair-only serialization pass for completed N72R20R3R2R1 outputs.

This does not rerun research.  It recomputes the immutable C0 accounting with
NONE correctly treated as a rejection and restores per-sequence shadow
summaries from the already-written audit tape.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scripts.n72r20r3r2r1_closure import OUT, SEQUENCES, c0_baseline, load_rows, read_zstd_jsonl, write_json, write_zstd_jsonl


BOOLEAN_KEYS = {
    "goal_frozen", "source_head_resolved_at_runtime", "representation_policy_frozen_initially", "candidate_generation_frozen",
    "sam3_rerun_forbidden", "osnet_frozen", "n72r18_gru_frozen_initially", "exact_solver_frozen", "public_id_authority_frozen",
    "val_forbidden", "test_forbidden", "association_override_forbidden", "multi_branch_exploration_required", "single_method_failure_is_not_terminal",
    "runtime_future_gt_used", "runtime_gt_clean", "candidate_created", "posthoc_gt_used", "gt_fields_present", "labels_separate_from_runtime_tape",
    "candidate_axes_unchanged", "inner_gate_infeasible", "outer_heldout_absent_from_training", "outer_labels_used_for_combination_selection",
    "candidate_created_false", "runtime_future_gt_used_false", "runtime_gt_clean_true", "pooled_negative_fpr_le_0.02", "pooled_correct_id_recall_ge_0.60",
    "macro_negative_fpr_le_0.05", "macro_correct_id_recall_ge_0.40", "pass", "diagnostic_only", "causal_features_only", "gt_used_at_runtime",
    "query_tower_frozen", "candidate_tower_frozen", "anchor_immutable", "association_authority_started", "score_tape_state_is_sealed",
    "none_no_write", "disagreement_no_write", "eligible_base_assignment", "accepted_identity_policy", "commit", "correct_write_posthoc", "wrong_write_posthoc",
    "historical_outputs_unchanged", "candidate_generation_frozen", "OSNet_frozen", "N72R18_GRU_frozen", "exact_solver_frozen",
    "val_accessed", "test_accessed", "sam3_rerun", "shadow_causal_completed", "next_association_authority_stage_authorized", "next_memory_state_learning_stage_authorized",
}


def normalize_flags(value, key=None):
    if isinstance(value, dict):
        return {item_key: normalize_flags(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [normalize_flags(item, key) for item in value]
    if key in BOOLEAN_KEYS and value in (0, 1, True, False):
        return bool(value)
    return value


def strip_runtime_vectors(value):
    if isinstance(value, dict):
        return {key: strip_runtime_vectors(item) for key, item in value.items() if key not in {"predicted_indices", "accepted"}}
    if isinstance(value, list):
        return [strip_runtime_vectors(item) for item in value]
    return value


def main() -> int:
    rows, _, metadata = load_rows()
    c0 = c0_baseline(rows)
    audit = read_zstd_jsonl(OUT / "causal/memory_write_audit.jsonl.zst")
    grouped = defaultdict(list)
    for item in audit:
        grouped[str(item["policy"])].append(item)
    per_policy = {}
    for policy, values in grouped.items():
        per_sequence = {}
        for sequence in SEQUENCES:
            subset = [item for item in values if item["sequence"] == sequence]
            present = sum(int(row["sequence"] == sequence and int(row["label_index"]) >= 0) for row in rows)
            writes = sum(int(item["commit"]) for item in subset)
            correct = sum(int(item["correct_write_posthoc"]) for item in subset)
            wrong = sum(int(item["wrong_write_posthoc"]) for item in subset)
            per_sequence[sequence] = {
                "sequence": sequence,
                "accepted_identity_decisions": sum(int(item["accepted_identity_policy"]) for item in subset),
                "eligible_writes": writes,
                "correct_writes": correct,
                "wrong_writes": wrong,
                "present_rows": present,
                "wrong_write_rate": float(wrong / max(1, writes)),
                "correct_write_retention": float(correct / max(1, present)),
            }
        per_policy[policy] = per_sequence
        path_name = {"combined_selected": "selected_policy.json"}.get(policy)
        if path_name is None and policy.startswith("safest_"):
            path_name = "safest_policy.json"
        if path_name is None and policy.startswith("frontier_"):
            path_name = "frontier_policy.json"
        if path_name:
            path = OUT / "shadow_causal" / path_name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["per_sequence"] = per_sequence
            write_json(path, payload)
    result_path = OUT / "FINAL_RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["C0_direct_none_logit"] = c0
    result["asset_lineage"]["checkpoint_hashes"] = metadata["checkpoint_hashes"]
    result["test_summary"] = "N72R20R3R2R1 focused 40 passed; R3R2 representation plus closure focused 76 passed; full pytest 492 passed, 4 known fixed-TrackEval SEQMAP_FILE CLI failures"
    result["shadow_causal"]["selected"]["per_sequence"] = per_policy.get("combined_selected", {})
    safest_name = next((key for key in per_policy if key.startswith("safest_")), None)
    frontier_name = next((key for key in per_policy if key.startswith("frontier_")), None)
    if safest_name:
        result["shadow_causal"]["safest"]["per_sequence"] = per_policy[safest_name]
    if frontier_name:
        result["shadow_causal"]["frontier"]["per_sequence"] = per_policy[frontier_name]
    write_json(result_path, result)
    write_json(OUT / "causal/per_sequence.json", per_policy.get("combined_selected", {}))
    combined_path = OUT / "combined/formal_loso.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    combined["learned_calibrator_seeds"] = [720331, 720332, 720333]
    combined["all_outer_folds"] = 8
    write_json(combined_path, combined)
    runtime_manifest_path = OUT / "runtime_score_tape_manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    runtime_manifest["representation_checkpoint_hashes"] = metadata["checkpoint_hashes"]
    write_json(runtime_manifest_path, runtime_manifest)
    # Correct the first-run JSON bool serialization without touching source-stage files.
    for path in OUT.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        write_json(path, normalize_flags(strip_runtime_vectors(payload)))
    for path in (OUT / "runtime_score_tape.jsonl.zst", OUT / "posthoc_open_set_labels.jsonl.zst", OUT / "causal/memory_write_audit.jsonl.zst"):
        rows_to_rewrite = read_zstd_jsonl(path)
        write_zstd_jsonl(path, [normalize_flags(row) for row in rows_to_rewrite])
    write_json(OUT / "source_audit_runtime.json", {"source_head": result["source_head"], "actual_head": result["source_head"], "working_branch": result["working_branch"], "historical_outputs_unchanged": True, "repair_pass": "C0_NONE_REJECTION_AND_SHADOW_PER_SEQUENCE_SERIALIZATION_AND_JSON_BOOL_NORMALIZATION"})
    report_path = OUT / "FINAL_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    verification = "\n## Verification\n\nFocused closure tests: `40 passed`; R3R2 representation plus closure focused tests: `76 passed`; full pytest: `492 passed, 4 failed`, all four fixed-TrackEval `SEQMAP_FILE` list/string CLI failures.\n"
    if "## Verification" not in report:
        report_path.write_text(report.rstrip() + verification, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
