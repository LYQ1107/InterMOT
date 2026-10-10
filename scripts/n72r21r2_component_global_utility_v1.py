"""Posthoc component-versus-global utility diagnostic, NEVER policy selection.

Use every original FIT/INNER sealed arm and its actual pinned future-H100
utility. Distinct-action/effective-action subsets are predefined bookkeeping,
not independent events, best-action oracle choices or full-video MOT results.
"""
from collections import Counter, defaultdict
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, preregistration, storage, append_log
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.evaluation.event_delivery_evidence import component_audit

EPSILON = 1e-10  # Existing API/CLI equality tolerance, not a decision threshold.
SUBSETS = ("ALL_ARMS", "DISTINCT_ACTION_CONFIGS", "DISTINCT_EFFECTIVE_ONE_SHOT_ARMS")


def sign(value):
    return "POSITIVE" if value > EPSILON else "NEGATIVE" if value < -EPSILON else "ZERO_WITHIN_API_CLI_TOLERANCE"


def paired_counts(labels, utility):
    if labels["future"]["H100"]["complete"] != utility["complete_H100"]:
        raise ValueError("Utility/component future boundary mismatch")
    if not utility["complete_H100"]:
        if any(utility[k] is not None for k in ("actual_nine_metrics", "paired_delta_vs_own_KEEP", "L5_value_label")):
            raise ValueError("Incomplete utility must not become a zero/negative reward")
        return {"all_arm_records": 1, "incomplete_H100_excluded_from_sign_comparison": 1}
    full, current = labels["future"]["H100"], labels["current_t"]
    if not full["complete"]:
        raise ValueError("Utility/component future boundary mismatch")
    safe = bool(full["benefit_label"]) and not any(current[k] for k in ("N10", "non_target_damage", "verified_OTHER_writes", "UNKNOWN_writes"))
    delta = utility["paired_delta_vs_own_KEEP"]
    both_positive = delta["HOTA"] > EPSILON and delta["AssA"] > EPSILON
    counts = {"all_arm_records": 1, "complete_H100_sign_pairs": 1,
              "target_value_" + sign(full["raw_target_value"]): 1,
              "global_component_proxy_" + sign(full["raw_global_component_proxy"]): 1,
              "actual_window_HOTA_delta_" + sign(delta["HOTA"]): 1,
              "actual_window_AssA_delta_" + sign(delta["AssA"]): 1,
              "safe_component_positive_arms_not_roots": int(safe),
              "safe_component_positive_and_both_global_metrics_positive_arms": int(safe and both_positive),
              "safe_component_positive_but_HOTA_or_AssA_nonpositive_arms": int(safe and not both_positive),
              "safe_component_positive_but_HOTA_or_AssA_negative_arms": int(safe and (delta["HOTA"] < -EPSILON or delta["AssA"] < -EPSILON)),
              "positive_target_value_but_negative_HOTA_arms": int(full["raw_target_value"] > 0 and delta["HOTA"] < -EPSILON),
              "positive_target_value_but_negative_AssA_arms": int(full["raw_target_value"] > 0 and delta["AssA"] < -EPSILON),
              "harmful_component_but_both_global_metrics_positive_arms": int((full["risk_label"] or any(current[k] for k in ("N10", "non_target_damage", "verified_OTHER_writes", "UNKNOWN_writes"))) and both_positive)}
    return counts


def run():
    p = preregistration()
    required = p["split"]["fit"] + p["split"]["inner"]
    # Entire raw-recomputation audit is prerequisite, never a successful subset.
    paths = [OUT / "events/delivery_audit_v1/sequences" / (s + ".json") for s in required]
    audit_rows = [read_json(path) for path in paths]
    protocol = read_json(OUT / "events/EVENT_DELIVERY_AUDIT_PROTOCOL_V1.json")
    assert protocol["source_sha256"] == {name: sha256(ROOT / name) for name in protocol["source_sha256"]}
    assert all(r["all_actual_label_fields_recomputed_exactly_from_raw_branches_and_original_GT"] and r["protocol_sha256"] == sha256(OUT / "events/EVENT_DELIVERY_AUDIT_PROTOCOL_V1.json") for r in audit_rows)
    storage(8 << 20)
    source_refs, census = [], []
    for sequence, recomputation_path, recomputation in zip(required, paths, audit_rows, strict=True):
        assert recomputation["sequence"] == sequence
        la_path = OUT / "events/label_audit" / (sequence + ".json")
        ua_path = OUT / "events/trajectory_utility_audit" / (sequence + ".json")
        la, ua = read_json(la_path), read_json(ua_path)
        verified = {r["path"]: r["sha256"] for r in recomputation["source_artifacts"]}
        assert verified[str(la_path)] == sha256(la_path) and verified[str(ua_path)] == sha256(ua_path)
        for record in (la, ua):
            assert verified[record["artifact"]["path"]] == record["artifact"]["sha256"] == sha256(record["artifact"]["path"])
        rows = read_zstd_jsonl(Path(la["artifact"]["path"]))
        urows = read_zstd_jsonl(Path(ua["artifact"]["path"]))
        utilities = {(r["event_uid"], r["branch"]): r for r in urows}
        assert len(utilities) == len(urows) == len(rows)
        assert set(utilities) == {(r["event_uid"], r["branch"]) for r in rows}
        subsets, by_branch = defaultdict(Counter), defaultdict(Counter)
        for row in rows:
            labels = row["offline_supervision_labels"]
            component_audit(labels, row["frame"], branch=row["branch"])
            utility = utilities[row["event_uid"], row["branch"]]
            assert utility["actual_branch_sha256"] == row["actual_branch_sha256"]
            assert utility["original_future_axis"] == [row["frame"] + 1, row["frame"] + 100]
            counts = paired_counts(labels, utility)
            by_branch[row["branch"]].update(counts)
            subsets[SUBSETS[0]].update(counts)
            if labels["same_executed_action_configuration_as"] is None:
                subsets[SUBSETS[1]].update(counts)
                if labels["effective_direct_action_frames"]:
                    subsets[SUBSETS[2]].update(counts)
        census.append({"sequence": sequence, "role": recomputation["role"],
                       "subsets": {key: dict(subsets[key]) for key in SUBSETS},
                       "all_arms_by_branch": {key: dict(value) for key, value in by_branch.items()}})
        source_refs.extend({"path": str(path), "sha256": sha256(path)} for path in (recomputation_path, la_path, ua_path))
        print({"component_global_pairs": sequence, "actual_arms": len(rows)}, flush=True)
    roles = {}
    for role in ("FIT", "INNER"):
        selected = [r for r in census if r["role"] == role]
        roles[role] = {key: dict(sum((Counter(r["subsets"][key]) for r in selected), Counter())) for key in SUBSETS}
    value = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "status": "COMPLETE_ACTUAL_ALL24_POSTHOC_COMPONENT_VS_PINNED_WINDOW_UTILITY_DIAGNOSTIC",
        "source_sha256": sha256(__file__), "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
        "source_receipts": source_refs, "all24_census": census, "by_role": roles,
        "numerical_zero_tolerance_from_original_API_CLI_validation": EPSILON,
        "scope": "Current t excluded from future utility. Components include separate current safety guard. All full H100 windows compared; truncated futures retained but not labeled negative.",
        "POSTHOC_DIAGNOSTIC_NOT_POLICY_OR_MODEL_SELECTION": True,
        "overlapping_arm_counts_not_independent_events_or_full_video_MOT": True,
        "no_best_branch_or_seed_selected": True, "new_runtime_rollouts_or_optimizer_steps": 0,
        "frozen_models_thresholds_actions_and_sampling_NOT_changed": True,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "independent_beneficial_correction_roots": None,
        "scientific_G1_G2_PASS": False, "full_goal_complete": False, "next_stage_authorized": False}
    write_json("events/COMPONENT_VS_GLOBAL_UTILITY_DIAGNOSTIC_V1.json", value)
    append_log("ALL24_ACTUAL_COMPONENT_GLOBAL_UTILITY_POSTHOC_DIAGNOSTIC", independent_roots=None)
    print({"by_role": roles, "not_policy_selection": True}, flush=True)


if __name__ == "__main__":
    run()
