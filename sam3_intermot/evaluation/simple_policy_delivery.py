"""Fixed-policy delivery: all videos, actual frames, never proxy root counts."""
from collections import Counter
from .development_closure import ONSET_KEYS, summarize_complete_group, development_gates
from .event_delivery_evidence import component_audit


POLICIES = {
    "KEEP": "P0_KEEP", "SHADOW": "P1_SHADOW", "RECOVERY_ONLY": "P2_RECOVERY_ONLY",
    "UNCERTAIN_ONLY": "P3_BASELINE_UNCERTAINTY", "GLOBAL_REGRET": "P4_IDENTITY_MARGIN_GLOBAL_REGRET",
    "COMPETITOR_PROTECTION": "P5_COMPETITOR_PROTECTION",
    "DELAYED_CONFIRMATION": "P6_PERSISTENT_CAUSAL_CHALLENGER",
    "NATIVE_RELIABILITY": "P7_SELECTIVE_NATIVE_RELIABILITY",
    "SELECTIVE_ABSTENTION": "P8_SELECTIVE_ABSTENTION",
    "GLOBAL_REATTACHMENT": "P9_CONSERVATIVE_GLOBAL_REATTACHMENT",
}


def onset_counts(entries):
    if len({e["episode_uid"] for e in entries}) != len(entries):
        raise ValueError("No duplicate registered click diagnostics")
    result = Counter({key: 0 for key in ONSET_KEYS})
    for entry in entries:
        decisions = entry["one_shot_diagnostics"]
        if entry["effective_decisions"] != len(decisions) or len({d["frame"] for d in decisions}) != len(decisions):
            raise ValueError("Every effective action needs one unique actual own-prefix diagnostic")
        if bool(decisions) != entry["nonvacuity"]:
            raise ValueError("Zero-action diagnostics cannot pass nonvacuity")
        for decision in decisions:
            labels = decision["labels"]
            complete = labels["future"]["H100"]["complete"]
            risk = any(any(row[k] for k in ("N10", "non_target_damage", "verified_OTHER_writes", "UNKNOWN_writes"))
                       for row in labels["raw_frame_components"])
            # Adapter metadata is audit-only; it does not change stored labels,
            # add interventions, or assert independence from chronological gaps.
            audit = component_audit({**labels, "effective_direct_action_frames": [decision["frame"]],
                "direct_action_onsets_in_this_one_shot_branch": 1,
                "distinct_correlated_window_not_independent_causal_origin": True,
                "same_executed_action_configuration_as": None,
                "H100_any_harm_including_current_t_label": risk if complete else None},
                decision["frame"], branch="ACTUAL_CURRENT_THEN_KEEP")
            result.update(effective_decisions=1, complete_H100_decisions=int(complete),
                incomplete_H100_decisions=int(not complete),
                beneficial_complete_H100_decisions=int(audit["safe_positive_H100_arm_not_independent_correction"]),
                risky_decisions=int(risk),
                severe_non_target_harm_decisions=int(audit["severe_other_harm_H100_arm"]))
    return dict(result)


def complete_role(cells, baselines, initializations, densities, sequences, gates):
    summary = summarize_complete_group(cells, baselines, initializations, densities,
                                       seeds=("FIXED",), sequences=sequences)
    return {**summary, "original_development_gates": development_gates(summary, gates),
            "zero_action_or_local_one_shot_benefit_NEVER_research_PASS": True}
