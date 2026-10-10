"""Full-task artifact census and preparation coverage, never science closure."""

GROUPS = {
    "": ("FINAL_GOAL.json", "PREREGISTRATION.json", "stage_status.json", "EXECUTION_LOG.md", "FINAL_RESULT.json", "FINAL_REPORT.md"),
    "audit": ("SOURCE_AUDIT.md", "BASELINE_REPRODUCTION.json", "CHECKPOINT_LINEAGE.json"),
    "data": ("FROZEN_SPLITS.json", "EXISTING_DATASETS.json", "CANDIDATE_EXTRACTION.json", "CANDIDATE_INTEGRITY.json", "VIDEO_DENSITY.json", "STORAGE_AUDIT.json"),
    "events": ("EVENT_SCHEMA.json", "EVENT_CORPUS.json", "EVENT_LABEL_AUDIT.json", "DIRECT_VS_PROPAGATED.json", "COUNTERFACTUAL_BRANCHES.json", "ACTION_FEASIBILITY.json"),
    "simple": ("KEEP.json", "SHADOW.json", "RECOVERY_ONLY.json", "UNCERTAIN_ONLY.json", "GLOBAL_REGRET.json", "COMPETITOR_PROTECTION.json", "NATIVE_RELIABILITY.json", "DELAYED_CONFIRMATION.json"),
    "training": ("TRAINING_DATA.json", "MODEL_MANIFEST.json", "SCALAR.json", "LOGISTIC.json", "MLP.json", "RANKER.json", "GLOBAL_RISK.json", "TEMPORAL.json", "RELATIONAL.json", "VALUE_MODEL.json", "DUAL_STATE.json"),
    "objectives": ("CURRENT_CORRECTNESS.json", "EVENT_CORRECTION.json", "H20.json", "H50.json", "H100.json", "GLOBAL_ASSOCIATION.json", "RISK_CONSTRAINED.json"),
    "on_policy": ("BASELINE_STATE.json", "TREATMENT_STATE.json", "MIXED_STATE.json", "CORRECTION_TRAINING.json"),
    "availability": ("OPEN_SET.json", "WRONG_TAKEOVER.json", "RISK_COVERAGE.json"),
    "memory": ("FROZEN.json", "DELAYED.json", "TRUSTED_BANK.json", "RISK_WRITE.json", "ROLLBACK.json", "SAFETY_FRONTIER.json"),
    "mot": ("BASELINE.json", "PILOT.json", "INNER_RESULTS.json", "PER_SEQUENCE.json", "HOTA_ASSA_IDF1.json", "N01_N10.json", "NON_TARGET_DAMAGE.json", "LATENCY.json"),
    "confirmation": ("FROZEN_POLICY.json", "RESULTS.json", "PAIRED_DELTAS.json"),
    "checkpoints": ("SHA256_MANIFEST.json",), "tests": ("FOCUSED.json", "REGRESSION.json"),
}
REQUIRED = tuple((group + "/" if group else "") + name for group, names in GROUPS.items() for name in names)

TABLE_REQUIREMENTS = {
    "T1_DATA_EVENTS": ("all40_train_roles", "independent_people_or_explicit_unavailable", "sole_clicks", "independent_correction_roots", "N10_onset_events", "candidate_coverage", "density", "integrity"),
    "T2_MECHANISMS": ("KEEP_Recovery_Confidence_GlobalRegret_CompetitorProtection_Delayed_RiskGated", "changed_events_vs_decisions", "N01_N10", "harm_onsets", "other_ID_damage", "persistent_correction"),
    "T3_OBJECTIVES": ("same_architecture_seven_losses", "all3_seeds", "independent_event_precision_recall_or_unavailable", "harmful_override_rate", "N01_N10", "delta_AssA_HOTA"),
    "T4_ARCHITECTURES": ("all9_families_or_justified_conditional_NOT_RUN", "parameters", "head_latency_separate_from_full_pipeline", "all3_seeds"),
    "T5_MEMORY_OPEN_SET": ("false_takeover", "Recall_at_verified_OTHER_FPR2", "physical_absence_vs_no_positive_candidate", "correct_write_retention", "wrong_plus_UNKNOWN_writes", "target_recall", "global_association", "cluster_uncertainty"),
    "T6_FULL_MOT": ("historical_C0_ACIB", "fresh_FIT_INNER_C0", "INNER_only_BestSimple_BestLearned_no_best_seed", "qualified_frozen_confirmation_or_NOT_RUN", "per_density", "all9_metrics", "paired_video_CI_improved_degraded_max_regression"),
}

TEST_REQUIREMENTS = (
    "source_SHA", "split_no_leakage", "confirmation_never_trains", "candidate_UID_axis", "frozen_encoder",
    "one_click", "no_second_click", "GT_free_runtime", "KEEP_AA", "full_global_assignment", "hard_negatives",
    "NONE", "candidate_unique", "public_ID_unique", "correct_state_propagation", "wrong_state_propagation",
    "first_divergence", "event_deduplication", "frame_vs_event_N01_N10", "non_target_damage", "GT_offline_labels",
    "same_prestate_CF", "no_future_features", "on_policy_isolation", "committed_only_memory_write",
    "no_zero_write_safety", "UNKNOWN_not_verified_negative", "checkpoint_SHA", "controller_schema",
    "full_video_TrackEval", "metric_parsing", "per_sequence_pairing", "resource_budget", "resumable_run", "code_only_Git",
)


def count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Actual nonnegative integer exposure count required")
    return value


def coverage_row(sequence, role, init, candidate, baseline, scene):
    if role not in ("fit", "inner") or any(value["sequence"] != sequence for value in (candidate, baseline, scene)):
        raise ValueError("Only matching frozen FIT/INNER source records")
    if "sequence" in init and init["sequence"] != sequence:
        raise ValueError("An optional initialization header cannot contradict its clicks")
    inputs = init["inputs"]
    episodes = {e["episode_uid"] for e in inputs}
    if len(episodes) != len(inputs) or any(e["role"] != role.upper() or e["sequence"] != sequence for e in inputs):
        raise ValueError("No duplicate or wrong-role click replacement")
    if any(not e["sole_click"] or e["runtime_future_GT_used"] for e in inputs):
        raise ValueError("Only sole human clicks with no future GT")
    valid = {e["episode_uid"] for e in inputs if not e["initialization_failure"]}
    if any(episode not in baseline["coverage"] for episode in valid):
        raise ValueError("Every valid click needs same-input actual baseline coverage")
    if init["candidate_index_sha256"] != candidate["index_sha256"] or baseline["candidate_index_sha256"] != candidate["index_sha256"]:
        raise ValueError("Coverage must use the original same candidate input")
    clicks = []
    for e in inputs:
        failure = bool(e["initialization_failure"])
        values = None if failure else baseline["coverage"][e["episode_uid"]]
        if values is not None:
            visible, available = count(values["visible_frames"]), count(values["strict_positive_available_frames"])
            if not 0 <= available <= visible or values["initialization_failure"]:
                raise ValueError("Actual candidate opportunity denominator required")
        clicks.append({"episode_uid": e["episode_uid"], "initialization_failure_retained": failure,
                       "actual_same_input_postclick_coverage": values})
    return {"sequence": sequence, "role": role, "status": "COMPLETE_ACTUAL_INPUT_COVERAGE_NOT_CAUSAL_EVENT_CLOSURE",
        "original_frames": count(candidate["frames"]), "real_candidate_count": count(candidate["candidate_count"]),
        "original_candidate_density": candidate["whole_video_density"],
        "sole_click_attempts": len(inputs), "valid_sole_clicks": len(valid), "failed_initializations_retained": len(inputs) - len(valid),
        "coverage_per_registered_click": clicks,
        "video_local_GT_identity_count_NOT_cross_scene_people": count(scene["original_complete_video"]["GT_video_local_identities_NOT_cross_scene_people"]),
        "independent_people_count": None, "independent_people_status": "UNPROVEN_CROSS_RECORDING_PERSON_IDENTITY",
        "independent_beneficial_correction_roots": None, "independent_N10_onset_events": None,
        "event_proof_status": "PENDING_SEPARATE_CAUSAL_ROOT_AUDIT; not replaced by frame runs, clicks or seeds",
        "candidate_integrity": "VERIFIED_FULL_ORIGINAL_AXIS_UID_FEATURE_OFFSETS", "scientific_success": None}


def unopened_row(sequence, role):
    if role not in ("confirmation", "historical_development"):
        raise ValueError("Not an unopened or already-exposed role")
    return {"sequence": sequence, "role": role,
        "status": "NOT_RUN_UNOPENED_CONFIRMATION_PENDING_G0_G1_G2" if role == "confirmation" else "HISTORICAL_ALREADY_EXPOSED_NOT_FRESH_POPULATION",
        "fresh_candidate_generation_or_truth_access": False, "independent_people_count": None,
        "independent_beneficial_correction_roots": None, "independent_N10_onset_events": None,
        "new_sole_click_attempts": None, "fresh_candidate_coverage": None, "fresh_density": None,
        "scientific_success": None}


def named_requirement(relative, present, evidence_sha256=None, narrow_proof=None):
    if relative not in REQUIRED:
        raise ValueError("Unrequested path is not a required artifact")
    if narrow_proof and not present:
        raise ValueError("Absent file cannot have a verified proof")
    return {"path": relative, "present": bool(present), "sha256": evidence_sha256,
            "evidence_status": "VERIFIED_NARROW_PREPARATION" if narrow_proof else "PRESENT_SEMANTICS_NOT_FINAL_VERIFIED" if present else "REQUIRED_REPORT_MISSING",
            "narrow_proof": narrow_proof, "counts_as_full_goal_completion": False,
            "completion_requirement": "Scope-matched contents and referenced actual experiments/conditional disposition; file existence or test totals alone never suffice"}
