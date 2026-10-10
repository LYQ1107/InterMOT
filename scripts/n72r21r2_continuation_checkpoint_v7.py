"""Actual process and result census, including diagnostic/support recoveries."""
import argparse
import subprocess
from scripts import n72r21r2_continuation_checkpoint_v6 as previous
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, storage, utcnow, update_status, append_log


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Narrow checkpoint suffix required")
    folders = {**previous.FOLDERS,
               "pure_margin_exact_CF": "availability/pure_margin_v1/CF_results",
               "support_ablation_full_MOT_pilot": "training/support_ablation_v1/results"}
    counts = {k: len(list((OUT / v).rglob("*.json"))) for k, v in folders.items()}
    counts["actual_main_fits"] = len(list((OUT / "training/event_authority").glob("MAIN__*.json")))
    counts["actual_state_source_fits"] = len(list((OUT / "on_policy/state_source_fits_v1").glob("*STATE__seed*.json")))
    markers = tuple(r for r in previous.DRIVERS if r != "events/offline_diagnostics_driver_v1.json") + (
        "events/offline_diagnostics_driver_v2.json", "availability/pure_margin_v1/driver.json", "training/support_ablation_v1/driver.json")
    drivers = {}
    for relative in markers:
        j = read_json(OUT / relative)
        drivers[relative] = {"status": j["status"], "marker_sha256": sha256(OUT / relative),
            "owner": previous.proc(j.get("pid")), "active": j.get("active"),
            "active_procs": previous.active_procs(j.get("active")), "failed_retained": j.get("failed_retained", []),
            "completed": j.get("completed", [])}
    publication_path = next(OUT / p for p in ("git_delivery/CHECKPOINT_LINEAR_SOURCE_SHIFT_V7.json",
                                              "git_delivery/CHECKPOINT_PURE_MARGIN_HAND_FILTER_SUPPORT_V6.json",
                                              "git_delivery/CHECKPOINT_CURRENT_RISK_TENSOR_RECOVERY_V5.json") if (OUT / p).exists())
    publication = read_json(publication_path)
    assert publication["local_HEAD"] == publication["fresh_remote_HEAD"] and not publication["force_push"]
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    regression_path = next(OUT / ("tests/REGRESSION_V" + v + ".json") for v in ("14", "13", "12") if (OUT / ("tests/REGRESSION_V" + v + ".json")).exists())
    regression = read_json(regression_path)
    support = read_json(OUT / "training/authority_support_v1/latest.json")
    assert sha256(support["path"]) == support["sha256"]
    resource = storage()
    checkpoint = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "application_goal_status": "ACTIVE",
        "scientific_decision": "PENDING", "counts": counts, "drivers": drivers, "resource": resource,
        "full_regression": regression, "pure_margin_current_calibration_sha256": sha256(OUT / "availability/pure_margin_v1/current_calibration.json"),
        "frozen_hand_filter_support_snapshot": support, "local_HEAD": local,
        "last_verified_published_HEAD": publication["fresh_remote_HEAD"], "publication_receipt_path": str(publication_path),
        "last_published_HEAD_is_current_local": local == publication["fresh_remote_HEAD"],
        "original_diagnostic_exit143_receipt_sha256": sha256(OUT / "events/offline_diagnostics_driver_v1_TERMINAL_143.json"),
        "read_only_linear_source_shift_diagnostic_sha256": sha256(OUT / "training/SUPPORT_LINEAR_SOURCE_SHIFT_DIAGNOSTIC_V2.json") if (OUT / "training/SUPPORT_LINEAR_SOURCE_SHIFT_DIAGNOSTIC_V2.json").exists() else None,
        "CONFIRM_VAL_TEST_SOT_unopened": True, "next_stage_authorized": False,
        "remaining_full_task_scope": "All24 CF/simple/memory/state-source closure;42 registered main fits with full own-MOT and objective/architecture comparison; matched/model-generated on-policy and staged branches when feasible; G4 uncertainty/open-set misses/density/generalization/final evidence and verified code-only delivery. No scope narrowing."}
    relative = "audit/CONTINUATION_CHECKPOINT_SUPPORT_V7" + suffix + ".json"
    write_json(relative, checkpoint)
    update_status(application_goal_status="ACTIVE", application_goal_active=True, scientific_decision="PENDING", next_stage_authorized=False,
        latest_continuation_checkpoint="outputs/N72R21R2/" + relative, current_actual_pipeline_counts=counts,
        local_code_commit=local, published_code_HEAD=publication["fresh_remote_HEAD"], published_code_commit=publication["fresh_remote_HEAD"],
        last_verified_published_code_commit=publication["fresh_remote_HEAD"], GitHub_remote_HEAD_verified=local == publication["fresh_remote_HEAD"],
        resource_snapshot=resource, frozen_hand_filter_support_snapshot=support,
        offline_diagnostics_original_status="TERMINATED_EXIT143_CAUSE_UNKNOWN_RESULTS_RETAINED", offline_diagnostics_current="events/offline_diagnostics_driver_v2.json",
        support_ablation_status="UNQUALIFIED_FIT_ONLY_FROZEN_MODELS_NO_FIT", actual_support_ablation_full_MOT_cells=counts["support_ablation_full_MOT_pilot"],
        actual_pure_margin_exact_CF_sequences=counts["pure_margin_exact_CF"],
        actual_CF_sequence_runtimes_completed=counts["CF_runtime"], actual_CF_labels_completed=counts["CF_labels"],
        actual_fresh_CF_sequences_complete=counts["CF_runtime"], actual_fresh_CF_label_sequences_complete=counts["CF_labels"],
        actual_pinned_event_utility_sequences_completed=counts["actual_pinned_utility"],
        actual_window_TrackEval_completed=counts["actual_window_TrackEval"],
        actual_simple_full_sequence_results_completed=counts["simple_full_video"],
        actual_simple_own_prefix_onset_audits_completed=counts["simple_own_prefix_onsets"],
        actual_M_A_memory_sequences_completed=counts["fixed_memory_full_video"], actual_memory_M_A_sequences_complete=counts["fixed_memory_full_video"],
        actual_fresh_memory_input_sequences_completed=counts["fresh_memory_inputs"],
        actual_fresh_memory_model_fits_completed=counts["fresh_memory_fits"], actual_fresh_memory_full_sequence_results_completed=counts["fresh_memory_full_video"],
        actual_main_model_fits_completed=counts["actual_main_fits"], new_state_source_fits_completed=counts["actual_state_source_fits"],
        actual_matched_joint_state_supervision_sequences_completed=counts["joint_state_supervision"],
        actual_exact_CF_verifier_sequences_completed=counts["exact_CF_verifiers"],
        latest_full_regression={"passed": regression["passed"], "failed": regression["failed"], "seconds": regression["seconds"], "all_passed": False},
        full_regression_passed=regression["passed"], full_regression_historical_failures=regression["failed"])
    append_log("CONTINUATION_CHECKPOINT_SUPPORT_V7", path=relative, counts=counts, published=publication["fresh_remote_HEAD"])
    print({"checkpoint": relative, "counts": counts, "live_exact_owners": sum(v["owner"]["is_live_exact_R2_process"] for v in drivers.values()),
           "required_owners": len(drivers), "published_matches_local": local == publication["fresh_remote_HEAD"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
