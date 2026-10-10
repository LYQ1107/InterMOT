"""Live actual state-policy/density census; no stale success/publication alias."""
import argparse
import subprocess
from scripts import n72r21r2_continuation_checkpoint_v6 as base
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, storage, utcnow, update_status, append_log

FOLDERS = {**base.FOLDERS,
    "pure_margin_exact_CF": "availability/pure_margin_v1/CF_results",
    "support_ablation_full_MOT_pilot": "training/support_ablation_v1/results",
    "support_own_prefix_onset_audits": "training/support_onset_v1/labels",
    "main_full_video_runtime": "mot/main_policy_v1/runtime", "main_full_video_MOT": "mot/main_policy_v1/results",
    "main_own_prefix_onset_runtime": "mot/main_policy_v1/onset_runtime", "main_INNER_shared_selections": "mot/main_policy_v1/selections",
    "state_policy_full_video_runtime": "mot/state_policy_v1/runtime", "state_policy_full_video_MOT": "mot/state_policy_v1/results",
    "state_policy_own_prefix_onset_runtime": "mot/state_policy_v1/onset_runtime", "state_policy_INNER_shared_selections": "mot/state_policy_v1/selections"}
MARKERS = tuple(r for r in base.DRIVERS if r != "events/offline_diagnostics_driver_v1.json") + (
    "events/offline_diagnostics_driver_v2.json", "availability/pure_margin_v1/driver.json",
    "training/support_ablation_v1/driver.json", "training/support_onset_v1/driver.json",
    "mot/main_policy_v1/driver.json", "mot/state_policy_v1/driver.json")
ALIASES = {"actual_CF_sequence_runtimes_completed": "CF_runtime", "actual_CF_labels_completed": "CF_labels",
    "actual_pinned_event_utility_sequences_completed": "actual_pinned_utility", "actual_window_TrackEval_completed": "actual_window_TrackEval",
    "actual_simple_full_sequence_results_completed": "simple_full_video", "actual_simple_own_prefix_onset_audits_completed": "simple_own_prefix_onsets",
    "actual_M_A_memory_sequences_completed": "fixed_memory_full_video", "actual_fresh_memory_input_sequences_completed": "fresh_memory_inputs",
    "actual_fresh_memory_model_fits_completed": "fresh_memory_fits", "actual_fresh_memory_full_sequence_results_completed": "fresh_memory_full_video",
    "actual_matched_joint_state_supervision_sequences_completed": "joint_state_supervision",
    "actual_exact_CF_verifier_sequences_completed": "exact_CF_verifiers", "actual_pure_margin_CF_sequences_completed": "pure_margin_exact_CF",
    "actual_support_ablation_results_completed": "support_ablation_full_MOT_pilot", "actual_support_onset_audits_completed": "support_own_prefix_onset_audits",
    "actual_main_model_fits_completed": "actual_main_fits", "actual_main_full_video_results_completed": "main_full_video_MOT",
    "actual_main_INNER_shared_selections": "main_INNER_shared_selections", "new_state_source_fits_completed": "actual_state_source_fits",
    "actual_state_policy_full_video_results_completed": "state_policy_full_video_MOT",
    "actual_state_policy_INNER_shared_selections": "state_policy_INNER_shared_selections"}


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Narrow immutable checkpoint suffix required")
    counts = {k: len(list((OUT / v).rglob("*.json"))) for k, v in FOLDERS.items()}
    counts["actual_main_fits"] = len(list((OUT / "training/event_authority").glob("MAIN__*.json")))
    counts["actual_state_source_fits"] = len(list((OUT / "on_policy/state_source_fits_v1").glob("*STATE__seed*.json")))
    counts["actual_pilot_fits"] = len(list((OUT / "training/event_authority").glob("PILOT__*.json")))
    counts["actual_current_axis_fits"] = len(list((OUT / "availability/current_axis_fits").glob("*__seed*.json")))
    counts["actual_total_optimizer_model_records"] = sum(counts[k] for k in (
        "actual_main_fits", "actual_state_source_fits", "actual_pilot_fits", "actual_current_axis_fits", "fresh_memory_fits"))
    drivers = {}
    for relative in MARKERS:
        path = OUT / relative
        if not path.exists():
            drivers[relative] = {"status": "NOT_LAUNCHED_MARKER_ABSENT"}
            continue
        j = read_json(path)
        drivers[relative] = {"status": j["status"], "marker_sha256": sha256(path), "owner": base.proc(j.get("pid")),
            "active": j.get("active"), "active_procs": base.active_procs(j.get("active")),
            "failed_retained": j.get("failed_retained", []), "completed_count": len(j.get("completed", []))}
    publication_path = next(OUT / p for p in (
        "git_delivery/CHECKPOINT_STATE_DENSITY_SCENES_V9.json", "git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json") if (OUT / p).exists())
    publication = read_json(publication_path)
    assert publication["local_HEAD"] == publication["fresh_remote_HEAD"] and not publication["force_push"]
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    clean = not subprocess.check_output(["git", "-c", "core.fsmonitor=false", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    resource = storage()
    value = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "application_goal_status": "ACTIVE",
        "scientific_decision": "PENDING", "counts": counts, "drivers": drivers, "resource": resource,
        "local_HEAD": local, "last_verified_published_HEAD": publication["fresh_remote_HEAD"],
        "publication_receipt": str(publication_path), "published_matches_local": local == publication["fresh_remote_HEAD"],
        "current_working_tree_clean": clean, "all_current_working_changes_published": clean and local == publication["fresh_remote_HEAD"],
        "publication_verification": {k: publication.get(k, False) for k in (
            "verified_by_fresh_normal_git_ref_identity", "API_canonical_readback_claimed", "verified_by_fresh_API_ref_and_canonical_commit")},
        "frozen_main_policy_protocol_sha256": sha256(OUT / "mot/MAIN_POLICY_PROTOCOL_V1.json"),
        "frozen_state_policy_protocol_sha256": sha256(OUT / "on_policy/STATE_POLICY_PROTOCOL_V1.json"),
        "CONFIRM_VAL_TEST_SOT_unopened": True, "next_stage_authorized": False,
        "scope_remaining": "All24 original CF and controlled-state closure; registered42 MAIN and9 state-source real three-seed own full-MOT/onset/density studies; qualified model-on-policy/staged and M-B branches; memory-risk/open-set misses; final six tables/G0-G4/independent causal-root audit/scientific closure and verified code-only delivery."}
    for key, relative in (("density_actual_all24", "mot/DENSITY_BASELINE_SIMPLE_FULL_VIDEO_V1.json"),
                          ("scene_characteristics_offline_all24", "data/SCENE_CHARACTERISTICS_OFFLINE_V1.json")):
        path = OUT / relative
        value[key] = {"path": str(path), "sha256": sha256(path), "status": read_json(path)["status"]} if path.exists() else {"status": "PENDING_ACTUAL_OUTPUT"}
    regression_path = next((OUT / ("tests/REGRESSION_V" + v + ".json") for v in ("17", "16", "15") if (OUT / ("tests/REGRESSION_V" + v + ".json")).exists()), None)
    if regression_path:
        value["full_regression"] = read_json(regression_path)
    relative = "audit/CONTINUATION_CHECKPOINT_STATE_DENSITY_V9" + suffix + ".json"
    write_json(relative, value)
    fields = {alias: counts[key] for alias, key in ALIASES.items()}
    fields.update(application_goal_status="ACTIVE", application_goal_active=True, scientific_decision="PENDING", next_stage_authorized=False,
        latest_continuation_checkpoint="outputs/N72R21R2/" + relative, current_actual_pipeline_counts=counts,
        local_code_commit=local, published_code_HEAD=publication["fresh_remote_HEAD"], published_code_commit=publication["fresh_remote_HEAD"],
        last_verified_published_code_commit=publication["fresh_remote_HEAD"], GitHub_remote_HEAD_verified=value["published_matches_local"],
        new_checkpoint_publication="VERIFIED_NORMAL_NON_FORCE_V9_FRESH_GIT_REF" if publication_path.name.endswith("V9.json") else "VERIFIED_NORMAL_NON_FORCE_V8_FRESH_GIT_REF",
        API_canonical_readback_claimed=publication.get("API_canonical_readback_claimed", False), resource_snapshot=resource,
        all_current_working_changes_published=value["all_current_working_changes_published"],
        new_model_fits_completed=counts["actual_total_optimizer_model_records"],
        actual_pilot_model_fits_completed=counts["actual_pilot_fits"], actual_open_set_model_fits_completed=counts["actual_current_axis_fits"],
        paired_state_fit_driver_status=drivers["on_policy/state_source_fits_v1/driver.json"]["status"],
        main_policy_driver="mot/main_policy_v1/driver.json", state_policy_driver="mot/state_policy_v1/driver.json",
        frozen_main_policy_protocol_sha256=value["frozen_main_policy_protocol_sha256"], frozen_state_policy_protocol_sha256=value["frozen_state_policy_protocol_sha256"],
        all24_density_table=value["density_actual_all24"], all24_offline_scene_characteristics=value["scene_characteristics_offline_all24"])
    if "full_regression" in value:
        regression = value["full_regression"]
        fields.update(full_regression_passed=regression["passed"], full_regression_historical_failures=regression["failed"],
            latest_full_regression={"passed": regression["passed"], "failed": regression["failed"], "seconds": regression["seconds"], "all_passed": False},
            focused_test_passed=regression["focused"]["passed"], latest_focused_regression=regression["focused"])
    update_status(**fields)
    append_log("CONTINUATION_CHECKPOINT_STATE_DENSITY_V9", path=relative, counts=counts)
    print({"checkpoint": relative, "counts": counts, "live_scoped_owners": sum(j.get("owner", {}).get("is_live_exact_R2_process", False) for j in drivers.values()),
           "published_matches_local": value["published_matches_local"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
