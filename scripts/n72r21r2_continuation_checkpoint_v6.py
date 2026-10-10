"""Actual live census, regression receipts and truthful Goal/publication state."""
import argparse
from pathlib import Path
import subprocess
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, utcnow, storage, update_status, append_log

FOLDERS = {
    "fresh_candidates": "data/candidate_integrity", "fresh_baselines": "mot/baseline_results",
    "CF_runtime": "events/counterfactual_sequences", "CF_labels": "events/label_audit",
    "actual_pinned_utility": "events/trajectory_utility_audit", "actual_window_TrackEval": "events/window_trackeval_results",
    "simple_full_video": "simple/results", "simple_own_prefix_onsets": "simple/onset_audit",
    "fixed_memory_full_video": "memory/M_A/results", "fresh_memory_inputs": "memory/risk_v1/supervision",
    "fresh_memory_fits": "memory/risk_v1/fits", "fresh_memory_full_video": "memory/risk_v1/results",
    "joint_state_supervision": "on_policy/joint_state_v1/supervision", "exact_CF_verifiers": "availability/exact_cf_v1/results",
}
DRIVERS = ("events/event_driver_v1.json", "simple/simple_driver_v1.json", "events/offline_diagnostics_driver_v1.json",
           "memory/memory_ma_driver_v1.json", "events/context_recovery_driver_v2.json", "training/main_fit_driver_v2.json",
           "on_policy/joint_state_v1/driver.json", "on_policy/state_source_fits_v1/driver.json",
           "availability/exact_cf_v1/driver.json", "memory/risk_v1/driver.json", "on_policy/joint_state_tensor_v2/driver.json")


def proc(pid):
    if not pid:
        return {"pid": pid, "is_live_exact_R2_process": False}
    try:
        args = Path("/proc/" + str(pid) + "/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        args = []
    own = any(a.startswith(b"scripts.n72r21r2_") for a in args)
    return {"pid": pid, "is_live_exact_R2_process": own,
            "exact_argv_if_owned": [a.decode() for a in args if a] if own else None}


def active_procs(active):
    entries = active if isinstance(active, list) else [] if active is None else [active]
    return [proc(entry.get("pid")) for entry in entries]


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Checkpoint suffix must be a narrow filename identifier")
    counts = {name: len(list((OUT / folder).glob("*.json"))) for name, folder in FOLDERS.items()}
    counts["actual_main_fits"] = len(list((OUT / "training/event_authority").glob("MAIN__*.json")))
    counts["actual_state_source_fits"] = len(list((OUT / "on_policy/state_source_fits_v1").glob("*STATE__seed*.json")))
    drivers = {}
    for relative in DRIVERS:
        path = OUT / relative
        if not path.exists():
            drivers[relative] = {"status": "MARKER_NOT_AT_REGISTERED_NAME_NEEDS_INSPECTION"}
            continue
        j = read_json(path)
        drivers[relative] = {"marker_sha256": sha256(path), "status": j["status"], "owner": proc(j.get("pid")),
                             "active": j.get("active"), "active_procs": active_procs(j.get("active")),
                             "failed_retained": j.get("failed_retained", []), "completed": j.get("completed", [])}
    regression = read_json(OUT / "tests/REGRESSION_V10.json")
    latest_memory = read_json(OUT / "memory/frontier_v1/latest.json")
    assert sha256(latest_memory["path"]) == latest_memory["sha256"]
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    publications = ("git_delivery/CHECKPOINT_CURRENT_RISK_TENSOR_RECOVERY_V5.json", "git_delivery/CHECKPOINT_MATCHED_JOINT_STATE_V4.json")
    publication = next(read_json(OUT / p) for p in publications if (OUT / p).exists())
    assert publication["local_HEAD"] == publication["fresh_remote_HEAD"] and not publication["force_push"]
    resource = storage()
    checkpoint = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "application_goal_status": "ACTIVE",
                  "scientific_decision": "PENDING", "counts": counts, "drivers": drivers, "resource": resource,
                  "regression": regression, "memory_partial_frontier": latest_memory,
                  "local_HEAD": local, "last_verified_published_HEAD": publication["fresh_remote_HEAD"],
                  "last_published_HEAD_is_current_local": local == publication["fresh_remote_HEAD"],
                  "main_wait_termination_receipt_sha256": sha256(OUT / "training/main_fit_driver_v1_TERMINAL_143.json"),
                  "CONFIRM_VAL_TEST_SOT_unopened": True, "next_stage_authorized": False,
                  "remaining": "All24 original controls/CF and own-state recovery,42 main fits+objectives, main own-MOT/qualified on-policy branches, all24 memory/risk uncertainty and open-set/density/generalization closure remain required. No scope narrowing."}
    relative = "audit/CONTINUATION_CHECKPOINT_RISK_TENSOR_V6" + suffix + ".json"
    write_json(relative, checkpoint)
    update_status(application_goal_status="ACTIVE", application_goal_active=True, scientific_decision="PENDING", next_stage_authorized=False,
                  latest_continuation_checkpoint="outputs/N72R21R2/" + relative, resource_snapshot=resource,
                  local_code_commit=local, published_code_HEAD=publication["fresh_remote_HEAD"],
                  last_verified_published_code_commit=publication["fresh_remote_HEAD"], published_code_commit=publication["fresh_remote_HEAD"],
                  published_code_only=True, GitHub_remote_HEAD_verified=local == publication["fresh_remote_HEAD"],
                  actual_CF_sequence_runtimes_completed=counts["CF_runtime"], actual_CF_labels_completed=counts["CF_labels"],
                  actual_fresh_CF_sequences_complete=counts["CF_runtime"], actual_fresh_CF_label_sequences_complete=counts["CF_labels"],
                  actual_pinned_event_utility_sequences_completed=counts["actual_pinned_utility"],
                  actual_window_TrackEval_completed=counts["actual_window_TrackEval"],
                  actual_simple_full_sequence_results_completed=counts["simple_full_video"],
                  actual_simple_own_prefix_onset_audits_completed=counts["simple_own_prefix_onsets"],
                  actual_M_A_memory_sequences_completed=counts["fixed_memory_full_video"], actual_memory_M_A_sequences_complete=counts["fixed_memory_full_video"],
                  actual_fresh_memory_input_sequences_completed=counts["fresh_memory_inputs"], actual_fresh_memory_model_fits_completed=counts["fresh_memory_fits"],
                  actual_fresh_memory_full_sequence_results_completed=counts["fresh_memory_full_video"],
                  actual_main_model_fits_completed=counts["actual_main_fits"], new_state_source_fits_completed=counts["actual_state_source_fits"],
                  actual_matched_joint_state_supervision_sequences_completed=counts["joint_state_supervision"],
                  actual_exact_CF_verifier_sequences_completed=counts["exact_CF_verifiers"],
                  main_fit_driver_v1_status="TERMINATED_EXIT143_CAUSE_UNKNOWN_ZERO_MAIN_FITS",
                  main_fit_driver_current="training/main_fit_driver_v2.json", memory_frontier_latest=latest_memory,
                  latest_full_regression={"passed": regression["passed"], "failed": regression["failed"], "seconds": regression["seconds"], "all_passed": False},
                  full_regression_passed=regression["passed"], full_regression_historical_failures=regression["failed"])
    append_log("CONTINUATION_CHECKPOINT_RISK_TENSOR_V6", path=relative, counts=counts, last_verified_published_HEAD=publication["fresh_remote_HEAD"])
    print({"checkpoint": relative, "counts": counts, "driver_census": {k: {"status": v["status"], "live": v.get("owner", {}).get("is_live_exact_R2_process")} for k, v in drivers.items()}}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
