"""Actual census includes own-onset and finite MAIN full-MOT queue owners."""
import argparse
import subprocess
from scripts import n72r21r2_continuation_checkpoint_v6 as base
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, storage, utcnow, update_status, append_log


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Narrow checkpoint suffix required")
    folders = {**base.FOLDERS, "pure_margin_exact_CF": "availability/pure_margin_v1/CF_results",
        "support_ablation_full_MOT_pilot": "training/support_ablation_v1/results",
        "support_own_prefix_onset_audits": "training/support_onset_v1/labels",
        "main_full_video_runtime": "mot/main_policy_v1/runtime", "main_full_video_MOT": "mot/main_policy_v1/results",
        "main_own_prefix_onset_runtime": "mot/main_policy_v1/onset_runtime", "main_INNER_shared_selections": "mot/main_policy_v1/selections"}
    counts = {k: len(list((OUT / v).rglob("*.json"))) for k, v in folders.items()}
    counts["actual_main_fits"] = len(list((OUT / "training/event_authority").glob("MAIN__*.json")))
    counts["actual_state_source_fits"] = len(list((OUT / "on_policy/state_source_fits_v1").glob("*STATE__seed*.json")))
    markers = tuple(r for r in base.DRIVERS if r != "events/offline_diagnostics_driver_v1.json") + (
        "events/offline_diagnostics_driver_v2.json", "availability/pure_margin_v1/driver.json",
        "training/support_ablation_v1/driver.json", "training/support_onset_v1/driver.json", "mot/main_policy_v1/driver.json")
    drivers = {}
    for relative in markers:
        if not (OUT / relative).exists():
            drivers[relative] = {"status": "NOT_LAUNCHED_MARKER_ABSENT"}
            continue
        j = read_json(OUT / relative)
        drivers[relative] = {"status": j["status"], "marker_sha256": sha256(OUT / relative),
            "owner": base.proc(j.get("pid")), "active": j.get("active"),
            "active_procs": base.active_procs(j.get("active")), "failed_retained": j.get("failed_retained", []),
            "completed_count": len(j.get("completed", []))}
    publication_path = next(OUT / p for p in ("git_delivery/CHECKPOINT_OWN_ONSET_MAIN_FULL_MOT_V8.json",
        "git_delivery/CHECKPOINT_LINEAR_SOURCE_SHIFT_V7.json", "git_delivery/CHECKPOINT_PURE_MARGIN_HAND_FILTER_SUPPORT_V6.json") if (OUT / p).exists())
    publication = read_json(publication_path)
    assert publication["local_HEAD"] == publication["fresh_remote_HEAD"] and not publication["force_push"]
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    resource = storage()
    value = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "application_goal_status": "ACTIVE",
        "scientific_decision": "PENDING", "counts": counts, "drivers": drivers, "resource": resource,
        "local_HEAD": local, "last_verified_published_HEAD": publication["fresh_remote_HEAD"],
        "publication_receipt": str(publication_path), "published_matches_local": local == publication["fresh_remote_HEAD"],
        "frozen_main_policy_protocol_sha256": sha256(OUT / "mot/MAIN_POLICY_PROTOCOL_V1.json"),
        "CONFIRM_VAL_TEST_SOT_unopened": True, "next_stage_authorized": False,
        "scope_remaining": "All24 CF/state-source closure; finite42-model MAIN own-video INNER/FIT nine-metric/onset/objective/architecture studies; qualified model-on-policy/staged and M-B branches; independent M-A learned risk/open-set misses/density/final G0-G4/root proof/closure and verified code-only delivery."}
    regression_path = next((OUT / ("tests/REGRESSION_V" + v + ".json") for v in ("15", "14") if (OUT / ("tests/REGRESSION_V" + v + ".json")).exists()), None)
    if regression_path:
        value["full_regression"] = read_json(regression_path)
    relative = "audit/CONTINUATION_CHECKPOINT_MAIN_V8" + suffix + ".json"
    write_json(relative, value)
    update_status(application_goal_status="ACTIVE", application_goal_active=True, scientific_decision="PENDING", next_stage_authorized=False,
        latest_continuation_checkpoint="outputs/N72R21R2/" + relative, current_actual_pipeline_counts=counts,
        local_code_commit=local, published_code_HEAD=publication["fresh_remote_HEAD"], published_code_commit=publication["fresh_remote_HEAD"],
        last_verified_published_code_commit=publication["fresh_remote_HEAD"], GitHub_remote_HEAD_verified=local == publication["fresh_remote_HEAD"],
        resource_snapshot=resource, actual_main_model_fits_completed=counts["actual_main_fits"],
        actual_main_full_video_results_completed=counts["main_full_video_MOT"],
        actual_main_INNER_shared_selections=counts["main_INNER_shared_selections"],
        actual_support_onset_audits_completed=counts["support_own_prefix_onset_audits"],
        main_policy_driver="mot/main_policy_v1/driver.json", frozen_main_policy_protocol_sha256=value["frozen_main_policy_protocol_sha256"])
    if "full_regression" in value:
        regression = value["full_regression"]
        update_status(full_regression_passed=regression["passed"], full_regression_historical_failures=regression["failed"],
            latest_full_regression={"passed": regression["passed"], "failed": regression["failed"], "seconds": regression["seconds"], "all_passed": False})
    append_log("CONTINUATION_CHECKPOINT_MAIN_V8", path=relative, counts=counts)
    print({"checkpoint": relative, "counts": counts, "live_scoped_owners": sum(j.get("owner", {}).get("is_live_exact_R2_process", False) for j in drivers.values()),
           "published_matches_local": value["published_matches_local"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
