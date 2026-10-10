"""Live optimizer/group/data-delivery census, not partial-table completion."""
import argparse
import subprocess
from scripts import n72r21r2_continuation_checkpoint_v11 as previous
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, storage, utcnow, update_status, append_log


def run(suffix, ledger):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Narrow immutable suffix required")
    ledger_path = OUT / "audit" / ledger
    if not ledger_path.resolve().is_relative_to((OUT / "audit").resolve()):
        raise ValueError("Only local requirement-ledger evidence")
    evidence = read_json(ledger_path)
    assert evidence["required_named_file_count"] == 74 and not evidence["overall_completion_proven"]
    assert len(evidence["required_six_final_tables"]) == 6 and len(evidence["required35_test_semantics"]) == 35
    groups = previous.optimizer_census()
    counts = {k: len(list((OUT / v).rglob("*.json"))) for k, v in previous.previous.prior.FOLDERS.items()}
    for key, name in (("actual_main_fits", "main"), ("actual_pilot_fits", "pilot"), ("actual_state_source_fits", "state_source"),
                      ("actual_current_axis_fits", "current_axis"), ("fresh_memory_fits", "fresh_memory")):
        counts[key] = groups[name]["verified_actual_optimizer_models"]
    counts["actual_total_optimizer_model_records"] = sum(g["verified_actual_optimizer_models"] for g in groups.values())
    counts["actual_total_fit_attempt_records"] = sum(g["attempt_records"] for g in groups.values())
    counts["actual_fit_attempts_without_optimization"] = sum(g["measured_no_optimization_records"] for g in groups.values())
    counts["actual_fit_records_unverified"] = sum(g["unverified_records"] for g in groups.values())
    drivers = {}
    for relative in previous.MARKERS:
        path = OUT / relative
        if not path.exists():
            drivers[relative] = {"status": "MARKER_ABSENT_NOT_LIVE"}
            continue
        j = read_json(path)
        active = j.get("active")
        entries = active if isinstance(active, list) else [active] if active else []
        owner = previous.previous.owned(j.get("pid"))
        drivers[relative] = {"status": j["status"], "marker_sha256": sha256(path), "owner": owner,
            "active": active, "active_procs": [previous.previous.owned(row.get("pid")) for row in entries],
            "completed_count": len(j.get("completed", [])), "failed_retained": j.get("failed_retained", []),
            "stale_ACTIVE_marker_without_live_owner": j["status"].startswith("ACTIVE") and not owner["is_live_exact_R2_process"]}
    pub_path = next(OUT / p for p in ("git_delivery/CHECKPOINT_ACTUAL_DATA_FULL_TASK_DELIVERY_V12.json",
                                   "git_delivery/CHECKPOINT_TERMINAL_STATE_OPTIMIZER_CENSUS_V11.json") if (OUT / p).exists())
    pub = read_json(pub_path)
    assert pub["local_HEAD"] == pub["fresh_remote_HEAD"] and not pub["force_push"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    clean = not subprocess.check_output(["git", "-c", "core.fsmonitor=false", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    regression = read_json(next(OUT / ("tests/REGRESSION_V" + n + ".json") for n in ("20", "19") if (OUT / ("tests/REGRESSION_V" + n + ".json")).exists()))
    memory, closure = (read_json(OUT / p) for p in ("memory/frontier_v1/latest.json", "mot/development_closure_v1/latest.json"))
    assert sha256(memory["path"]) == memory["sha256"] and sha256(closure["path"]) == closure["sha256"]
    counts["actual_complete_development_groups"] = closure["actual_complete_groups"]
    data = {key: {"path": str(OUT / key), "sha256": sha256(OUT / key), "status": read_json(OUT / key)["status"]}
            for key in ("data/CANDIDATE_EXTRACTION.json", "data/CANDIDATE_INTEGRITY.json", "tables/TABLE1_DATA_COVERAGE_PREPARATION_V1.json")}
    resource = storage()
    value = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "application_goal_status": "ACTIVE", "scientific_decision": "PENDING",
        "counts": counts, "actual_optimizer_evidence_census": groups, "drivers": drivers, "resource": resource,
        "full_task_deliverable_ledger": {"path": str(ledger_path), "sha256": sha256(ledger_path)},
        "required_files": 74, "present_required_files_NOT_verified_final": evidence["present_named_file_count"],
        "narrow_verified_preparations_NOT_goal_completion": evidence["verified_narrow_preparation_count"],
        "actual_all24_data_deliverables": data, "six_final_tables_complete": False, "full_task_complete": False,
        "latest_actual_memory_frontier": memory, "latest_all17_development_closure": closure, "full_regression": regression,
        "local_HEAD": head, "last_verified_published_HEAD": pub["fresh_remote_HEAD"], "publication_receipt": str(pub_path),
        "current_working_tree_clean": clean, "published_matches_local": head == pub["fresh_remote_HEAD"],
        "all_current_working_changes_published": clean and head == pub["fresh_remote_HEAD"],
        "API_canonical_readback_claimed": pub.get("API_canonical_readback_claimed", False),
        "CONFIRM_VAL_TEST_SOT_unopened": True, "next_stage_authorized": False,
        "scope_remaining": "All original M0-M11: remaining MAIN/STATE training and own full-MOT/all17 whole-group closure, all24 controlled-state/exact-CF/pure-margin/risk replay, actual independent-root proof or valid negative bounds, justified conditional branches, all74 semantically complete named files/six final tables/G0-G4 scientific conclusion/final verified code-only delivery."}
    relative = "audit/CONTINUATION_CHECKPOINT_FULL_TASK_DATA_V12" + suffix + ".json"
    write_json(relative, value)
    fields = {alias: counts[key] for alias, key in previous.previous.prior.ALIASES.items()}
    fields.update(application_goal_status="ACTIVE", application_goal_active=True, scientific_decision="PENDING", next_stage_authorized=False,
        latest_continuation_checkpoint="outputs/N72R21R2/" + relative, current_actual_pipeline_counts=counts,
        actual_optimizer_evidence_census={k: {n: v for n, v in g.items() if n != "records"} for k, g in groups.items()},
        new_model_fits_completed=counts["actual_total_optimizer_model_records"], actual_fit_attempt_records=counts["actual_total_fit_attempt_records"],
        actual_fit_attempts_without_optimization=counts["actual_fit_attempts_without_optimization"], actual_unverified_fit_records=counts["actual_fit_records_unverified"],
        resource_snapshot=resource, full_task_deliverable_ledger=value["full_task_deliverable_ledger"], actual_all24_data_deliverables=data,
        six_final_tables_complete=False, required_named_files=74, present_named_files_not_semantic_completion=evidence["present_named_file_count"],
        latest_all17_development_closure=closure, latest_actual_memory_frontier=memory, local_code_commit=head,
        published_code_HEAD=pub["fresh_remote_HEAD"], published_code_commit=pub["fresh_remote_HEAD"], last_verified_published_code_commit=pub["fresh_remote_HEAD"],
        GitHub_remote_HEAD_verified=value["published_matches_local"], all_current_working_changes_published=value["all_current_working_changes_published"],
        new_checkpoint_publication="VERIFIED_NORMAL_NON_FORCE_V12_FRESH_GIT_REF" if pub_path.name.endswith("V12.json") else "VERIFIED_NORMAL_NON_FORCE_V11_FRESH_GIT_REF",
        API_canonical_readback_claimed=value["API_canonical_readback_claimed"], full_regression_passed=regression["passed"], full_regression_historical_failures=regression["failed"],
        latest_full_regression={"passed": regression["passed"], "failed": regression["failed"], "seconds": regression["seconds"], "all_passed": False},
        focused_test_passed=regression["focused"]["passed"], latest_focused_regression=regression["focused"])
    update_status(**fields)
    append_log("FULL_TASK_DATA_CENSUS_V12", path=relative, counts=counts)
    print({"checkpoint": relative, "counts": counts, "live_owners": sum(j.get("owner", {}).get("is_live_exact_R2_process", False) for j in drivers.values()),
           "all_current_changes_published": value["all_current_working_changes_published"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    run(args.suffix, args.ledger)
