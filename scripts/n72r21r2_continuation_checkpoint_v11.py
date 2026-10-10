"""Verified optimizer census, dead-owner observation, isolated V11 delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scripts import n72r21r2_continuation_checkpoint_v10 as previous
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, utcnow, storage, update_status, append_log
from sam3_intermot.evaluation.optimization_census import assess_fit, census

FITS = {
    "main": ("training/event_authority", "MAIN__*.json", "training/EVENT_AUTHORITY_PROTOCOL_V1.json"),
    "pilot": ("training/event_authority", "PILOT__*.json", "training/EVENT_AUTHORITY_PROTOCOL_V1.json"),
    "state_source": ("on_policy/state_source_fits_v1", "*STATE__seed*.json", "on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json"),
    "current_axis": ("availability/current_axis_fits", "*__seed*.json", "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json"),
    "fresh_memory": ("memory/risk_v1/fits", "*__seed*.json", "memory/RISK_WRITE_PROTOCOL_V1.json"),
}
MARKERS = (*previous.prior.MARKERS, "on_policy/joint_state_terminal_resume_v3/driver.json")


def scoped_hash(path, root):
    path = Path(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Census artifact escaped its scoped root")
    return sha256(path)


def optimizer_census():
    groups = {}
    for name, (folder, pattern, protocol) in FITS.items():
        rows = []
        for path in sorted((OUT / folder).glob(pattern)):
            raw = path.read_bytes()
            result = assess_fit(json.loads(raw), protocol_sha256=sha256(OUT / protocol),
                artifact_sha256=lambda p: scoped_hash(p, ASSETS), source_sha256=lambda p: scoped_hash(ROOT / p, ROOT))
            rows.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), **result})
        groups[name] = {**census(rows), "records": rows}
    return groups


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Narrow immutable suffix required")
    groups = optimizer_census()
    counts = {k: len(list((OUT / v).rglob("*.json"))) for k, v in previous.prior.FOLDERS.items()}
    for key, name in (("actual_main_fits", "main"), ("actual_pilot_fits", "pilot"),
                      ("actual_state_source_fits", "state_source"), ("actual_current_axis_fits", "current_axis"),
                      ("fresh_memory_fits", "fresh_memory")):
        counts[key] = groups[name]["verified_actual_optimizer_models"]
    counts["actual_total_optimizer_model_records"] = sum(g["verified_actual_optimizer_models"] for g in groups.values())
    counts["actual_total_fit_attempt_records"] = sum(g["attempt_records"] for g in groups.values())
    counts["actual_fit_attempts_without_optimization"] = sum(g["measured_no_optimization_records"] for g in groups.values())
    counts["actual_fit_records_unverified"] = sum(g["unverified_records"] for g in groups.values())
    drivers = {}
    for relative in MARKERS:
        path = OUT / relative
        if not path.exists():
            drivers[relative] = {"status": "MARKER_ABSENT_NOT_A_LIVE_JOB"}
            continue
        raw = path.read_bytes()
        value = json.loads(raw)
        active = value.get("active")
        entries = active if isinstance(active, list) else [active] if active else []
        owner = previous.owned(value.get("pid"))
        drivers[relative] = {"status": value["status"], "marker_sha256": hashlib.sha256(raw).hexdigest(),
            "owner": owner, "active": active, "active_procs": [previous.owned(e.get("pid")) for e in entries],
            "completed_count": len(value.get("completed", [])), "failed_retained": value.get("failed_retained", []),
            "stale_ACTIVE_marker_without_live_owner": value["status"].startswith("ACTIVE") and not owner["is_live_exact_R2_process"],
            "marker_status_never_substitutes_process_liveness": True}
    pub_path = next(OUT / p for p in ("git_delivery/CHECKPOINT_TERMINAL_STATE_OPTIMIZER_CENSUS_V11.json",
                                   "git_delivery/CHECKPOINT_ALL17_DEVELOPMENT_CLOSURE_V10.json") if (OUT / p).exists())
    pub = read_json(pub_path)
    assert pub["local_HEAD"] == pub["fresh_remote_HEAD"] and not pub["force_push"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    clean = not subprocess.check_output(["git", "-c", "core.fsmonitor=false", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    regression = read_json(next(OUT / ("tests/REGRESSION_V" + n + ".json") for n in ("19", "18") if (OUT / ("tests/REGRESSION_V" + n + ".json")).exists()))
    memory, closure = (read_json(OUT / name) for name in ("memory/frontier_v1/latest.json", "mot/development_closure_v1/latest.json"))
    assert sha256(memory["path"]) == memory["sha256"] and sha256(closure["path"]) == closure["sha256"]
    counts["actual_complete_development_groups"] = closure["actual_complete_groups"]
    terminal_path = OUT / "on_policy/JOINT_STATE_V2_DEAD_OWNER_OBSERVATION_V1.json"
    terminal = {"path": str(terminal_path), "sha256": sha256(terminal_path), "actual_exit_code_observed": False,
                "termination_cause": "UNKNOWN", "old_marker_and_partial_NOT_overwritten": True}
    resource = storage()
    value = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "application_goal_status": "ACTIVE",
        "scientific_decision": "PENDING", "counts": counts, "optimizer_evidence_census": groups, "drivers": drivers,
        "old_tensor_owner_terminal_observation": terminal, "resource": resource, "full_regression": regression,
        "latest_all17_development_closure": closure, "latest_actual_memory_frontier": memory,
        "local_HEAD": head, "last_verified_published_HEAD": pub["fresh_remote_HEAD"], "publication_receipt": str(pub_path),
        "current_working_tree_clean": clean, "published_matches_local": head == pub["fresh_remote_HEAD"],
        "all_current_working_changes_published": clean and head == pub["fresh_remote_HEAD"],
        "API_canonical_readback_claimed": pub.get("API_canonical_readback_claimed", False),
        "failed_attempt_records_NEVER_counted_as_optimized_models": True,
        "verified_fit_or_runtime_NEVER_research_PASS": True, "CONFIRM_VAL_TEST_SOT_unopened": True,
        "next_stage_authorized": False, "scope_remaining": "Full original M0-M11 task: all24 controlled-state/exact-CF/pure-margin closure, registered42 MAIN/9 state fits and all17 own-policy full-MOT groups/density, justified conditional on-policy/M-B/confirmation dispositions, six final tables, named artifacts, independent causal-root/G0-G4 scientific closure and final verified code-only delivery."}
    relative = "audit/CONTINUATION_CHECKPOINT_OPTIMIZER_STATE_V11" + suffix + ".json"
    write_json(relative, value)
    fields = {alias: counts[key] for alias, key in previous.prior.ALIASES.items()}
    fields.update(application_goal_status="ACTIVE", application_goal_active=True, scientific_decision="PENDING", next_stage_authorized=False,
        latest_continuation_checkpoint="outputs/N72R21R2/" + relative, current_actual_pipeline_counts=counts,
        actual_optimizer_evidence_census={k: {n: v for n, v in g.items() if n != "records"} for k, g in groups.items()},
        new_model_fits_completed=counts["actual_total_optimizer_model_records"],
        actual_fit_attempt_records=counts["actual_total_fit_attempt_records"],
        actual_fit_attempts_without_optimization=counts["actual_fit_attempts_without_optimization"],
        actual_unverified_fit_records=counts["actual_fit_records_unverified"],
        failed_fit_records_not_trained_models=True, old_tensor_owner_terminal_observation=terminal,
        terminal_state_recovery_driver="outputs/N72R21R2/on_policy/joint_state_terminal_resume_v3/driver.json",
        resource_snapshot=resource, local_code_commit=head, published_code_HEAD=pub["fresh_remote_HEAD"],
        published_code_commit=pub["fresh_remote_HEAD"], last_verified_published_code_commit=pub["fresh_remote_HEAD"],
        GitHub_remote_HEAD_verified=value["published_matches_local"], all_current_working_changes_published=value["all_current_working_changes_published"],
        new_checkpoint_publication="VERIFIED_NORMAL_NON_FORCE_V11_FRESH_GIT_REF" if pub_path.name.endswith("V11.json") else "VERIFIED_NORMAL_NON_FORCE_V10_FRESH_GIT_REF",
        API_canonical_readback_claimed=value["API_canonical_readback_claimed"],
        latest_all17_development_closure=closure, latest_actual_memory_frontier=memory,
        full_regression_passed=regression["passed"], full_regression_historical_failures=regression["failed"],
        latest_full_regression={"passed": regression["passed"], "failed": regression["failed"], "seconds": regression["seconds"], "all_passed": False},
        focused_test_passed=regression["focused"]["passed"], latest_focused_regression=regression["focused"])
    update_status(**fields)
    append_log("CONTINUATION_CHECKPOINT_OPTIMIZER_STATE_V11", path=relative, counts=counts)
    print({"checkpoint": relative, "counts": counts, "live_scoped_owners": sum(j.get("owner", {}).get("is_live_exact_R2_process", False) for j in drivers.values()),
           "all_current_changes_published": value["all_current_working_changes_published"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
