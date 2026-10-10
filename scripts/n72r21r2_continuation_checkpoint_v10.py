"""Actual all-group closure/risk census and V10-only delivery readback state."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scripts import n72r21r2_continuation_checkpoint_v9 as prior
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, storage, utcnow, update_status, append_log


def owned(pid):
    value = prior.base.proc(pid)
    try:
        value["is_live_exact_R2_process"] &= Path("/proc/" + str(pid) + "/cwd").resolve(strict=True) == ROOT.resolve()
    except (FileNotFoundError, PermissionError):
        value["is_live_exact_R2_process"] = False
    if not value["is_live_exact_R2_process"]:
        value["exact_argv_if_owned"] = None
    return value


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Narrow immutable checkpoint suffix required")
    counts = {k: len(list((OUT / v).rglob("*.json"))) for k, v in prior.FOLDERS.items()}
    counts.update(actual_main_fits=len(list((OUT / "training/event_authority").glob("MAIN__*.json"))),
        actual_state_source_fits=len(list((OUT / "on_policy/state_source_fits_v1").glob("*STATE__seed*.json"))),
        actual_pilot_fits=len(list((OUT / "training/event_authority").glob("PILOT__*.json"))),
        actual_current_axis_fits=len(list((OUT / "availability/current_axis_fits").glob("*__seed*.json"))))
    counts["actual_total_optimizer_model_records"] = sum(counts[k] for k in (
        "actual_main_fits", "actual_state_source_fits", "actual_pilot_fits", "actual_current_axis_fits", "fresh_memory_fits"))
    drivers = {}
    for rel in prior.MARKERS:
        path = OUT / rel
        if not path.exists():
            drivers[rel] = {"status": "MARKER_ABSENT_NOT_A_LIVE_JOB"}; continue
        raw = path.read_bytes(); j = json.loads(raw)
        active = j.get("active"); entries = active if isinstance(active, list) else [active] if active else []
        drivers[rel] = {"status": j["status"], "marker_sha256": hashlib.sha256(raw).hexdigest(),
            "owner": owned(j.get("pid")), "active": active, "active_procs": [owned(e.get("pid")) for e in entries],
            "completed_count": len(j.get("completed", [])), "failed_retained": j.get("failed_retained", [])}
    pub_path = next(OUT / n for n in ("git_delivery/CHECKPOINT_ALL17_DEVELOPMENT_CLOSURE_V10.json",
        "git_delivery/CHECKPOINT_STATE_DENSITY_SCENES_V9.json") if (OUT / n).exists())
    pub = read_json(pub_path)
    assert pub["local_HEAD"] == pub["fresh_remote_HEAD"] and not pub["force_push"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    clean = not subprocess.check_output(["git", "-c", "core.fsmonitor=false", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    regression = read_json(next(OUT / ("tests/REGRESSION_V" + n + ".json") for n in ("18", "17") if (OUT / ("tests/REGRESSION_V" + n + ".json")).exists()))
    memory = read_json(OUT / "memory/frontier_v1/latest.json")
    closure = read_json(OUT / "mot/development_closure_v1/latest.json")
    assert sha256(memory["path"]) == memory["sha256"] and sha256(closure["path"]) == closure["sha256"]
    counts["actual_complete_development_groups"] = closure["actual_complete_groups"]
    resource = storage()
    value = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "application_goal_status": "ACTIVE",
        "scientific_decision": "PENDING", "counts": counts, "drivers": drivers, "resource": resource,
        "full_regression": regression, "latest_all17_development_closure": closure, "latest_actual_memory_frontier": memory,
        "local_HEAD": head, "last_verified_published_HEAD": pub["fresh_remote_HEAD"], "publication_receipt": str(pub_path),
        "published_matches_local": head == pub["fresh_remote_HEAD"], "current_working_tree_clean": clean,
        "all_current_working_changes_published": clean and head == pub["fresh_remote_HEAD"],
        "API_canonical_readback_claimed": pub.get("API_canonical_readback_claimed", False),
        "CONFIRM_VAL_TEST_SOT_unopened": True, "next_stage_authorized": False,
        "scope_remaining": "All24 original CF/controlled-state closure; registered42 MAIN and9 state-source own full-MOT/all17-group summaries/density; actual qualified on-policy/staged/M-B branches or properly measured conditional NOT_RUN; all24 learned-risk and open-set missed repairs; final six tables/named deliverables/causal-root/G0-G4/full-task scientific closure."}
    rel = "audit/CONTINUATION_CHECKPOINT_ALL17_V10" + suffix + ".json"
    write_json(rel, value)
    fields = {alias: counts[key] for alias, key in prior.ALIASES.items()}
    fields.update(application_goal_status="ACTIVE", application_goal_active=True, scientific_decision="PENDING", next_stage_authorized=False,
        latest_continuation_checkpoint="outputs/N72R21R2/" + rel, current_actual_pipeline_counts=counts,
        local_code_commit=head, published_code_HEAD=pub["fresh_remote_HEAD"], published_code_commit=pub["fresh_remote_HEAD"],
        last_verified_published_code_commit=pub["fresh_remote_HEAD"], GitHub_remote_HEAD_verified=value["published_matches_local"],
        all_current_working_changes_published=value["all_current_working_changes_published"],
        new_checkpoint_publication="VERIFIED_NORMAL_NON_FORCE_V10_FRESH_GIT_REF" if pub_path.name.endswith("V10.json") else "VERIFIED_NORMAL_NON_FORCE_V9_FRESH_GIT_REF",
        API_canonical_readback_claimed=value["API_canonical_readback_claimed"], resource_snapshot=resource,
        new_model_fits_completed=counts["actual_total_optimizer_model_records"], actual_complete_development_groups=counts["actual_complete_development_groups"],
        latest_all17_development_closure=closure, latest_actual_memory_frontier=memory,
        full_regression_passed=regression["passed"], full_regression_historical_failures=regression["failed"],
        latest_full_regression={"passed": regression["passed"], "failed": regression["failed"], "seconds": regression["seconds"], "all_passed": False},
        focused_test_passed=regression["focused"]["passed"], latest_focused_regression=regression["focused"])
    update_status(**fields)
    append_log("CONTINUATION_CHECKPOINT_ALL17_V10", path=rel, counts=counts)
    print({"checkpoint": rel, "counts": counts, "live_scoped_owners": sum(j.get("owner", {}).get("is_live_exact_R2_process", False) for j in drivers.values()),
           "all_current_changes_published": value["all_current_working_changes_published"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--suffix", required=True)
    run(parser.parse_args().suffix)
