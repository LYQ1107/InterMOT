"""All17 dispositions without changing frozen V1 metrics, source or gates.

The original STATE driver keys group_status by mode, not mode+objective.
An all3-seed measured no-fit is a terminal experiment disposition, NEVER a
completed full-MOT group, fabricated zero metrics or general baseline failure.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from scripts.n72r21r2_common import OUT, GOAL, read_json, write_json, sha256, utcnow, update_status
from scripts.n72r21r2_state_fit_storage_resume_v3 import valid_no_fit


def status_key(study, group):
    return group.split("__", 1)[0] if study == "STATE" else group


def enrich(snapshot, state_driver, failures, seeds, fit_protocol_sha):
    result = deepcopy(snapshot["all17_group_census"])
    if len(result) != 17:
        raise ValueError("Original full17 census is mandatory")
    key = "STATE/BASELINE_STATE__L3_HARM_AWARE"
    mode = status_key("STATE", key.split("/", 1)[1])
    reason = state_driver["group_status"].get(mode)
    if reason != "NOT_RUN_ACTUAL_SUPERVISION_FAILURE_NOT_A_WEIGHT":
        raise ValueError("Actual canonical no-fit policy disposition required")
    if len(failures) != len(seeds) or {r["seed"] for r in failures} != set(seeds):
        raise ValueError("All original seed dispositions required")
    if not all(valid_no_fit(r, r["seed"], fit_protocol_sha) for r in failures):
        raise ValueError("Only actual measured zero-benefit no-optimizer failures")
    if result[key]["status"].startswith("COMPLETE_ACTUAL"):
        raise ValueError("Untrained baseline cannot have a complete MOT effect")
    result[key] = {"status": "TERMINAL_MEASURED_ALL3_SUPERVISION_PREFIT_FAILURE_NOT_RUN_FULL_MOT",
        "original_V1_classification": result[key], "actual_driver_mode_key": mode,
        "actual_driver_mode_status": reason, "actual_optimizer_steps": 0,
        "actual_models": 0, "actual_full_MOT_cells": 0, "MOT_metrics": None,
        "specific_probe_support_failure_NOT_general_baseline_training_failure": True,
        "counts_as_complete_full_MOT_group": False, "scientific_success": False}
    return result


def run():
    latest = read_json(OUT / "mot/development_closure_v1/latest.json")
    assert sha256(latest["path"]) == latest["sha256"]
    snapshot = read_json(latest["path"])
    assert snapshot["goal"] == GOAL and snapshot["registered_groups"] == 17
    fit_protocol = OUT / "on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json"
    p = read_json(fit_protocol)
    state_path = OUT / "mot/state_policy_v1/driver.json"
    state = read_json(state_path)
    failures, refs = [], {latest["path"]: latest["sha256"], str(state_path): sha256(state_path), str(fit_protocol): sha256(fit_protocol)}
    for seed in p["seeds"]:
        path = OUT / "on_policy/state_source_fits_v1" / ("BASELINE_STATE__seed" + str(seed) + ".json")
        failures.append(read_json(path)); refs[str(path)] = sha256(path)
    groups = enrich(snapshot, state, failures, p["seeds"], sha256(fit_protocol))
    complete = sum(v["status"].startswith("COMPLETE_ACTUAL") for v in groups.values())
    assert complete == snapshot["actual_complete_groups"]
    body = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
        "status": "PARTIAL_ALL17_REGISTERED_GROUP_DISPOSITIONS_NOT_SCIENTIFIC_CLOSURE",
        "all17_group_census": groups, "actual_complete_full_MOT_groups": complete,
        "measured_no_fit_terminal_groups": 1, "remaining_group_dispositions": 17-complete-1,
        "original_frozen_V1_metrics_gates_sources_unchanged": True,
        "source_sha256": sha256(__file__), "evidence_sha256": refs,
        "scientific_goal_complete": False, "confirmation_authorized": False}
    path = write_json("mot/development_dispositions_v2/" + sha256(latest["path"]) + ".json", body)
    reference = {"path": str(path), "sha256": sha256(path)}
    write_json("mot/development_dispositions_v2/latest.json", reference, mutable=True)
    update_status(latest_all17_actual_group_dispositions=reference,
                  actual_complete_full_MOT_groups=complete, actual_terminal_no_fit_groups=1)
    print({"full_MOT_groups": complete, "terminal_no_fit_groups": 1, "remaining": 17-complete-1}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    run()
