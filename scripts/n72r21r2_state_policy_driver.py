"""One CPU owner: finite9 matched-state models, identical MAIN evaluation."""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, PYTHON, read_json, write_json, sha256, storage, append_log
from scripts.n72r21r2_state_policy import PROTOCOL, PREFIX, CODE, OBJECTIVE


def run_cell(state, marker, mode, seed, point, sequence):
    uid = "STATE__" + mode + "__" + OBJECTIVE + "__seed" + str(seed) + "__" + point
    key = uid + "/" + sequence
    if key in state["completed"] or any(r["key"] == key for r in state["failed_retained"]):
        return
    storage(32 << 20)
    log = ASSETS / "state_policy_v1_logs" / uid / (sequence + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        for action, directory in (("runtime", "runtime"), ("onsets", "onset_runtime"), ("evaluate", "results")):
            if (OUT / PREFIX / directory / uid / (sequence + ".json")).exists():
                raise FileExistsError("Preserve unowned state-policy partial and completed evidence")
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_state_policy", action,
                "--mode", mode, "--seed", str(seed), "--point", point, "--sequence", sequence], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"key": key, "action": action, "pid": process.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = process.wait()
            if code:
                state["failed_retained"].append({"key": key, "action": action, "returncode": code, "log": str(log)})
                break
        else:
            state["completed"].append(key)
    state["active"] = None
    write_json(marker, state, mutable=True)
    print({"actual_matched_state_full_video_cells": len(state["completed"]), "failed_retained": len(state["failed_retained"])}, flush=True)


def run():
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual state-policy owner before duplicate launch")
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    state = {"stage": "N72R21R2", "goal": GOAL, "pid": os.getpid(), "CPU_workers": 1,
        "status": "WAITING_ALL24_CONTROLLED_SOURCE_AND_ALL3_REGISTERED_STATE_WEIGHTS",
        "protocol_sha256": sha256(PROTOCOL), "completed": [], "failed_retained": [], "group_status": {},
        "active": None, "actual_model_generated_on_policy_training": False, "confirmation_authorized": False}
    write_json(marker, state)
    while not all((OUT / "on_policy/joint_state_v1/supervision" / (s + ".json")).exists() for s in p["split"]["fit"] + p["split"]["inner"]):
        time.sleep(2)
    for mode in p["modes"]:
        paths = [OUT / "on_policy/state_source_fits_v1" / (mode + "__seed" + str(seed) + ".json") for seed in p["seeds"]]
        while not all(path.exists() for path in paths):
            fitting = read_json(OUT / "on_policy/state_source_fits_v1/driver.json")
            if fitting["status"] in ("COMPLETE_REGISTERED_STATE_SOURCE_ATTEMPTS_NOT_ON_POLICY_CLOSURE", "ACTUAL_AVAILABLE_FITS_FAILURES_RETAINED"):
                state["group_status"][mode] = "NOT_RUN_MISSING_ACTUAL_WEIGHT_FAILED_FIT_RETAINED"
                break
            time.sleep(2)
        if mode in state["group_status"]:
            write_json(marker, state, mutable=True)
            continue
        fits = [read_json(path) for path in paths]
        if any(fit["status"] != "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE" for fit in fits):
            state["group_status"][mode] = "NOT_RUN_ACTUAL_SUPERVISION_FAILURE_NOT_A_WEIGHT"
            write_json(marker, state, mutable=True)
            continue
        state["status"] = "ACTIVE_FINITE9_MATCHED_STATE_OWN_FULL_MOT"
        for point in p["points"]:
            for seed in p["seeds"]:
                for sequence in p["split"]["inner"]:
                    run_cell(state, marker, mode, seed, point, sequence)
        expected = [("STATE__" + mode + "__" + OBJECTIVE + "__seed" + str(seed) + "__" + point + "/" + s)
                    for point in p["points"] for seed in p["seeds"] for s in p["split"]["inner"]]
        if not set(expected).issubset(state["completed"]):
            state["group_status"][mode] = "NOT_SELECTED_ACTUAL_INNER_FAILURES_RETAINED"
            write_json(marker, state, mutable=True)
            continue
        log = ASSETS / "state_policy_v1_logs" / (mode + "__INNER_selection.log")
        with log.open("x") as handle:
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_state_policy", "select", "--mode", mode],
                cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"key": mode, "action": "select", "pid": process.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = process.wait()
        state["active"] = None
        if code:
            state["group_status"][mode] = "NOT_SELECTED_ACTUAL_SELECTION_FAILURE_RETAINED"
            state["failed_retained"].append({"key": mode, "action": "select", "returncode": code, "log": str(log)})
            write_json(marker, state, mutable=True)
            continue
        selected = read_json(OUT / PREFIX / "selections" / (mode + "__" + OBJECTIVE + ".json"))["selected_point"]
        for seed in p["seeds"]:
            for sequence in p["split"]["fit"]:
                run_cell(state, marker, mode, seed, selected, sequence)
        state["group_status"][mode] = "ACTUAL_CONTROLLED_STATE_OWN_MOT_ATTEMPTS_NOT_ON_POLICY_OR_G1_G2_CLOSURE"
        write_json(marker, state, mutable=True)
    state["status"] = "AVAILABLE_STATE_SOURCE_FULL_MOT_FINISHED_SEPARATE_ON_POLICY_AND_GATE_CLOSURE_REQUIRED"
    write_json(marker, state, mutable=True)
    append_log("CONTROLLED_STATE_FULL_MOT_DRIVER_FINISHED", completed=len(state["completed"]), failed=state["failed_retained"], groups=state["group_status"])


if __name__ == "__main__":
    run()
