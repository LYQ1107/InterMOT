"""One CPU owner for finite42-model, two-point INNER / selected-point FIT.

No unowned partial adoption, no best seed, no automatic confirmation. Source
training may progress independently; a group needs all three actual weights.
"""
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256, storage, append_log
from scripts.n72r21r2_main_policy import PREFIX, PROTOCOL, CODE


def run_cell(state, marker, job, point, sequence):
    model = "MAIN__" + job["family"] + "__" + job["objective"] + "__seed" + str(job["seed"])
    uid = model + "__" + point
    key = uid + "/" + sequence
    if key in state["completed"] or any(r["key"] == key for r in state["failed_retained"]):
        return
    storage(32 << 20)
    log = ASSETS / "main_policy_v1_logs" / uid / (sequence + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        for action, directory in (("runtime", "runtime"), ("onsets", "onset_runtime"), ("evaluate", "results")):
            if (OUT / PREFIX / directory / uid / (sequence + ".json")).exists():
                raise FileExistsError("No implicit adoption of an unowned MAIN policy attempt")
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_main_policy", action,
                "--sequence", sequence, "--family", job["family"], "--objective", job["objective"],
                "--seed", str(job["seed"]), "--point", point], cwd=ROOT,
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
    print({"actual_MAIN_full_video_cells": len(state["completed"]), "failures_retained": len(state["failed_retained"])}, flush=True)


def run():
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual scoped ownership before another MAIN policy driver")
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in CODE}
    groups = list(dict.fromkeys((j["family"], j["objective"]) for j in p["jobs"]))
    state = {"stage": "N72R21R2", "goal": GOAL, "pid": os.getpid(), "status": "WAITING_ACTUAL_PILOT_ONSETS_ALL24_MAIN_INPUTS_AND_THREE_SEED_WEIGHTS",
             "protocol_sha256": sha256(PROTOCOL), "CPU_workers": 1, "active": None, "completed": [],
             "failed_retained": [], "group_status": {}, "confirmation_authorized": False}
    write_json(marker, state)
    while len(list((OUT / "training/support_ablation_v1/results").rglob("*.json"))) != 36 or \
          len(list((OUT / "training/support_onset_v1/labels").rglob("*.json"))) != 36 or \
          not (OUT / "training/ALL24_MAIN_SOURCE_MANIFEST_V1.json").exists():
        time.sleep(2)
    state["status"] = "ACTIVE_FINITE_REGISTERED_MAIN_OWN_POLICY_FULL_MOT"
    write_json(marker, state, mutable=True)
    for family, objective in groups:
        jobs = [j for j in p["jobs"] if (j["family"], j["objective"]) == (family, objective)]
        assert {j["seed"] for j in jobs} == set(p["seeds"])
        group = family + "__" + objective
        while not all((OUT / "training/event_authority" / ("MAIN__" + family + "__" + objective + "__seed" + str(j["seed"]) + ".json")).exists() for j in jobs):
            training = read_json(OUT / "training/main_fit_driver_v2.json")
            if training["status"] in ("COMPLETE_ACTUAL_UNCALIBRATED_MAIN_FITS_ONLY", "ACTUAL_FITS_AVAILABLE_CASES_FAILURES_RETAINED"):
                state["group_status"][group] = "NOT_RUN_MISSING_ACTUAL_SEED_WEIGHT_FIT_FAILURE_RETAINED"
                break
            time.sleep(2)
        if group in state["group_status"]:
            write_json(marker, state, mutable=True)
            continue
        for point in p["points"]:
            for job in jobs:
                for sequence in p["split"]["inner"]:
                    run_cell(state, marker, job, point, sequence)
        expected = [("MAIN__" + family + "__" + objective + "__seed" + str(j["seed"]) + "__" + point + "/" + s)
                    for point in p["points"] for j in jobs for s in p["split"]["inner"]]
        if not set(expected).issubset(state["completed"]):
            state["group_status"][group] = "NOT_SELECTED_ACTUAL_INNER_CELL_FAILURES_RETAINED"
            write_json(marker, state, mutable=True)
            continue
        log = ASSETS / "main_policy_v1_logs" / (group + "__INNER_selection.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_main_policy", "select", "--family", family, "--objective", objective],
                cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"key": group, "action": "select", "pid": process.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = process.wait()
        state["active"] = None
        if code:
            state["group_status"][group] = "NOT_SELECTED_ACTUAL_SELECTION_FAILURE_RETAINED"
            state["failed_retained"].append({"key": group, "action": "select", "returncode": code, "log": str(log)})
            write_json(marker, state, mutable=True)
            continue
        choice = read_json(OUT / PREFIX / "selections" / (group + ".json"))["selected_point"]
        for job in jobs:
            for sequence in p["split"]["fit"]:
                run_cell(state, marker, job, choice, sequence)
        state["group_status"][group] = "ACTUAL_INNER_SELECTION_AND_SELECTED_FIT_CELLS_ATTEMPTED_NOT_G1_G2_QUALIFIED"
        write_json(marker, state, mutable=True)
    state["status"] = "AVAILABLE_MAIN_FULL_VIDEO_DEVELOPMENT_COMPLETE_SEPARATE_CLOSURE_REQUIRED"
    write_json(marker, state, mutable=True)
    append_log("MAIN_FULL_VIDEO_POLICY_DRIVER_FINISHED", completed=len(state["completed"]), failed=state["failed_retained"], groups=state["group_status"])


if __name__ == "__main__":
    run()
