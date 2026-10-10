"""Recover a dead wait-only owner; preserve the registered42 training jobs.

The old marker/termination receipt are immutable, all24 prerequisites are
unchanged, and existing main attempts are never silently adopted/refitted.
"""
import os
from pathlib import Path
import subprocess
import time
from scripts import n72r21r2_main_fit_driver as original
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage, append_log

TERMINAL = OUT / "training/main_fit_driver_v1_TERMINAL_143.json"
PROTOCOL = OUT / "training/MAIN_WAIT_RECOVERY_PROTOCOL_V2.json"


def original_owner_alive(pid):
    try:
        return b"scripts.n72r21r2_main_fit_driver\x00" in Path("/proc" + "/" + str(pid) + "/cmdline").read_bytes()
    except FileNotFoundError:
        return False


def freeze():
    previous_path = OUT / "training/main_fit_driver_v1.json"
    previous = read_json(previous_path)
    terminal = read_json(TERMINAL)
    p = read_json(OUT / "training/MAIN_EXECUTION_PROTOCOL_V1.json")
    assert terminal["old_marker_sha256"] == sha256(previous_path) and terminal["unified_exec_exit_code"] == 143
    if original_owner_alive(previous["pid"]) or previous["active"] is not None or previous["completed"] or previous["failed_retained"]:
        raise ValueError("Only a verified dead wait-only driver is recoverable here")
    assert previous["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_main_fit_driver.py")
    assert p["source_sha256"] == previous["source_sha256"]
    if any((OUT / "training/event_authority" / ("MAIN__" + j["family"] + "__" + j["objective"] + "__seed" + str(j["seed"]) + ".json")).exists() for j in p["jobs"]):
        raise ValueError("No implicit adoption of any prior main fit")
    write_json("training/MAIN_WAIT_RECOVERY_PROTOCOL_V2.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "old_marker_path": str(previous_path), "old_marker_sha256": sha256(previous_path),
        "termination_receipt_sha256": sha256(TERMINAL), "old_exit_code": 143, "termination_cause": "UNKNOWN_NOT_INFERRED_FROM_RETURN_CODE",
        "unchanged_main_job_protocol_sha256": sha256(OUT / "training/MAIN_EXECUTION_PROTOCOL_V1.json"),
        "jobs": p["jobs"], "all24_prerequisites": list(original.PREREQUISITES), "CPU_workers": 1,
        "source_sha256": {name: sha256(ROOT / name) for name in ("scripts/n72r21r2_main_fit_driver_v2.py", "scripts/n72r21r2_main_fit_driver.py", "scripts/n72r21r2_train_event_authority.py")},
        "no_subset_training_or_authority_or_confirmation_permission": True,
        "previous_attempts_logs_and_mutable_marker_NOT_overwritten": True})


def run():
    marker = "training/main_fit_driver_v2.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect recovery ownership before another driver")
    freeze()
    p = read_json(PROTOCOL)
    required = preregistration()["split"]["fit"] + preregistration()["split"]["inner"]
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "WAITING_COMPLETE_ALL24_UNCHANGED_MAIN_PREREQUISITES",
             "protocol_sha256": sha256(PROTOCOL), "active": None, "completed": [], "failed_retained": [], "CPU_workers": 1}
    write_json(marker, state)
    previous_ready = None
    while not original.all_required_ready(required):
        ready = [s for s in required if original.all_required_ready([s])]
        if ready != previous_ready:
            state["ready_videos_all_prerequisites"] = ready
            write_json(marker, state, mutable=True)
            print({"recovered_main_fit_driver": "WAITING_ALL24", "ready": len(ready), "required": 24}, flush=True)
            previous_ready = ready
        time.sleep(2)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in p["source_sha256"]}
    source = original.verify_prerequisites(required)
    write_json("training/ALL24_MAIN_SOURCE_MANIFEST_V1.json", {
        "stage": "N72R21R2", "status": "ALL24_ACTUAL_MAIN_PREREQUISITES_VERIFIED_BEFORE_FIRST_OPTIMIZER_STEP",
        "protocol_sha256": p["unchanged_main_job_protocol_sha256"], "receipts": source,
        "no_ready_subset_or_old_scene_substitution": True})
    state["status"] = "ACTIVE_UNCHANGED_REGISTERED42_MAIN_FITS"
    for job in p["jobs"]:
        family, objective, seed = job["family"], job["objective"], job["seed"]
        uid = "MAIN__" + family + "__" + objective + "__seed" + str(seed)
        destination = OUT / "training/event_authority" / (uid + ".json")
        if destination.exists():
            raise FileExistsError("Do not silently overwrite/adopt a previous main fit")
        storage(32 << 20)
        log = ASSETS / "main_fit_driver_v2_logs" / (uid + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_train_event_authority", "--family", family, "--objective", objective, "--seed", str(seed)], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"experiment_uid": uid, "pid": process.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = process.wait()
        if code:
            state["failed_retained"].append({"experiment_uid": uid, "returncode": code, "log": str(log)})
        else:
            assert destination.exists()
            state["completed"].append(uid)
        state["active"] = None
        write_json(marker, state, mutable=True)
        print({"actual_main_fits_complete": len(state["completed"]), "failed_retained": len(state["failed_retained"]), "not_MOT_qualification": True}, flush=True)
    state.update(status="COMPLETE_ACTUAL_UNCALIBRATED_MAIN_FITS_ONLY" if not state["failed_retained"] else "ACTUAL_FITS_AVAILABLE_CASES_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M5_M6_RECOVERED_UNCHANGED42_MAIN_FIT_DRIVER_FINISHED", completed=len(state["completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
