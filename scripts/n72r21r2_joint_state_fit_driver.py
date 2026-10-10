"""Wait for ALL24 controlled-state sources, then the fixed9 paired fits."""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage, append_log


def all_sources_ready(sequences):
    return all((OUT / "on_policy/joint_state_v1/supervision" / (s + ".json")).exists() for s in sequences)


def run():
    marker = "on_policy/state_source_fits_v1/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect existing state-source fit ownership")
    p = read_json(OUT / "on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json")
    split = preregistration()["split"]
    required = split["fit"] + split["inner"]
    state = {"stage": "N72R21R2", "status": "WAITING_ALL24_CONTROLLED_STATE_SOURCES", "pid": os.getpid(),
             "source_sha256": sha256(__file__), "protocol_sha256": sha256(OUT / "on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json"),
             "completed": [], "failed_retained": [], "active": None, "CPU_workers": 1}
    write_json(marker, state)
    last = 0.
    while not all_sources_ready(required):
        if time.monotonic() - last > 20:
            state["ready_videos"] = sum((OUT / "on_policy/joint_state_v1/supervision" / (s + ".json")).exists() for s in required)
            write_json(marker, state, mutable=True)
            print({"state_source_fit_driver": state["status"], "ready": state["ready_videos"], "required": 24}, flush=True)
            last = time.monotonic()
        time.sleep(2)
    state["status"] = "ACTIVE_FIT_ONLY_PAIRED_STATE_SOURCE_CONTRAST"
    for mode in p["modes"]:
        for seed in p["seeds"]:
            uid = mode + "__seed" + str(seed)
            destination = OUT / "on_policy/state_source_fits_v1" / (uid + ".json")
            if destination.exists():
                raise FileExistsError("No implicit state-source refit or receipt substitution")
            storage(32 << 20)
            log = ASSETS / "state_source_fit_driver_v1_logs" / (uid + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x") as handle:
                process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_train_joint_state", "--mode", mode, "--seed", str(seed)], cwd=ROOT,
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
    state.update(status="COMPLETE_REGISTERED_STATE_SOURCE_ATTEMPTS_NOT_ON_POLICY_CLOSURE" if not state["failed_retained"] else "ACTUAL_AVAILABLE_FITS_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M7_PAIRED_STATE_SOURCE_FIT_DRIVER_FINISHED", completed=len(state["completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
