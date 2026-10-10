"""One bounded CPU policy worker, all fresh registered FIT/INNER inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, append_log, storage
from scripts.n72r21r2_input_driver import alive_owned_worker


def worker(sequence):
    for action, prefix in (("runtime", "simple/runtime_sequences"), ("evaluate", "simple/results")):
        if not (OUT / prefix / (sequence + ".json")).exists():
            subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_simple", action, "--sequence", sequence], cwd=ROOT, check=True)


def run(adopt_pid):
    protocol = preregistration()
    sequences = protocol["split"]["fit"] + protocol["split"]["inner"]
    marker = "simple/simple_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Preserve previous driver ownership and evidence")
    state = {"stage": "N72R21R2", "status": "ACTIVE", "pid": os.getpid(), "source_sha256": sha256(Path(__file__)),
             "sequences": sequences, "max_CPU_workers": 1, "adopt_existing_runtime_pid": adopt_pid,
             "completed": [], "failed_retained": [], "active": None, "same_frozen_inputs_full_global_runtime": True}
    write_json(marker, state)
    current, last = None, 0.
    while True:
        storage(512 << 20)
        if current is not None and current["process"].poll() is not None:
            current["handle"].close()
            if current["process"].returncode:
                state["failed_retained"].append({"sequence": current["sequence"], "returncode": current["process"].returncode, "log_path": current["log"]})
            current = None
        state["completed"] = [s for s in sequences if (OUT / "simple/results" / (s + ".json")).exists()]
        failed = {r["sequence"] for r in state["failed_retained"]}
        existing_alive = adopt_pid is not None and alive_owned_worker(adopt_pid, "scripts.n72r21r2_simple runtime")
        if current is None and not existing_alive:
            ready = [s for s in sequences if s not in state["completed"] and s not in failed and (OUT / "mot/baseline_results" / (s + ".json")).exists()]
            if ready:
                sequence = ready[0]
                log = ASSETS / "simple_driver_v1_logs" / (sequence + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("x")
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_simple_driver", "--worker", sequence], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                current = {"sequence": sequence, "process": process, "handle": handle, "log": str(log)}
        if time.monotonic() - last >= 20:
            state["active"] = {"adopted_pid": adopt_pid} if existing_alive else None if current is None else {"sequence": current["sequence"], "pid": current["process"].pid, "log": current["log"]}
            write_json(marker, state, mutable=True)
            print(json.dumps({"simple_driver": "ACTIVE", "complete": len(state["completed"]), "active": state["active"], "failures": state["failed_retained"]}), flush=True)
            last = time.monotonic()
        if current is None and not existing_alive and len(state["completed"]) + len(failed) == len(sequences):
            break
        time.sleep(2)
    state.update(status="COMPLETE" if not failed else "AVAILABLE_SIMPLE_RESULTS_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M4_FIXED_SIMPLE_DRIVER_FINISHED", completed=len(state["completed"]), failures=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--adopt-running-pid", type=int)
    args = parser.parse_args()
    worker(args.worker) if args.worker else run(args.adopt_running_pid)
