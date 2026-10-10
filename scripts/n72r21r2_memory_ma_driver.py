"""One incremental CPU worker for all frozen fresh M-A memory controls."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, storage, append_log, development_sequence


def worker(sequence):
    development_sequence(sequence)
    for action, prefix in (("runtime", "memory/M_A/runtime_sequences"), ("evaluate", "memory/M_A/results")):
        destination = OUT / prefix / (sequence + ".json")
        if destination.exists():
            read_json(destination)
            continue
        process = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_zero_authority_memory", action, "--sequence", sequence], cwd=ROOT, check=False)
        if process.returncode:
            raise RuntimeError("Retain failed M-A operation: " + action + "/" + sequence)
        assert destination.exists()


def run():
    protocol = preregistration()
    sequences = protocol["split"]["fit"] + protocol["split"]["inner"]
    marker = "memory/memory_ma_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect existing M-A driver ownership before another launch")
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE", "sequences": sequences,
             "source_sha256": sha256(Path(__file__)), "CPU_workers": 1, "active": None, "completed": [], "failed_retained": []}
    write_json(marker, state)
    child, last = None, 0.
    while True:
        storage(512 << 20)
        if child and child["process"].poll() is not None:
            child["handle"].close()
            if child["process"].returncode:
                state["failed_retained"].append({"sequence": child["sequence"], "returncode": child["process"].returncode, "log": child["log"]})
            child = None
        completed = [s for s in sequences if (OUT / "memory/M_A/results" / (s + ".json")).exists()]
        failed = {r["sequence"] for r in state["failed_retained"]}
        if child is None:
            selected = next((s for s in sequences if s not in completed and s not in failed and (OUT / "mot/baseline_results" / (s + ".json")).exists()), None)
            if selected:
                log = ASSETS / "memory_ma_driver_v1_logs" / (selected + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("x")
                process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_memory_ma_driver", "--worker", selected], cwd=ROOT,
                                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                child = {"sequence": selected, "process": process, "handle": handle, "log": str(log)}
        if time.monotonic() - last > 20:
            state.update(completed=completed, active=None if child is None else {"sequence": child["sequence"], "pid": child["process"].pid, "log": child["log"]})
            write_json(marker, state, mutable=True)
            print({"M_A_driver": "ACTIVE", "complete": len(completed), "active": state["active"], "failures": state["failed_retained"]}, flush=True)
            last = time.monotonic()
        if child is None and len(completed) + len(failed) == len(sequences):
            break
        time.sleep(2)
    state.update(status="COMPLETE" if not failed else "AVAILABLE_INPUTS_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M9_ZERO_AUTHORITY_MEMORY_DRIVER_FINISHED", complete=len(completed), failed=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
