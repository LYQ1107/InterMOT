"""One CPU collector for all24 fresh current-axis datasets; no holdout open."""
import argparse
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, storage, append_log, development_sequence


def worker(sequence):
    development_sequence(sequence)
    for action, prefix in (("runtime", "availability/current_axis_v1/runtime_sequences"), ("label", "availability/current_axis_v1/supervision")):
        path = OUT / prefix / (sequence + ".json")
        if path.exists():
            read_json(path)
            continue
        result = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_open_set_data", action, "--sequence", sequence], cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError("Retain actual current-axis failure: " + action + "/" + sequence)
        assert path.exists()


def run():
    protocol = preregistration()
    sequences = protocol["split"]["fit"] + protocol["split"]["inner"]
    marker = "availability/open_set_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual prior driver ownership before another launch")
    state = {"stage": "N72R21R2", "status": "ACTIVE", "pid": os.getpid(), "source_sha256": sha256(__file__),
             "CPU_workers": 1, "sequences": sequences, "completed": [], "failed_retained": [], "active": None}
    write_json(marker, state)
    child, last = None, 0.
    while True:
        storage(256 << 20)
        if child and child["process"].poll() is not None:
            child["handle"].close()
            if child["process"].returncode:
                state["failed_retained"].append({"sequence": child["sequence"], "returncode": child["process"].returncode, "log": child["log"]})
            child = None
        completed = [s for s in sequences if (OUT / "availability/current_axis_v1/supervision" / (s + ".json")).exists()]
        failed = {r["sequence"] for r in state["failed_retained"]}
        if child is None:
            selected = next((s for s in sequences if s not in completed and s not in failed and (OUT / "mot/baseline_results" / (s + ".json")).exists()), None)
            if selected:
                log = ASSETS / "open_set_driver_v1_logs" / (selected + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("x")
                process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_open_set_driver", "--worker", selected], cwd=ROOT,
                                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                child = {"sequence": selected, "process": process, "handle": handle, "log": str(log)}
        if time.monotonic() - last > 20:
            state.update(completed=completed, active=None if child is None else {"sequence": child["sequence"], "pid": child["process"].pid, "log": child["log"]})
            write_json(marker, state, mutable=True)
            print({"fresh_current_axis_driver": "ACTIVE", "completed": len(completed), "active": state["active"], "failures": state["failed_retained"]}, flush=True)
            last = time.monotonic()
        if child is None and len(completed) + len(failed) == len(sequences):
            break
        time.sleep(2)
    state.update(status="COMPLETE" if not failed else "AVAILABLE_INPUTS_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M8_FRESH_CURRENT_AXIS_DRIVER_FINISHED", completed=len(completed), failed=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
