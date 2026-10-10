"""One bounded CPU worker for ready metric labels and own-policy onset audits.

Only already-sealed FIT/INNER runtime records become eligible. A failed
operation is retained and is not silently retried or used to stop other work.
"""
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, storage, append_log


def run():
    sequences = preregistration()["split"]["fit"] + preregistration()["split"]["inner"]
    marker = "events/offline_diagnostics_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect existing diagnostic driver before launching another")
    operations = (
        ("events/trajectory_utility_audit", "events/label_audit", "scripts.n72r21r2_trajectory_utility", "label"),
        ("simple/onset_runtime", "simple/runtime_sequences", "scripts.n72r21r2_simple_onset_audit", "runtime"),
        ("simple/onset_audit", "simple/onset_runtime", "scripts.n72r21r2_simple_onset_audit", "label"),
    )
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE", "CPU_workers": 1,
             "sequences": sequences, "source_sha256": sha256(Path(__file__)), "failed_retained": [], "active": None}
    write_json(marker, state)
    child = None
    last = 0.
    while True:
        storage(256 << 20)
        if child and child["process"].poll() is not None:
            child["handle"].close()
            if child["process"].returncode:
                state["failed_retained"].append({"key": child["key"], "returncode": child["process"].returncode, "log": child["log"]})
            child = None
        failed = {r["key"] for r in state["failed_retained"]}
        done = [prefix + "/" + sequence for prefix, _, _, _ in operations for sequence in sequences if (OUT / prefix / (sequence + ".json")).exists()]
        if child is None:
            task = next(((prefix, sequence, module, action) for sequence in sequences for prefix, dependency, module, action in operations
                         if prefix + "/" + sequence not in done and prefix + "/" + sequence not in failed
                         and (OUT / dependency / (sequence + ".json")).exists()), None)
            if task:
                prefix, sequence, module, action = task
                key = prefix + "/" + sequence
                path = ASSETS / "offline_diagnostics_driver_v1_logs" / (prefix.replace("/", "__") + "__" + sequence + ".log")
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("x")
                process = subprocess.Popen([str(PYTHON), "-u", "-m", module, action, "--sequence", sequence], cwd=ROOT,
                                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                child = {"process": process, "handle": handle, "key": key, "log": str(path)}
        if time.monotonic() - last > 20:
            state.update(completed=done, active=None if child is None else {"key": child["key"], "pid": child["process"].pid, "log": child["log"]})
            write_json(marker, state, mutable=True)
            print({"offline_diagnostic_driver": "ACTIVE", "complete_operations": len(done), "active": state["active"], "failures": state["failed_retained"]}, flush=True)
            last = time.monotonic()
        if child is None and len(done) + len(failed) == len(sequences) * len(operations):
            break
        time.sleep(2)
    state.update(status="COMPLETE" if not failed else "AVAILABLE_OPERATIONS_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M3_M4_OFFLINE_DIAGNOSTIC_DRIVER_FINISHED", complete=len(done), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
