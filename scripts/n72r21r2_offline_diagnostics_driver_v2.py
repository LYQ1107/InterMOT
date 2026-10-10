"""Resume a verified dead diagnostic owner without changing scientific jobs.

The V1 marker, successful operation receipts and failed attempts are retained.
Only missing, dependency-ready operations are launched under a new log namespace.
"""
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage, append_log

OLD = "events/offline_diagnostics_driver_v1.json"
TERMINAL = "events/offline_diagnostics_driver_v1_TERMINAL_143.json"
PROTOCOL = "events/DIAGNOSTICS_WAIT_RECOVERY_PROTOCOL_V2.json"
MARKER = "events/offline_diagnostics_driver_v2.json"
OPERATIONS = (
    ("events/trajectory_utility_audit", "events/label_audit", "scripts.n72r21r2_trajectory_utility", "label"),
    ("simple/onset_runtime", "simple/runtime_sequences", "scripts.n72r21r2_simple_onset_audit", "runtime"),
    ("simple/onset_audit", "simple/onset_runtime", "scripts.n72r21r2_simple_onset_audit", "label"),
)
CODE = ("scripts/n72r21r2_offline_diagnostics_driver_v2.py", "scripts/n72r21r2_offline_diagnostics_driver.py",
        "scripts/n72r21r2_trajectory_utility.py", "scripts/n72r21r2_simple_onset_audit.py")


def live_scoped_pid(pid):
    try:
        args = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return any(a.startswith(b"scripts.n72r21r2_") for a in args)


def next_task(sequences, completed, failed, dependency_ready):
    return next(((prefix, sequence, module, action) for sequence in sequences
                 for prefix, dependency, module, action in OPERATIONS
                 if prefix + "/" + sequence not in completed and prefix + "/" + sequence not in failed
                 and dependency_ready(dependency, sequence)), None)


def freeze():
    old, terminal = read_json(OUT / OLD), read_json(OUT / TERMINAL)
    sequences = preregistration()["split"]["fit"] + preregistration()["split"]["inner"]
    assert terminal["unified_exec_exit_code"] == 143 and terminal["old_marker_sha256"] == sha256(OUT / OLD)
    assert old["sequences"] == sequences and old["source_sha256"] == sha256(ROOT / CODE[1])
    if live_scoped_pid(old["pid"]) or (old["active"] and live_scoped_pid(old["active"]["pid"])):
        raise ValueError("Never overlap a live previous diagnostic owner/child")
    existing = [{"path": str(path), "sha256": sha256(path)} for prefix, _, _, _ in OPERATIONS
                for sequence in sequences if (path := OUT / prefix / (sequence + ".json")).exists()]
    write_json(PROTOCOL, {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "old_marker_sha256": sha256(OUT / OLD), "termination_receipt_sha256": sha256(OUT / TERMINAL),
        "termination_cause": "UNKNOWN_NOT_INFERRED_FROM_RETURN_CODE", "old_exit_code": 143,
        "unchanged_all24_sequences": sequences, "unchanged_operations": [list(op) for op in OPERATIONS],
        "existing_successful_operation_receipts_not_reexecuted": existing,
        "old_failed_attempts_retained_not_silently_retried": old["failed_retained"],
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "CPU_workers": 1,
        "new_log_namespace": "offline_diagnostics_driver_v2_logs",
        "no_scientific_gate_selection_or_split_change": True, "old_marker_and_logs_NOT_overwritten": True})


def run():
    if (OUT / MARKER).exists():
        raise FileExistsError("Inspect actual V2 diagnostic ownership before another launch")
    freeze()
    p = read_json(OUT / PROTOCOL)
    sequences = p["unchanged_all24_sequences"]
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE_UNCHANGED_DIAGNOSTIC_OPERATIONS",
             "CPU_workers": 1, "protocol_sha256": sha256(OUT / PROTOCOL), "active": None,
             "completed": [], "failed_retained": list(p["old_failed_attempts_retained_not_silently_retried"])}
    write_json(MARKER, state)
    last = None
    while True:
        assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
        storage(256 << 20)
        done = [prefix + "/" + sequence for prefix, _, _, _ in OPERATIONS for sequence in sequences
                if (OUT / prefix / (sequence + ".json")).exists()]
        failed = {r["key"] for r in state["failed_retained"]}
        task = next_task(sequences, set(done), failed, lambda dep, seq: (OUT / dep / (seq + ".json")).exists())
        state["completed"] = done
        if done != last:
            write_json(MARKER, state, mutable=True)
            print({"offline_diagnostics_V2_complete_operations": len(done), "required": 72,
                   "old_exit143_retained": True}, flush=True)
            last = done.copy()
        if task is not None:
            prefix, sequence, module, action = task
            key = prefix + "/" + sequence
            log = ASSETS / "offline_diagnostics_driver_v2_logs" / (prefix.replace("/", "__") + "__" + sequence + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x") as handle:
                process = subprocess.Popen([str(PYTHON), "-u", "-m", module, action, "--sequence", sequence], cwd=ROOT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                state["active"] = {"key": key, "pid": process.pid, "log": str(log)}
                write_json(MARKER, state, mutable=True)
                code = process.wait()
            if code:
                state["failed_retained"].append({"key": key, "returncode": code, "log": str(log)})
            elif not (OUT / prefix / (sequence + ".json")).exists():
                raise RuntimeError("A successful process must produce its sealed operation receipt")
            state["active"] = None
            write_json(MARKER, state, mutable=True)
        elif len(set(done) | failed) == len(sequences) * len(OPERATIONS):
            break
        else:
            time.sleep(2)
    state.update(status="COMPLETE_ALL24_UNCHANGED_DIAGNOSTICS" if not failed else "AVAILABLE_OPERATIONS_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(MARKER, state, mutable=True)
    append_log("M3_M4_RECOVERED_DIAGNOSTICS_FINISHED", complete=len(done), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
