"""Single bounded CPU owner of the36 frozen FIT support-ablation pilot cells."""
import os
import subprocess
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, storage, append_log
from scripts.n72r21r2_support_ablation_pilot import PROTOCOL, PREFIX, inputs


def run():
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect support-ablation owner/partials before another driver")
    p = read_json(PROTOCOL)
    jobs = [(seed, policy, s) for seed in p["seeds"] for policy in p["policies"] for s in p["sequences"]]
    assert len(jobs) == 36
    state = {"stage": "N72R21R2", "status": "ACTIVE36_FIT_SUPPORT_ABLATION_NOT_QUALIFIED",
             "pid": os.getpid(), "protocol_sha256": sha256(PROTOCOL), "CPU_workers": 1,
             "active": None, "completed": [], "failed_retained": []}
    write_json(marker, state)
    for seed, policy, s in jobs:
        _, uid, _ = inputs(s, seed, policy)
        key = uid + "/" + s
        log = ASSETS / "support_ablation_v1_logs" / uid / (s + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        storage(128 << 20)
        with log.open("x") as handle:
            for action, directory in (("runtime", "runtime"), ("evaluate", "results")):
                if (OUT / PREFIX / directory / uid / (s + ".json")).exists():
                    raise FileExistsError("No silent adoption/refit of prior support-ablation attempt")
                process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_support_ablation_pilot", action,
                    "--sequence", s, "--seed", str(seed), "--policy", policy], cwd=ROOT,
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
        print({"actual_support_ablation_cells": len(state["completed"]), "required": 36,
               "failed_retained": len(state["failed_retained"])}, flush=True)
    state["status"] = "COMPLETE36_ACTUAL_FIT_DIAGNOSTIC_NOT_QUALIFIED" if not state["failed_retained"] else "AVAILABLE_CASES_COMPLETE_FAILURES_RETAINED"
    write_json(marker, state, mutable=True)
    append_log("FROZEN_HAND_FILTER_SUPPORT_ABLATION_PILOT_FINISHED", complete=len(state["completed"]), failures=state["failed_retained"])


if __name__ == "__main__":
    run()
