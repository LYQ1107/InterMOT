"""One scoped CPU owner, no fourth CF worker and no silent partial adoption."""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage
from scripts.n72r21r2_support_onset_audit import PROTOCOL, PREFIX
from scripts.n72r21r2_support_ablation_pilot import PREFIX as SOURCE_PREFIX, inputs


def run():
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual onset owner before duplicate launch")
    p = read_json(PROTOCOL)
    jobs = [(seed, policy, s) for seed in p["seeds"] for policy in p["policies"] for s in p["sequences"]]
    required = preregistration()["split"]["fit"] + preregistration()["split"]["inner"]
    state = {"stage": "N72R21R2", "status": "ACTIVE_READY_CELL_ONSETS_WAIT_NONZERO_CF_RESOURCE", "pid": os.getpid(),
             "protocol_sha256": sha256(PROTOCOL), "CPU_workers": 1, "active": None, "completed": [], "failed_retained": []}
    write_json(marker, state)
    last = None
    while True:
        ready = []
        for seed, policy, s in jobs:
            _, uid, _ = inputs(s, seed, policy)
            key = uid + "/" + s
            if key in state["completed"] or key in {r["key"] for r in state["failed_retained"]}:
                continue
            path = OUT / SOURCE_PREFIX / "runtime" / uid / (s + ".json")
            if not path.exists():
                continue
            seal = read_json(path)
            nonzero = any(e["effective_direct_decisions_NOT_independent_onsets"] for e in seal["episodes"])
            if not nonzero or all((OUT / "events/counterfactual_sequences" / (seq + ".json")).exists() for seq in required):
                ready.append((key, seed, policy, s, uid))
        if ready:
            key, seed, policy, s, uid = ready[0]
            storage(256 << 20)
            log = ASSETS / "support_onset_v1_logs" / uid / (s + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x") as handle:
                for action, directory in (("runtime", "runtime"), ("label", "labels")):
                    if (OUT / PREFIX / directory / uid / (s + ".json")).exists():
                        raise FileExistsError("No silent adoption of an unowned onset attempt")
                    process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_support_onset_audit", action,
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
        if len(state["completed"]) != last:
            print({"actual_support_onset_cells": len(state["completed"]), "required": 36, "failed_retained": len(state["failed_retained"])}, flush=True)
            last = len(state["completed"])
        if len(state["completed"]) + len(state["failed_retained"]) == len(jobs):
            break
        if not ready:
            time.sleep(2)
    state["status"] = "COMPLETE36_ACTUAL_ONSET_AUDITS_NOT_QUALIFIED" if not state["failed_retained"] else "AVAILABLE_CASES_COMPLETE_FAILURES_RETAINED"
    write_json(marker, state, mutable=True)


if __name__ == "__main__":
    run()
