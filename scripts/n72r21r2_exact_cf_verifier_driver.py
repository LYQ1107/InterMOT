"""Exact-frame current-verifier diagnostics as each fresh source is sealed."""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, append_log, storage


def run():
    p = preregistration()
    sequences = p["split"]["fit"] + p["split"]["inner"]
    marker = "availability/exact_cf_v1/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect exact-CF driver before duplicate launch")
    protocol = OUT / "availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json"
    state = {"stage": "N72R21R2", "status": "ACTIVE", "pid": os.getpid(), "source_sha256": sha256(__file__),
             "protocol_sha256": sha256(protocol), "completed": [], "failed_retained": [], "active": None,
             "CPU_workers": 1, "association_authority": False, "all_seeds_retained": True}
    write_json(marker, state)
    last = 0.
    while True:
        state["completed"] = [s for s in sequences if (OUT / "availability/exact_cf_v1/results" / (s + ".json")).exists()]
        failed = {r["sequence"] for r in state["failed_retained"]}
        ready = [s for s in sequences if s not in state["completed"] and s not in failed
                 and all((OUT / folder / (s + ".json")).exists() for folder in
                         ("events/label_audit", "on_policy/joint_state_v1/supervision"))]
        if ready:
            sequence = ready[0]
            storage(32 << 20)
            log = ASSETS / "exact_cf_verifier_driver_v1_logs" / (sequence + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x") as handle:
                for action, folder in (("score", "predictions"), ("report", "results")):
                    if (OUT / "availability/exact_cf_v1" / folder / (sequence + ".json")).exists():
                        continue
                    process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_exact_cf_verifier", action, "--sequence", sequence], cwd=ROOT,
                                               env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                    state["active"] = {"sequence": sequence, "action": action, "pid": process.pid, "log": str(log)}
                    write_json(marker, state, mutable=True)
                    code = process.wait()
                    if code:
                        state["failed_retained"].append({"sequence": sequence, "action": action, "returncode": code, "log": str(log)})
                        break
            state["active"] = None
        if time.monotonic() - last > 20:
            write_json(marker, state, mutable=True)
            print({"exact_CF_verifier_driver": "ACTIVE", "completed": len(state["completed"]), "required": 24,
                   "failed_retained": state["failed_retained"]}, flush=True)
            last = time.monotonic()
        if len(state["completed"]) + len(failed) == 24:
            break
        time.sleep(2)
    state.update(status="COMPLETE_ALL24_EXACT_CF_VERIFIER_DIAGNOSTICS" if not failed else "AVAILABLE_CASES_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M8_EXACT_CF_VERIFIER_DRIVER_FINISHED", completed=len(state["completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
