"""One bounded short CPU exact-frame diagnostic owner, all24 still required."""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, storage, preregistration
from scripts.n72r21r2_pure_margin_probe import PROTOCOL, SELECTED, PREFIX


def run():
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual pure-margin owner before duplicate launch")
    p = preregistration()
    sequences = p["split"]["fit"] + p["split"]["inner"]
    assert SELECTED.exists()
    state = {"stage": "N72R21R2", "status": "ACTIVE_INCREMENTAL_EXACT_CF_CURRENT_DIAGNOSTIC", "pid": os.getpid(),
             "source_sha256": sha256(__file__), "protocol_sha256": sha256(PROTOCOL), "completed": [], "failed_retained": [], "active": None, "CPU_workers": 1}
    write_json(marker, state)
    last_complete = None
    while True:
        state["completed"] = [s for s in sequences if (OUT / PREFIX / "CF_results" / (s + ".json")).exists()]
        rejected = {r["sequence"] for r in state["failed_retained"]}
        if state["completed"] != last_complete:
            write_json(marker, state, mutable=True)
            print({"pure_margin_exact_CF_ready": len(state["completed"]), "required": 24}, flush=True)
            last_complete = state["completed"].copy()
        ready = [s for s in sequences if s not in state["completed"] and s not in rejected and
                 all((OUT / folder / (s + ".json")).exists() for folder in ("on_policy/joint_state_v1/runtime_sequences", "events/label_audit"))]
        if ready:
            s = ready[0]
            storage(16 << 20)
            log = ASSETS / "pure_margin_v1_logs" / (s + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x") as handle:
                for action, folder in (("score_cf", "CF_predictions"), ("report_cf", "CF_results")):
                    if (OUT / PREFIX / folder / (s + ".json")).exists():
                        continue
                    process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_pure_margin_probe", action, "--sequence", s], cwd=ROOT,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                    state["active"] = {"sequence": s, "action": action, "pid": process.pid, "log": str(log)}
                    write_json(marker, state, mutable=True)
                    code = process.wait()
                    if code:
                        state["failed_retained"].append({"sequence": s, "action": action, "returncode": code, "log": str(log)})
                        break
            state["active"] = None
            write_json(marker, state, mutable=True)
        if not ready and len(state["completed"]) + len(rejected) == 24:
            break
        time.sleep(2)
    state.update(status="COMPLETE_ALL24_ACTUAL_PURE_MARGIN_EXACT_CF_DIAGNOSTIC" if not rejected else "ACTUAL_AVAILABLE_DIAGNOSTICS_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)


if __name__ == "__main__":
    run()
