"""One CPU joint-state source worker, retaining all24 fresh development videos."""
import argparse
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, storage, append_log, development_sequence


def worker(sequence):
    development_sequence(sequence)
    for action, folder in (("prepare", "plans"), ("runtime", "runtime_sequences"), ("label", "supervision")):
        path = OUT / "on_policy/joint_state_v1" / folder / (sequence + ".json")
        if path.exists():
            old = read_json(path)
            assert old["protocol_sha256"] == sha256(OUT / "on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json")
            if "artifact" in old:
                assert sha256(old["artifact"]["path"]) == old["artifact"]["sha256"]
            continue
        completed = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_curriculum", action, "--sequence", sequence], cwd=ROOT, check=False)
        if completed.returncode:
            raise RuntimeError("Joint-state source failed; preserve all partials: " + action + "/" + sequence)
        assert path.exists()


def run():
    p = preregistration()
    sequences = p["split"]["fit"] + p["split"]["inner"]
    marker = "on_policy/joint_state_v1/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual driver ownership before launching another")
    state = {"stage": "N72R21R2", "status": "ACTIVE", "pid": os.getpid(), "source_sha256": sha256(__file__),
             "protocol_sha256": sha256(OUT / "on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json"),
             "sequences": sequences, "CPU_workers": 1, "active": None, "completed": [], "failed_retained": [],
             "controlled_treatment_source_not_model_generated_on_policy": True}
    write_json(marker, state)
    child, handle, sequence = None, None, None
    last = 0.
    while True:
        storage(256 << 20)
        if child is not None and child.poll() is not None:
            handle.close()
            if child.returncode:
                state["failed_retained"].append({"sequence": sequence, "returncode": child.returncode, "log": state["active"]["log"]})
            child, handle, sequence = None, None, None
            state["active"] = None
        state["completed"] = [s for s in sequences if (OUT / "on_policy/joint_state_v1/supervision" / (s + ".json")).exists()]
        failed = {r["sequence"] for r in state["failed_retained"]}
        ready = [s for s in sequences if s not in state["completed"] and s not in failed
                 and (OUT / "events/counterfactual_sequences" / (s + ".json")).exists()]
        if child is None and ready:
            sequence = ready[0]
            log = ASSETS / "joint_state_driver_v1_logs" / (sequence + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("x")
            child = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_driver", "--worker", sequence],
                                     cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"sequence": sequence, "pid": child.pid, "log": str(log)}
        if time.monotonic() - last > 20:
            write_json(marker, state, mutable=True)
            print({"joint_state_source_driver": state["status"], "complete": len(state["completed"]), "required": 24,
                   "active": state["active"], "failed_retained": state["failed_retained"]}, flush=True)
            last = time.monotonic()
        if child is None and len(state["completed"]) + len(failed) == 24:
            break
        time.sleep(2)
    state.update(status="COMPLETE_ALL24_CONTROLLED_STATE_SOURCES" if not failed else "AVAILABLE_CASES_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M7_CONTROLLED_STATE_SOURCE_DRIVER_FINISHED", completed=len(state["completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
