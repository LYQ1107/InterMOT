"""Serial bounded own-policy execution of all frozen pilot seeds/videos."""
import os
from pathlib import Path
import subprocess
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, storage, append_log
from scripts.n72r21r2_learned_pilot import identity, PROTOCOL


def run():
    marker = "training/learned_pilot_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect previous pilot driver ownership before another attempt")
    protocol = read_json(PROTOCOL)
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE", "CPU_workers": 1,
             "source_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(PROTOCOL), "completed": [], "active": None}
    write_json(marker, state)
    for family in protocol["families"]:
        for seed in protocol["seeds"]:
            uid = identity(family, seed)
            for sequence in protocol["sequences"]:
                storage(128 << 20)
                key = uid + "/" + sequence
                for action, prefix in (("runtime", "training/pilot_policy_runtime"), ("evaluate", "training/pilot_policy_results")):
                    if (OUT / prefix / uid / (sequence + ".json")).exists():
                        continue
                    log = ASSETS / "learned_pilot_driver_v1_logs" / (uid + "__" + sequence + "__" + action + ".log")
                    log.parent.mkdir(parents=True, exist_ok=True)
                    command = [str(PYTHON), "-u", "-m", "scripts.n72r21r2_learned_pilot", action,
                               "--sequence", sequence, "--family", family, "--seed", str(seed)]
                    with log.open("x") as handle:
                        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                                   env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                        state["active"] = {"key": key, "action": action, "pid": process.pid, "log": str(log)}
                        write_json(marker, state, mutable=True)
                        print({"pilot_policy_active": state["active"]}, flush=True)
                        returncode = process.wait()
                    if returncode:
                        state.update(status="FAILED_ENGINEERING_OPERATION_RETAINED", failure={"key": key, "action": action, "returncode": returncode, "log": str(log)}, active=None)
                        write_json(marker, state, mutable=True)
                        raise RuntimeError("Pilot engineering failure retained; inspect before any explicit versioned recovery")
                state["completed"].append(key)
                state["active"] = None
                write_json(marker, state, mutable=True)
    state.update(status="COMPLETE_ACTUAL_ALL_FROZEN_FIT_PILOT_POLICIES", active=None)
    write_json(marker, state, mutable=True)
    append_log("M5_FIT_PILOT_POLICY_DRIVER_COMPLETE", actual_model_video_combinations=len(state["completed"]))


if __name__ == "__main__":
    run()
