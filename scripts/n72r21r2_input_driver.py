"""Bounded fresh-candidate preparation with immediate full baseline checking.

Adopts the two initial V2 workers by verified PID; never starts duplicates.
Pilot is fixed before effects. Full phase requires every pilot input baseline,
not a scientific improvement gate, and then prepares all FIT16 and INNER8.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage, update_status, append_log


def alive_owned_worker(pid, module=None):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        owner = Path(f"/proc/{pid}").stat().st_uid
        return owner == os.getuid() and (module is None or module in command)
    except (OSError, UnicodeError):
        return False


def idle_physical(gpu):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    values = {int(line.split(",")[0]): int(line.split(",")[1]) for line in raw.splitlines()}
    return values.get(gpu, 1 << 30) < 200


def run_baseline(sequence):
    path = ASSETS / "input_driver_logs" / (sequence + "__baseline.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for action in ("prepare", "replay", "evaluate"):
            completed = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_baseline", action, "--sequence", sequence],
                                       cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError("Baseline action failed: " + sequence + "/" + action + "; retained log " + str(path))
    return path


def run(phase):
    protocol = preregistration()
    marker = OUT / "data" / ("input_driver_" + phase + ".json")
    if marker.exists():
        old = read_json(marker)
        if alive_owned_worker(old["pid"], "scripts.n72r21r2_input_driver"):
            raise RuntimeError("A live driver already owns this phase")
        if old["status"] == "COMPLETE":
            assert old["source_sha256"] == sha256(Path(__file__))
            for s in old["sequences"]:
                assert (OUT / "mot/baseline_results" / (s + ".json")).exists()
            print(json.dumps({"already_complete": phase}), flush=True)
            return
        raise RuntimeError("Preserve interrupted driver; audit and version explicit recovery")
    if phase == "full":
        for sequence in protocol["fresh_candidate_pilot"]:
            if not (OUT / "mot/baseline_results" / (sequence + ".json")).exists():
                raise RuntimeError("Complete fixed input pilot first; this is integrity, not an effect gate")
    sequences = protocol["fresh_candidate_pilot"] if phase == "pilot" else protocol["split"]["fit"] + protocol["split"]["inner"]
    state = {"stage": "N72R21R2", "phase": phase, "status": "ACTIVE", "pid": os.getpid(), "sequences": sequences,
             "source_sha256": sha256(Path(__file__)), "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
             "max_GPU_workers": 2, "GPU_pool": [0, 1], "scientific_gate_required_for_fresh_data_preparation": False,
             "completed_baselines": [], "adopted_existing_workers": [], "new_generation_workers": []}
    write_json("data/input_driver_" + phase + ".json", state)
    children, handles = {}, {}
    for sequence in sequences:
        preflight = OUT / "data/extraction_preflight_v2" / (sequence + ".json")
        if preflight.exists() and not (OUT / "data/candidate_integrity" / (sequence + ".json")).exists():
            record = read_json(preflight)
            if alive_owned_worker(record["pid"], "scripts.n72r21r2_candidates_v2"):
                children[sequence] = (record["pid"], int(record["visible_physical_GPU"]), None)
                state["adopted_existing_workers"].append(record)
            else:
                raise RuntimeError("Partial/failed worker requires explicit recovery audit: " + sequence)
    last_report = 0.
    while len(state["completed_baselines"]) < len(sequences):
        storage(8 << 30)
        for sequence, (pid, gpu, process) in list(children.items()):
            running = process.poll() is None if process is not None else alive_owned_worker(pid, "scripts.n72r21r2_candidates_v2")
            if running:
                continue
            if sequence in handles:
                handles.pop(sequence).close()
            del children[sequence]
            if not (OUT / "data/candidate_integrity" / (sequence + ".json")).exists():
                state.update(status="FAILED_RETAINED", failed_sequence=sequence)
                write_json("data/input_driver_" + phase + ".json", state, mutable=True)
                raise RuntimeError("Worker failed; no blind retry: " + sequence)
        for sequence in sequences:
            complete = OUT / "mot/baseline_results" / (sequence + ".json")
            if complete.exists() and sequence not in state["completed_baselines"]:
                state["completed_baselines"].append(sequence)
            elif (OUT / "data/candidate_integrity" / (sequence + ".json")).exists() and sequence not in state["completed_baselines"]:
                run_baseline(sequence)
                state["completed_baselines"].append(sequence)
                print(json.dumps({"fresh_input_and_baseline_complete": sequence, "completed": len(state["completed_baselines"]), "total": len(sequences)}), flush=True)
        free = [gpu for gpu in (0, 1) if gpu not in [p[1] for p in children.values()] and idle_physical(gpu)]
        waiting = [s for s in sequences if s not in children and not (OUT / "data/candidate_integrity" / (s + ".json")).exists()]
        for sequence, gpu in zip(waiting, free):
            if len(children) >= 2:
                break
            existing = OUT / "data/extraction_preflight_v2" / (sequence + ".json")
            if existing.exists():
                raise RuntimeError("Prior attempt requires explicit versioned recovery: " + sequence)
            log_path = ASSETS / "input_driver_logs" / (sequence + "__generation_worker.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("x")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_candidates_v2", "--sequence", sequence], cwd=ROOT,
                                       env=env, stdout=handle, stderr=subprocess.STDOUT)
            children[sequence] = (process.pid, gpu, process)
            handles[sequence] = handle
            state["new_generation_workers"].append({"sequence": sequence, "pid": process.pid, "physical_GPU": gpu, "log_path": str(log_path)})
        if time.monotonic() - last_report >= 20:
            state["active_workers"] = [{"sequence": s, "pid": p[0], "physical_GPU": p[1]} for s, p in children.items()]
            write_json("data/input_driver_" + phase + ".json", state, mutable=True)
            update_status(status="ACTIVE_M1_FRESH_" + phase.upper() + "_CANDIDATES_AND_BASELINES",
                          fresh_input_phase=phase, fresh_baselines_completed=len(state["completed_baselines"]), fresh_current_workers=state["active_workers"])
            print(json.dumps({"phase": phase, "completed_baselines": len(state["completed_baselines"]), "workers": state["active_workers"], "storage": storage()}), flush=True)
            last_report = time.monotonic()
        time.sleep(2)
    state.update(status="COMPLETE", active_workers=[])
    write_json("data/input_driver_" + phase + ".json", state, mutable=True)
    append_log("M1_INPUT_DRIVER_COMPLETE", phase=phase, fresh_input_and_baseline_sequences=len(sequences))
    print(json.dumps({"input_driver_complete": phase, "sequences": len(sequences)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "full"], required=True)
    args = parser.parse_args()
    run(args.phase)
