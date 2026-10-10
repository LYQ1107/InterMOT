"""Versioned input-driver recovery, preserving the interrupted V1 marker.

Completed real inference is integrity-rechecked, not rerun for a JSON writer
failure. New workers use explicit physical GPU isolation and native counters.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage, update_status, append_log
from scripts.n72r21r2_input_driver import alive_owned_worker, idle_physical, run_baseline
from scripts.n72r21r2_integrity_v2 import integrity


def run(phase):
    protocol = preregistration()
    relative = "data/input_driver_v2_" + phase + ".json"
    marker = OUT / relative
    if marker.exists():
        raise FileExistsError("Preserve existing driver evidence; inspect live ownership before recovery")
    if phase == "full":
        assert all((OUT / "mot/baseline_results" / (s + ".json")).exists() for s in protocol["fresh_candidate_pilot"])
    sequences = protocol["fresh_candidate_pilot"] if phase == "pilot" else protocol["split"]["fit"] + protocol["split"]["inner"]
    state = {"stage": "N72R21R2", "phase": phase, "status": "ACTIVE", "pid": os.getpid(), "sequences": sequences,
             "source_sha256": sha256(Path(__file__)), "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
             "completed_baselines": [], "adopted_existing_workers": [], "new_workers": [], "max_GPU_workers": 2,
             "old_driver_marker_preserved": "data/input_driver_pilot.json", "science_improvement_gate_before_fresh_inputs": False}
    write_json(relative, state)
    children, handles = {}, {}
    for sequence in sequences:
        path = OUT / "data/extraction_preflight_v2" / (sequence + ".json")
        if path.exists() and not (OUT / "data/candidate_integrity" / (sequence + ".json")).exists():
            record = read_json(path)
            if alive_owned_worker(record["pid"], "scripts.n72r21r2_candidates"):
                children[sequence] = (record["pid"], int(record["visible_physical_GPU"]), None)
                state["adopted_existing_workers"].append(record)
            elif (ASSETS / "candidates" / sequence / "done.json").exists():
                integrity(sequence)
            else:
                raise RuntimeError("No completed tape and no live worker: audit before another attempt: " + sequence)
    last = 0.
    while len(state["completed_baselines"]) != len(sequences):
        storage(8 << 30)
        for sequence, (pid, gpu, process) in list(children.items()):
            running = process.poll() is None if process is not None else alive_owned_worker(pid, "scripts.n72r21r2_candidates")
            if running:
                continue
            if sequence in handles:
                handles.pop(sequence).close()
            del children[sequence]
            if not (OUT / "data/candidate_integrity" / (sequence + ".json")).exists():
                if (ASSETS / "candidates" / sequence / "done.json").exists():
                    integrity(sequence)
                else:
                    state.update(status="FAILED_RETAINED", failed_sequence=sequence)
                    write_json(relative, state, mutable=True)
                    raise RuntimeError("Generation failed; preserve attempt: " + sequence)
        for sequence in sequences:
            if sequence in state["completed_baselines"]:
                continue
            if (OUT / "mot/baseline_results" / (sequence + ".json")).exists():
                state["completed_baselines"].append(sequence)
            elif (OUT / "data/candidate_integrity" / (sequence + ".json")).exists():
                run_baseline(sequence)
                state["completed_baselines"].append(sequence)
                print(json.dumps({"input_baseline_complete": sequence, "completed": len(state["completed_baselines"]), "total": len(sequences)}), flush=True)
        free = [g for g in (0, 1) if g not in [v[1] for v in children.values()] and idle_physical(g)]
        waiting = [s for s in sequences if s not in children and not (OUT / "data/candidate_integrity" / (s + ".json")).exists()]
        for sequence, gpu in zip(waiting, free):
            if len(children) >= 2:
                break
            if (OUT / "data/extraction_preflight_v2" / (sequence + ".json")).exists():
                raise RuntimeError("A partial previous attempt must not be overwritten: " + sequence)
            log = ASSETS / "input_driver_logs" / (sequence + "__generation_worker.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("x")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_candidates_v3", "--sequence", sequence], cwd=ROOT, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT)
            handles[sequence] = handle
            children[sequence] = (process.pid, gpu, process)
            state["new_workers"].append({"sequence": sequence, "pid": process.pid, "physical_GPU": gpu, "log_path": str(log)})
        if time.monotonic() - last > 20:
            state["active_workers"] = [{"sequence": s, "pid": p[0], "physical_GPU": p[1]} for s, p in children.items()]
            write_json(relative, state, mutable=True)
            update_status(status="ACTIVE_M1_FRESH_" + phase.upper() + "_CANDIDATES_AND_BASELINES", fresh_baselines_completed=len(state["completed_baselines"]),
                          fresh_input_phase=phase, fresh_current_workers=state["active_workers"])
            print(json.dumps({"phase": phase, "completed_baselines": len(state["completed_baselines"]), "workers": state["active_workers"]}), flush=True)
            last = time.monotonic()
        time.sleep(2)
    state.update(status="COMPLETE", active_workers=[])
    write_json(relative, state, mutable=True)
    append_log("M1_INPUT_DRIVER_V2_COMPLETE", phase=phase, completed_sequences=len(sequences))
    print(json.dumps({"input_phase_complete": phase, "sequences": len(sequences)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "full"], required=True)
    args = parser.parse_args()
    run(args.phase)
