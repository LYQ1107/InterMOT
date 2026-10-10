"""Resume all frozen fresh inputs; preserve failures, decouple GPU and CPU.

At most two idle physical GPUs and one baseline CPU worker are owned here.
An isolated failed candidate does not kill the other worker or prevent
remaining registered fresh videos from being prepared. Failed attempts are
not silently retried or counted complete.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage, append_log, update_status
from scripts.n72r21r2_input_driver import idle_physical, alive_owned_worker
from scripts.n72r21r2_integrity_v2 import integrity


def run():
    protocol = preregistration()
    sequences = protocol["split"]["fit"] + protocol["split"]["inner"]
    marker = "data/input_driver_v3_full.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect existing V3 ownership, never overwrite a driver attempt")
    assert read_json(OUT / "data/TRIM_SCHEMA_COMPATIBILITY_PREFIX_V4.json")["raw_candidates_UIDs_boxes_scores_features_exactly_equal"]
    state = {"stage": "N72R21R2", "status": "ACTIVE", "pid": os.getpid(), "sequences": sequences,
             "source_sha256": sha256(Path(__file__)), "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
             "completed_baselines": [], "failed_sequences_retained": [], "active_GPU_workers": [],
             "max_GPU_workers": 2, "max_baseline_CPU_workers": 1, "prior_driver_marker_preserved": "data/input_driver_v2_full.json",
             "old_reproduction_effect_is_not_fresh_data_gate": True}
    write_json(marker, state)
    children, baseline, last = {}, None, 0.
    # Refuse duplicates even when an earlier parent has terminated.
    for namespace in ("extraction_preflight_v2", "extraction_v4_preflight"):
        for path in (OUT / "data" / namespace).glob("*.json"):
            item = read_json(path)
            if alive_owned_worker(item["pid"], "scripts.n72r21r2_candidates"):
                children[item["sequence"]] = {"pid": item["pid"], "gpu": int(item["visible_physical_GPU"]), "process": None, "handle": None}
    while True:
        storage(8 << 30)
        for sequence, worker in list(children.items()):
            process = worker["process"]
            running = process.poll() is None if process is not None else alive_owned_worker(worker["pid"], "scripts.n72r21r2_candidates")
            if running:
                continue
            if worker["handle"] is not None:
                worker["handle"].close()
            del children[sequence]
            if not (OUT / "data/candidate_integrity" / (sequence + ".json")).exists():
                if (ASSETS / "candidates" / sequence / "done.json").exists():
                    integrity(sequence)
                else:
                    state["failed_sequences_retained"].append({"sequence": sequence, "phase": "generation", "returncode": process.returncode if process else None})
        if baseline is not None and baseline["process"].poll() is not None:
            baseline["handle"].close()
            if baseline["process"].returncode:
                state["failed_sequences_retained"].append({"sequence": baseline["sequence"], "phase": "baseline", "returncode": baseline["process"].returncode})
            else:
                assert (OUT / "mot/baseline_results" / (baseline["sequence"] + ".json")).exists()
            baseline = None
        state["completed_baselines"] = [s for s in sequences if (OUT / "mot/baseline_results" / (s + ".json")).exists()]
        failed = {r["sequence"] for r in state["failed_sequences_retained"]}
        if baseline is None:
            ready = [s for s in sequences if s not in state["completed_baselines"] and s not in failed and (OUT / "data/candidate_integrity" / (s + ".json")).exists()]
            if ready:
                sequence = ready[0]
                log = ASSETS / "input_driver_logs_v3" / (sequence + "__baseline.log")
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("x")
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_input_driver_v3", "--baseline-sequence", sequence], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                baseline = {"sequence": sequence, "process": process, "handle": handle}
        free = [g for g in (0, 1) if g not in [w["gpu"] for w in children.values()] and idle_physical(g)]
        waiting = [s for s in sequences if s not in children and s not in failed and not (OUT / "data/candidate_integrity" / (s + ".json")).exists()]
        for sequence, gpu in zip(waiting, free):
            if len(children) >= 2:
                break
            if (OUT / "data/extraction_v4_preflight" / (sequence + ".json")).exists():
                raise RuntimeError("Previous V4 attempt cannot be overwritten: " + sequence)
            log = ASSETS / "input_driver_logs_v3" / (sequence + "__generation.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("x")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_candidates_v4", "--sequence", sequence], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            children[sequence] = {"process": process, "pid": process.pid, "gpu": gpu, "handle": handle}
        if time.monotonic() - last >= 20:
            state["active_GPU_workers"] = [{"sequence": s, "pid": w["pid"], "physical_GPU": w["gpu"]} for s, w in children.items()]
            state["active_baseline_worker"] = None if baseline is None else {"sequence": baseline["sequence"], "pid": baseline["process"].pid}
            write_json(marker, state, mutable=True)
            update_status(status="ACTIVE_M1_RESUMED_FRESH_INPUTS_AND_M3_CAUSAL_LABELS", fresh_current_workers=state["active_GPU_workers"],
                          fresh_baselines_completed=len(state["completed_baselines"]), fresh_input_driver_pid=os.getpid(), scientific_success=None)
            print(json.dumps({"V3_input_driver": "ACTIVE", "baselines_complete": len(state["completed_baselines"]), "GPU_workers": state["active_GPU_workers"],
                              "baseline_worker": state["active_baseline_worker"], "failures_retained": state["failed_sequences_retained"]}), flush=True)
            last = time.monotonic()
        if not children and baseline is None and len(state["completed_baselines"]) + len(failed) == len(sequences):
            break
        time.sleep(2)
    state.update(status="COMPLETE" if not state["failed_sequences_retained"] else "COMPLETE_AVAILABLE_INPUTS_WITH_FAILURES_RETAINED", active_GPU_workers=[], active_baseline_worker=None)
    write_json(marker, state, mutable=True)
    append_log("M1_INPUT_DRIVER_V3_FINISHED", completed=len(state["completed_baselines"]), failures=state["failed_sequences_retained"])


def baseline_sequence(sequence):
    for action in ("prepare", "replay", "evaluate"):
        subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_baseline", action, "--sequence", sequence], cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sequence")
    args = parser.parse_args()
    baseline_sequence(args.baseline_sequence) if args.baseline_sequence else run()
