"""Incremental, bounded fresh causal-event execution; no CONFIRM/VAL input."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, append_log, storage, development_sequence
from scripts.n72r21r2_input_driver import alive_owned_worker


def worker(sequence):
    development_sequence(sequence)
    tasks = [
        ("events/corpus/", "scripts.n72r21r2_events", []),
        ("events/corpus_v2/", "scripts.n72r21r2_events_future_v2", []),
        ("events/counterfactual_plans/", "scripts.n72r21r2_counterfactual", ["prepare"]),
        ("events/counterfactual_sequences/", "scripts.n72r21r2_counterfactual_v2", []),
        ("events/current_context_sequences/", "scripts.n72r21r2_event_context", []),
        ("events/label_audit/", "scripts.n72r21r2_label_counterfactual", []),
        ("events/window_trackeval_results/", "scripts.n72r21r2_window_trackeval", []),
    ]
    for prefix, module, actions in tasks:
        path = OUT / prefix / (sequence + ".json")
        if path.exists():
            read_json(path)
            continue
        completed = subprocess.run([str(PYTHON), "-u", "-m", module, *actions, "--sequence", sequence], cwd=ROOT, check=False)
        if completed.returncode:
            raise RuntimeError("Retain failed causal stage: " + module + "/" + sequence)
        assert path.exists()
    write_json("events/driver_sequence_complete/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_EVENT_RUNTIME_OFFLINE_LABELS_ACTUAL_WINDOW_EVAL",
        "source_sha256": sha256(Path(__file__)), "not_full_policy_MOT_success": True,
        "label_audit_sha256": sha256(OUT / "events/label_audit" / (sequence + ".json")),
        "window_evaluation_sha256": sha256(OUT / "events/window_trackeval_results" / (sequence + ".json"))})


def run():
    protocol = preregistration()
    sequences = protocol["split"]["fit"] + protocol["split"]["inner"]
    marker = "events/event_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect existing driver ownership before a new attempt")
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE", "source_sha256": sha256(Path(__file__)),
             "sequences": sequences, "max_CPU_workers": 2, "source_distribution": "Fresh C0-shadow P0",
             "completed": [], "failed_retained": [], "active": []}
    write_json(marker, state)
    children = {}
    last = 0.
    while True:
        storage(1 << 30)
        for sequence, item in list(children.items()):
            if item["process"].poll() is None:
                continue
            item["handle"].close()
            if item["process"].returncode:
                state["failed_retained"].append({"sequence": sequence, "returncode": item["process"].returncode, "log_path": item["log"]})
            del children[sequence]
        state["completed"] = [s for s in sequences if (OUT / "events/driver_sequence_complete" / (s + ".json")).exists()]
        failed = {r["sequence"] for r in state["failed_retained"]}
        ready = [s for s in sequences if s not in state["completed"] and s not in children and s not in failed and (OUT / "mot/baseline_results" / (s + ".json")).exists()]
        for sequence in ready[:max(0, 2 - len(children))]:
            log = ASSETS / "event_driver_v1_logs" / (sequence + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("x")
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_event_driver", "--worker", sequence], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            children[sequence] = {"process": process, "handle": handle, "log": str(log)}
        if time.monotonic() - last > 20:
            state["active"] = [{"sequence": s, "pid": c["process"].pid, "log": c["log"]} for s, c in children.items()]
            write_json(marker, state, mutable=True)
            print(json.dumps({"event_driver": "ACTIVE", "complete": len(state["completed"]), "active": state["active"], "failed_retained": state["failed_retained"]}), flush=True)
            last = time.monotonic()
        if not children and len(state["completed"]) + len(failed) == len(sequences):
            break
        time.sleep(2)
    state.update(status="COMPLETE" if not failed else "AVAILABLE_CORPUS_COMPLETE_FAILURES_RETAINED", active=[])
    write_json(marker, state, mutable=True)
    append_log("M2_M3_FRESH_EVENT_DRIVER_FINISHED", complete=len(state["completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
