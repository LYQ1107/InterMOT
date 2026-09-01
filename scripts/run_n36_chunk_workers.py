#!/usr/bin/env python3
"""Run N36 chunks with at most one child process per physical GPU.

The scheduler itself does not own a SAM3 session.  Every work item launches a
fresh ``run_n36_export_chunk.py`` child, so a completed range cannot retain a
CUDA context or official inference state into the next range.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_tape_common import ROOT, atomic_json


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_child(
    plan_path: Path,
    output_root: Path,
    chunk: dict[str, Any],
    gpu: int,
    log_root: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    sequence = str(chunk["sequence"])
    chunk_id = str(chunk["chunk_id"])
    log_path = log_root / sequence / f"{chunk_id}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_n36_export_chunk.py"),
        "--plan",
        str(plan_path),
        "--chunk-id",
        chunk_id,
        "--output-root",
        str(output_root),
        "--gpu",
        "0",
    ]
    if skip_existing:
        command.append("--skip-existing")
    started = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = f"{ROOT / 'third_party/sam3'}:{ROOT}:{env.get('PYTHONPATH', '')}"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    done_path = output_root / "chunk_done" / sequence / f"{chunk_id}.json"
    done_status = None
    if done_path.is_file():
        try:
            done_status = json.loads(done_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            done_status = {"status": "CORRUPT_DONE_JSON"}
    status = "PASS" if completed.returncode == 0 and done_status and done_status.get("status") == "PASS" else "FAIL"
    return {
        "sequence": sequence,
        "chunk_id": chunk_id,
        "gpu": int(gpu),
        "return_code": int(completed.returncode),
        "status": status,
        "done": done_status,
        "log": str(log_path.relative_to(ROOT)),
        "elapsed_sec": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/n36/real_tape")
    parser.add_argument("--log-root", type=Path, default=ROOT / "outputs/n36/logs/chunks")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-label", default="latest")
    args = parser.parse_args()
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) > 4 or any(value < 0 or value > 3 for value in gpus):
        raise ValueError("N36 scheduler accepts one to four physical GPUs from 0,1,2,3")
    plan_path = args.plan.resolve()
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()
    plan = load_plan(plan_path)
    selected = {item.strip() for item in args.sequences.split(",") if item.strip()}
    chunks = [
        dict(chunk)
        for chunk in plan.get("all_chunks", [])
        if not selected or str(chunk.get("sequence")) in selected
    ]
    chunks.sort(key=lambda item: (str(item["sequence"]), int(item["chunk_index"])))
    if not chunks:
        raise ValueError("no chunks selected")
    work: queue.Queue[dict[str, Any]] = queue.Queue()
    for chunk in chunks:
        work.put(chunk)
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                chunk = work.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_child(
                    plan_path,
                    output_root,
                    chunk,
                    gpu,
                    log_root,
                    bool(args.skip_existing),
                )
                with results_lock:
                    results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    started = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results.sort(key=lambda item: (item["sequence"], item["chunk_id"]))
    payload = {
        "protocol": "N36_CHUNK_WORKER_RUN",
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "PARTIAL",
        "plan": str(plan_path.relative_to(ROOT)),
        "output_root": str(output_root.relative_to(ROOT)),
        "sequences": sorted({item["sequence"] for item in results}),
        "chunk_count_expected": len(chunks),
        "chunk_count_finished": len(results),
        "chunk_count_pass": sum(item["status"] == "PASS" for item in results),
        "chunk_count_fail": sum(item["status"] != "PASS" for item in results),
        "physical_gpus": gpus,
        "allocator": "expandable_segments:True",
        "independent_process_per_chunk": True,
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    artifact = ROOT / "outputs/n36/chunk_manifests" / f"worker_run_{args.run_label}.json"
    atomic_json(artifact, payload)
    print(json.dumps({"artifact": str(artifact), **{key: payload[key] for key in ("status", "chunk_count_expected", "chunk_count_pass", "chunk_count_fail")}}, sort_keys=True), flush=True)
    # All work items were attempted.  The validator, not the scheduler exit
    # code, is the completeness gate and will preserve every failed child.


if __name__ == "__main__":
    main()
