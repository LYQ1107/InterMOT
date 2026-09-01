#!/usr/bin/env python
"""N5 parallel orchestrator: persistent worker pool across safe idle GPUs.

Each job is one (protocol, budget, sequence) unit.  Jobs are ordered so the
longest-running protocols (P2/P3) start first and are interleaved across
sequences, keeping all GPUs busy.  The orchestrator is a single blocking
process; it does not poll from the agent.
"""

import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(".")
PY = ROOT / "envs/sam3_intermot/bin/python"

SEQS = os.environ.get("N5_SEQS", "dancetrack0004 dancetrack0005 dancetrack0007").split()
GPUS = [int(x) for x in os.environ.get("N5_GPUS", "1 2 3 5 6 7").split()]
OUT_ROOT = Path(os.environ.get("N5_OUT_ROOT", ROOT / "outputs/n5"))
FRAME_LIMIT = os.environ.get("N5_FRAMES", "")
TAG = os.environ.get("N5_TAG", "trend")


def build_jobs() -> List[Dict]:
    jobs = []
    for seq in SEQS:
        jobs.append({"protocol": "p2", "budget": 0, "seq": seq})
        jobs.append({"protocol": "p3", "budget": 0, "seq": seq})
    for b in (8, 4, 2, 1):
        for seq in SEQS:
            jobs.append({"protocol": "p4", "budget": b, "seq": seq})
    return jobs


def out_dir_for(job: Dict) -> Path:
    if job["protocol"] == "p2":
        return OUT_ROOT / "p2_oracle_state_all" / job["seq"]
    if job["protocol"] == "p3":
        return OUT_ROOT / "p3_continuous_id_miss" / job["seq"]
    return OUT_ROOT / f"p4_budget_b{job['budget']}" / job["seq"]


def run_job(job: Dict, gpu: int) -> Dict:
    outdir = out_dir_for(job)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(outdir) + ".run.log")
    env = dict(os.environ)
    env.update(
        {
            "N5_PROTOCOL": job["protocol"],
            "N5_BUDGET": str(job["budget"]),
            "N5_SEQ": job["seq"],
            "N5_OUT_DIR": str(outdir),
            "N5_SKIP_TRACKEVAL": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(ROOT),
        }
    )
    if FRAME_LIMIT:
        env["N5_FRAMES"] = FRAME_LIMIT
    start = time.time()
    retry_count = 0
    exit_code = -1
    while True:
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                [str(PY), str(ROOT / "scripts/run_n5_continuous_observer.py")],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        exit_code = proc.returncode
        if exit_code == 0 or retry_count >= 1:
            break
        retry_count += 1
        time.sleep(2)
    return {
        "job_id": f"{job['protocol']}_b{job['budget']}_{job['seq']}",
        "protocol": job["protocol"],
        "budget": job["budget"],
        "sequence": job["seq"],
        "physical_gpu": gpu,
        "logical_gpu": gpu,
        "output_dir": str(outdir),
        "start_time": start,
        "end_time": time.time(),
        "wall_seconds": time.time() - start,
        "exit_code": exit_code,
        "retry_count": retry_count,
    }


def gpu_usage() -> Dict[int, int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    usage: Dict[int, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) == 2:
            try:
                usage[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                continue
    return usage


def main() -> None:
    jobs = build_jobs()
    status: Dict[str, Dict] = {}
    lock = threading.Lock()
    in_use = set()
    job_queue: "queue.Queue[Dict]" = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    def pick_gpu() -> int:
        with lock:
            usage = gpu_usage()
            free = [
                g
                for g in GPUS
                if g not in in_use and usage.get(g, 10 ** 9) <= 100
            ]
            if free:
                gpu = free[0]
            else:
                candidates = sorted(
                    [g for g in GPUS if g not in in_use],
                    key=lambda g: usage.get(g, 10 ** 9),
                )
                if not candidates:
                    raise RuntimeError("no GPU available for a new job")
                gpu = candidates[0]
            in_use.add(gpu)
            return gpu

    def worker() -> None:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            gpu = pick_gpu()
            try:
                result = run_job(job, gpu)
            finally:
                with lock:
                    in_use.discard(gpu)
            with lock:
                status[result["job_id"]] = result
                print(
                    json.dumps(result, ensure_ascii=False),
                    flush=True,
                )
            job_queue.task_done()

    started = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    manifest_dir = OUT_ROOT
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(status.values(), key=lambda r: r["job_id"])
    with (manifest_dir / "job_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (manifest_dir / "job_status.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["job_id", "sequence", "protocol", "budget", "exit_code", "retry_count"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "job_id": r["job_id"],
                    "sequence": r["sequence"],
                    "protocol": r["protocol"],
                    "budget": r["budget"],
                    "exit_code": r["exit_code"],
                    "retry_count": r["retry_count"],
                }
            )
    with (manifest_dir / "gpu_allocation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["job_id", "physical_gpu", "logical_gpu", "wall_seconds"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "job_id": r["job_id"],
                    "physical_gpu": r["physical_gpu"],
                    "logical_gpu": r["logical_gpu"],
                    "wall_seconds": round(r["wall_seconds"], 2),
                }
            )
    with (manifest_dir / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "start_time", "end_time", "wall_seconds"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "job_id": r["job_id"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "wall_seconds": round(r["wall_seconds"], 2),
                }
            )
    (manifest_dir / "process_cleanup.txt").write_text(
        "N5 orchestrator completed; all worker subprocesses exited.\n", encoding="utf-8"
    )
    total_wall = time.time() - started
    summary = {
        "tag": TAG,
        "jobs_total": len(rows),
        "jobs_ok": sum(1 for r in rows if r["exit_code"] == 0),
        "jobs_failed": [r["job_id"] for r in rows if r["exit_code"] != 0],
        "total_wall_seconds": total_wall,
        "gpu_time_seconds": round(sum(r["wall_seconds"] for r in rows), 2),
    }
    (manifest_dir / f"orchestrator_{TAG}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["jobs_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
