#!/usr/bin/env python
"""N7 parallel orchestrator for Route A real-SAM jobs.

One blocking process: builds remaining jobs, launches them on idle GPUs
(one SAM job per GPU), waits for all, retries each environment failure once,
and writes job_manifest / gpu allocation / runtime summaries.
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
SEQS = os.environ.get(
    "N7_SEQS", "dancetrack0004 dancetrack0005 dancetrack0007"
).split()
GPUS = [int(x) for x in os.environ.get("N7_GPUS", "0 1 8 9").split()]
BUDGETS = [int(x) for x in os.environ.get("N7_BUDGETS", "0 1 2 4 8").split()]
OUT_ROOT = Path(os.environ.get("N7_OUT_ROOT", ROOT / "outputs/n7"))
SEGMENT_LEN = os.environ.get("N7_SEGMENT_LEN", "30")
FRAME_LIMIT = os.environ.get("N7_FRAMES", "")
TAG = os.environ.get("N7_TAG", "three_seq")


def build_jobs() -> List[Dict]:
    jobs = []
    for b in BUDGETS:
        for seq in SEQS:
            jobs.append({"budget": b, "seq": seq})
    return jobs


def out_dir_for(job: Dict) -> Path:
    return OUT_ROOT / "real" / f"route_a_b{job['budget']}" / job["seq"]


def gpu_usage() -> Dict[int, int]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    usage = {}
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) == 2:
            try:
                usage[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                continue
    return usage


def run_job(job: Dict, gpu: int) -> Dict:
    outdir = out_dir_for(job)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(outdir) + ".run.log")
    env = dict(os.environ)
    env.update(
        {
            "N7_PROTOCOL": "p4",
            "N7_BUDGET": str(job["budget"]),
            "N7_SEQ": job["seq"],
            "N7_OUT_DIR": str(outdir),
            "N7_SEGMENT_LEN": SEGMENT_LEN,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(ROOT),
        }
    )
    if FRAME_LIMIT:
        env["N7_FRAMES"] = FRAME_LIMIT
    start = time.time()
    retry = 0
    rc = -1
    while True:
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                [str(PY), str(ROOT / "scripts/run_n7_real.py")],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        rc = proc.returncode
        if rc == 0 or retry >= 1:
            break
        retry += 1
        time.sleep(2)
    return {
        "job_id": f"route_a_b{job['budget']}_{job['seq']}",
        "route": "A",
        "budget": job["budget"],
        "sequence": job["seq"],
        "physical_gpu": gpu,
        "output_dir": str(outdir),
        "start_time": start,
        "end_time": time.time(),
        "wall_seconds": time.time() - start,
        "exit_code": rc,
        "retry_count": retry,
    }


def main() -> None:
    jobs = [j for j in build_jobs() if not (out_dir_for(j) / "summary.json").exists()]
    skipped = len(build_jobs()) - len(jobs)
    print(json.dumps({"skipped_completed": skipped, "jobs_to_run": len(jobs)}), flush=True)
    q: "queue.Queue[Dict]" = queue.Queue()
    for j in jobs:
        q.put(j)
    status: Dict[str, Dict] = {}
    lock = threading.Lock()
    in_use = set()

    def pick_gpu() -> int:
        with lock:
            usage = gpu_usage()
            free = [g for g in GPUS if g not in in_use and usage.get(g, 10**9) <= 100]
            if free:
                gpu = free[0]
            else:
                cand = sorted(
                    [g for g in GPUS if g not in in_use],
                    key=lambda g: usage.get(g, 10**9),
                )
                if not cand:
                    raise RuntimeError("no GPU available")
                gpu = cand[0]
            in_use.add(gpu)
            return gpu

    def worker() -> None:
        while True:
            try:
                job = q.get_nowait()
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
                print(json.dumps(result, ensure_ascii=False), flush=True)
            q.task_done()

    started = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = sorted(status.values(), key=lambda r: r["job_id"])
    out_real = OUT_ROOT / "real"
    with (out_real / "job_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    failed = [r["job_id"] for r in rows if r["exit_code"] != 0]
    summary = {
        "tag": TAG,
        "jobs_total": len(rows),
        "jobs_ok": sum(1 for r in rows if r["exit_code"] == 0),
        "jobs_failed": failed,
        "total_wall_seconds": time.time() - started,
        "gpu_time_seconds": round(sum(r["wall_seconds"] for r in rows), 2),
    }
    (out_real / f"orchestrator_{TAG}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
