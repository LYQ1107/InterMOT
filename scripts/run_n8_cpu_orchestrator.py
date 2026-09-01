#!/usr/bin/env python
"""N8 CPU orchestrator: chronological verified-error interaction on P0."""

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
PY = "python"
SEQS = sorted(os.environ.get("N8_SEQS", "").split()) or sorted(
    p.name
    for p in Path("/path/to/dancetrack/val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
BUDGETS = [int(x) for x in os.environ.get("N8_BUDGETS", "0 1 2 4 8 -1").split()]
OUT_ROOT = Path(os.environ.get("N8_OUT_ROOT", ROOT / "outputs/n8"))
WORKERS = int(os.environ.get("N8_CPU_WORKERS", str(os.cpu_count() or 8)))
TAG = os.environ.get("N8_TAG", "cpu")
FRAME_LIMIT = int(os.environ.get("N8_FRAMES", "0")) or None


def budget_name(b: int) -> str:
    return "unlimited" if b < 0 else f"b{b}"


def build_jobs() -> List[Dict]:
    return [{"budget": b, "seq": s} for b in BUDGETS for s in SEQS]


def out_dir_for(job: Dict) -> Path:
    return OUT_ROOT / "real" / f"route_a_{budget_name(job['budget'])}" / job["seq"]


def run_job(job: Dict) -> Dict:
    outdir = out_dir_for(job)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(outdir) + ".run.log")
    env = dict(os.environ)
    env.update(
        {
            "N8_BUDGET": str(job["budget"]),
            "N8_SEQ": job["seq"],
            "N8_OUT_DIR": str(outdir),
            "PYTHONPATH": str(ROOT),
        }
    )
    if FRAME_LIMIT is not None:
        env["N8_FRAMES"] = str(FRAME_LIMIT)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [PY, str(ROOT / "scripts/run_n8_real.py")],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return {
        "job_id": f"route_a_{budget_name(job['budget'])}_{job['seq']}",
        "route": "A-CPU",
        "budget": job["budget"],
        "sequence": job["seq"],
        "output_dir": str(outdir),
        "start_time": start,
        "end_time": time.time(),
        "wall_seconds": round(time.time() - start, 3),
        "exit_code": proc.returncode,
        "retry_count": 0,
    }


def main() -> None:
    jobs = [
        j
        for j in build_jobs()
        if not (out_dir_for(j) / "summary.json").exists()
    ]
    print(
        json.dumps(
            {
                "tag": TAG,
                "sequences": len(SEQS),
                "budgets": BUDGETS,
                "jobs_total": len(build_jobs()),
                "jobs_to_run": len(jobs),
            }
        ),
        flush=True,
    )
    q: "queue.Queue[Dict]" = queue.Queue()
    for j in jobs:
        q.put(j)
    status: Dict[str, Dict] = {}
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            result = run_job(job)
            with lock:
                status[result["job_id"]] = result
                print(json.dumps(result, ensure_ascii=False), flush=True)
            q.task_done()

    started = time.time()
    threads = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(min(WORKERS, len(jobs) or 1))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = sorted(status.values(), key=lambda r: r["job_id"])
    with (OUT_ROOT / "job_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    with (OUT_ROOT / "job_status.csv").open("w", newline="", encoding="utf-8") as f:
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
        "total_wall_seconds": round(time.time() - started, 2),
        "cpu_time_seconds": round(sum(r["wall_seconds"] for r in rows), 2),
    }
    with (OUT_ROOT / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    (OUT_ROOT / "orchestrator.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT_ROOT / "process_cleanup.txt").write_text(
        "No GPU processes were spawned. All N8 jobs are CPU-only and joined "
        "before this orchestrator returned.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
