#!/usr/bin/env python
"""N9 CPU orchestrator: variants x budgets x sequences on val P0."""

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
SEQS = sorted(os.environ.get("N9_SEQS", "").split()) or sorted(
    p.name
    for p in Path("/path/to/dancetrack/val").iterdir()
    if p.is_dir() and (p / "gt" / "gt.txt").is_file()
)
BUDGETS = [int(x) for x in os.environ.get("N9_BUDGETS", "0 1 2 4 8").split()]
VARIANTS = os.environ.get("N9_VARIANTS", "reid pairwise auto proposed").split()
OUT_ROOT = Path(os.environ.get("N9_OUT_ROOT", ROOT / "outputs/n9"))
WORKERS = int(os.environ.get("N9_CPU_WORKERS", str(os.cpu_count() or 8)))
TAG = os.environ.get("N9_TAG", "cpu")


def build_jobs() -> List[Dict]:
    return [
        {"variant": v, "budget": b, "seq": s}
        for v in VARIANTS
        for b in BUDGETS
        for s in SEQS
    ]


def out_dir_for(job: Dict) -> Path:
    return OUT_ROOT / "real" / f"{job['variant']}_b{job['budget']}" / job["seq"]


def run_job(job: Dict) -> Dict:
    outdir = out_dir_for(job)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(outdir) + ".run.log")
    env = dict(os.environ)
    env.update(
        {
            "N9_SEQ": job["seq"],
            "N9_BUDGET": str(job["budget"]),
            "N9_VARIANT": job["variant"],
            "N9_OUT_DIR": str(outdir),
            "PYTHONPATH": str(ROOT),
        }
    )
    model = ROOT / "outputs/n9/models"
    if job["variant"] == "pairwise":
        env["N9_MODEL_PATH"] = str(model / "pairwise_mlp.pt")
    elif job["variant"] == "auto":
        env["N9_MODEL_PATH"] = str(model / "set_associator.pt")
    elif job["variant"] == "proposed":
        env["N9_MODEL_PATH"] = str(model / "hcpim.pt")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [PY, str(ROOT / "scripts/run_n9_real.py")],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return {
        "job_id": f"{job['variant']}_b{job['budget']}_{job['seq']}",
        "variant": job["variant"],
        "budget": job["budget"],
        "sequence": job["seq"],
        "output_dir": str(outdir),
        "wall_seconds": round(time.time() - start, 3),
        "exit_code": proc.returncode,
    }


def main() -> None:
    jobs = [j for j in build_jobs() if not (out_dir_for(j) / "summary.json").exists()]
    print(
        json.dumps(
            {
                "tag": TAG,
                "variants": VARIANTS,
                "budgets": BUDGETS,
                "sequences": len(SEQS),
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
            r = run_job(job)
            with lock:
                status[r["job_id"]] = r
                print(json.dumps(r, ensure_ascii=False), flush=True)
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
    with (OUT_ROOT / "n9_job_manifest.csv").open("w", newline="", encoding="utf-8") as f:
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
    (OUT_ROOT / "n9_orchestrator.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
