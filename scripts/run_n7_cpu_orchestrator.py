#!/usr/bin/env python
"""N7 CPU orchestrator: Route A on the frozen P0 backbone (identity-only).

The real-SAM reset/rehydration path is documented as
FAIL_BASELINE_PRESERVATION on full-length sequences with the pinned SAM 3.1
API (see docs/N7_SPARSE_BUDGET_ROOT_CAUSE.md).  The compliant route that
satisfies ZERO_INTERACTION_EQUIVALENCE exactly is the identity-layer sparse
interaction on the frozen P0 canonical backbone (N6BackboneObserver p4).
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
PY = "python"
SEQS = os.environ.get(
    "N7_SEQS", "dancetrack0004 dancetrack0005 dancetrack0007"
).split()
BUDGETS = [int(x) for x in os.environ.get("N7_BUDGETS", "1 2 4 8").split()]
OUT_ROOT = Path(os.environ.get("N7_OUT_ROOT", ROOT / "outputs/n7"))
WORKERS = int(os.environ.get("N7_CPU_WORKERS", str(os.cpu_count() or 8)))
TAG = os.environ.get("N7_TAG", "three_seq_cpu")


def build_jobs() -> List[Dict]:
    return [{"budget": b, "seq": s} for b in BUDGETS for s in SEQS]


def out_dir_for(job: Dict) -> Path:
    return OUT_ROOT / "real_cpu" / f"route_a_b{job['budget']}" / job["seq"]


def run_job(job: Dict) -> Dict:
    outdir = out_dir_for(job)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(outdir) + ".run.log")
    env = dict(os.environ)
    env.update(
        {
            "N6_PROTOCOL": "p4",
            "N6_BUDGET": str(job["budget"]),
            "N6_SEQ": job["seq"],
            "N6_OUT_DIR": str(outdir),
            "PYTHONPATH": str(ROOT),
        }
    )
    start = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [PY, str(ROOT / "scripts/run_n6_backbone.py")],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return {
        "job_id": f"route_a_b{job['budget']}_{job['seq']}",
        "route": "A-CPU",
        "budget": job["budget"],
        "sequence": job["seq"],
        "output_dir": str(outdir),
        "start_time": start,
        "end_time": time.time(),
        "wall_seconds": time.time() - start,
        "exit_code": proc.returncode,
        "retry_count": 0,
    }


def main() -> None:
    jobs = [j for j in build_jobs() if not (out_dir_for(j) / "summary.json").exists()]
    print(json.dumps({"skipped_completed": len(build_jobs()) - len(jobs), "jobs_to_run": len(jobs)}), flush=True)
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
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(WORKERS, len(jobs) or 1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = sorted(status.values(), key=lambda r: r["job_id"])
    out_real = OUT_ROOT / "real_cpu"
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
        "cpu_time_seconds": round(sum(r["wall_seconds"] for r in rows), 2),
    }
    (out_real / f"orchestrator_{TAG}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
