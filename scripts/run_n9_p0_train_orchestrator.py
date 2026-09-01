#!/usr/bin/env python
"""N9 P0-on-train orchestrator: one blocking process, GPU workers 8/9."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(".")
PY = ROOT / "envs/sam3_intermot/bin/python"
GPUS = [int(x) for x in os.environ.get("N9_GPUS", "8 9").split()]
OUT = ROOT / "outputs/n9/p0_train"
TRAIN = Path("/path/to/dancetrack/train")
SEQS = sorted(
    p.name for p in TRAIN.iterdir() if p.is_dir() and (p / "img1").is_dir()
)


def run_job(seq: str, gpu: int) -> dict:
    outdir = OUT / seq
    outdir.parent.mkdir(parents=True, exist_ok=True)
    log = Path(str(outdir) + ".run.log")
    env = dict(os.environ)
    env.update(
        {
            "N9_SEQ": seq,
            "N9_OUT_DIR": str(OUT),
            "N9_SPLIT": "train",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(ROOT),
        }
    )
    start = time.time()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [str(PY), str(ROOT / "scripts/run_n9_p0_train.py")],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return {
        "sequence": seq,
        "gpu": gpu,
        "exit_code": proc.returncode,
        "wall_seconds": round(time.time() - start, 2),
        "log": str(log),
    }


def main() -> None:
    jobs = [
        s
        for s in SEQS
        if not (OUT / f"{s}.txt").exists() or not (OUT / f"{s}.summary.json").exists()
    ]
    print(json.dumps({"total": len(SEQS), "to_run": len(jobs), "gpus": GPUS}), flush=True)
    q: "queue.Queue[str]" = queue.Queue()
    for s in jobs:
        q.put(s)
    results = {}
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                s = q.get_nowait()
            except queue.Empty:
                return
            r = run_job(s, gpu)
            with lock:
                results[s] = r
                print(json.dumps(r, ensure_ascii=False), flush=True)
            q.task_done()

    started = time.time()
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS[: len(jobs) or 1]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [s for s, r in results.items() if r["exit_code"] != 0]
    summary = {
        "jobs_total": len(results),
        "jobs_ok": sum(1 for r in results.values() if r["exit_code"] == 0),
        "jobs_failed": failed,
        "total_wall_seconds": round(time.time() - started, 2),
        "gpu_time_seconds": round(sum(r["wall_seconds"] for r in results.values()), 2),
    }
    (OUT / "orchestrator.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
