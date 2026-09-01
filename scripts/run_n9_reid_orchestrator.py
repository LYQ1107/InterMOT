"""Extract ReID features for a sequence set across idle GPUs (blocking)."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(".")
PY = "python"
SPLIT = os.environ.get("N9_SPLIT", "val")
GPUS = [int(x) for x in os.environ.get("N9_GPUS", "0 1 2 3").split()]
SEQS = sorted(os.environ.get("N9_SEQS", "").split())
FEAT = ROOT / "outputs/n9/features"


def run_job(seq, gpu):
    env = dict(os.environ)
    env.update(
        {
            "N9_SEQ": seq,
            "N9_SPLIT": SPLIT,
            "N9_FEAT_DIR": str(FEAT),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": str(ROOT),
        }
    )
    if SPLIT == "train":
        env["N9_P0_DIR"] = str(ROOT / "outputs/n9/p0_train")
    else:
        env["N9_P0_DIR"] = str(ROOT / "outputs/n5/integrity/canonical_mot_results/b0")
    log = Path(f"/tmp/n9_reid_{seq}.log")
    start = time.time()
    for attempt in range(2):
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                [PY, str(ROOT / "scripts/run_n9_reid_features.py")],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode == 0:
            break
        time.sleep(3)
    return {"sequence": seq, "gpu": gpu, "exit_code": proc.returncode, "wall": round(time.time() - start, 2)}


def main():
    if not SEQS:
        split_dir = Path("/path/to/dancetrack") / SPLIT
        SEQS.extend(sorted(p.name for p in split_dir.iterdir() if p.is_dir()))
    jobs = [s for s in SEQS if not (FEAT / f"{s}.npz").exists()]
    print(json.dumps({"split": SPLIT, "to_run": len(jobs), "gpus": GPUS}), flush=True)
    q: "queue.Queue[str]" = queue.Queue()
    for s in jobs:
        q.put(s)
    results = {}
    lock = threading.Lock()

    def worker(gpu):
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

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS[: max(1, len(jobs))]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [s for s, r in results.items() if r["exit_code"] != 0]
    summary = {"jobs": len(results), "ok": len(results) - len(failed), "failed": failed}
    (FEAT / "orchestrator.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
