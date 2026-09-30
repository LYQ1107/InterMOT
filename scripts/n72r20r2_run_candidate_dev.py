#!/usr/bin/env python3
"""Run the eight R2 train candidate workers on explicitly idle GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = (
    "dancetrack0001",
    "dancetrack0002",
    "dancetrack0023",
    "dancetrack0024",
    "dancetrack0039",
    "dancetrack0057",
    "dancetrack0062",
    "dancetrack0072",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20R2/asset_manifest.json")
    parser.add_argument("--output-root", type=Path, default=Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets"))
    parser.add_argument("--log-root", type=Path, default=ROOT / "outputs/N72R20R2/candidate_logs")
    args = parser.parse_args()

    args.log_root.mkdir(parents=True, exist_ok=True)
    command_base = [
        sys.executable,
        str(ROOT / "scripts/n72r20r2_candidate_stream.py"),
        "--asset-manifest",
        str(args.asset_manifest.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--device",
        "cuda:0",
    ]
    pending = list(args.sequences)
    running: dict[str, tuple[subprocess.Popen[str], int, object, float]] = {}
    available_gpus = list(dict.fromkeys(int(gpu) for gpu in args.gpus))
    results: list[dict[str, object]] = []
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(ROOT) + os.pathsep + base_env.get("PYTHONPATH", "")

    while pending or running:
        while pending and available_gpus:
            sequence = pending.pop(0)
            gpu = available_gpus.pop(0)
            log_path = args.log_root / f"{sequence}.gpu{gpu}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            started = time.time()
            process = subprocess.Popen(
                command_base + ["--sequence", sequence],
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[sequence] = (process, gpu, handle, started)
            print(json.dumps({"event": "started", "sequence": sequence, "gpu": gpu, "pid": process.pid, "log": str(log_path)}, sort_keys=True), flush=True)

        finished: list[str] = []
        for sequence, (process, gpu, handle, started) in running.items():
            code = process.poll()
            if code is None:
                continue
            handle.close()
            elapsed = time.time() - started
            result = {"sequence": sequence, "gpu": gpu, "returncode": code, "elapsed_sec": elapsed}
            results.append(result)
            print(json.dumps({"event": "finished", **result}, sort_keys=True), flush=True)
            finished.append(sequence)
        for sequence in finished:
            # A GPU is returned only with its exact child, never by current
            # running-count modulo.  This prevents a newly queued sequence
            # from overlapping a still-running child on the same GPU when a
            # different GPU finishes first.
            available_gpus.append(int(running[sequence][1]))
            del running[sequence]
        available_gpus.sort()
        if running:
            time.sleep(5)

    failed = [item for item in results if int(item["returncode"]) != 0]
    summary = {
        "stage": "N72R20R2",
        "status": "PASS_N72R20R2_CANDIDATE_DEV" if not failed else "FAIL_N72R20R2_CANDIDATE_DEV",
        "sequences": list(args.sequences),
        "gpus": list(args.gpus),
        "results": sorted(results, key=lambda item: str(item["sequence"])),
        "runtime_future_gt_used": False,
    }
    output = ROOT / "outputs/N72R20R2/candidate_dev_run.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "failed": len(failed), "output": str(output)}, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
