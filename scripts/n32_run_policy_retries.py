#!/usr/bin/env python3
"""Schedule N32 policy-level retries across at most four GPUs.

Each scheduled item is executed by a separate Python process through
``n32_retry_one_policy.py``.  Thus one process owns exactly one
``(episode_id, policy)`` and is gone before the next item on that GPU starts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/n32/policy_rollouts/retry_manifest.json"
RETRY_DIR = ROOT / "outputs/n32/policy_rollouts/policy_retries"
LOG_DIR = ROOT / "outputs/n32/logs"
CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
PYTHON = Path("python")
ITEM_SCRIPT = ROOT / "scripts/n32_retry_one_policy.py"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_path(output_dir: Path, item: dict[str, Any]) -> Path:
    import hashlib

    digest = hashlib.sha256(f"{item['episode_id']}|{item['policy']}".encode("utf-8")).hexdigest()
    return output_dir / f"{digest}.json"


def _artifact_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "PASS" and payload.get("strict_complete") is True


def _run_shard(
    *,
    gpu: str,
    item_indices: list[int],
    manifest_path: Path,
    checkpoint: Path,
    output_dir: Path,
    attempt: int,
    force: bool,
) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"policy_retry_gpu_{gpu}.log"
    completed = 0
    skipped = 0
    failed = 0
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        for position, item_index in enumerate(item_indices, 1):
            # Re-read only the small manifest in the supervisor; the child
            # reads the frozen item itself.  Existing PASS artifacts are never
            # overwritten unless --force is explicitly requested.
            item = _ITEMS[item_index]
            artifact = _artifact_path(output_dir, item)
            if not force and _artifact_is_complete(artifact):
                skipped += 1
                print(f"N32_RETRY gpu={gpu} {position}/{len(item_indices)} {item['episode_id']} {item['policy']} SKIP_PASS", flush=True)
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["PYTHONPATH"] = f"{ROOT / 'third_party/sam3'}:{ROOT}:{env.get('PYTHONPATH', '')}"
            command = [
                str(PYTHON),
                str(ITEM_SCRIPT),
                "--manifest", str(manifest_path),
                "--checkpoint", str(checkpoint),
                "--output-dir", str(output_dir),
                "--item-index", str(item_index),
                "--attempt", str(attempt),
            ]
            log.write(f"\nN32_RETRY_START gpu={gpu} position={position}/{len(item_indices)} item_index={item_index} episode={item['episode_id']} policy={item['policy']}\n")
            log.flush()
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode == 0 and _artifact_is_complete(artifact):
                completed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"
            print(f"N32_RETRY gpu={gpu} {position}/{len(item_indices)} {item['episode_id']} {item['policy']} {status}", flush=True)
    return {
        "gpu": str(gpu),
        "item_count": len(item_indices),
        "completed_pass": completed,
        "skipped_existing_pass": skipped,
        "failed_or_incomplete": failed,
        "log": str(log_path),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


_ITEMS: list[dict[str, Any]] = []


def run(
    *,
    manifest_path: Path = MANIFEST,
    checkpoint: Path = CHECKPOINT,
    output_dir: Path = RETRY_DIR,
    gpu_ids: list[str] | None = None,
    max_items: int | None = None,
    attempt: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    global _ITEMS
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("retry_items_unique") is not True:
        raise ValueError("retry manifest is not a unique frozen PASS artifact")
    _ITEMS = list(payload.get("items", []))
    if max_items is not None:
        indices = list(range(min(int(max_items), len(_ITEMS))))
    else:
        indices = list(range(len(_ITEMS)))
    if gpu_ids is None:
        gpu_ids = ["0", "1", "2", "3"]
    gpu_ids = [str(gpu) for gpu in gpu_ids[:4]]
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    shards = [indices[offset::len(gpu_ids)] for offset in range(len(gpu_ids))]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(
                _run_shard,
                gpu=gpu,
                item_indices=shard,
                manifest_path=manifest_path,
                checkpoint=checkpoint,
                output_dir=output_dir,
                attempt=attempt,
                force=force,
            )
            for gpu, shard in zip(gpu_ids, shards)
            if shard
        ]
        for future in as_completed(futures):
            results.append(future.result())
    summary = {
        "protocol": "N32-C-POLICY-LEVEL-RETRY-SUPERVISOR",
        "status": "PASS" if all(item["failed_or_incomplete"] == 0 for item in results) else "PARTIAL",
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "gpu_ids": gpu_ids,
        "attempt": int(attempt),
        "requested_item_count": len(indices),
        "completed_pass": sum(item["completed_pass"] for item in results),
        "skipped_existing_pass": sum(item["skipped_existing_pass"] for item in results),
        "failed_or_incomplete": sum(item["failed_or_incomplete"] for item in results),
        "shards": sorted(results, key=lambda item: item["gpu"]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
    }
    _write(output_dir / f"retry_supervisor_attempt_{int(attempt)}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=RETRY_DIR)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        gpu_ids=[item for item in args.gpus.split(",") if item.strip()],
        max_items=args.max_items,
        attempt=args.attempt,
        force=args.force,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
