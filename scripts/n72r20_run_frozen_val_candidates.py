#!/usr/bin/env python3
"""Generate the frozen DanceTrack-val SAM3 candidate cache.

This is intentionally a separate command from the two-sequence train smoke.
It requires a passed smoke/storage gate, explicit confirmation, all 25 frozen
val sequences, and full-sequence caches.  It never reads GT in the worker.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "n72r20_candidate_stream_smoke.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-frozen-val", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20/asset_manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R20/protocol.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.run_frozen_val:
        raise SystemExit("refusing val generation without explicit --run-frozen-val")
    manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if manifest.get("stage") != "N72R20" or protocol.get("stage") != "N72R20":
        raise SystemExit("asset manifest and protocol must both be N72R20")
    sequences = list(protocol.get("val_evaluation_sequences", []))
    if len(sequences) != 25 or len(set(sequences)) != 25:
        raise SystemExit(f"frozen val protocol must contain exactly 25 unique sequences, got {len(sequences)}")
    checkpoint = (args.checkpoint or Path(manifest.get("sam3_checkpoint", {}).get("path", ""))).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing authenticated SAM3 checkpoint: {checkpoint}")

    gate_path = ROOT / "outputs/N72R20/candidate_storage_profile.json"
    if not gate_path.is_file():
        raise SystemExit(f"missing passed train smoke/storage gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_N72R20_TWO_SEQUENCE_SMOKE":
        raise SystemExit(f"train smoke gate is not passed: {gate.get('status')}")
    projected_bytes = gate.get("projected_full_val_cache_bytes")
    if projected_bytes is None:
        raise SystemExit("train smoke did not produce a full-val storage projection")
    if int(projected_bytes) > 25 * (1024**3):
        raise SystemExit(
            "projected full val cache exceeds the 25 GiB cache limit; use a bounded streaming evaluator instead"
        )
    if not gate.get("projected_full_val_cache_allowed", False):
        raise SystemExit("storage gate does not permit a full val candidate cache with the 100 GiB reserve")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = ROOT / "outputs/N72R20/frozen_val_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    for sequence in sequences:
        command = [
            sys.executable,
            str(WORKER),
            "--sequence",
            sequence,
            "--split",
            "val",
            "--max-frames",
            "0",
            "--frozen-eval",
            "--asset-manifest",
            str(args.asset_manifest),
            "--output-root",
            str(output_root),
            "--checkpoint",
            str(checkpoint),
            "--device",
            str(args.device),
        ]
        if args.dataset_root is not None:
            command.extend(["--dataset-root", str(args.dataset_root)])
        if args.osnet_checkpoint is not None:
            command.extend(["--osnet-checkpoint", str(args.osnet_checkpoint)])
        completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, check=False)
        log_path = log_root / f"{sequence}.log"
        log_path.write_text(
            f"command={json.dumps(command, ensure_ascii=False)}\nreturncode={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n",
            encoding="utf-8",
        )
        result: dict[str, Any] = {
            "sequence": sequence,
            "returncode": completed.returncode,
            "log": str(log_path),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if completed.returncode != 0:
            results.append(result)
            break
        results.append(result)

    free_bytes = shutil.disk_usage(output_root).free
    report = {
        "stage": "N72R20",
        "status": "PASS_N72R20_FROZEN_VAL_CANDIDATES" if len(results) == len(sequences) else "FAIL_N72R20_FROZEN_VAL_CANDIDATES",
        "sequences": sequences,
        "completed_sequence_count": len(results),
        "results": results,
        "free_space_bytes_after_run": free_bytes,
        "free_space_gib_after_run": free_bytes / (1024**3),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_root": str(output_root),
        "asset_manifest": str(args.asset_manifest.expanduser().resolve()),
        "protocol": str(args.protocol.expanduser().resolve()),
    }
    report_path = ROOT / "outputs/N72R20/frozen_val_candidate_run.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(report_path), "completed": len(results)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS_N72R20_FROZEN_VAL_CANDIDATES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
