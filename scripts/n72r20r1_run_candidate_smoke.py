#!/usr/bin/env python3
"""Run the N72R20R1 two-sequence train-only smoke, one child per sequence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "n72r20r1_candidate_stream_smoke.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", help="comma-separated; defaults to the first two frozen train protocol sequences")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20R1/asset_manifest.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "outputs/N72R20R1/protocol.json").read_text(encoding="utf-8"))
    sequences = (
        [item.strip() for item in args.sequences.split(",") if item.strip()]
        if args.sequences
        else list(protocol["development_sequences"])
    )
    if len(sequences) != 2:
        raise SystemExit(f"N72R20R1 smoke requires exactly two train sequences, got {sequences}")
    output_env = os.environ.get("N72R20R1_ASSET_ROOT")
    if args.output_root is None and not output_env:
        raise SystemExit("provide --output-root or set N72R20R1_ASSET_ROOT")
    output_root = (args.output_root or Path(output_env)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = ROOT / "outputs/N72R20R1/smoke_logs"
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
            "train",
            "--max-frames",
            str(int(args.max_frames)),
            "--asset-manifest",
            str(args.asset_manifest),
            "--output-root",
            str(output_root),
            "--device",
            str(args.device),
        ]
        for flag, value in (
            ("--dataset-root", args.dataset_root),
            ("--checkpoint", args.checkpoint),
            ("--osnet-checkpoint", args.osnet_checkpoint),
        ):
            if value is not None:
                command.extend([flag, str(value)])
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
        try:
            result["profile"] = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            result["profile"] = None
        results.append(result)
    profiles = [item["profile"] for item in results if isinstance(item.get("profile"), dict)]
    free_bytes = shutil.disk_usage(output_root).free
    profile = {
        "stage": "N72R20R1",
        "status": "PASS_N72R20R1_TWO_SEQUENCE_SMOKE" if len(profiles) == 2 else "FAIL_N72R20R1_TWO_SEQUENCE_SMOKE",
        "sequences": sequences,
        "completed_profiles": profiles,
        "child_results": results,
        "frames": sum(int(item["frames"]) for item in profiles),
        "candidates": sum(int(item["candidates"]) for item in profiles),
        "bytes": sum(int(item["total_bytes"]) for item in profiles),
        "bytes_per_frame": (
            sum(int(item["total_bytes"]) for item in profiles) / sum(int(item["frames"]) for item in profiles)
            if profiles and sum(int(item["frames"]) for item in profiles)
            else None
        ),
        "bytes_per_candidate": (
            sum(int(item["total_bytes"]) for item in profiles) / sum(int(item["candidates"]) for item in profiles)
            if profiles and sum(int(item["candidates"]) for item in profiles)
            else None
        ),
        "projected_10000_frames_bytes": (
            sum(int(item["total_bytes"]) for item in profiles)
            / sum(int(item["frames"]) for item in profiles)
            * 10000
            if profiles
            else None
        ),
        "free_space_bytes_after_smoke": free_bytes,
        "free_space_gib_after_smoke": free_bytes / (1024**3),
        "hard_stop_triggered": free_bytes < 80 * (1024**3),
        "warning_below_normal_target": free_bytes < 100 * (1024**3),
        "projected_full_val_cache_allowed": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "asset_manifest": str(args.asset_manifest),
        "dataset_root": manifest.get("DANCETRACK_ROOT"),
    }
    output = ROOT / "outputs/N72R20R1/candidate_storage_profile.json"
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": profile["status"], "output": str(output), "free_space_gib": profile["free_space_gib_after_smoke"]}, ensure_ascii=False))
    return 0 if profile["status"] == "PASS_N72R20R1_TWO_SEQUENCE_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
