#!/usr/bin/env python3
"""Run frozen val one sequence at a time when a full cache is too large."""

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
EVALUATOR = ROOT / "scripts" / "n72r20_identity_bridge_eval.py"
AGGREGATOR = ROOT / "scripts" / "n72r20_aggregate_identity_bridge.py"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_threshold(path: Path) -> float:
    payload = load_json(path)
    if payload.get("stage") != "N72R20" or payload.get("split") != "train" or payload.get("frozen") is not True:
        raise ValueError("threshold manifest must be a frozen N72R20 train manifest")
    return float(payload["confirmation_margin_threshold"])


def run_command(command: list[str], log_path: Path) -> int:
    completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, check=False)
    log_path.write_text(
        f"command={json.dumps(command, ensure_ascii=False)}\nreturncode={completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-streaming-val", action="store_true")
    parser.add_argument("--delete-sequence-cache-after-evaluation", action="store_true")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--confirmation-threshold-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20/asset_manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R20/protocol.json")
    parser.add_argument("--anchor-device", default="cpu")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.run_streaming_val:
        raise SystemExit("refusing streaming val without explicit --run-streaming-val")
    if not args.delete_sequence_cache_after_evaluation:
        raise SystemExit("streaming val requires explicit --delete-sequence-cache-after-evaluation")
    manifest = load_json(args.asset_manifest)
    protocol = load_json(args.protocol)
    if manifest.get("stage") != "N72R20" or protocol.get("stage") != "N72R20":
        raise SystemExit("asset manifest and protocol must both be N72R20")
    sequences = list(protocol.get("val_evaluation_sequences", []))
    if len(sequences) != 25 or len(set(sequences)) != 25:
        raise SystemExit("frozen val protocol must contain exactly 25 unique sequences")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing SAM3 checkpoint: {checkpoint}")
    threshold = load_threshold(args.confirmation_threshold_manifest.expanduser().resolve())
    candidate_root = args.candidate_root.expanduser().resolve()
    records_dir = args.records_dir.expanduser().resolve()
    records_dir.mkdir(parents=True, exist_ok=True)
    log_dir = records_dir / "stream_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    for sequence in sequences:
        candidate_dir = candidate_root / "candidates" / sequence
        if candidate_dir.exists():
            raise SystemExit(f"refusing to reuse or delete pre-existing candidate directory: {candidate_dir}")
        worker_command = [
            sys.executable,
            str(WORKER),
            "--sequence", sequence,
            "--split", "val",
            "--max-frames", "0",
            "--frozen-eval",
            "--asset-manifest", str(args.asset_manifest),
            "--output-root", str(candidate_root),
            "--checkpoint", str(checkpoint),
            "--device", str(args.device),
        ]
        if args.dataset_root is not None:
            worker_command.extend(["--dataset-root", str(args.dataset_root)])
        if args.osnet_checkpoint is not None:
            worker_command.extend(["--osnet-checkpoint", str(args.osnet_checkpoint)])
        result: dict[str, Any] = {"sequence": sequence}
        worker_log = log_dir / f"{sequence}.worker.log"
        result["worker_returncode"] = run_command(worker_command, worker_log)
        result["worker_log"] = str(worker_log)
        if result["worker_returncode"] != 0:
            results.append(result)
            break
        for policy in ("immediate", "2frame"):
            output = records_dir / f"{sequence}.{policy}.json"
            eval_command = [
                sys.executable,
                str(EVALUATOR),
                "--sequence", sequence,
                "--split", "val",
                "--asset-manifest", str(args.asset_manifest),
                "--protocol", str(args.protocol),
                "--candidate-root", str(candidate_root),
                "--anchor-device", str(args.anchor_device),
                "--update-policy", policy,
                "--output", str(output),
                "--confirmation-margin-threshold", str(threshold),
            ]
            if args.dataset_root is not None:
                eval_command.extend(["--dataset-root", str(args.dataset_root)])
            if args.osnet_checkpoint is not None:
                eval_command.extend(["--osnet-checkpoint", str(args.osnet_checkpoint)])
            eval_log = log_dir / f"{sequence}.{policy}.eval.log"
            result[f"{policy}_returncode"] = run_command(eval_command, eval_log)
            result[f"{policy}_log"] = str(eval_log)
            if result[f"{policy}_returncode"] != 0:
                break
        if any(result.get(f"{policy}_returncode") != 0 for policy in ("immediate", "2frame")):
            results.append(result)
            break
        if not candidate_dir.is_dir():
            result["cache_cleanup"] = "missing_after_successful_evaluation"
            results.append(result)
            break
        shutil.rmtree(candidate_dir)
        result["cache_cleanup"] = "deleted_after_successful_evaluation"
        free_bytes = shutil.disk_usage(candidate_root).free
        result["free_space_bytes_after_sequence"] = free_bytes
        result["free_space_gib_after_sequence"] = free_bytes / (1024**3)
        results.append(result)
        if free_bytes < 80 * (1024**3):
            result["hard_stop_triggered"] = True
            break

    aggregate_output = records_dir / "identity_bridge_summary.json"
    aggregate_command = [
        sys.executable,
        str(AGGREGATOR),
        "--split", "val",
        "--records-dir", str(records_dir),
        "--asset-manifest", str(args.asset_manifest),
        "--protocol", str(args.protocol),
        "--require-two-policies",
        "--output", str(aggregate_output),
    ]
    aggregate_log = log_dir / "aggregate.log"
    aggregate_returncode = run_command(aggregate_command, aggregate_log)
    report = {
        "stage": "N72R20",
        "status": "PASS_N72R20_STREAMING_VAL_BRIDGE" if len(results) == len(sequences) and aggregate_returncode == 0 else "FAIL_N72R20_STREAMING_VAL_BRIDGE",
        "sequences": sequences,
        "completed_sequence_count": len(results),
        "results": results,
        "aggregate": str(aggregate_output),
        "aggregate_log": str(aggregate_log),
        "confirmation_threshold_manifest": str(args.confirmation_threshold_manifest.expanduser().resolve()),
        "candidate_root": str(candidate_root),
        "records_dir": str(records_dir),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "val_tuning_used": False,
    }
    output = ROOT / "outputs/N72R20/streaming_val_bridge_run.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "completed": len(results), "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS_N72R20_STREAMING_VAL_BRIDGE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
