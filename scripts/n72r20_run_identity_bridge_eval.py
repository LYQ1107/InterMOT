#!/usr/bin/env python3
"""Run the N72R20 per-sequence bridge evaluator and compact aggregation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "n72r20_identity_bridge_eval.py"
AGGREGATOR = ROOT / "scripts" / "n72r20_aggregate_identity_bridge.py"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_threshold(path: Path) -> tuple[float, str]:
    payload = load_json(path)
    if payload.get("stage") != "N72R20" or payload.get("split") != "train":
        raise ValueError("confirmation threshold manifest must be a frozen N72R20 train manifest")
    if payload.get("frozen") is not True:
        raise ValueError("confirmation threshold manifest is not frozen")
    threshold = float(payload["confirmation_margin_threshold"])
    if threshold < 0.0:
        raise ValueError("confirmation margin threshold must be non-negative")
    return threshold, str(path.expanduser().resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-evaluation", action="store_true")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20/asset_manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R20/protocol.json")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--gru-checkpoint", type=Path)
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--anchor-device", default="cpu")
    parser.add_argument("--confirmation-threshold-manifest", type=Path)
    parser.add_argument("--max-anchors", type=int, default=0)
    args = parser.parse_args()

    if not args.run_evaluation:
        raise SystemExit("refusing identity evaluation without explicit --run-evaluation")
    manifest = load_json(args.asset_manifest)
    protocol = load_json(args.protocol)
    if manifest.get("stage") != "N72R20" or protocol.get("stage") != "N72R20":
        raise SystemExit("asset manifest and protocol must both be N72R20")
    sequences = list(protocol["val_evaluation_sequences"] if args.split == "val" else protocol["train_dev_sequences"])
    if args.split == "val" and len(sequences) != 25:
        raise SystemExit("frozen val evaluation requires all 25 protocol sequences")
    threshold = None
    threshold_source = None
    if args.split == "val":
        if args.confirmation_threshold_manifest is None:
            raise SystemExit("frozen val evaluation requires a train-frozen confirmation threshold manifest")
        threshold, threshold_source = load_threshold(args.confirmation_threshold_manifest)
    elif args.confirmation_threshold_manifest is not None:
        threshold, threshold_source = load_threshold(args.confirmation_threshold_manifest)

    candidate_root = args.candidate_root.expanduser().resolve()
    records_dir = args.records_dir.expanduser().resolve()
    records_dir.mkdir(parents=True, exist_ok=True)
    log_dir = records_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    for policy in ("immediate", "2frame"):
        if policy == "2frame" and threshold is None:
            raise SystemExit("2-frame evaluation requires a frozen confirmation threshold")
        for sequence in sequences:
            output = records_dir / f"{sequence}.{policy}.json"
            command = [
                sys.executable,
                str(EVALUATOR),
                "--sequence",
                sequence,
                "--split",
                args.split,
                "--asset-manifest",
                str(args.asset_manifest),
                "--protocol",
                str(args.protocol),
                "--candidate-root",
                str(candidate_root),
                "--anchor-device",
                str(args.anchor_device),
                "--update-policy",
                policy,
                "--output",
                str(output),
            ]
            for flag, value in (
                ("--dataset-root", args.dataset_root),
                ("--gru-checkpoint", args.gru_checkpoint),
                ("--osnet-checkpoint", args.osnet_checkpoint),
            ):
                if value is not None:
                    command.extend([flag, str(value)])
            if args.max_anchors > 0:
                command.extend(["--max-anchors", str(args.max_anchors)])
            if policy == "2frame":
                command.extend(["--confirmation-margin-threshold", str(threshold)])
            completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, check=False)
            log = log_dir / f"{sequence}.{policy}.log"
            log.write_text(
                f"command={json.dumps(command, ensure_ascii=False)}\nreturncode={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "sequence": sequence,
                    "policy": policy,
                    "returncode": completed.returncode,
                    "output": str(output),
                    "log": str(log),
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-4000:],
                }
            )
            if completed.returncode != 0:
                break
        if results[-1]["returncode"] != 0:
            break

    aggregate_output = records_dir / "identity_bridge_summary.json"
    aggregate_command = [
        sys.executable,
        str(AGGREGATOR),
        "--split",
        args.split,
        "--records-dir",
        str(records_dir),
        "--asset-manifest",
        str(args.asset_manifest),
        "--protocol",
        str(args.protocol),
        "--require-two-policies",
        "--output",
        str(aggregate_output),
    ]
    aggregate = subprocess.run(aggregate_command, cwd=str(ROOT), text=True, capture_output=True, check=False)
    aggregate_log = log_dir / "aggregate.log"
    aggregate_log.write_text(
        f"command={json.dumps(aggregate_command, ensure_ascii=False)}\nreturncode={aggregate.returncode}\n"
        f"stdout:\n{aggregate.stdout}\nstderr:\n{aggregate.stderr}\n",
        encoding="utf-8",
    )
    report = {
        "stage": "N72R20",
        "status": "PASS_N72R20_IDENTITY_BRIDGE_EVALUATION" if aggregate.returncode == 0 and all(item["returncode"] == 0 for item in results) else "FAIL_N72R20_IDENTITY_BRIDGE_EVALUATION",
        "split": args.split,
        "sequences": sequences,
        "results": results,
        "aggregate": str(aggregate_output),
        "aggregate_log": str(aggregate_log),
        "confirmation_margin_threshold": threshold,
        "confirmation_threshold_source": threshold_source,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "val_tuning_used": False,
    }
    report_path = records_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "aggregate": str(aggregate_output), "output": str(report_path)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS_N72R20_IDENTITY_BRIDGE_EVALUATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
