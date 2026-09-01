#!/usr/bin/env python3
"""Supervise isolated N42 T0/T1 replay workers sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
WORKER = ROOT / "scripts/n42_t1_replay_worker.py"
OUT = ROOT / "outputs/n42/replay"
ATTEMPTS = ROOT / "outputs/n42/attempts"
PROTOCOL = "N42_T1_REPLAY_SUPERVISOR_V1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_events() -> list[dict[str, Any]]:
    payload = json.loads(N37_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("event_count") != 24:
        raise RuntimeError("N37 manifest is not PASS/24")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 24:
        raise RuntimeError("N37 events invalid")
    return sorted(events, key=lambda item: str(item["event"]["event_id"]))


def smoke_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in events:
        action = str(item["event"]["action_type"])
        if action in seen:
            continue
        selected.append(item)
        seen.add(action)
    if len(selected) != 4:
        raise RuntimeError(f"could not select one smoke event per action: {len(selected)}")
    return selected


def run_workers(events: list[dict[str, Any]], mode: str, output_dir: Path, label: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = now()
    records = []
    failures = []
    for item in events:
        event_id = str(item["event"]["event_id"])
        output = output_dir / f"{event_id}.json"
        command = [sys.executable, str(WORKER), "--event-id", event_id, "--mode", mode, "--output", str(output)]
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "N42_REPLAY_WORKER": "1"})
        completed = subprocess.run(command, cwd=str(ROOT), env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        record = {
            "event_id": event_id,
            "mode": mode,
            "command": command,
            "returncode": int(completed.returncode),
            "output": str(output.relative_to(ROOT)),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "started_at": started,
            "finished_at": now(),
        }
        if completed.returncode != 0 or not output.is_file():
            failures.append(record)
            failure_path = ATTEMPTS / f"replay_supervisor_{label}_{mode}_{event_id.replace('/', '_')}_failure.json"
            if not failure_path.exists():
                atomic_json(failure_path, {"protocol": PROTOCOL, "status": "FAIL_WORKER", "record": record, "failure_preserved": True})
            break
        artifact = json.loads(output.read_text(encoding="utf-8"))
        record["artifact_status"] = artifact.get("status")
        record["runtime_future_gt_used"] = artifact.get("runtime_boundary", {}).get("runtime_future_gt_used")
        if artifact.get("status") != "PASS" or record["runtime_future_gt_used"] is not False:
            failures.append(record)
            failure_path = ATTEMPTS / f"replay_supervisor_{label}_{mode}_{event_id.replace('/', '_')}_audit_failure.json"
            if not failure_path.exists():
                atomic_json(failure_path, {"protocol": PROTOCOL, "status": "FAIL_WORKER_AUDIT", "record": record, "failure_preserved": True})
            break
        records.append(record)
    return {
        "protocol": PROTOCOL,
        "status": "PASS" if len(records) == len(events) and not failures else "FAIL",
        "label": label,
        "mode": mode,
        "expected_event_count": len(events),
        "completed_event_count": len(records),
        "worker_records": records,
        "failures": failures,
        "runtime_future_gt_used": False,
        "started_at": started,
        "finished_at": now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    started = now()
    try:
        events = load_events()
        selected = smoke_events(events)
        if args.phase == "smoke":
            summaries = []
            for mode in ("t0", "t1"):
                summaries.append(run_workers(selected, mode, OUT / "smoke" / mode, "smoke"))
            status = "SMOKE_PASS" if all(item["status"] == "PASS" for item in summaries) else "SMOKE_FAIL"
            payload = {"protocol": PROTOCOL, "stage": "N42-03-smoke", "status": status, "event_ids": [str(item["event"]["event_id"]) for item in selected], "summaries": summaries, "started_at": started, "finished_at": now(), "runtime_future_gt_used": False}
            atomic_json(OUT / "smoke_manifest.json", payload)
            atomic_json(ROOT / "outputs/n42/stage_03_status.json", payload)
        else:
            summaries = []
            for mode in ("t0", "t1"):
                summary = run_workers(events, mode, OUT / "runtime" / mode, "full")
                atomic_json(OUT / f"runtime_{mode}_manifest.json", summary)
                summaries.append(summary)
                if summary["status"] != "PASS":
                    raise RuntimeError(f"{mode} full replay failed after preserving its worker evidence")
            payload = {"protocol": PROTOCOL, "stage": "N42-03-runtime", "status": "RUNTIME_PASS", "expected_worker_count": 48, "completed_worker_count": sum(item["completed_event_count"] for item in summaries), "summaries": summaries, "started_at": started, "finished_at": now(), "runtime_future_gt_used": False, "posthoc_gt_loaded": False}
            atomic_json(ROOT / "outputs/n42/stage_03_status.json", payload)
        print(json.dumps({"status": payload["status"], "phase": args.phase}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {"protocol": PROTOCOL, "phase": args.phase, "status": "FAIL", "started_at": started, "finished_at": now(), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "failure_preserved": True}
        failure_path = ATTEMPTS / f"replay_supervisor_{args.phase}_failure.json"
        if not failure_path.exists():
            atomic_json(failure_path, failure)
        atomic_json(ROOT / "outputs/n42/stage_03_status.json", failure)
        raise


if __name__ == "__main__":
    main()
