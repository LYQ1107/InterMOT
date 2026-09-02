#!/usr/bin/env python3
"""Run the frozen N72R3 official correction/memory events in isolated workers.

At most four child processes are live at once.  The first event was already
completed by the targeted smoke; its PASS artifact is promoted without
re-running it.  Every remaining event gets a fresh Python process, SAM3
session, and atomic event artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RUNNER = ROOT / "scripts/n72r3_stage16_official_correction.py"
EVENT_MANIFEST = ROOT / "outputs/N72R3/simulation/real_event_manifest.json"
OUTPUT_ROOT = ROOT / "outputs/N72R3/official_correction"
SMOKE = OUTPUT_ROOT / "smoke_attempt6_dancetrack0001_0296.json"
EVENT_ROOT = OUTPUT_ROOT / "events"
LOG_ROOT = OUTPUT_ROOT / "logs"
STAGE16_STATUS = ROOT / "outputs/N72R3/stage_16_status.json"
STAGE17_STATUS = ROOT / "outputs/N72R3/stage_17_status.json"
BATCH_MANIFEST = OUTPUT_ROOT / "stage16_17_batch_manifest.json"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def load_events() -> list[dict[str, Any]]:
    payload = json.loads(EVENT_MANIFEST.read_text(encoding="utf-8"))
    events = [dict(item) for item in payload.get("events", [])]
    if len(events) != 6:
        raise RuntimeError(f"frozen Stage14 event count changed: expected 6, got {len(events)}")
    ids = [str(item["event_id"]) for item in events]
    if len(ids) != len(set(ids)):
        raise RuntimeError("frozen Stage14 event IDs are not unique")
    return events


def is_pass(payload: dict[str, Any]) -> bool:
    correction = payload.get("official_current_correction") or {}
    transaction = payload.get("transaction") or {}
    causal = payload.get("causal_audit") or {}
    memory_write = causal.get("memory_write") or {}
    record = payload.get("appearance_memory") or {}
    positives = record.get("positive") if isinstance(record, dict) else None
    feature = positives[0].get("feature") if positives and isinstance(positives[0], dict) else None
    return bool(
        payload.get("status") == "PASS_STAGE16_OFFICIAL_CORRECTION_AND_STAGE17_MEMORY"
        and payload.get("official_backend") is True
        and payload.get("runtime_future_gt_used") is False
        and int(payload.get("future_frames_loaded", -1)) == 0
        and payload.get("future_read_executed") is False
        and correction.get("success") is True
        and bool(correction.get("official_prompt_observation_available"))
        and float(correction.get("post_box_iou_to_simulated_human_box", 0.0)) >= 0.98
        and transaction.get("committed") is True
        and transaction.get("rolled_back") is False
        and set(transaction.get("completed_phases", [])) == {"backend", "identity", "memory"}
        and causal.get("event_frame_read") is False
        and causal.get("current_frame_write_hidden") is True
        and int(causal.get("first_visible_frame", -1)) == int(payload.get("event_frame", -2)) + 1
        and causal.get("memory_reads") == []
        and memory_write.get("feature_dim") == 512
        and memory_write.get("current_frame_write_hidden") is True
        and isinstance(feature, list)
        and len(feature) == 512
        and all(isinstance(value, (int, float)) and value == value for value in feature)
    )


def promote_smoke(event_id: str, destination: Path) -> dict[str, Any]:
    payload = json.loads(SMOKE.read_text(encoding="utf-8"))
    if str(payload.get("event_id")) != event_id or not is_pass(payload):
        raise RuntimeError("the targeted smoke artifact is not a valid PASS for the first frozen event")
    if not destination.exists():
        atomic_json(destination, payload)
    return {"event_id": event_id, "mode": "promoted_targeted_smoke", "return_code": 0, "status": payload["status"], "output": str(destination)}


def launch_batch(events: list[dict[str, Any]], gpu_ids: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    smoke_event_id = str(events[0]["event_id"])
    records.append(promote_smoke(smoke_event_id, EVENT_ROOT / f"{smoke_event_id}.json"))
    pending = events[1:]
    for offset in range(0, len(pending), len(gpu_ids)):
        group = pending[offset : offset + len(gpu_ids)]
        processes: list[tuple[dict[str, Any], int, subprocess.Popen, Any, Path]] = []
        for event, gpu in zip(group, gpu_ids):
            event_id = str(event["event_id"])
            output = EVENT_ROOT / f"{event_id}.json"
            log_path = LOG_ROOT / f"{event_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            command = [
                str(PYTHON),
                str(RUNNER),
                "--event-id",
                event_id,
                "--gpu",
                str(gpu),
                "--output",
                str(output),
            ]
            started = time.time()
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((event, gpu, process, log_handle, log_path))
            records.append(
                {
                    "event_id": event_id,
                    "gpu": gpu,
                    "mode": "fresh_process",
                    "pid": process.pid,
                    "output": str(output),
                    "log": str(log_path),
                    "started_unix": started,
                }
            )
        for event, gpu, process, log_handle, log_path in processes:
            return_code = process.wait()
            log_handle.close()
            event_id = str(event["event_id"])
            output = EVENT_ROOT / f"{event_id}.json"
            payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
            item = next(item for item in records if item.get("event_id") == event_id and item.get("pid") == process.pid)
            item.update(
                {
                    "return_code": return_code,
                    "status": payload.get("status", "MISSING_ARTIFACT"),
                    "artifact_validated_pass": is_pass(payload) if payload else False,
                    "elapsed_sec": time.time() - item["started_unix"],
                }
            )
    return records


def write_status(records: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    payloads = []
    for record in records:
        path = Path(record["output"])
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    valid = len(payloads) == len(events) and all(is_pass(payload) for payload in payloads)
    unique = len({str(payload.get("event_id")) for payload in payloads}) == len(payloads)
    batch_payload = {
        "schema_version": "N72R3_STAGE16_17_BATCH_MANIFEST_V1",
        "event_count_expected": len(events),
        "event_count_artifacts": len(payloads),
        "event_ids_expected": [str(event["event_id"]) for event in events],
        "event_ids_artifacts": sorted(str(payload.get("event_id")) for payload in payloads),
        "unique_event_ids": unique,
        "all_artifacts_pass": valid,
        "records": records,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "max_concurrent_gpus": 4,
        "one_fresh_process_per_event": True,
        "batch_input_sha256": digest_json(events),
        "scientific_result": "CURRENT_FRAME_AND_MEMORY_ARTIFACT_ONLY_NOT_FUTURE_EFFECT",
    }
    atomic_json(BATCH_MANIFEST, batch_payload)
    stage16 = {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": "16_OFFICIAL_CURRENT_FRAME_CORRECTION",
        "status": "PASS_STAGE16_OFFICIAL_CORRECTION" if valid and unique else "BLOCKED_STAGE16_OFFICIAL_CORRECTION",
        "event_count_expected": len(events),
        "event_count_pass": sum(is_pass(payload) for payload in payloads),
        "correction_success_rate": sum(bool((payload.get("official_current_correction") or {}).get("success")) for payload in payloads) / len(events) if events else 0.0,
        "official_backend": True,
        "checkpoint_unchanged": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "batch_manifest": str(BATCH_MANIFEST),
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    stage17 = {
        "schema_version": "N72R3_STAGE_STATUS_V1",
        "stage": "17_REAL_HUMAN_ROI_APPEARANCE_MEMORY_WRITE",
        "status": "PASS_STAGE17_REAL_512D_ROI_MEMORY" if valid and unique else "BLOCKED_STAGE17_REAL_512D_ROI_MEMORY",
        "event_count_expected": len(events),
        "event_count_pass": sum(is_pass(payload) for payload in payloads),
        "finite_512d_memory_writes": sum(
            bool(((payload.get("causal_audit") or {}).get("memory_write") or {}).get("feature_dim") == 512)
            for payload in payloads
        ),
        "event_frame_write_hidden": all((payload.get("causal_audit") or {}).get("current_frame_write_hidden") is True for payload in payloads),
        "first_visible_frame_all_event_plus_one": all(int((payload.get("causal_audit") or {}).get("first_visible_frame", -1)) == int(payload.get("event_frame", -2)) + 1 for payload in payloads),
        "machine_candidate_feature_substitution": False,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "batch_manifest": str(BATCH_MANIFEST),
        "scientific_result": "NOT_A_SCIENTIFIC_RESULT",
    }
    atomic_json(STAGE16_STATUS, stage16)
    atomic_json(STAGE17_STATUS, stage17)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpu_ids = [int(item) for item in str(args.gpus).split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) > 4 or len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("--gpus must contain one to four unique GPU IDs")
    events = load_events()
    records = launch_batch(events, gpu_ids)
    write_status(records, events)
    all_pass = json.loads(STAGE16_STATUS.read_text(encoding="utf-8"))["status"].startswith("PASS") and json.loads(STAGE17_STATUS.read_text(encoding="utf-8"))["status"].startswith("PASS")
    print(json.dumps({"status": "PASS_STAGE16_17_BATCH" if all_pass else "BLOCKED_STAGE16_17_BATCH", "batch_manifest": str(BATCH_MANIFEST)}, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
