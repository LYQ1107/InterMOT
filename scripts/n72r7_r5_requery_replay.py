#!/usr/bin/env python3
"""Run a CPU, paired R5 re-query replay for one frozen N72R7 event.

The decoder, admission rule, global solver, and metric inputs are held fixed.
The treatment adds only the sealed multi-query target-session candidates to the
existing B0 + single target-session pool.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r7_dev_replay import (  # noqa: E402
    DEV_PROTOCOL,
    read_json,
    run_event,
    validate_inputs,
)
from scripts.n72r7_learned_replay import LearnedTargetCandidateSelector  # noqa: E402


CHECKPOINT = ROOT / "outputs/N72R7/training_v2/HumanConditionedTargetIDDecoder_v1.pt"
REPLAY_PROTOCOL = ROOT / "outputs/N72R7/training_v2/learned_replay_protocol.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--requery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    requery_root = args.requery_root if args.requery_root.is_absolute() else ROOT / args.requery_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    event_id = str(args.event_id)
    result: dict[str, Any] = {
        "schema_version": "N72R7_R5_REQUERY_REPLAY_WORKER_V1",
        "status": "FAIL",
        "event_id": event_id,
        "attempt": int(args.attempt),
        "started_at_utc": now_utc(),
    }
    try:
        _batch, policy, frozen = validate_inputs()
        if event_id not in frozen:
            raise ValueError(f"event is not in frozen N72R6 set: {event_id}")
        requery_done_path = requery_root / f"attempt_{int(args.attempt)}" / event_id / "done.json"
        requery_done = read_json(requery_done_path)
        if requery_done.get("status") != "PASS_N72R7_CANDIDATE_GENERATOR_REQUERY":
            raise RuntimeError(f"requery artifact is not PASS: {requery_done_path}")
        requery_frames = Path(str(requery_done["frames"]))
        if not requery_frames.is_absolute():
            requery_frames = ROOT / requery_frames
        if sha256(requery_frames) != str(requery_done["frames_sha256"]):
            raise RuntimeError(f"requery frame hash mismatch: {requery_frames}")
        replay_protocol = read_json(REPLAY_PROTOCOL)
        checkpoint = Path(str(replay_protocol["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if sha256(checkpoint) != str(replay_protocol["checkpoint_sha256"]):
            raise RuntimeError("R2 decoder checkpoint hash mismatch")
        device = torch.device(str(args.device))
        selector = LearnedTargetCandidateSelector(checkpoint, device=device, protocol=replay_protocol)
        event = next(item for item in policy["events"] if str(item["event_id"]) == event_id)
        frozen_manifest = dict(frozen[event_id])
        current_root = output_root / "current"
        treatment_root = output_root / "requery"
        current_manifest = run_event(
            event,
            frozen_manifest,
            variant="D2",
            output_root=current_root,
            selector=selector,
            protocol=replay_protocol,
        )
        treatment_manifest = run_event(
            event,
            frozen_manifest,
            variant="R5_REQUERY",
            output_root=treatment_root,
            selector=selector,
            protocol=replay_protocol,
            target_requery_path=requery_frames,
        )
        result.update({
            "status": "PASS_N72R7_R5_REQUERY_REPLAY",
            "current_manifest": str(current_root / event_id / "event_manifest.json"),
            "treatment_manifest": str(treatment_root / event_id / "event_manifest.json"),
            "requery_source": str(requery_frames),
            "requery_source_sha256": sha256(requery_frames),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": str(replay_protocol["checkpoint_sha256"]),
            "replay_protocol": str(REPLAY_PROTOCOL),
            "replay_protocol_sha256": sha256(REPLAY_PROTOCOL),
            "selector": "HumanConditionedTargetIDDecoder_v2_greedy",
            "device": str(device),
            "variant_pair": {"D1": "same R2 decoder + current target session", "D2": "same R2 decoder + current target session + R5 re-query pool"},
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "finished_at_utc": now_utc(),
        })
        atomic_json(output_root / event_id / "worker_status.json", result)
        print(json.dumps({"status": result["status"], "event_id": event_id}, sort_keys=True))
        return 0
    except Exception as exc:
        result.update({
            "status": "FAIL_N72R7_R5_REQUERY_REPLAY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
            "finished_at_utc": now_utc(),
        })
        atomic_json(output_root / "attempts" / f"{event_id}.attempt{int(args.attempt)}.failure.json", result)
        print(json.dumps({"status": result["status"], "event_id": event_id, "failure": result["error"]}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
