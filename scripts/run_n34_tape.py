#!/usr/bin/env python3
"""Materialize the N34 candidate/event tape gate without fabricating rows.

The current project has real multi-ID DanceTrack train data, but no source
that exports every SAM3 candidate for every frame together with a valid
public-ID mapping.  This script therefore writes an explicit NOT_AVAILABLE
real tape sentinel and a separate synthetic fallback declaration.  It never
uses future GT to construct a runtime candidate row.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n34"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    selected_path = OUT / "selected_sequences.json"
    inventory_path = OUT / "sequence_inventory.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {}
    selected_sequences = [str(row["sequence"]) for row in selected.get("sequences", []) if isinstance(row, dict)]

    reason_codes = [
        "NO_PER_FRAME_ALL_SAM3_CANDIDATE_EXPORT",
        "N25R_CACHE_IS_EPISODE_WINDOW_TOP_K_ONLY",
        "N25R_SELECTED_OBJ_ID_COVERAGE_IS_ZERO",
        "N31_ROLLOUT_IS_POLICY_SELECTED_NOT_FRAME_COMPLETE",
        "NO_VALID_PUBLIC_ID_CANDIDATE_MAPPING",
    ]
    tape_sentinel = {
        "record_type": "real_candidate_complete_tape_status",
        "protocol": "N34_CANDIDATE_COMPLETE_TAPE",
        "status": "NOT_AVAILABLE",
        "candidate_complete": False,
        "candidate_set_complete": False,
        "future_gt_used_runtime": False,
        "reason_codes": reason_codes,
        "selected_sequence_count": len(selected_sequences),
        "selected_sequences": selected_sequences,
        "evidence": {
            "sequence_inventory": "outputs/n34/sequence_inventory.json",
            "selected_sequences": "outputs/n34/selected_sequences.json",
            "n25r_alignment": "outputs/n25r/feature_alignment.json",
            "n31_candidate_rollouts": "outputs/n31/candidate_rollout_index.json",
        },
        "note": "This sentinel is not a frame record and must not be consumed as a candidate row.",
    }
    atomic_text(OUT / "candidate_complete_tape.jsonl", json.dumps(tape_sentinel, sort_keys=True) + "\n")

    event_types = {
        "ADD_NEW_IDENTITY": "INTERFACE_PRESENT_BUT_REAL_EVENT_TAPE_NOT_AVAILABLE",
        "AUTHORITATIVE_REASSIGN": "INTERFACE_PRESENT_BUT_REAL_EVENT_TAPE_NOT_AVAILABLE",
        "ATOMIC_ID_SWAP": "INTERFACE_PRESENT_BUT_REAL_EVENT_TAPE_NOT_AVAILABLE",
        "RECOVER_IDENTITY": "INTERFACE_PRESENT_BUT_REAL_EVENT_TAPE_NOT_AVAILABLE",
    }
    event_tape = {
        "protocol": "N34_HUMAN_EVENT_TAPE",
        "status": "NOT_AVAILABLE",
        "interaction_source": "real_event_source_not_available",
        "future_gt_used_runtime": False,
        "events": [],
        "event_types": event_types,
        "reason": "No real candidate-complete per-frame interaction tape with independent human ROI evidence is available.",
        "synthetic_fallback": {
            "available": True,
            "path": "outputs/n34/synthetic_event_tape.json",
            "interaction_source": "simulated_from_gt",
            "not_a_real_data_result": True,
        },
    }
    atomic_json(OUT / "human_event_tape.json", event_tape)

    manifest = {
        "protocol": "N34_TAPE_MANIFEST",
        "status": "NOT_AVAILABLE",
        "real_multi_id_data": bool(inventory.get("real_multi_id_data", False)),
        "candidate_complete": False,
        "candidate_set_complete": False,
        "num_sequences": len(selected_sequences),
        "num_frames": 0,
        "num_events": 0,
        "num_ids": 0,
        "future_gt_used_runtime": False,
        "real_candidate_tape": "outputs/n34/candidate_complete_tape.jsonl",
        "real_event_tape": "outputs/n34/human_event_tape.json",
        "reason_codes": reason_codes,
        "synthetic_fallback": {
            "candidate_complete": True,
            "interaction_source": "simulated_from_gt",
            "metrics_are_not_real_data_claims": True,
        },
    }
    atomic_json(OUT / "tape_manifest.json", manifest)
    stage = {
        "stage": "N34-2",
        "status": "NOT_AVAILABLE",
        "commands": ["python scripts/run_n34_tape.py"],
        "artifacts": [
            "outputs/n34/candidate_complete_tape.jsonl",
            "outputs/n34/human_event_tape.json",
            "outputs/n34/tape_manifest.json",
        ],
        "errors": [],
        "reason": "Real multi-ID data exists, but the required candidate-complete public-ID tape source is unavailable.",
        "next_action": "Run the explicit synthetic fallback for transaction and paired replay code paths; keep real-data claims NOT_AVAILABLE.",
        "candidate_complete": False,
        "future_gt_used_runtime": False,
    }
    atomic_json(OUT / "stage_02_status.json", stage)
    return {"manifest": manifest, "stage": stage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(json.dumps({"manifest": "outputs/n34/tape_manifest.json", **result["stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
