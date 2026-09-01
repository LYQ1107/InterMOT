#!/usr/bin/env python3
"""Resolve the N36 completion/learning authorization gate without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import ROOT, atomic_json, jsonable


OUT = ROOT / "outputs/n36"
STAGE = OUT / "stage_05_status.json"
GATE = OUT / "n36_final_gate.json"


def read(name: str) -> dict[str, Any]:
    path = OUT / name
    if not path.is_file():
        return {"status": "NOT_RUN", "missing_artifact": name}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    stage01 = read("stage_01_status.json")
    stage02 = read("stage_02_status.json")
    audit = read("all24_integrity_audit.json")
    events = read("real_event_manifest.json")
    full_loop = read("full_loop_results.json")
    replay = read("ccam_paired_replay_results.json")

    checks = {
        "stage_01_audit_pass": stage01.get("status") == "PASS",
        "third_party_unmodified": stage01.get("third_party_modified") is False,
        "real_tape_status_pass": stage02.get("status") == "PASS",
        "real_tape_24_sequences": stage02.get("sequence_count_pass") == 24,
        "real_tape_26691_frames": stage02.get("frame_count") == 26691,
        "real_tape_203_chunks": stage02.get("chunk_count_pass") == 203,
        "real_tape_no_errors": stage02.get("error_count") == 0,
        "real_tape_audit_no_duplicate": audit.get("duplicate_frames") == 0,
        "real_tape_audit_no_missing": audit.get("missing_frames") == 0,
        "real_tape_audit_no_unavailable": audit.get("unavailable_chunks") == 0,
        "event_manifest_pass": events.get("status") == "PASS",
        "event_manifest_six_sequences": events.get("independent_sequence_count") == 6,
        "event_manifest_four_action_types": all(
            int(events.get("action_counts", {}).get(action, 0)) > 0
            for action in (
                "ADD_NEW_IDENTITY",
                "ATOMIC_ID_SWAP",
                "AUTHORITATIVE_REASSIGN",
                "RECOVER_IDENTITY",
            )
        ),
        "full_loop_pass": full_loop.get("status") == "PASS",
        "full_loop_all_events_pass": full_loop.get("event_pass_count") == full_loop.get("event_count") == 7,
        "full_loop_six_sequences": full_loop.get("independent_sequence_count") == 6,
        "full_loop_runtime_future_gt_false": full_loop.get("runtime_future_gt_used") is False,
        "paired_replay_pass": replay.get("status") == "PASS",
        "paired_replay_all_events_pass": replay.get("successful_event_count") == replay.get("event_count") == 7,
        "paired_replay_six_clusters": replay.get("independent_sequence_count") == 6,
        "paired_replay_runtime_future_gt_false": replay.get("runtime_future_gt_used") is False,
        "paired_replay_leakage_free": replay.get("future_effect_gate", {}).get("checks", {}).get("paired_replay_post_treatment_leakage_free") is True,
    }
    effect_checks = replay.get("future_effect_gate", {}).get("checks", {})
    for variant in ("M2", "M3", "M4"):
        checks[f"{variant}_h20_lower_ci_gt_zero"] = effect_checks.get(
            f"{variant}_h20_sequence_cluster_lower_ci_gt_zero"
        ) is True
        checks[f"{variant}_protected_no_obvious_regression"] = effect_checks.get(
            f"{variant}_protected_no_obvious_regression"
        ) is True
    execution_complete = all(
        value
        for key, value in checks.items()
        if not any(token in key for token in ("lower_ci_gt_zero", "protected_no_obvious_regression"))
    )
    future_effect_gate_pass = all(
        checks.get(f"{variant}_h20_lower_ci_gt_zero", False)
        and checks.get(f"{variant}_protected_no_obvious_regression", False)
        for variant in ("M2", "M3", "M4")
    )
    authorization = bool(execution_complete and future_effect_gate_pass)
    payload = {
        "protocol": "N36_COMPLETION_AND_LEARNING_AUTHORIZATION_GATE",
        "status": "PASS" if execution_complete else "FAIL",
        "execution_complete": execution_complete,
        "future_effect_gate": "PASS" if future_effect_gate_pass else "NOT_AUTHORIZED",
        "ccam_future_effect": replay.get("ccam_future_effect", "NOT_COMPUTABLE"),
        "calibration_head": "AUTHORIZED" if authorization else "NOT_AUTHORIZED",
        "decoder_lora": "AUTHORIZED_PILOT_ONLY" if authorization else "NOT_AUTHORIZED",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "evidence": {
            "stage_02": "outputs/n36/stage_02_status.json",
            "tape_audit": "outputs/n36/all24_integrity_audit.json",
            "event_manifest": "outputs/n36/real_event_manifest.json",
            "full_loop": "outputs/n36/full_loop_results.json",
            "paired_replay": "outputs/n36/ccam_paired_replay_results.json",
        },
        "decision": (
            "No calibration or LoRA: M2/M3/M4 strict sequence-cluster H20 lower CI is not > 0."
            if not authorization
            else "A small calibration pilot is authorized by the explicit N36 checks."
        ),
    }
    atomic_json(GATE, jsonable(payload))
    stage = {
        "stage": "N36-07",
        "status": payload["status"],
        "execution_complete": execution_complete,
        "future_effect_gate": payload["future_effect_gate"],
        "ccam_future_effect": payload["ccam_future_effect"],
        "calibration_head": payload["calibration_head"],
        "decoder_lora": payload["decoder_lora"],
        "checks": checks,
        "failed_checks": payload["failed_checks"],
        "artifacts": ["outputs/n36/n36_final_gate.json"],
        "errors": [] if execution_complete else payload["failed_checks"],
        "next_action": (
            "Do not train; preserve real write-only/full-loop/replay evidence and report the null strict future effect."
            if not authorization
            else "Run only the authorized small calibration pilot, then reapply the no-regression gate."
        ),
    }
    atomic_json(STAGE, jsonable(stage))
    print(json.dumps({"status": payload["status"], "execution_complete": execution_complete, "future_effect_gate": payload["future_effect_gate"], "calibration_head": payload["calibration_head"], "decoder_lora": payload["decoder_lora"], "output": "outputs/n36/n36_final_gate.json"}, sort_keys=True))


if __name__ == "__main__":
    main()
