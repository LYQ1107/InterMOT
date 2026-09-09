#!/usr/bin/env python3
"""Write the final N72R11R4 machine-readable decision without hiding failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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
        if temporary.exists():
            temporary.unlink()


def ref(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "status": payload.get("status")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1a", type=Path, required=True)
    parser.add_argument("--e1b", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--merge-gate", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--stage-output", type=Path, required=True)
    parser.add_argument("--controller-output", type=Path, required=True)
    parser.add_argument("--final-gate-output", type=Path, required=True)
    args = parser.parse_args()

    paths = {}
    payloads = {}
    for name in ("e1a", "e1b", "readiness", "merge_gate", "failure_audit", "comparison"):
        raw = getattr(args, name)
        path = raw if raw.is_absolute() else ROOT / raw
        paths[name] = path
        payloads[name] = read(path)
    merge_gate = payloads["merge_gate"]
    e1b = payloads["e1b"]
    readiness = payloads["readiness"]
    failure_audit = payloads["failure_audit"]
    created = now_utc()
    refs = {name: ref(paths[name], payloads[name]) for name in paths}
    e1b_horizons = e1b.get("by_horizon", {})
    e1b_ci = {h: e1b_horizons.get(h, {}).get("sequence_cluster_bootstrap_95ci") for h in ("20", "50", "100")}
    e1b_strict = all(isinstance(ci, dict) and isinstance(ci.get("lower"), (int, float)) and float(ci["lower"]) > 0.0 for ci in e1b_ci.values())
    e2_complete = bool(merge_gate.get("oracle_authorized"))
    stage = {
        "schema_version": "N72R11R4_FINAL_STAGE_STATUS_V1",
        "stage": "N72R11R4-17-FINAL-E2-DECISION",
        "status": "BLOCKED_RESOURCE_INCOMPLETE_E2",
        "created_at_utc": created,
        "e1a_complete": bool(payloads["e1a"].get("complete")),
        "e1b_complete": bool(e1b.get("complete")),
        "e1b_strict_future_effect_gate": e1b_strict,
        "e2_complete": e2_complete,
        "e2_merge_gate": merge_gate.get("completeness"),
        "e2_metrics_generated": False,
        "oracle_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "pctis_readiness": readiness,
        "failure_classes": failure_audit.get("class_counts"),
        "minimal_next_step": "Fix or explicitly verify official SAM3 internal device placement/streaming memory for the unresolved event, then rerun only that frozen event under the same H100 protocol; do not use this partial E2 corpus for Oracle.",
        "not_real_human_evidence": True,
        "source_artifacts": refs,
    }
    controller = {
        "schema_version": "N72R11R4_CONTROLLER_STATUS_V1",
        "created_at_utc": created,
        "status": stage["status"],
        "exact_solver_on_policy": True,
        "candidate_features_recomputed": True,
        "causal_scores_recomputed": True,
        "raw_scope_preserved": True,
        "v3_stage_a": payloads["e1a"].get("status"),
        "pctis_stage_b_e1b": e1b.get("status"),
        "pctis_readiness_status": readiness.get("status"),
        "e2_live_status": merge_gate.get("status"),
        "production_authorized": False,
        "oracle_authorized": False,
        "real_human_event_tape": False,
        "interaction_source": "simulated_from_gt",
        "failure_preserved": True,
        "source_artifacts": refs,
    }
    final_gate = {
        "schema_version": "N72R11R4_FINAL_GATE_V1",
        "status": "BLOCKED_RESOURCE_INCOMPLETE_E2_FAIL_FUTURE_EFFECT",
        "created_at_utc": created,
        "research_gate": "FAIL_FUTURE_EFFECT",
        "resource_gate": "BLOCKED_E2_MISSING_ONE_EVENT_AFTER_FINAL_EQUIVALENT_RETRY",
        "e1a_complete": bool(payloads["e1a"].get("complete")),
        "e1b_complete": bool(e1b.get("complete")),
        "e1b_h20_h50_h100_ci": e1b_ci,
        "e1b_strict_future_effect_gate": e1b_strict,
        "e2_complete": e2_complete,
        "e2_merge_completeness": merge_gate.get("completeness"),
        "e2_metrics_available": False,
        "oracle_authorized": False,
        "production_authorized": False,
        "calibration_head_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "real_human_evidence": False,
        "not_real_human_evidence": True,
        "failure_classes": failure_audit.get("class_counts"),
        "source_artifacts": refs,
    }
    for raw, value in ((args.stage_output, stage), (args.controller_output, controller), (args.final_gate_output, final_gate)):
        output = raw if raw.is_absolute() else ROOT / raw
        atomic_json(output, value)
    print(json.dumps({"status": final_gate["status"], "stage": str(args.stage_output), "controller": str(args.controller_output), "final_gate": str(args.final_gate_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
