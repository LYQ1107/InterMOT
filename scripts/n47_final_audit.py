#!/usr/bin/env python3
"""Final read-only preservation and gate audit for the N47 probe."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import CHECKPOINT, OUT, load, sha256, write_json


AUDIT = OUT / "final_audit.json"
LEGACY = ROOT / "outputs/n46/n46_integrity_report.json"


def main() -> None:
    failures = []
    final_gate = load(OUT / "n47_final_gate.json")
    final_status = load(OUT / "stage_05_status.json")
    required_stage_fields = ("status", "command", "inputs", "outputs", "metrics", "gate_checks", "failure_root_cause", "next_action")
    stage_statuses = {}
    for number in (1, 2, 3, 4, 5):
        path = OUT / f"stage_{number:02d}_status.json"
        if not path.is_file():
            failures.append(f"missing stage {number}"); continue
        payload = load(path); stage_statuses[path.name] = payload.get("status")
        if any(key not in payload for key in required_stage_fields): failures.append(f"stage {number} schema")
    for path in [OUT / "probe_protocol.json", OUT / "stage_03_targeted_regression.json", OUT / "stage_04_smoke.json", OUT / "stage_04_integrity.json", OUT / "replay/runtime_status.json", OUT / "replay/probe_results.json", OUT / "training/training_manifest.json", OUT / "training/n47_global_fusion_probe.pt"]:
        if not path.is_file(): failures.append(f"missing artifact {path}")
    integrity = load(OUT / "stage_04_integrity.json")
    if integrity.get("status") != "PASS": failures.append("stage 04 integrity")
    if final_gate.get("status") != "N47_COMPLETED_GATE_FAILED": failures.append("final gate status")
    if final_gate.get("authorization", {}).get("checkpoint_production_authorized") is not False: failures.append("authorization")
    if final_status.get("status") != final_gate.get("status"): failures.append("stage/gate status mismatch")
    if load(OUT / "training/training_manifest.json").get("actual_full_training") is not True: failures.append("actual training")
    if load(OUT / "training/training_manifest.json").get("production_authorized") is not False: failures.append("training authorization")
    if sha256(ROOT / "outputs/n44/training/n44_assignment_aware.pt") != "0b5e750f5d9569f71ae887595c1d88d4d625f120f8a3811f2598a852cf82348f": failures.append("N44 checkpoint hash")
    legacy = load(LEGACY).get("legacy_n43_n44_n45_preservation_sha256", {})
    changed_legacy = []
    for raw_path, expected in legacy.items():
        path = Path(raw_path)
        if not path.is_file(): changed_legacy.append(raw_path)
        elif sha256(path) != expected: changed_legacy.append(raw_path)
    if changed_legacy: failures.append("legacy N43/N44/N45 hash mismatch")
    report = {"status": "PASS" if not failures else "FAIL", "protocol": "N47_FINAL_PRESERVATION_AUDIT_V1", "inputs": {"legacy_manifest": str(LEGACY), "n47_gate": str(OUT / "n47_final_gate.json"), "n47_integrity": str(OUT / "stage_04_integrity.json")}, "outputs": {"audit": str(AUDIT)}, "metrics": {"legacy_files_checked": len(legacy), "legacy_hash_mismatches": changed_legacy, "n47_attempts_retained": len(list((OUT / "attempts").glob("*")))}, "gate_checks": {"all_stage_statuses_schema": not any("stage " in x and "schema" in x for x in failures), "required_n47_artifacts": not any("missing artifact" in x for x in failures), "full_integrity_pass": integrity.get("status") == "PASS", "all_legacy_n43_n44_n45_hashes_stable": not changed_legacy, "n44_checkpoint_unchanged": "N44 checkpoint hash" not in failures, "actual_training": load(OUT / "training/training_manifest.json").get("actual_full_training") is True, "production_authorized_false": final_gate.get("authorization", {}).get("checkpoint_production_authorized") is False, "broader_objective_open": True}, "failure_root_cause": "A final preservation audit fails if any frozen N43/N44/N45 hash, N44 checkpoint hash, N47 artifact, stage schema or strict gate is inconsistent.", "next_action": "Keep the broader objective open; use only provenance-complete real human tape/full-loop to motivate any future experiment.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    write_json(AUDIT, report); print(json.dumps({"status": report["status"], "legacy_files_checked": len(legacy), "mismatches": len(changed_legacy), "attempts": report["metrics"]["n47_attempts_retained"]}))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
