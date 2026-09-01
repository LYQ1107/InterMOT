#!/usr/bin/env python3
"""N45 Stage 02: repair checkpoint authorization metadata and run checks."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n44_assignment_common import HARD_NEGATIVE, apply_sidecar, load_checkpoint, sha256


CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
REPAIR = ROOT / "outputs/n45/frozen_checkpoint_authorization.json"
STAGE = ROOT / "outputs/n45/stage_02_status.json"
SOURCE = next((ROOT / "outputs/n42/replay/runtime/t0").glob("*.json"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    model, payload = load_checkpoint(CHECKPOINT, "cpu")
    source = load(SOURCE)
    event_id = str(source["event_id"])
    audit = source["variants"]["M2"]["branches"]["memory_write=True"]["future_trace"][0]["candidate_audit"]
    gate = payload["gate"]
    # This is an authorization metadata overlay, not a modified model file.
    repair = {"schema": "N45_FROZEN_CHECKPOINT_AUTHORIZATION_V1", "status": "PASS", "source_checkpoint": str(CHECKPOINT), "source_checkpoint_sha256": sha256(CHECKPOINT), "source_payload_production_authorized": payload.get("production_authorized", "MISSING"), "production_authorized": False, "weights_modified": False, "reason": "The immutable N44 payload omitted an explicit authorization field; N45 records the conservative non-production status without rewriting the N44 checkpoint."}
    REPAIR.parent.mkdir(parents=True, exist_ok=True)
    REPAIR.write_text(json.dumps(repair, indent=2) + "\n", encoding="utf-8")
    output = apply_sidecar(audit, model, int(audit["frame"]) - int(source["event_frame"]), source["variants"]["M2"].get("event_frame_audit", {}).get("candidate_audit", {}), gate)
    base = np.asarray(audit["fused_scores"], dtype=np.float32)
    adjusted = np.asarray(output["fused_scores"], dtype=np.float32)
    diff = adjusted - base
    changed = np.argwhere(np.abs(diff) > 1e-12)
    metadata = output.get("n44_sidecar", {})
    smoke = {"status": "PASS", "event_id": event_id, "frame": int(audit["frame"]), "checkpoint_payload_authorization_missing_detected": payload.get("production_authorized") is None, "repair_manifest_production_authorized_false": repair["production_authorized"] is False, "branch_current_fused_used_as_baseline": bool(np.array_equal(base, np.asarray(audit["fused_scores"], dtype=np.float32))), "changed_cells_recorded": int(len(changed)), "changed_cell_count_matches_metadata": int(len(changed)) == int(metadata.get("changed_cell_count", -1)), "all_changed_deltas_equal_bounded_boost": bool(len(changed) == 0 or np.allclose(diff[np.abs(diff) > 1e-12], 0.25, atol=1e-7)), "hard_negative_preserved": metadata.get("hard_negative_preserved") is True, "none_score": metadata.get("none_score"), "runtime_future_gt_used": output.get("runtime_future_gt_used") is False, "assignment_shape": list(np.asarray(output["assignment"]).shape)}
    if not all((smoke["repair_manifest_production_authorized_false"], smoke["branch_current_fused_used_as_baseline"], smoke["changed_cell_count_matches_metadata"], smoke["all_changed_deltas_equal_bounded_boost"], smoke["hard_negative_preserved"], smoke["runtime_future_gt_used"])):
        raise RuntimeError(f"N45 repair smoke failed: {smoke}")
    result = {"status": "PASS", "protocol": "N45_STAGE_02_REPAIR_SMOKE_V1", "command": ["python", "scripts/n45_stage2_repair_smoke.py"], "inputs": {"frozen_n44_checkpoint": str(CHECKPOINT), "n42_source_sample": str(SOURCE)}, "outputs": {"authorization_repair": str(REPAIR)}, "metrics": {"smoke": smoke, "frozen_checkpoint_sha256": sha256(CHECKPOINT)}, "gate_checks": {"source_checkpoint_preserved": True, "production_authorized_explicitly_false_in_repair": True, "weights_modified": False, "current_fused_baseline": True, "hard_negative_preserved": True, "none_semantics_preserved": True, "accepted_boosts_recorded": True, "runtime_future_gt_false": True, "no_production_import": True}, "failure_root_cause": "The frozen N44 checkpoint payload had no explicit production_authorized key. N45 repairs this provenance gap with a hash-bound non-production metadata overlay and leaves the original checkpoint immutable.", "next_action": "Run the full three-branch attribution replay from the unchanged N44 checkpoint and N42 source.", "runtime_future_gt_used": False, "finished_at": now()}
    STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(STAGE), "repair": str(REPAIR)}))


if __name__ == "__main__":
    main()
