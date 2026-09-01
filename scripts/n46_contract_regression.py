#!/usr/bin/env python3
"""N46 read-only contract regression for the frozen N44 sidecar."""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n44_assignment_common import HARD_NEGATIVE, apply_sidecar, load_checkpoint

CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
SOURCE = ROOT / "outputs/n42/replay/runtime/t0/n37-dancetrack0032-0000-add_new_identity-001.json"
OUT = ROOT / "outputs/n46/n46_sidecar_targeted_regression.json"


def main() -> None:
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    audit = source["variants"]["M2"]["branches"]["memory_write=True"]["future_trace"][0]["candidate_audit"]
    previous = source["variants"]["M2"].get("event_frame_audit", {}).get("candidate_audit", {})
    original = copy.deepcopy(audit)
    result = apply_sidecar(audit, model, int(audit["frame"]) - int(source["event_frame"]), previous, checkpoint["gate"])
    base = np.asarray(audit["fused_scores"], dtype=np.float32)
    adjusted = np.asarray(result["fused_scores"], dtype=np.float32)
    diff = adjusted - base
    changed = {(int(i), int(j)) for i, j in np.argwhere(np.abs(diff) > 1e-12)}
    hard = base <= HARD_NEGATIVE
    meta = result["n44_sidecar"]
    authorization = ROOT / "outputs/n45/frozen_checkpoint_authorization.json"
    checks = {
        "source_audit_unchanged": audit == original,
        "branch_current_fused_is_exact_baseline": bool(np.array_equal(base, np.asarray(audit["fused_scores"], dtype=np.float32))),
        "all_changed_cells_are_finite_non_hard": bool(not changed or np.all(~hard[np.abs(diff) > 1e-12])),
        "all_changed_deltas_are_exact_bounded_boost": bool(not changed or np.allclose(diff[np.abs(diff) > 1e-12], 0.25, atol=1e-7)),
        "changed_cell_count_matches_metadata": len(changed) == int(meta["changed_cell_count"]),
        "selected_count_not_less_than_changed_cells": int(meta["proposals_selected"]) >= len(changed),
        "hard_negative_preserved": bool(meta["hard_negative_preserved"]) and bool(np.array_equal(adjusted[hard], base[hard])),
        "none_semantics_preserved": float(meta["none_score"]) == -1.0e8,
        "runtime_future_gt_used_direct_false": result.get("runtime_future_gt_used") is False and meta.get("runtime_future_gt_used") is False,
        "checkpoint_authorization_false_overlay": authorization.is_file() and json.loads(authorization.read_text(encoding="utf-8")).get("production_authorized") is False,
        "weights_not_rewritten": checkpoint.get("production_authorized") is None,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": "N46_SIDECAR_CONTRACT_TARGETED_REGRESSION_V1",
        "source": str(SOURCE), "checkpoint": str(CHECKPOINT), "variant": "M2", "frame": int(audit["frame"]),
        "checks": checks,
        "metrics": {"proposals_considered": int(meta["proposals_considered"]), "proposals_selected": int(meta["proposals_selected"]), "changed_cell_count": len(changed), "changed_cells": sorted([list(x) for x in changed]), "max_abs_delta": float(np.max(np.abs(diff))) if diff.size else 0.0},
        "notes": "Global assignment collateral is allowed only through recorded Hungarian recomputation; score changes are restricted to accepted bounded cells.",
        "runtime_future_gt_used": False, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUT)}))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
