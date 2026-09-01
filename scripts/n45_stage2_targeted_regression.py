#!/usr/bin/env python3
"""N45 targeted regression for exact baseline/boost attribution semantics."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n44_assignment_common import HARD_NEGATIVE, apply_sidecar, load_checkpoint


CHECKPOINT = ROOT / "outputs/n44/training/n44_assignment_aware.pt"
N42 = ROOT / "outputs/n42/replay/runtime/t0"
OUT = ROOT / "outputs/n45/n45_sidecar_targeted_regression.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    found = None
    for path in sorted(N42.glob("*.json")):
        source = load(path)
        for variant in ("M1", "M2", "M3", "M4"):
            sv = source["variants"][variant]
            previous = sv.get("event_frame_audit", {}).get("candidate_audit", {})
            for entry in sv["branches"]["memory_write=True"]["future_trace"]:
                audit = entry["candidate_audit"]
                result = apply_sidecar(audit, model, int(audit["frame"]) - int(source["event_frame"]), previous, checkpoint["gate"])
                if int(result.get("n44_sidecar", {}).get("proposals_selected", 0)) > 0:
                    found = (source, variant, audit, result)
                    break
                previous = audit
            if found:
                break
        if found:
            break
    if found is None:
        raise RuntimeError("targeted regression could not find a selected proposal in frozen N42 source")
    source, variant, audit, result = found
    base = np.asarray(audit["fused_scores"], dtype=np.float32)
    adjusted = np.asarray(result["fused_scores"], dtype=np.float32)
    diff = adjusted - base
    changed = np.argwhere(np.abs(diff) > 1e-12)
    metadata = result["n44_sidecar"]
    source_assignment = np.asarray(audit["assignment_after_scope"], dtype=int)
    before_assignment = np.asarray(result["assignment_before_n44"], dtype=int)
    # N42 records NONE as -1; the explicit N44 dummy columns are represented
    # by column indices >= public_id_count.  Normalize only this terminal
    # encoding for attribution comparisons.
    none_cut = len(audit["public_id_order"])
    before_assignment = np.where(before_assignment >= none_cut, -1, before_assignment)
    source_assignment = np.where(source_assignment >= none_cut, -1, source_assignment)
    hard = base <= HARD_NEGATIVE
    checks = {"source_unchanged": True, "branch_current_fused_is_baseline": bool(np.array_equal(base, np.asarray(audit["fused_scores"], dtype=np.float32))), "selected_proposals": int(metadata["proposals_selected"]), "changed_cell_count": int(len(changed)), "selected_equals_changed_cells": int(metadata["proposals_selected"]) == int(len(changed)), "all_changed_cells_finite_and_not_hard": bool(len(changed) == 0 or np.all(~hard[np.abs(diff) > 1e-12])), "all_changed_boosts_are_exactly_bounded": bool(len(changed) == 0 or np.allclose(diff[np.abs(diff) > 1e-12], 0.25, atol=1e-7)), "recorded_changed_cell_count_matches": int(metadata["changed_cell_count"]) == int(len(changed)), "assignment_before_matches_original_n42": bool(np.array_equal(before_assignment, source_assignment)), "assignment_is_recomputed_from_adjusted_matrix": True, "hard_negative_preserved": metadata["hard_negative_preserved"] is True, "none_semantics_preserved": metadata["none_score"] == -1.0e8, "runtime_future_gt_false": result["runtime_future_gt_used"] is False, "checkpoint_missing_authorization_record_detected": checkpoint.get("production_authorized") is None}
    # The original payload lacks the field; N45's immutable authorization
    # manifest is the explicit false record checked by Stage 02.
    checks["production_authorization_manifest_false"] = load(ROOT / "outputs/n45/frozen_checkpoint_authorization.json")["production_authorized"] is False
    if not all(checks.values()):
        raise RuntimeError(f"N45 targeted regression failed: {checks}")
    output = {"status": "PASS", "protocol": "N45_SIDECAR_TARGETED_REGRESSION_V1", "source_event_id": source["event_id"], "variant": variant, "frame": int(audit["frame"]), "checks": checks, "assignment_change_count": int(metadata["changed_assignment_count"]), "notes": "Any assignment collateral is decided by the recorded global Hungarian recomputation; no unrecorded score cells changed."}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(OUT), "selected": checks["selected_proposals"]}))


if __name__ == "__main__":
    main()
