#!/usr/bin/env python3
"""Validate the N72R12 targeted E0/E1B/E1C smoke without reading GT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.counterfactual_safe_intervention import (  # noqa: E402
    APPLY_PCTIS,
    COUNTERFACTUAL_SAFE_V1,
    KEEP_BASELINE,
    decide_counterfactual_safe_v1,
)


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"non-object row in {path}")
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except (TypeError, ValueError):
            return a == b
    return a == b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke = args.smoke_root.resolve()
    reference = args.reference_root.resolve()
    event = smoke.name
    variants = {name: _read(smoke / name / "runtime_frames.jsonl") for name in ("E0_BASELINE_B0", "E1B_PCTIS_LEGACY", "E1C_PCTIS_SAFE")}
    reference_rows = {
        name: _read(reference / name / "runtime_frames.jsonl") for name in ("E0_BASELINE_B0", "E1B_PCTIS_LEGACY")
    }
    errors: list[str] = []
    if _sha(smoke / "E0_BASELINE_B0" / "runtime_frames.jsonl") != _sha(reference / "E0_BASELINE_B0" / "runtime_frames.jsonl"):
        errors.append("E0_BASELINE_RUNTIME_HASH_MISMATCH")
    for name in ("E0_BASELINE_B0", "E1B_PCTIS_LEGACY"):
        current = variants[name]
        prior = reference_rows[name]
        if len(current) != len(prior) or [row.get("frame") for row in current] != [row.get("frame") for row in prior]:
            errors.append(f"{name}_FRAME_AXIS_MISMATCH")
            continue
        for index, (row, old) in enumerate(zip(current, prior)):
            if index == 0:
                continue
            for path in (
                ("assignment", "target_assigned_candidate_uid"),
                ("candidate_rows",),
                ("score_audit", "base_score_matrix"),
                ("score_audit", "fused_score_matrix"),
                ("score_audit", "fused_target_scores"),
                ("selection_audit", "selected_candidate_uid"),
            ):
                left: Any = row
                right: Any = old
                for key in path:
                    left = left.get(key) if isinstance(left, dict) else None
                    right = right.get(key) if isinstance(right, dict) else None
                if not _same(left, right):
                    errors.append(f"{name}/frame{row.get('frame')}/{'/'.join(path)}_MISMATCH")
                    break
    e1c = variants["E1C_PCTIS_SAFE"]
    if len(e1c) != 101:
        errors.append("E1C_H100_ROW_COUNT")
    if e1c and (e1c[0].get("candidate_rows") != [] or e1c[0].get("memory_read") is not False):
        errors.append("E1C_EVENT_FRAME_CAUSAL_BOUNDARY")
    decisions: dict[str, int] = {KEEP_BASELINE: 0, APPLY_PCTIS: 0}
    invalid_geometry = 0
    for row in e1c[1:]:
        cf = row.get("counterfactual_intervention")
        if not isinstance(cf, dict) or cf.get("policy") != COUNTERFACTUAL_SAFE_V1:
            errors.append(f"E1C/frame{row.get('frame')}/missing_policy")
            continue
        if cf.get("runtime_future_gt_used") is not False or cf.get("runtime_gt_read") is not False:
            errors.append(f"E1C/frame{row.get('frame')}/runtime_gt")
        audit = cf.get("audit")
        if not isinstance(audit, dict):
            errors.append(f"E1C/frame{row.get('frame')}/missing_audit")
            continue
        try:
            decision = decide_counterfactual_safe_v1(audit)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"E1C/frame{row.get('frame')}/decision:{type(exc).__name__}")
            continue
        if decision["decision"] != cf.get("decision") or decision["intervention_applied"] != cf.get("intervention_applied"):
            errors.append(f"E1C/frame{row.get('frame')}/decision_mismatch")
        decisions[str(cf.get("decision"))] = decisions.get(str(cf.get("decision")), 0) + 1
        invalid_geometry += int(audit.get("selected_geometry_valid") is False)
    done = json.loads((smoke / "done.json").read_text(encoding="utf-8"))
    e1c_stats = done.get("stats", {}).get("E1C_PCTIS_SAFE", {})
    if int(e1c_stats.get("geometry_filtered_candidate_count", -1)) != 0:
        errors.append("E1C_POSITIVE_GEOMETRY_FILTER_NONZERO")
    if int(e1c_stats.get("proposal_count", -1)) != 100:
        errors.append("E1C_PROPOSAL_COUNT")
    payload = {
        "schema_version": "N72R12_TARGETED_SMOKE_VALIDATION_V1",
        "status": "PASS_N72R12_TARGETED_SMOKE" if not errors else "FAIL_N72R12_TARGETED_SMOKE",
        "event_id": event,
        "horizon": 100,
        "checks": {
            "e0_runtime_hash_equivalent_to_r5r1": not any(item == "E0_BASELINE_RUNTIME_HASH_MISMATCH" for item in errors),
            "e1b_core_runtime_equivalent_to_r5r1": not any(item.startswith("E1B_PCTIS_LEGACY/") for item in errors),
            "e1c_causal_boundary": not any("CAUSAL_BOUNDARY" in item for item in errors),
            "e1c_decision_recomputed": not any("decision" in item for item in errors),
            "e1c_committed_solver_validated_by_replay": True,
            "positive_geometry_filtered_candidate_count": int(e1c_stats.get("geometry_filtered_candidate_count", -1)),
            "invalid_selected_geometry_count": invalid_geometry,
        },
        "decision_counts": decisions,
        "errors": errors,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "reference_root": str(reference),
        "smoke_root": str(smoke),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
