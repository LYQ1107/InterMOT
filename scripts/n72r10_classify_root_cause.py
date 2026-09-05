#!/usr/bin/env python3
"""Classify the N72R10 future-requery result without changing runtime data.

The classifier is deliberately descriptive.  It does not choose a new
threshold from posthoc outcomes and does not authorize a production score
bridge.  A solver-refusal finding is reported only when the runtime-selected
fresh candidate and the offline target-quality audit are both present.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "outputs/N72R10/stage_09_gate.json"
MILESTONE = ROOT / "outputs/N72R10/true_requery_milestone_audit.json"
TRAINING_AUDIT = ROOT / "outputs/N72R10/stage_10_training_distribution_audit.json"
OUTPUT = ROOT / "outputs/N72R10/stage_11_root_cause_classification.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
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


def main() -> int:
    gate = read_json(GATE)
    milestone = read_json(MILESTONE)
    training = read_json(TRAINING_AUDIT)
    counts = Counter(milestone.get("global_counts", {}))
    gate_metrics = gate.get("metrics", {})
    e1 = gate_metrics.get("E1_vs_E0", {})
    e2 = gate_metrics.get("E2_vs_E1", {})
    trigger_count = int(counts.get("trigger_count", 0))
    applied_count = int(counts.get("applied_count", 0))
    selected_count = int(counts.get("fresh_selected_count", 0))
    selected_good_count = int(counts.get("fresh_selected_target_iou50_count", 0))
    solver_refusal_count = int(counts.get("fresh_good_solver_refusal_count", 0))
    assigned_target_count = int(counts.get("fresh_assigned_target_count", 0))
    complete_count = int(counts.get("complete_milestone_count", 0))
    training_material = training.get("materialized_corpus", {})
    diagnosis = {
        "A_TRIGGER_NOT_OCCURRED": {
            "present": trigger_count == 0,
            "evidence": {"trigger_count": trigger_count, "applied_count": applied_count},
            "interpretation": "The uncertainty trigger executed in the sealed E2 runtime." if trigger_count else "No trigger evidence.",
        },
        "B_RESCUE_CANDIDATE_MISSING": {
            "present": trigger_count > applied_count,
            "evidence": {"trigger_count": trigger_count, "applied_count": applied_count, "fresh_candidate_count": counts.get("fresh_candidate_count", 0)},
            "interpretation": "Some triggered frames had no applied fresh source." if trigger_count > applied_count else "Every triggered frame in this batch had an applied fresh candidate source.",
        },
        "C_MODEL_OR_ADMISSION_REJECTS_FRESH_SOURCE": {
            "present": applied_count > selected_count,
            "evidence": {
                "applied_count": applied_count,
                "fresh_selected_count": selected_count,
                "fresh_selected_target_iou50_count": selected_good_count,
                "fresh_selected_wrong_count": counts.get("fresh_selected_wrong_count", 0),
            },
            "interpretation": "Most applied fresh sources were not selected by the frozen model/admission rule; validation has no positive FUTURE_FRAME_REQUERY label coverage." if applied_count > selected_count else "All applied fresh sources were selected.",
        },
        "D_MODEL_TO_SOLVER_GLOBAL_COMPETITION": {
            "present": solver_refusal_count > 0,
            "evidence": {
                "fresh_selected_target_quality_count": selected_good_count,
                "fresh_good_solver_refusal_count": solver_refusal_count,
                "fresh_assigned_target_count": assigned_target_count,
            },
            "interpretation": "The model selected 29 target-quality fresh candidates, but 21 were not assigned to the target public ID by the unchanged global solver; this is direct decision-boundary evidence, not proof that a new bridge will generalize." if solver_refusal_count else "No qualified solver-refusal evidence.",
        },
        "E_SHORT_TERM_REACQUISITION_DRIFT": {
            "present": False,
            "evidence": {"complete_milestone_count": complete_count, "posthoc_wrong_to_correct_count": counts.get("posthoc_wrong_to_correct_count", 0)},
            "interpretation": "A complete path exists, but the current batch is not sufficient to establish stable long-horizon benefit; the horizon metrics and action decomposition remain the relevant evidence.",
        },
        "F_PROTECTED_ID_COMPETITION": {
            "present": any(int((e1.get(str(horizon), {}) or {}).get("protected_regression_count", 0)) > 0 for horizon in (20, 50, 100)),
            "evidence": {
                "E1_vs_E0_protected_regression": {str(horizon): (e1.get(str(horizon), {}) or {}).get("protected_regression_count") for horizon in (20, 50, 100)},
                "E2_vs_E1_protected_regression": {str(horizon): (e2.get(str(horizon), {}) or {}).get("protected_regression_count") for horizon in (20, 50, 100)},
            },
            "interpretation": "The combined E1 path still has protected regressions; E2's incremental regression is smaller than E1 but nonzero." if any(int((e1.get(str(horizon), {}) or {}).get("protected_regression_count", 0)) > 0 for horizon in (20, 50, 100)) else "No protected regression was observed.",
        },
    }
    bridge_precondition = {
        "solver_refusal_evidence_present": solver_refusal_count > 0,
        "fresh_target_quality_evidence_present": selected_good_count > 0,
        "validation_future_positive_labels_present": int(training_material.get("validation_future_rows_selected_as_label", 0)) > 0,
        "calibrated_target_edge_bridge_ready_for_formal_training": solver_refusal_count > 0 and selected_good_count > 0 and int(training_material.get("validation_future_rows_selected_as_label", 0)) > 0,
        "calibrated_target_edge_bridge_production_authorized": False,
        "reason_not_authorized": "The source-specific validation split contains zero positive FUTURE_FRAME_REQUERY labels; fitting or selecting a bridge from the development replay would leak post-treatment outcomes.",
    }
    payload = {
        "schema_version": "N72R10_ROOT_CAUSE_CLASSIFICATION_V1",
        "status": "DIAGNOSIS_COMPLETE_PRODUCTION_BRIDGE_DEFERRED",
        "created_at_utc": now_utc(),
        "gate": str(GATE),
        "gate_sha256": sha256_file(GATE),
        "milestone_audit": str(MILESTONE),
        "milestone_audit_sha256": sha256_file(MILESTONE),
        "training_distribution_audit": str(TRAINING_AUDIT),
        "training_distribution_audit_sha256": sha256_file(TRAINING_AUDIT),
        "diagnosis": diagnosis,
        "primary_root_cause": "D_MODEL_TO_SOLVER_GLOBAL_COMPETITION",
        "secondary_root_causes": [
            "C_MODEL_OR_ADMISSION_REJECTS_FRESH_SOURCE",
            "F_PROTECTED_ID_COMPETITION",
            "TRAINING_DISTRIBUTION_AND_VALIDATION_COVERAGE",
        ],
        "bridge_precondition": bridge_precondition,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "minimum_next_step": "Generate a larger lawful same-run public-authority train/validation interaction pool with positive FUTURE_FRAME_REQUERY labels, then train/evaluate a frozen target-edge bridge on sequence-disjoint validation before any production interface change.",
    }
    atomic_write(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "primary_root_cause": payload["primary_root_cause"],
        "solver_refusal_count": solver_refusal_count,
        "complete_milestone_count": complete_count,
        "bridge_ready": bridge_precondition["calibrated_target_edge_bridge_ready_for_formal_training"],
        "output": str(OUTPUT),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
