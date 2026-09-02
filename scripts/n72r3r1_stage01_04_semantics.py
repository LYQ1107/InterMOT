#!/usr/bin/env python3
"""Run the prereplay semantic probes and write one artifact per repair stage."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment
from sam3_intermot.evaluation.interaction_effect_metrics import (
    AssignmentChangeType,
    metric_record,
    sequence_cluster_bootstrap,
)


OUT = ROOT / "outputs/N72R3R1"
STAGES = OUT / "stage_status"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(stage: str, value: str, artifact: Path, **extra: object) -> None:
    atomic_json(
        STAGES / f"{stage}.json",
        {
            "schema_version": "N72R3R1_STAGE_STATUS_V1",
            "stage": stage,
            "status": value,
            "artifact": str(artifact),
            "runtime_future_gt_used": False,
            "scientific_result": "NO_SCIENTIFIC_RESULT",
            **extra,
        },
    )


def stage01_metric() -> None:
    wrong_to_correct = metric_record(
        baseline_iou=0.1,
        treatment_iou=0.7,
        baseline_correct=False,
        treatment_correct=True,
        assignment_changed=True,
    )
    correct_to_wrong = metric_record(
        baseline_iou=0.7,
        treatment_iou=0.1,
        baseline_correct=True,
        treatment_correct=False,
        assignment_changed=True,
    )
    unchanged_error = metric_record(
        baseline_iou=0.1,
        treatment_iou=0.1,
        baseline_correct=False,
        treatment_correct=False,
        assignment_changed=False,
    )
    assert wrong_to_correct["identity_error_reduction"] == 1
    assert correct_to_wrong["identity_error_reduction"] == -1
    assert unchanged_error["identity_error_reduction"] == 0
    # This is the old sign error, evaluated only as a diagnostic contrast.
    old_wrong_to_correct_identity_term = 0.5 * (int(False) - int(True))
    new_wrong_to_correct_identity_term = 0.5 * (int(True) - int(False))
    artifact = OUT / "metric_direction_tests.json"
    atomic_json(
        artifact,
        {
            "schema_version": "N72R3R1_METRIC_DIRECTION_TESTS_V1",
            "status": "PASS_METRIC_DIRECTION_SEMANTICS",
            "primary": "identity_error_reduction = baseline_error - treatment_error",
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "unchanged": unchanged_error,
            "old_wrong_to_correct_identity_term": old_wrong_to_correct_identity_term,
            "new_wrong_to_correct_identity_term": new_wrong_to_correct_identity_term,
            "optional_composite_is_secondary": True,
            "runtime_future_gt_used": False,
        },
    )
    status("stage_01_metric_direction", "PASS_METRIC_DIRECTION_SEMANTICS", artifact)


def stage02_solver() -> None:
    rows = [
        {"candidate_index": 0, "candidate_uid": "toy:c0"},
        {"candidate_index": 1, "candidate_uid": "toy:c1"},
    ]
    states = [
        {"association_state_id": 17, "public_id": 1007},
        {"association_state_id": 23, "public_id": 1042},
    ]
    scores = np.asarray([[0.9, 0.0], [0.8, -1.0]], dtype=np.float64)
    artifact_value = solve_effect_assignment(
        candidate_rows=rows,
        persistent_states=states,
        fused_state_candidate_scores=scores,
        source_run_id="n72r3r1-toy",
        session_id="toy-session",
    )
    by_candidate = {int(item["candidate_index"]): item for item in artifact_value["assignment_rows"]}
    assert by_candidate[0]["public_id"] == 1007
    assert by_candidate[1]["public_id"] is None
    assert by_candidate[1]["status"] == "EXPLICIT_NONE"
    assert artifact_value["public_id_axis"] == [1007, 1042]
    assert artifact_value["association_state_axis"] == [17, 23]
    all_negative = solve_effect_assignment(
        candidate_rows=rows,
        persistent_states=states,
        fused_state_candidate_scores=[[-0.1, -0.2], [-0.3, -0.4]],
        source_run_id="n72r3r1-toy-negative",
        session_id="toy-session",
    )
    assert all(item["status"] == "EXPLICIT_NONE" for item in all_negative["assignment_rows"])
    artifact = OUT / "exact_none_solver_tests.json"
    atomic_json(
        artifact,
        {
            "schema_version": "N72R3R1_EXACT_NONE_SOLVER_TESTS_V1",
            "status": "PASS_EXACT_NONE_SOLVER",
            "solver_source": "sam3_intermot.association.public_assignment.solve_exact_public_assignment",
            "wrapper": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
            "axis_separation": {"association_state_axis": [17, 23], "public_id_axis": [1007, 1042]},
            "counterexample": {
                "state_x_candidate_scores": scores.tolist(),
                "explicit_none_assignment_rows": artifact_value["assignment_rows"],
                "explicit_none_count": artifact_value["explicit_none_count"],
            },
            "all_negative_assignment_rows": all_negative["assignment_rows"],
            "outer_birth_is_downstream_of_solver": True,
            "runtime_future_gt_used": False,
        },
    )
    status("stage_02_exact_none_solver", "PASS_EXACT_NONE_SOLVER", artifact)


def stage03_crossing() -> None:
    cases = {
        "unchanged": metric_record(
            baseline_iou=0.7, treatment_iou=0.7, baseline_correct=True, treatment_correct=True, assignment_changed=False
        ),
        "true_correct": metric_record(
            baseline_iou=0.1, treatment_iou=0.7, baseline_correct=False, treatment_correct=True, assignment_changed=True
        ),
        "true_incorrect": metric_record(
            baseline_iou=0.7, treatment_iou=0.1, baseline_correct=True, treatment_correct=False, assignment_changed=True
        ),
        "directional_improvement": metric_record(
            baseline_iou=0.1, treatment_iou=0.2, baseline_correct=False, treatment_correct=False, assignment_changed=True
        ),
        "directional_regression": metric_record(
            baseline_iou=0.7, treatment_iou=0.6, baseline_correct=False, treatment_correct=False, assignment_changed=True
        ),
        "neutral": metric_record(
            baseline_iou=0.2, treatment_iou=0.2, baseline_correct=False, treatment_correct=False, assignment_changed=True
        ),
    }
    expected = {
        "unchanged": AssignmentChangeType.UNCHANGED.value,
        "true_correct": AssignmentChangeType.TRUE_CORRECT_CROSSING.value,
        "true_incorrect": AssignmentChangeType.TRUE_INCORRECT_CROSSING.value,
        "directional_improvement": AssignmentChangeType.DIRECTIONAL_IMPROVEMENT.value,
        "directional_regression": AssignmentChangeType.DIRECTIONAL_REGRESSION.value,
        "neutral": AssignmentChangeType.NEUTRAL_CHANGE.value,
    }
    assert {key: value["assignment_change_type"] for key, value in cases.items()} == expected
    assert cases["directional_improvement"]["true_correct_crossing"] is False
    artifact = OUT / "crossing_taxonomy_tests.json"
    atomic_json(
        artifact,
        {
            "schema_version": "N72R3R1_CROSSING_TAXONOMY_TESTS_V1",
            "status": "PASS_CROSSING_TAXONOMY",
            "cases": cases,
            "true_correct_requires": "assignment_changed and baseline_correct=false and treatment_correct=true",
            "runtime_future_gt_used": False,
        },
    )
    status("stage_03_crossing_taxonomy", "PASS_CROSSING_TAXONOMY", artifact)


def stage04_bootstrap() -> None:
    artifact_value = sequence_cluster_bootstrap(
        {"sequence-A": [1.0, -1.0], "sequence-B": [1.0]}, seed=7202, repetitions=2000
    )
    assert artifact_value["sequence_means"] == {"sequence-A": 0.0, "sequence-B": 1.0}
    assert artifact_value["clusters"] == 2
    artifact = OUT / "bootstrap_tests.json"
    atomic_json(
        artifact,
        {
            "schema_version": "N72R3R1_BOOTSTRAP_TESTS_V1",
            "status": "PASS_SEQUENCE_CLUSTER_BOOTSTRAP",
            "input_event_values": {"sequence-A": [1.0, -1.0], "sequence-B": [1.0]},
            "result": artifact_value,
            "coverage": "all events retained before within-sequence mean; sequence is bootstrap unit",
            "runtime_future_gt_used": False,
        },
    )
    status("stage_04_sequence_bootstrap", "PASS_SEQUENCE_CLUSTER_BOOTSTRAP", artifact)


def main() -> int:
    stage01_metric()
    stage02_solver()
    stage03_crossing()
    stage04_bootstrap()
    audit = {
        "schema_version": "N72R3R1_SEMANTIC_AUDIT_V1",
        "status": "PASS_STAGE01_04_SEMANTIC_REPAIRS",
        "stages": [
            "stage_01_metric_direction",
            "stage_02_exact_none_solver",
            "stage_03_crossing_taxonomy",
            "stage_04_sequence_bootstrap",
        ],
        "production_model_changed": False,
        "candidate_stream_changed": False,
        "event_protocol_changed": False,
        "runtime_future_gt_used": False,
        "artifacts": {
            name: sha256(OUT / name)
            for name in (
                "metric_direction_tests.json",
                "exact_none_solver_tests.json",
                "crossing_taxonomy_tests.json",
                "bootstrap_tests.json",
            )
        },
    }
    atomic_json(OUT / "semantic_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
