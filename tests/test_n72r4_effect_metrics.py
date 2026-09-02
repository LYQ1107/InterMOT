"""Toy tests for sign and crossing semantics; not scientific results."""

from __future__ import annotations

from sam3_intermot.evaluation.interaction_effect_metrics import (
    AssignmentChangeType,
    classify_assignment_change,
    identity_error_reduction,
    metric_record,
    sequence_cluster_bootstrap,
)


def test_identity_error_reduction_sign() -> None:
    assert identity_error_reduction(baseline_correct=False, treatment_correct=True) == 1
    assert identity_error_reduction(baseline_correct=True, treatment_correct=False) == -1
    assert identity_error_reduction(baseline_correct=False, treatment_correct=False) == 0


def test_wrong_to_correct_is_true_crossing() -> None:
    value = metric_record(
        baseline_iou=0.1,
        treatment_iou=0.7,
        baseline_correct=False,
        treatment_correct=True,
        assignment_changed=True,
    )
    assert value["assignment_change_type"] == AssignmentChangeType.TRUE_CORRECT_CROSSING.value
    assert value["true_correct_crossing"] is True


def test_correct_to_wrong_is_true_incorrect_crossing() -> None:
    value = metric_record(
        baseline_iou=0.7,
        treatment_iou=0.1,
        baseline_correct=True,
        treatment_correct=False,
        assignment_changed=True,
    )
    assert value["assignment_change_type"] == AssignmentChangeType.TRUE_INCORRECT_CROSSING.value
    assert value["true_incorrect_crossing"] is True


def test_directional_improvement_is_not_true_correct() -> None:
    value = metric_record(
        baseline_iou=0.1,
        treatment_iou=0.2,
        baseline_correct=False,
        treatment_correct=False,
        assignment_changed=True,
    )
    assert value["assignment_change_type"] == AssignmentChangeType.DIRECTIONAL_IMPROVEMENT.value
    assert value["true_correct_crossing"] is False


def test_sequence_bootstrap_preserves_multiple_events() -> None:
    result = sequence_cluster_bootstrap({"A": [1.0, -1.0], "B": [1.0]}, seed=7202, repetitions=2000)
    assert result["sequence_means"]["A"] == 0.0
    assert result["sequence_means"]["B"] == 1.0
    assert result["clusters"] == 2
    assert result["within_cluster_aggregation"] == "mean_event_value"

