"""Separated, sign-correct interaction-effect metrics and sequence bootstrap."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np


BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 7202


class AssignmentChangeType(str, Enum):
    UNCHANGED = "UNCHANGED"
    TRUE_CORRECT_CROSSING = "TRUE_CORRECT_CROSSING"
    TRUE_INCORRECT_CROSSING = "TRUE_INCORRECT_CROSSING"
    DIRECTIONAL_IMPROVEMENT = "DIRECTIONAL_IMPROVEMENT"
    DIRECTIONAL_REGRESSION = "DIRECTIONAL_REGRESSION"
    NEUTRAL_CHANGE = "NEUTRAL_CHANGE"


def identity_error_reduction(*, baseline_correct: bool, treatment_correct: bool) -> int:
    """Return +1 for wrong→correct and -1 for correct→wrong."""

    return int(not baseline_correct) - int(not treatment_correct)


def delta_iou(*, baseline_iou: float, treatment_iou: float) -> float:
    return float(treatment_iou) - float(baseline_iou)


def optional_composite_utility(*, baseline_iou: float, treatment_iou: float, baseline_correct: bool, treatment_correct: bool) -> float:
    """Historical composite with the corrected identity term; secondary only."""

    return 0.5 * delta_iou(baseline_iou=baseline_iou, treatment_iou=treatment_iou) + 0.5 * (
        int(treatment_correct) - int(baseline_correct)
    )


def classify_assignment_change(
    *,
    assignment_changed: bool,
    baseline_correct: bool,
    treatment_correct: bool,
    baseline_iou: float,
    treatment_iou: float,
    epsilon: float = 1.0e-9,
) -> AssignmentChangeType:
    """Classify assignment changes without calling directional change true."""

    if not assignment_changed:
        return AssignmentChangeType.UNCHANGED
    if not baseline_correct and treatment_correct:
        return AssignmentChangeType.TRUE_CORRECT_CROSSING
    if baseline_correct and not treatment_correct:
        return AssignmentChangeType.TRUE_INCORRECT_CROSSING
    difference = float(treatment_iou) - float(baseline_iou)
    if difference > float(epsilon) and not treatment_correct:
        return AssignmentChangeType.DIRECTIONAL_IMPROVEMENT
    if difference < -float(epsilon) and not (baseline_correct and not treatment_correct):
        return AssignmentChangeType.DIRECTIONAL_REGRESSION
    return AssignmentChangeType.NEUTRAL_CHANGE


def _sequence_means(values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for sequence, event_values in values.items():
        array = np.asarray(list(event_values), dtype=np.float64)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"sequence {sequence} must have finite non-empty event values")
        means[str(sequence)] = float(np.mean(array))
    return means


def sequence_cluster_bootstrap(
    values_by_sequence: Mapping[str, Sequence[float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Bootstrap independent sequences after averaging all events per sequence."""

    if int(repetitions) <= 0:
        raise ValueError("repetitions must be positive")
    means = _sequence_means(values_by_sequence)
    names = sorted(means)
    if not names:
        return {
            "mean": None,
            "lower": None,
            "upper": None,
            "clusters": 0,
            "unit": "independent_sequence",
            "within_cluster_aggregation": "mean_event_value",
            "seed": int(seed),
            "repetitions": int(repetitions),
        }
    values = np.asarray([means[name] for name in names], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        draw = rng.integers(0, len(values), size=len(values))
        samples[index] = float(np.mean(values[draw]))
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "clusters": len(names),
        "unit": "independent_sequence",
        "within_cluster_aggregation": "mean_event_value",
        "sequence_means": means,
        "seed": int(seed),
        "repetitions": int(repetitions),
    }


def metric_record(
    *,
    baseline_iou: float,
    treatment_iou: float,
    baseline_correct: bool,
    treatment_correct: bool,
    assignment_changed: bool,
    epsilon: float = 1.0e-9,
) -> dict[str, Any]:
    classification = classify_assignment_change(
        assignment_changed=assignment_changed,
        baseline_correct=baseline_correct,
        treatment_correct=treatment_correct,
        baseline_iou=baseline_iou,
        treatment_iou=treatment_iou,
        epsilon=epsilon,
    )
    return {
        "identity_error_reduction": identity_error_reduction(
            baseline_correct=baseline_correct, treatment_correct=treatment_correct
        ),
        "delta_iou": delta_iou(baseline_iou=baseline_iou, treatment_iou=treatment_iou),
        "composite_utility_secondary": optional_composite_utility(
            baseline_iou=baseline_iou,
            treatment_iou=treatment_iou,
            baseline_correct=baseline_correct,
            treatment_correct=treatment_correct,
        ),
        "assignment_change_type": classification.value,
        "true_correct_crossing": classification is AssignmentChangeType.TRUE_CORRECT_CROSSING,
        "true_incorrect_crossing": classification is AssignmentChangeType.TRUE_INCORRECT_CROSSING,
        "directional_improvement": classification is AssignmentChangeType.DIRECTIONAL_IMPROVEMENT,
        "directional_regression": classification is AssignmentChangeType.DIRECTIONAL_REGRESSION,
    }


__all__ = [
    "AssignmentChangeType",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "classify_assignment_change",
    "delta_iou",
    "identity_error_reduction",
    "metric_record",
    "optional_composite_utility",
    "sequence_cluster_bootstrap",
]
