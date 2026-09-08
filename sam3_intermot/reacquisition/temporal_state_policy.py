"""Single causal temporal-state contract for N72R11R3.

The corpus builder, V3 self-rollout, and on-demand replay must update exactly
the same state.  This module intentionally contains no GT, public-ID labels,
or posthoc values.  ``target_uid`` is the result of the exact runtime solver;
``selected_uid`` is the scorer's own candidate selection and is never inferred
from GT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .target_candidate_pool import FUTURE_FRAME_REQUERY
from .target_candidate_selector import box_iou


MEMORY_FEATURE_DIM = 512
TEMPORAL_FEATURE_DIM = 8
TEMPORAL_HORIZON = 100
TEMPORAL_HORIZON_INDEX = 0
TEMPORAL_CAUSAL_TOP_INDEX = 1
TEMPORAL_CAUSAL_SECOND_INDEX = 2
TEMPORAL_CAUSAL_MARGIN_INDEX = 3
TEMPORAL_PREVIOUS_SCORE_INDEX = 4
TEMPORAL_PREVIOUS_UNCERTAINTY_INDEX = 5
TEMPORAL_TRUSTED_AGE_INDEX = 6
TEMPORAL_HAS_FUTURE_REQUERY_INDEX = 7

RECENT_TRUSTED_SLOTS = 4
LONG_TERM_TRUSTED_SLOTS = 4
DISTRACTOR_SLOTS = 8
ADMISSION_SCORE = 0.50
ADMISSION_MARGIN = 0.20
LONG_TERM_MARGIN = 0.30
LONG_TERM_STRIDE = 5

TEMPORAL_FEATURE_SCHEMA = (
    "frame_horizon_over_100",
    "tanh_causal_top_score",
    "tanh_causal_second_score",
    "tanh_causal_margin",
    "previous_fused_target_score_clipped",
    "previous_assignment_uncertainty",
    "trusted_age_over_100",
    "has_future_frame_requery_candidate",
)


def _unit_feature(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != MEMORY_FEATURE_DIM or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite {MEMORY_FEATURE_DIM}-D")
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-6:
        raise ValueError(f"{label} has zero norm")
    return (array / norm).astype(np.float32)


def _box(value: Sequence[float], label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be four finite coordinates")
    return [float(item) for item in array]


@dataclass
class TemporalIdentityState:
    """Causal state shared by corpus generation and runtime replay."""

    predicted_box: list[float]
    previous_raw_sam_id: int | None = None
    previous_native_scope: str | None = None
    previous_score: float = 0.0
    previous_uncertainty: float = 1.0
    trusted_age: int = 0
    recent_trusted: list[np.ndarray] = field(default_factory=list)
    long_term_trusted: list[np.ndarray] = field(default_factory=list)
    distractors: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.predicted_box = _box(self.predicted_box, "predicted_box")
        self.previous_score = float(self.previous_score)
        self.previous_uncertainty = float(self.previous_uncertainty)
        self.trusted_age = int(self.trusted_age)
        if not np.isfinite(self.previous_score) or not np.isfinite(self.previous_uncertainty):
            raise ValueError("temporal scalar state must be finite")
        self.recent_trusted = [_unit_feature(value, "recent trusted feature") for value in self.recent_trusted]
        self.long_term_trusted = [_unit_feature(value, "long-term trusted feature") for value in self.long_term_trusted]
        self.distractors = [_unit_feature(value, "distractor feature") for value in self.distractors]


def initialize_temporal_state(
    *,
    anchor_feature: Sequence[float],
    anchor_box: Sequence[float],
    previous_raw_sam_id: int | None = None,
    previous_native_scope: str | None = None,
) -> TemporalIdentityState:
    """Initialize state from the correction anchor, before event+1."""

    anchor = _unit_feature(anchor_feature, "anchor feature")
    return TemporalIdentityState(
        predicted_box=_box(anchor_box, "anchor_box"),
        previous_raw_sam_id=None if previous_raw_sam_id is None else int(previous_raw_sam_id),
        previous_native_scope=None if previous_native_scope is None else str(previous_native_scope),
        recent_trusted=[anchor.copy()],
        long_term_trusted=[anchor.copy()],
    )


def build_temporal_features(
    state: TemporalIdentityState,
    *,
    frame_horizon: int,
    causal_top_score: float,
    causal_second_score: float,
    causal_margin: float,
    has_future_requery: bool,
) -> np.ndarray:
    """Build the exact eight-dimensional temporal feature schema."""

    values = np.asarray(
        [
            float(max(int(frame_horizon), 0)) / float(TEMPORAL_HORIZON),
            float(np.tanh(float(causal_top_score))),
            float(np.tanh(float(causal_second_score))),
            float(np.tanh(float(causal_margin))),
            float(np.clip(float(state.previous_score), -1.0, 1.0)),
            float(np.clip(float(state.previous_uncertainty), 0.0, 1.0)),
            float(min(max(int(state.trusted_age), 0), TEMPORAL_HORIZON)) / float(TEMPORAL_HORIZON),
            float(bool(has_future_requery)),
        ],
        dtype=np.float32,
    )
    if values.shape != (TEMPORAL_FEATURE_DIM,) or not np.isfinite(values).all():
        raise RuntimeError("temporal feature construction produced a non-finite 8-D vector")
    return values


def _candidate_feature(candidate: Mapping[str, Any], label: str) -> np.ndarray | None:
    value = candidate.get("feature")
    if value is None:
        return None
    return _unit_feature(value, label)


def _candidate_uid(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_uid")
    if value is None:
        raise ValueError("candidate is missing candidate_uid")
    return str(value)


def _candidate_binding(candidate: Mapping[str, Any]) -> tuple[int | None, str | None]:
    raw = candidate.get("official_raw_sam_id")
    scope = candidate.get("native_scope", candidate.get("native_tid_scope"))
    return (None if raw is None else int(raw), None if scope is None else str(scope))


def _append_bounded(values: list[np.ndarray], value: np.ndarray, slots: int) -> None:
    values.append(_unit_feature(value, "state update feature"))
    del values[:-int(slots)]


def update_temporal_state(
    state: TemporalIdentityState,
    *,
    candidates: Sequence[Mapping[str, Any]],
    target_uid: str | None,
    selected_uid: str | None,
    selected_score: float | None,
    selected_margin: float | None,
    fused_target_scores: Sequence[float],
    frame_horizon: int,
    assigned_candidate: Mapping[str, Any] | None = None,
    base_top_score: float | None = None,
    base_second_score: float | None = None,
) -> dict[str, Any]:
    """Apply one causal update using exact-solver assignment and scorer choice.

    Geometry/native binding follows the exact solver assignment even when the
    scorer did not select that candidate.  Only a scorer selection that agrees
    with the exact target assignment can enter trusted memory.  A selected
    non-target candidate enters the distractor bank; unselected candidates do
    not pollute it.
    """

    candidate_by_uid = {_candidate_uid(candidate): candidate for candidate in candidates}
    if len(candidate_by_uid) != len(candidates):
        raise ValueError("temporal update candidate UIDs are not unique")
    if len(fused_target_scores) != len(candidates):
        raise ValueError("fused target score axis does not match candidate axis")
    fused = np.asarray(fused_target_scores, dtype=np.float64)
    if not np.isfinite(fused).all():
        raise ValueError("fused target scores are non-finite")
    assigned = assigned_candidate
    if assigned is None and target_uid is not None:
        assigned = candidate_by_uid.get(str(target_uid))
    if target_uid is not None and assigned is None:
        raise ValueError("target_uid is not present in candidate axis")
    selected = None if selected_uid is None else candidate_by_uid.get(str(selected_uid))
    if selected_uid is not None and selected is None:
        raise ValueError("selected_uid is not present in candidate axis")

    old_box = list(state.predicted_box)
    old_raw = state.previous_raw_sam_id
    old_scope = state.previous_native_scope
    trusted_before = len(state.recent_trusted)
    long_before = len(state.long_term_trusted)
    distractor_before = len(state.distractors)
    target_agreement = bool(target_uid is not None and selected_uid is not None and str(target_uid) == str(selected_uid))
    score_ok = selected_score is not None and np.isfinite(float(selected_score)) and float(selected_score) >= ADMISSION_SCORE
    margin_ok = selected_margin is not None and np.isfinite(float(selected_margin)) and float(selected_margin) >= ADMISSION_MARGIN
    feature = None if selected is None else _candidate_feature(selected, "selected candidate feature")
    trusted_admitted = bool(target_agreement and score_ok and margin_ok and feature is not None)

    if assigned is not None:
        state.predicted_box = _box(assigned.get("box_xyxy", assigned.get("box")), "assigned candidate box")
        state.previous_raw_sam_id, state.previous_native_scope = _candidate_binding(assigned)
    if trusted_admitted:
        assert feature is not None
        _append_bounded(state.recent_trusted, feature, RECENT_TRUSTED_SLOTS)
        source = str(selected.get("candidate_source", selected.get("source_kind", "UNKNOWN")))
        if float(selected_margin) >= LONG_TERM_MARGIN and source != FUTURE_FRAME_REQUERY and int(frame_horizon) % LONG_TERM_STRIDE == 0:
            _append_bounded(state.long_term_trusted, feature, LONG_TERM_TRUSTED_SLOTS)
        state.trusted_age = 0
    else:
        state.trusted_age += 1

    distractor_added = False
    if selected is not None and not target_agreement:
        distractor_feature = _candidate_feature(selected, "selected distractor feature")
        if distractor_feature is not None:
            _append_bounded(state.distractors, distractor_feature, DISTRACTOR_SLOTS)
            distractor_added = True

    state.previous_score = float(np.max(fused, initial=0.0))
    margin_for_uncertainty = float(max(float(base_top_score or 0.0) - float(base_second_score or 0.0), 0.0))
    state.previous_uncertainty = float(1.0 / (1.0 + margin_for_uncertainty))
    return {
        "target_uid": None if target_uid is None else str(target_uid),
        "selected_uid": None if selected_uid is None else str(selected_uid),
        "selected_target_agreement": target_agreement,
        "trusted_admitted": trusted_admitted,
        "trusted_admission_score": None if selected_score is None else float(selected_score),
        "trusted_admission_margin": None if selected_margin is None else float(selected_margin),
        "distractor_added": distractor_added,
        "assigned_candidate_used_for_geometry": assigned is not None,
        "binding_before": {"official_raw_sam_id": old_raw, "native_scope": old_scope},
        "binding_after": {"official_raw_sam_id": state.previous_raw_sam_id, "native_scope": state.previous_native_scope},
        "predicted_box_before": old_box,
        "predicted_box_after": list(state.predicted_box),
        "memory_sizes_before": {"recent_trusted": trusted_before, "long_term_trusted": long_before, "distractors": distractor_before},
        "memory_sizes_after": {"recent_trusted": len(state.recent_trusted), "long_term_trusted": len(state.long_term_trusted), "distractors": len(state.distractors)},
        "previous_score_after": float(state.previous_score),
        "previous_uncertainty_after": float(state.previous_uncertainty),
        "trusted_age_after": int(state.trusted_age),
        "runtime_future_gt_used": False,
        "gt_used_for_runtime_decision": False,
    }


def state_memory_arrays(state: TemporalIdentityState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-size memory arrays and masks for the V3 model."""

    def pad(values: Sequence[np.ndarray], slots: int) -> tuple[np.ndarray, np.ndarray]:
        array = np.zeros((int(slots), MEMORY_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(int(slots), dtype=np.bool_)
        for index, value in enumerate(list(values)[-int(slots):]):
            array[index] = _unit_feature(value, "memory array value")
            mask[index] = True
        return array, mask

    recent, recent_mask = pad(state.recent_trusted, RECENT_TRUSTED_SLOTS)
    long_term, long_mask = pad(state.long_term_trusted, LONG_TERM_TRUSTED_SLOTS)
    distractors, distractor_mask = pad(state.distractors, DISTRACTOR_SLOTS)
    return recent, recent_mask, long_term, long_mask, distractors, distractor_mask


def state_audit(state: TemporalIdentityState) -> dict[str, Any]:
    return {
        "predicted_box": list(state.predicted_box),
        "previous_raw_sam_id": state.previous_raw_sam_id,
        "previous_native_scope": state.previous_native_scope,
        "previous_score": float(state.previous_score),
        "previous_uncertainty": float(state.previous_uncertainty),
        "trusted_age": int(state.trusted_age),
        "recent_trusted_size": len(state.recent_trusted),
        "long_term_trusted_size": len(state.long_term_trusted),
        "distractor_size": len(state.distractors),
        "runtime_future_gt_used": False,
    }


__all__ = [
    "TEMPORAL_FEATURE_DIM",
    "TEMPORAL_HORIZON",
    "TEMPORAL_FEATURE_SCHEMA",
    "RECENT_TRUSTED_SLOTS",
    "LONG_TERM_TRUSTED_SLOTS",
    "DISTRACTOR_SLOTS",
    "ADMISSION_SCORE",
    "ADMISSION_MARGIN",
    "LONG_TERM_MARGIN",
    "LONG_TERM_STRIDE",
    "TemporalIdentityState",
    "initialize_temporal_state",
    "build_temporal_features",
    "update_temporal_state",
    "state_memory_arrays",
    "state_audit",
]
