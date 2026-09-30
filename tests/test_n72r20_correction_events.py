from __future__ import annotations

import copy

import pytest

from sam3_intermot.identity_memory.correction_update import CorrectionMemoryUpdater
from sam3_intermot.interaction.correction_event import HumanCorrectionEvent


def _vector(value: float) -> tuple[float, ...]:
    return (value,) + (0.0,) * 511


def test_correction_event_round_trips_and_requires_correction_target_for_switch() -> None:
    event = HumanCorrectionEvent(
        frame_id=12,
        sequence_id="dancetrack0001",
        identity_id=17,
        predicted_candidate_id="wrong",
        corrected_candidate_id="correct",
        predicted_identity_state=_vector(1.0),
        corrected_embedding=_vector(0.5),
        candidate_context=({"candidate_id": "wrong", "rank": 1},),
        timestamp="2026-09-30T00:00:00+00:00",
        error_type="identity_switch",
        hard_negative=True,
    )

    restored = HumanCorrectionEvent.from_dict(event.to_dict())
    assert restored == event
    with pytest.raises(ValueError, match="requires corrected_candidate_id"):
        HumanCorrectionEvent(
            frame_id=12,
            sequence_id="dancetrack0001",
            identity_id=17,
            predicted_candidate_id="wrong",
            corrected_candidate_id=None,
            predicted_identity_state=_vector(1.0),
            corrected_embedding=_vector(0.5),
            candidate_context=(),
            timestamp="now",
            error_type="identity_switch",
        )


def test_missed_target_accepts_human_crop_without_candidate_id() -> None:
    event = HumanCorrectionEvent(
        frame_id=12,
        sequence_id="dancetrack0001",
        identity_id=17,
        predicted_candidate_id=None,
        corrected_candidate_id=None,
        predicted_identity_state=_vector(1.0),
        corrected_embedding=_vector(0.5),
        candidate_context=(),
        timestamp="now",
        error_type="missed_target",
    )
    assert event.to_dict()["corrected_candidate_id"] is None


def test_dummy_correction_updater_preserves_memory_and_records_io() -> None:
    state = {"embedding": [1.0, 0.0]}
    wrong = {"candidate_id": "wrong"}
    corrected = {"candidate_id": "correct"}
    before = copy.deepcopy(state)
    result = CorrectionMemoryUpdater().update_from_correction(state, wrong, corrected)

    assert result["updated"] is False
    assert result["memory_state_before"] == before
    assert result["memory_state_after"] == before
    state["embedding"][0] = 0.0
    assert result["memory_state_after"] == before
