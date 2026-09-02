"""Focused tests for the official adapter candidate-to-prompt bridge."""

from __future__ import annotations

import numpy as np
import pytest

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.backend.sam3_backend import Sam3Backend


def _backend() -> Sam3Backend:
    backend = Sam3Backend(device="cpu")
    # Explicit toy model guard: this focused unit test exercises only the
    # adapter registry bridge and must not load the official checkpoint.
    backend._ensure_model = lambda: None
    backend._session_id = "toy-session"
    backend._frame_w = 1920
    backend._frame_h = 1080
    return backend


def _observation(sam_id: int, raw_id: int, box: list[float]) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=10,
        sam_object_id=sam_id,
        raw_sam_object_id=raw_id,
        mask=np.ones((2, 2), dtype=bool),
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.9,
        source="concept_detection",
    )


def test_detected_observation_registers_explicit_native_mapping() -> None:
    backend = _backend()
    observation = _observation(4, 17, [100, 100, 180, 240])

    assert backend.register_detected_observation(observation) == 4
    assert backend._objects[4]["source"] == "detected_candidate_registration"
    assert backend._ext_to_sam == {4: 17}
    assert backend._sam_to_ext == {17: 4}


def test_detected_observation_registration_rejects_conflicting_reuse() -> None:
    backend = _backend()
    backend.register_detected_observation(_observation(4, 17, [100, 100, 180, 240]))

    with pytest.raises(ValueError, match="different box"):
        backend.register_detected_observation(_observation(4, 17, [300, 300, 380, 440]))
