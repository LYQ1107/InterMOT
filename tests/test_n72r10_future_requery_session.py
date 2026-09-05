"""Toy-only contracts for the N72R10 future-frame re-query adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.reacquisition.future_requery_session import (
    FUTURE_FRAME_REQUERY,
    FutureFrameRequerySession,
    query_box,
)
from sam3_intermot.reacquisition.target_candidate_pool import (
    build_candidate_pool_with_future_requery,
)


def _observation(
    frame: int,
    box: list[float],
    raw_id: int,
    *,
    mask: np.ndarray | None = None,
) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=frame,
        sam_object_id=raw_id + 100,
        raw_sam_object_id=raw_id,
        mask=np.ones((8, 8), dtype=bool) if mask is None else mask,
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.9,
        presence_score=0.8,
        source="toy_official_prompt",
    )


@dataclass
class _FakeBackend:
    serial: int
    closed: bool = False

    def start_video(self, _source: str) -> str:
        return f"fake-backend-session-{self.serial}"

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, *, backend, event_id, sequence, public_id, event_frame, target_session_scope, frame_offset, isolate_official_target_state=True, preserve_official_action_history=False):
        self.backend = backend
        self.event_id = str(event_id)
        self.sequence = str(sequence)
        self.public_id = int(public_id)
        self.event_frame = int(event_frame)
        self.frame_offset = int(frame_offset)
        self.target_session_scope = str(target_session_scope)
        self.isolate_official_target_state = bool(isolate_official_target_state)
        self.preserve_official_action_history = bool(preserve_official_action_history)
        self.session_id = None
        self._box = None
        self._raw_id = 1000 + backend.serial
        self.closed = False

    def start(self, video_source, *, main_y_pre_frozen):
        assert main_y_pre_frozen is True
        self.session_id = self.backend.start_video(str(video_source))
        return self.session_id

    def seed_from_human_box(self, human_box):
        self._box = [float(value) for value in human_box]
        return [_observation(0, self._box, self._raw_id)]

    def candidate_at(self, frame):
        assert int(frame) == self.event_frame
        return _observation(0, self._box, self._raw_id)

    def propagate_to(self, end_frame):
        outputs = {}
        for frame in range(self.event_frame, int(end_frame) + 1):
            delta = float(frame - self.event_frame)
            box = [self._box[0] + delta, self._box[1], self._box[2] + delta, self._box[3]]
            outputs[frame] = [_observation(frame - self.frame_offset, box, self._raw_id)]
        return outputs

    def audit(self):
        return {
            "schema_version": "TOY_FAKE_SESSION_AUDIT",
            "session_id": self.session_id,
            "target_raw_sam_id": self._raw_id,
            "event_frame_memory_read": False,
            "first_memory_visible_frame": self.event_frame + 1,
            "runtime_future_gt_used": False,
            "closed": self.closed,
        }

    def close(self):
        if not self.closed:
            self.backend.close()
            self.closed = True


class _ZeroAreaAbsentSession(_FakeSession):
    """Official-like missing row: ID retained, empty mask, zero-area box."""

    def propagate_to(self, end_frame):
        outputs = super().propagate_to(end_frame)
        outputs[7] = [
            _observation(
                2,
                [0.0, 0.0, 0.0, 0.0],
                self._raw_id,
                mask=np.zeros((8, 8), dtype=bool),
            )
        ]
        return outputs


class _ZeroBoxNonEmptyMaskSession(_FakeSession):
    """Official-like inconsistent row: a usable mask with a bad box."""

    def propagate_to(self, end_frame):
        outputs = super().propagate_to(end_frame)
        outputs[7] = [
            _observation(
                2,
                [0.0, 0.0, 0.0, 0.0],
                self._raw_id,
                mask=np.ones((8, 8), dtype=bool),
            )
        ]
        return outputs


def _make_frames(tmp_path: Path, count: int = 12) -> list[Path]:
    paths = []
    for frame in range(count):
        path = tmp_path / f"frame_{frame:03d}.jpg"
        path.write_bytes(f"toy-frame-{frame}".encode("ascii"))
        paths.append(path)
    return paths


def _make_session(tmp_path: Path, *, session_factory=_FakeSession):
    backends: list[_FakeBackend] = []

    def backend_factory():
        backend = _FakeBackend(len(backends))
        backends.append(backend)
        return backend

    def feature_fn(_frame: int, _box):
        feature = np.zeros(512, dtype=np.float32)
        feature[0] = 1.0
        return feature

    session = FutureFrameRequerySession(
        backend_factory=backend_factory,
        sequence="toy-sequence",
        event_id="toy-event",
        event_frame=4,
        target_public_id=1007,
        frame_paths=_make_frames(tmp_path),
        feature_fn=feature_fn,
        session_factory=session_factory,
    )
    return session, backends


def test_query_uses_causal_prediction_and_closes_every_probe(tmp_path: Path):
    session, backends = _make_session(tmp_path)
    started = session.start_from_frame(
        5,
        [10.0, 20.0, 30.0, 60.0],
        {"previous_raw_sam_id": 17, "velocity": [1.0, 0.0, 1.0, 0.0]},
        end_frame=8,
    )
    assert started["local_frame_zero_global"] == 5
    candidates = session.query_current_frame()
    assert len(candidates) == 4
    assert candidates[0]["requery_box_xyxy"] == query_box(
        [10.0, 20.0, 30.0, 60.0], {"dx_fraction": 0.0, "dy_fraction": 0.0, "scale": 1.0, "name": "x"}
    )
    assert candidates[0]["candidate_source"] == FUTURE_FRAME_REQUERY
    assert all(row["public_id"] is None for row in candidates)
    assert all(row["runtime_future_gt_used"] is False for row in candidates)
    assert all(backend.closed for backend in backends)
    assert all(item["status"] == "PASS_QUERY_CURRENT_FRAME" for item in session.audit()["query_audits"])
    session.close()


def test_selected_query_is_rerun_once_and_rebind_keeps_public_id(tmp_path: Path):
    session, backends = _make_session(tmp_path)
    session.start_from_frame(5, [10.0, 20.0, 30.0, 60.0], {"last_raw_sam_id": 17}, end_frame=8)
    candidates = session.query_current_frame()
    result = session.propagate_if_selected(
        selected_candidate_uid=candidates[2]["candidate_uid"],
        selection_audit={"selector": "toy_fixed", "runtime_future_gt_used": False},
        margin=0.2,
    )
    assert len(result) == 4
    assert {row["frame"] for row in result} == {5, 6, 7, 8}
    audit = session.audit()
    assert audit["status"] == "PASS_SELECTED"
    assert audit["selected_query_name"] == "PREDICTED_LEFT"
    assert audit["raw_rebinding"]["old_raw_sam_id"] == 17
    assert audit["raw_rebinding"]["new_source"] == FUTURE_FRAME_REQUERY
    assert audit["raw_rebinding"]["public_id"] == 1007
    assert audit["raw_rebinding"]["public_id_changed"] is False
    assert backends[-1].closed is False
    session.close()
    assert backends[-1].closed is True


def test_zero_area_empty_mask_is_explicit_absence_not_synthetic_candidate(tmp_path: Path):
    session, backends = _make_session(tmp_path, session_factory=_ZeroAreaAbsentSession)
    session.start_from_frame(5, [10.0, 20.0, 30.0, 60.0], {"last_raw_sam_id": 17}, end_frame=8)
    candidates = session.query_current_frame()
    future = session.propagate_if_selected(
        selected_candidate_uid=candidates[0]["candidate_uid"],
        selection_audit={"selector": "toy_fixed"},
    )
    audit = session.audit()
    coverage = {int(item["global_frame"]): item for item in audit["future_frame_coverage"]}
    assert coverage[7]["candidate_count"] == 0
    assert not any(int(row["frame"]) == 7 for row in future)
    assert len(audit["invalid_observation_audit"]) == 1
    assert audit["invalid_observation_audit"][0]["status"] == (
        "LEGITIMATELY_ABSENT_OFFICIAL_ZERO_AREA_EMPTY_MASK"
    )
    assert audit["invalid_observation_audit"][0]["action"] == (
        "EXCLUDE_FROM_CANDIDATE_STREAM_NO_SYNTHETIC_BOX"
    )
    session.close()
    assert all(backend.closed for backend in backends)


def test_nonempty_official_mask_repairs_only_its_box(tmp_path: Path):
    session, _ = _make_session(tmp_path, session_factory=_ZeroBoxNonEmptyMaskSession)
    session.start_from_frame(5, [10.0, 20.0, 30.0, 60.0], {"last_raw_sam_id": 17}, end_frame=8)
    candidates = session.query_current_frame()
    future = session.propagate_if_selected(
        selected_candidate_uid=candidates[0]["candidate_uid"],
        selection_audit={"selector": "toy_fixed"},
    )
    audit = session.audit()
    repaired = [item for item in audit["invalid_observation_audit"] if item["frame"] == 7]
    assert len(repaired) == 1
    assert repaired[0]["status"] == "REPAIRED_BOX_FROM_OFFICIAL_NONEMPTY_MASK"
    row = next(item for item in future if int(item["frame"]) == 7)
    assert row["box_xyxy"] == [0.0, 0.0, 8.0, 8.0]
    assert row["official_box_xyxy_raw"] == [0.0, 0.0, 0.0, 0.0]
    assert row["box_provenance"] == "DETERMINISTIC_OFFICIAL_MASK_TO_BOX_REPAIR"
    session.close()


def test_none_closes_without_future_candidates(tmp_path: Path):
    session, backends = _make_session(tmp_path)
    session.start_from_frame(5, [10.0, 20.0, 30.0, 60.0], {"current_raw_sam_id": 17}, end_frame=8)
    session.query_current_frame()
    assert session.propagate_if_selected(selection_audit={"selector": "toy_none"}) == []
    audit = session.audit()
    assert audit["status"] == "CLOSED_NONE"
    assert audit["selection"]["status"] == "NONE"
    assert audit["future_candidate_count"] == 0
    assert all(backend.closed for backend in backends)


def test_runtime_gt_metadata_is_rejected(tmp_path: Path):
    session, _ = _make_session(tmp_path)
    with pytest.raises(ValueError, match="forbidden runtime metadata"):
        session.start_from_frame(5, [10.0, 20.0, 30.0, 60.0], {"future_gt_box": [0, 0, 1, 1]})


def test_future_pool_is_distinct_from_historical_requery_and_public_id_free():
    feature = [0.0] * 512
    feature[0] = 1.0
    main = [{"candidate_uid": "main", "box_xyxy": [0, 0, 2, 2], "feature": feature, "confidence": 0.5}]
    future = [
        {
            "candidate_uid": "future",
            "candidate_kind": "FUTURE_FRAME_REQUERY_CANDIDATE",
            "box_xyxy": [0, 0, 2, 2],
            "feature": feature,
            "confidence": 0.8,
            "public_id": None,
        }
    ]
    pool, audit = build_candidate_pool_with_future_requery(
        main,
        future_requery_candidates=future,
        sequence="toy",
        frame=5,
    )
    assert pool[-1]["candidate_source"] == FUTURE_FRAME_REQUERY
    assert audit["future_frame_requery_candidate_count"] == 1
    assert pool[-1]["public_id"] is None
