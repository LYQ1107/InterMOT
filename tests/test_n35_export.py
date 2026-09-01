import numpy as np
import pytest

from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.backend.base import NotSupportedError
from sam3_intermot.backend.mock_backend import MockBackend


def _obs(oid, box):
    mask = np.zeros((20, 30), dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in box]
    mask[y1:y2, x1:x2] = True
    return PromptObjectObservation(
        frame_idx=3,
        sam_object_id=oid,
        mask=mask,
        box_xyxy=np.asarray(box, dtype=float),
        confidence=0.8,
        presence_score=0.7,
        source="automatic_propagation",
    )


def test_adapter_export_retains_candidates_and_marks_unexposed_features():
    backend = Sam3Backend(device="cpu")
    backend._output_cache[3] = [_obs(11, [1, 2, 8, 12]), _obs(22, [10, 3, 18, 14])]
    rows = backend.export_frame_candidates(3)
    assert [row["native_tid"] for row in rows] == [11, 22]
    assert all(row["embedding"] is None for row in rows)
    assert all(row["embedding_status"] == "NOT_EXPOSED" for row in rows)
    assert all(row["feature_source"] == "official_response_no_embedding" for row in rows)
    assert all(np.asarray(row["mask"]).any() for row in rows)


def test_adapter_export_normalizes_machine_roi_features_without_dropping_rows():
    backend = Sam3Backend(device="cpu")
    backend._output_cache[3] = [_obs(11, [1, 2, 8, 12]), _obs(22, [10, 3, 18, 14])]
    rows = backend.export_frame_candidates(
        3,
        embeddings=[np.asarray([3.0, 4.0]), np.asarray([0.0, -2.0])],
    )
    np.testing.assert_allclose(rows[0]["embedding"], [0.6, 0.8])
    np.testing.assert_allclose(rows[1]["embedding"], [0.0, -1.0])
    assert all(row["embedding_status"] == "MACHINE_ROI_FALLBACK" for row in rows)
    assert all(row["feature_source"] == "machine_roi_fallback" for row in rows)


def test_state_audit_contains_public_candidate_matrix_and_mapping():
    manager = StateManager(StateManagerConfig(variant="reid", score_threshold=-100.0))
    obs = [
        {"obs_id": 0, "feat": np.asarray([1.0, 0.0]), "has_feat": 1.0, "box": np.asarray([0, 0, 4, 8]), "native_tid": 11, "native_age": 0.0},
        {"obs_id": 1, "feat": np.asarray([0.0, 1.0]), "has_feat": 1.0, "box": np.asarray([8, 0, 12, 8]), "native_tid": 22, "native_age": 0.0},
    ]
    manager.rollout_frame(0, obs)
    audit = manager.candidate_log[-1]
    assert audit["public_id_order"] == []
    manager.rollout_frame(1, obs)
    audit = manager.candidate_log[-1]
    assert audit["public_id_order"]
    assert len(audit["public_id_score_matrix"]) == len(audit["public_id_order"])
    assert len(audit["public_id_score_matrix"][0]) == len(audit["candidate_order"])
    assert set(audit["public_id_to_native_tid"]) == {str(pid) for pid in audit["public_id_order"]}
    assert audit["assignment_pairs_after_scope"]


def test_old_test_double_explicitly_lacks_candidate_export():
    backend = MockBackend()
    with pytest.raises(NotSupportedError):
        backend.export_frame_candidates(0)
