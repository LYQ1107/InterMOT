"""CPU-only causal/mapping regression tests for N72.

These are toy contract fixtures, not scientific tracking results.  They guard
the event-frame write/read boundary and ensure explicit mapping failures stay
failures rather than being inferred from visual similarity.
"""

from __future__ import annotations

import numpy as np

from sam3_intermot.association.appearance_memory import AppearanceMemory
from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.association.online_associator import score_matrix_pairwise
from sam3_intermot.provenance.mapping import resolve_exact_mapping


def test_human_memory_write_is_hidden_on_event_frame_and_visible_at_t_plus_one() -> None:
    memory = AppearanceMemory(feat_dim=4, human_weight=1.0, machine_weight=0.35)
    state = IdentityState(7, np.asarray([1, 0, 0, 0], dtype=np.float32), np.asarray([0, 0, 2, 2], dtype=float), 0, 10)
    observation = {
        "obs_id": 1,
        "native_tid": 11,
        "native_age": 1,
        "conf": 1.0,
        "box": np.asarray([0, 0, 2, 2], dtype=float),
        "feat": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "has_feat": 1.0,
    }

    before = {}
    event_scores_before = score_matrix_pairwise([state], [observation], 10, None, appearance_memory=memory, score_audit=before)
    assert before["appearance_score_deltas"] == [[0.0]]

    assert memory.update_from_human(7, 10, observation["feat"], quality=1.0, write_event_id="toy-event")
    after = {}
    event_scores_after = score_matrix_pairwise([state], [observation], 10, None, appearance_memory=memory, score_audit=after)
    future = {}
    future_scores = score_matrix_pairwise([state], [observation], 11, None, appearance_memory=memory, score_audit=future)

    # The full current-frame output and assignment score remain unchanged after
    # the write; only the next frame may observe the human anchor.
    np.testing.assert_array_equal(event_scores_before, event_scores_after)
    assert after["appearance_score_deltas"] == [[0.0]]
    assert future["appearance_score_deltas"][0][0] > 0.0
    assert future_scores[0, 0] > event_scores_after[0, 0]


def test_hard_negative_remains_non_overridable_by_appearance() -> None:
    memory = AppearanceMemory(feat_dim=4, human_weight=8.0)
    state = IdentityState(7, np.asarray([1, 0, 0, 0], dtype=np.float32), np.asarray([0, 0, 2, 2], dtype=float), 0, 10)
    state.add_negative(11)
    observation = {
        "obs_id": 1,
        "native_tid": 11,
        "native_age": 1,
        "conf": 1.0,
        "box": np.asarray([0, 0, 2, 2], dtype=float),
        "feat": np.asarray([0, 1, 0, 0], dtype=np.float32),
        "has_feat": 1.0,
    }
    assert memory.update_from_human(7, 0, observation["feat"], write_event_id="toy-event")
    audit = {}
    scores = score_matrix_pairwise([state], [observation], 1, None, appearance_memory=memory, appearance_score_weight=8.0, score_audit=audit)
    assert scores[0, 0] == -1e9
    assert audit["appearance_score_deltas"][0][0] > 0.0


def test_exact_mapping_requires_authoritative_source_and_all_axes() -> None:
    row = {
        "sequence": "toy",
        "frame": 4,
        "raw_native_id": 17,
        "adapter_external_id": 9001,
        "segment_local_id": "chunk0:17",
        "sequence_global_id": "toy:g17",
    }
    exact = resolve_exact_mapping(row, exact_sources=[{"source": "direct_user_public_id", "public_id": 101}])
    assert exact["status"] == "EXACT"
    assert exact["public_id"] == 101

    no_source = resolve_exact_mapping(row)
    assert no_source["status"] == "UNMAPPED_NO_SOURCE"
    incomplete = dict(row)
    incomplete.pop("sequence_global_id")
    assert resolve_exact_mapping(incomplete, exact_sources=[{"source": "direct_user_public_id", "public_id": 101}])["status"] == "AXIS_MISMATCH"


def test_ambiguous_and_public_absent_are_first_class_not_inferred() -> None:
    row = {
        "sequence": "toy",
        "frame": 4,
        "raw_native_id": 17,
        "adapter_external_id": 9001,
        "segment_local_id": "chunk0:17",
        "sequence_global_id": "toy:g17",
    }
    ambiguous = resolve_exact_mapping(
        row,
        exact_sources=[
            {"source": "identity_registry_binding", "public_id": 101},
            {"source": "explicit_runtime_assignment", "public_id": 202},
        ],
    )
    assert ambiguous["status"] == "AMBIGUOUS_ONE_TO_MANY"
    assert ambiguous["public_id"] is None
    absent = resolve_exact_mapping(row, public_assignment_absent=True)
    assert absent["status"] == "PUBLIC_ASSIGNMENT_ABSENT"
    assert absent["public_id"] is None
