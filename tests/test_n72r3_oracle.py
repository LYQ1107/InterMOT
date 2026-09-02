"""CPU-only toy tests for the isolated N72R3 simulated-human oracle."""

import numpy as np

from sam3_intermot.simulation.human_oracle import SimulatedHumanOracle


def gt(gt_id: int, box=(0.0, 0.0, 10.0, 10.0)) -> dict:
    return {"boxes": [np.asarray(box, dtype=float)], "gt_ids": [gt_id]}


def test_unknown_identity_requires_explicit_outer_allocation() -> None:
    oracle = SimulatedHumanOracle("toy")
    decisions = oracle.choose_actions(3, gt(7), [{"candidate_uid": "c7", "box": [0, 0, 10, 10]}])
    assert decisions[0].action_type == "ADD_NEW_IDENTITY"
    assert decisions[0].target_public_id is None
    assert decisions[0].mapping_confirmation_required is False
    oracle.commit_mapping(7, 1007, reason="outer_allocator_birth")
    assert oracle.gt_to_public == {7: 1007}


def test_known_wrong_assignment_is_reassign_not_native_id_inference() -> None:
    oracle = SimulatedHumanOracle("toy", known_gt_to_public={7: 1007})
    decisions = oracle.choose_actions(
        4,
        gt(7),
        [{"candidate_uid": "native-88", "native_tid": 88, "public_id": 2002, "box": [0, 0, 10, 10]}],
    )
    assert decisions[0].action_type == "AUTHORITATIVE_REASSIGN"
    assert decisions[0].target_public_id == 1007


def test_reciprocal_known_assignments_are_one_atomic_swap() -> None:
    oracle = SimulatedHumanOracle("toy", known_gt_to_public={7: 1007, 8: 1008})
    decisions = oracle.choose_actions(
        5,
        {"boxes": [[0, 0, 10, 10], [20, 0, 30, 10]], "gt_ids": [7, 8]},
        [
            {"candidate_uid": "a", "public_id": 1008, "box": [0, 0, 10, 10]},
            {"candidate_uid": "b", "public_id": 1007, "box": [20, 0, 30, 10]},
        ],
    )
    swaps = [item for item in decisions if item.action_type == "ATOMIC_ID_SWAP"]
    assert len(swaps) == 1
    assert swaps[0].other_public_id in {1007, 1008}


def test_future_gt_field_is_rejected() -> None:
    oracle = SimulatedHumanOracle("toy")
    try:
        oracle.choose_actions(1, {"boxes": [], "gt_ids": [], "future_boxes": []}, [])
    except ValueError as exc:
        assert "current-frame GT only" in str(exc)
    else:
        raise AssertionError("future GT field was accepted")
