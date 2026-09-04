"""Focused toy contracts for the N72R7 candidate-pool/selector adapter."""

from __future__ import annotations

import numpy as np
import pytest

from sam3_intermot.reacquisition.target_candidate_pool import (
    MAIN_B0_CANDIDATE,
    TARGET_SESSION_CURRENT_RAW,
    build_candidate_pool,
)
from sam3_intermot.reacquisition.target_candidate_selector import (
    TargetCandidateSelector,
    TargetSelectionContext,
)


def _feature(value: float) -> list[float]:
    result = [0.0] * 512
    result[0] = value
    result[1] = (1.0 - value * value) ** 0.5
    return result


def _main(uid: str = "main:0") -> dict:
    return {
        "candidate_uid": uid,
        "box_xyxy": [0.0, 0.0, 10.0, 10.0],
        "official_raw_sam_id": 3,
        "native_scope": "scope-a",
        "confidence": 0.9,
        "feature": _feature(1.0),
        "solver_public_id": 1007,
    }


def _target(uid: str = "target:0", *, public_id=None) -> dict:
    return {
        "candidate_uid": uid,
        "candidate_kind": "TARGET_CORRECTION_SESSION_CANDIDATE",
        "box_xyxy": [0.0, 0.0, 10.0, 10.0],
        "official_raw_sam_id": 8,
        "native_scope": "target-session",
        "source_session_id": "toy-session",
        "confidence": 0.9,
        "feature": _feature(1.0),
        "public_id": public_id,
    }


def test_pool_keeps_complete_sources_but_hides_identity_authority() -> None:
    pool, audit = build_candidate_pool(
        [_main()],
        [_target()],
        sequence="toy",
        frame=11,
        include_target_session=True,
    )

    assert [item["candidate_source"] for item in pool] == [
        MAIN_B0_CANDIDATE,
        TARGET_SESSION_CURRENT_RAW,
    ]
    assert pool[0]["incumbent_public_id_if_any"] == 1007
    assert all(item["public_id"] is None for item in pool)
    assert audit["all_candidate_public_ids_null_before_solver"] is True
    assert audit["public_id_inference"] is False


def test_target_source_rejects_public_id() -> None:
    with pytest.raises(ValueError, match="public ID"):
        build_candidate_pool(
            [_main()],
            [_target(public_id=1007)],
            sequence="toy",
            frame=11,
            include_target_session=True,
        )


def test_pool_rejects_duplicate_candidate_uid() -> None:
    with pytest.raises(ValueError, match="candidate UID collision"):
        build_candidate_pool(
            [_main("same")],
            [_target("same")],
            sequence="toy",
            frame=11,
            include_target_session=True,
        )


def test_pool_retains_finite_degenerate_box_as_audited_candidate() -> None:
    candidate = _main()
    candidate["box_xyxy"] = [5.0, 5.0, 5.0, 6.0]
    pool, _ = build_candidate_pool(
        [candidate],
        sequence="toy",
        frame=11,
        include_target_session=False,
    )
    assert len(pool) == 1
    assert pool[0]["geometry_valid"] is False


def test_selector_is_future_only_and_returns_candidate_without_public_id() -> None:
    pool, _ = build_candidate_pool(
        [_main()],
        [_target()],
        sequence="toy",
        frame=11,
        include_target_session=True,
    )
    selector = TargetCandidateSelector()
    context = TargetSelectionContext(
        human_anchor=np.asarray(_feature(1.0), dtype=np.float32),
        predicted_box=[0.0, 0.0, 10.0, 10.0],
        previous_raw_sam_id=8,
        previous_native_scope="target-session",
        frame=11,
        event_frame=10,
        memory_read=True,
    )
    result = selector.select(
        pool,
        context=context,
        base_target_scores={item["candidate_uid"]: 0.0 for item in pool},
    )

    assert result["selected_candidate_uid"] in {item["candidate_uid"] for item in pool}
    assert result["event_frame_memory_read"] is False
    assert result["public_id_inference"] is False
    assert all(item["public_id"] is None for item in pool)

    context.frame = 10
    with pytest.raises(ValueError, match="after the event frame"):
        selector.select(
            pool,
            context=context,
            base_target_scores={item["candidate_uid"]: 0.0 for item in pool},
        )
