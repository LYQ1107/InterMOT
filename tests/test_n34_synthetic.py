import sys

import numpy as np

sys.path.insert(0, "scripts")

from n34_synthetic import build_tape
from sam3_intermot.association.ccam_replay import paired_replay, validate_candidate_tape
from sam3_intermot.association.state_manager import StateManagerConfig


def test_all_n34_event_tapes_are_explicitly_candidate_complete_synthetic():
    for action in (
        "ADD_NEW_IDENTITY",
        "AUTHORITATIVE_REASSIGN",
        "ATOMIC_ID_SWAP",
        "RECOVER_IDENTITY",
    ):
        validation = validate_candidate_tape(build_tape(action, future_frames=3))
        assert validation["valid"]
        assert validation["candidate_complete"]
        assert validation["issues"] == []


def test_n34_m0_is_disabled_and_m2_differs_only_after_future_boundary():
    tape = build_tape("AUTHORITATIVE_REASSIGN", future_frames=3)
    base = StateManagerConfig(variant="reid")
    m0 = paired_replay(
        tape,
        config=base,
        write_branch_uses_appearance_memory=False,
    )
    m2 = paired_replay(
        tape,
        config=StateManagerConfig(
            variant="reid",
            use_appearance_memory=True,
            appearance_anchor_cap=8,
            appearance_negative_cap=0,
        ),
        write_branch_uses_appearance_memory=True,
    )
    assert all(item["max_abs_score_delta"] == 0.0 for item in m0["comparison"])
    assert m2["comparison"][0]["frame"] == 1
    assert m2["comparison"][0]["max_abs_score_delta"] > 0.0
    assert np.isfinite(m2["comparison"][0]["max_abs_score_delta"])
