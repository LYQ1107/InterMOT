import numpy as np
from sam3_intermot.one_click.simple_event_intervention import approve_simple, SelectiveNativeTracker
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.association.opportunity_tracker import OpportunityTracker


def features(**changes):
    return dict(proposal_NONE=0., proposed_probability=.99, candidate_margin=.2, quality=.9, anchor_cosine=.9,
                KEEP_NONE=0., target_LOST=0., base_KEEP_margin=.1, anchor_advantage_vs_KEEP=.3, global_regret=.1,
                displaced_count=0, pending_confirmation_count=3, track_gap=1, KEEP_anchor_cosine=.5,
                NONE_probability=.01, NONE_advantage=-.9) | changes


def test_strict_recovery_does_not_use_low_margin_as_lost():
    assert not approve_simple("P2_RECOVERY_ONLY", features(base_KEEP_margin=-.1), feasible=True, changed=True)[0]
    assert approve_simple("P2_RECOVERY_ONLY", features(KEEP_NONE=1.), feasible=True, changed=True)[0]


def test_competitor_protected_and_hard_feasibility_precede_confidence():
    assert not approve_simple("P5_COMPETITOR_PROTECTION", features(displaced_count=1), feasible=True, changed=True)[0]
    assert not approve_simple("P4_IDENTITY_MARGIN_GLOBAL_REGRET", features(), feasible=False, changed=True)[0]


def test_causal_confirmation_needs_three_not_two():
    assert not approve_simple("P6_PERSISTENT_CAUSAL_CHALLENGER", features(pending_confirmation_count=2), feasible=True, changed=True)[0]
    assert approve_simple("P6_PERSISTENT_CAUSAL_CHALLENGER", features(), feasible=True, changed=True)[0]


def test_selective_none_not_always_none():
    assert not approve_simple("P8_SELECTIVE_ABSTENTION", features(proposal_NONE=1.), feasible=True, changed=True)[0]
    assert approve_simple("P8_SELECTIVE_ABSTENTION", features(proposal_NONE=1., NONE_probability=.98, NONE_advantage=.2), feasible=True, changed=True)[0]


def test_native_discount_keeps_other_columns_core_and_hard_negatives():
    vector = np.zeros(512, np.float32)
    vector[0] = 1.
    event = {"event_frame": 0, "human_anchor": vector, "target_candidate_uid": "0", "target_box_xyxy": [0, 0, 20, 80]}
    tracker = SelectiveNativeTracker(config=AuthorityConfig(mode="off", source="raw", memory="P0", lifecycle="dynamic"), event=event)
    states = [IdentityState(i + 1, vector, [0, 0, 20, 80], 0, native_tid=4) for i in range(2)]
    for i, state in enumerate(states):
        state.public_id = i + 100001
    tracker.target_public = states[0].public_id
    states[0].add_negative(8)
    rows = [{"candidate_uid": str(i), "feature": vector, "box_xyxy": [0, 0, 20, 80], "native_tid": native, "conf": .9} for i, native in enumerate((4, 8))]
    original = OpportunityTracker._scores(tracker, states, rows, 1)
    assert np.array_equal(tracker._scores(states, rows, 1), original)
    tracker.native_discount_active = True
    adjusted = tracker._scores(states, rows, 1)
    assert adjusted[0, 0] == original[0, 0] - 1.5
    assert np.array_equal(adjusted[:, 1], original[:, 1])
    assert adjusted[1, 0] == -1e9
    assert states[0].has_negative(8, 1)
