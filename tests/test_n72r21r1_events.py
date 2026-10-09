from sam3_intermot.evaluation.safe_intervention_events import contiguous_intervals, analyze_target_events, identity_outcome


def test_harm_frames_are_not_independent_initiating_actions():
    rows = [{'frame': f, 'C0_correct': True, 'actual_correct': f == 6,
             'own_KEEP_correct': f == 1 or f == 6, 'effective_action': f == 1} for f in range(1, 7)]
    result = analyze_target_events(rows)
    assert result['N10_frames'] == 5
    assert result['direct_harmful_decisions_against_own_KEEP'] == 1
    assert result['N10_frames_without_current_direct_harmful_action'] == 4
    assert result['N10_observed_intervals'][0]['recovery_frame'] == 6
    assert not result['N10_observed_intervals'][0]['causal_origin_proven']


def test_n10_run_may_start_when_baseline_recovers_without_action():
    rows = [{'frame': 7, 'C0_correct': True, 'actual_correct': False,
             'own_KEEP_correct': False, 'effective_action': False}]
    result = analyze_target_events(rows)
    assert result['first_N10'] == 7 and result['first_direct_harmful_action'] is None
    assert not result['N10_observed_intervals'][0]['starts_with_direct_harmful_action']


def test_unknown_is_not_known_other_or_false_negative_presence():
    labels = {'a': 1, 'b': 2, 'u': None}
    assert identity_outcome('a', labels, 1) == 'TARGET'
    assert identity_outcome('b', labels, 1) == 'VERIFIED_OTHER'
    assert identity_outcome('u', labels, 1) == 'UNKNOWN_UNMATCHED'
    assert identity_outcome(None, labels, 1) == 'NONE'
    assert contiguous_intervals([9, 2, 1, 1, 10]) == [
        {'start_frame': 1, 'end_frame': 2, 'frames': 2}, {'start_frame': 9, 'end_frame': 10, 'frames': 2}]
