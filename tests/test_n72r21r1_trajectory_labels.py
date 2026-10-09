from sam3_intermot.evaluation.joint_trajectory_labels import non_target_damage, memory_safety, label_branch


def row(frame, target='p', other='q'):
    return {'frame': frame, 'target_public_id': 1, 'outputs': [{'public_id': 1, 'candidate_uid': target}, {'public_id': 2, 'candidate_uid': other}],
            'births': [], 'deaths': [], 'memory_write': False, 'memory_write_uid': None,
            'target_native_after': 1 if target == 'p' else 2, 'target_prototype_anchor_cosine_after': 1. if target == 'p' else .5}


def test_non_target_damage_counts_ids_not_target_and_preserves_UNKNOWN():
    result = non_target_damage(row(1, 'q', 'u'), row(1), {'p': 11, 'q': 22, 'u': None}, {1: 11, 2: 22}, 1)
    assert result['harmed_public_ids'] == [2] and not result['helped_public_ids']
    assert result['UNKNOWN_damage_is_conservative_origin_proxy_not_verified_takeover']


def test_zero_writes_never_pass_memory_safety():
    result = memory_safety(0, 0, 0, 100)
    assert result['wrong_or_unknown_rate'] is None and not result['safety_and_usefulness_pass']
    assert memory_safety(60, 0, 0, 100)['safety_and_usefulness_pass']
    assert not memory_safety(60, 0, 2, 100)['safety_and_usefulness_pass']


def test_current_correctness_does_not_replace_future_value_or_full_horizon():
    keep = [row(f, 'q', 'p') for f in range(3)]
    treatment = [row(0), row(1, 'q', 'p'), row(2, 'q', 'p')]
    labels = label_branch(treatment, keep, {f: {'p': 11, 'q': 22} for f in range(3)}, 11)
    assert labels['time_zero']['N01'] == 1
    assert labels['future']['H1']['target_value'] == 0
    assert not labels['future']['H100']['complete']
    assert labels['future']['H100']['actual_future_frames'] == 2
    assert labels['future']['H100']['proxy_not_actual_HOTA_or_AssA']


def test_uncommitted_write_fails_closed():
    import pytest
    bad = row(0); bad.update(memory_write=True, memory_write_uid='q')
    with pytest.raises(AssertionError): label_branch([bad], [row(0)], {0: {'p': 11, 'q': 22}}, 11)


def test_restoring_original_other_identity_is_not_bad_continuity():
    keep = [row(f, 'q', 'p') for f in range(2)]
    corrected = [row(f) for f in range(2)]
    truth = {f: {'p': 11, 'q': 22} for f in range(2)}
    local = label_branch(corrected, keep, truth, 11)
    canonical = label_branch(corrected, keep, truth, 11, prestate_origins={1: 11, 2: 22})
    assert local['origin_reference'] == 'SHORT_KEEP_WINDOW_CONTINUITY_ONLY'
    assert canonical['future']['H1']['non_target_benefit'] == 1
    assert canonical['future']['H1']['non_target_damage'] == 0
    assert canonical['future']['H1']['benefit_label']


def test_recovering_same_target_fragment_is_not_other_person_damage():
    keep = {'outputs': [{'public_id': 2, 'candidate_uid': 'p'}, {'public_id': 3, 'candidate_uid': 'q'}]}
    recovered = {'outputs': [{'public_id': 1, 'candidate_uid': 'p'}, {'public_id': 3, 'candidate_uid': 'q'}]}
    result = non_target_damage(recovered, keep, {'p': 11, 'q': 22}, {1: 11, 2: 11, 3: 22}, 1, 11)
    assert result['same_target_fragment_displaced_public_ids'] == [2]
    assert not result['harmed_public_ids'] and not result['helped_public_ids']
