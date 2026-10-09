import pytest
from sam3_intermot.evaluation.delivery_tables import (
    count_metrics, recovery_summary, seed_average_sequence_interval,
    require_complete_validation, unavailable_cross_recording,
)


def counts(**changes):
    c = dict(frames=10, visible_frames=8, available_frames=5, strict_correct=3,
             box_correct=4, writes=0, strict_target_writes=0,
             verified_other_identity_writes=0, UNKNOWN_unmatched_writes=0)
    return {**c, **changes}


def test_zero_write_rate_is_undefined_and_usefulness_fails():
    r = count_metrics(counts())
    assert r['conservative_non_target_or_unverified_write_rate'] is None
    assert r['strict_correct_observation_retention'] == 0 and not r['memory_joint_gate']


def test_missing_candidate_visible_frames_remain_in_recall():
    r = count_metrics(counts())
    assert r['strict_UID_target_recall'] == 3/8
    assert r['strict_recognition_recall_given_available'] == 3/5
    assert r['candidate_coverage_given_visible'] == 5/8


def test_UNKNOWN_write_is_not_verified_other_but_fails_conservative_gate():
    r = count_metrics(counts(writes=4, strict_target_writes=3, UNKNOWN_unmatched_writes=1))
    assert r['conservative_non_target_or_unverified_write_rate'] == .25
    assert not r['memory_joint_gate'] and r['counts']['verified_other_identity_writes'] == 0


@pytest.mark.parametrize('changes', [{'frames':-1}, {'strict_correct':6}, {'writes':1}])
def test_inconsistent_actual_count_evidence_is_rejected(changes):
    with pytest.raises(ValueError): count_metrics(counts(**changes))


def test_seed_average_inside_sequence_is_not_seed_IID_and_no_partial_cases():
    b = {'seed':72104, 'replicates':2000}
    r = seed_average_sequence_interval({'seed1':{'A':0.,'B':1.},'seed2':{'A':1.,'B':1.}}, b)
    assert r['sequence_macro_mean'] == .75 and r['eligible_sequences'] == ['A','B']
    with pytest.raises(ValueError, match='incomplete'): seed_average_sequence_interval({'s1':{'A':0.},'s2':{}}, b)


def test_failed_recovery_windows_are_not_dropped_from_recall():
    def w(ok): return dict(strict_identity_recovered=ok, strict_delay_seconds=.5 if ok else None,
                           gap_bin='short_le5', first_accept_strict_correct=ok,
                           first_accept_verified_wrong_identity=not ok, no_accept_in_return_window=False)
    r = recovery_summary([{'strict_UID_reappearance_windows':[w(True),w(False)],'strict_verified_other_identity_takeover_runs':[]}])
    assert r['all_return_windows']['strict_identity_recovery_recall_all_windows'] == .5
    assert r['all_return_windows']['failed_recovery_windows'] == 1
    assert r['all_return_windows']['recovered_only_median_strict_delay_seconds'] == .5


def test_partial_frozen_VAL_cannot_be_named_final():
    r = {'cases':{'A':{'complete_registered_cohort':False,'per_sequence':{'seq1':{}}}}}
    with pytest.raises(ValueError,match='all375'): require_complete_validation(r,['A'],['seq1','seq2'])
    with pytest.raises(ValueError,match='registered VAL'): require_complete_validation(r,['A','B'],['seq1'])


def test_no_media_never_means_zero_error_or_cross_day_success():
    r = unavailable_cross_recording('CROSS_DAY')
    assert r['valid_episodes'] == 0 and r['target_recall'] is None and r['false_takeover'] is None
