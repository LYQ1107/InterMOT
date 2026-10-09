import numpy as np
import pytest
from sam3_intermot.evaluation.one_click_breakdowns import Counts,density_bin,gap_bin,frame_facts,recovery_events


def test_real_density_and_seconds_boundaries():
    assert [density_bin(i) for i in (0,4,5,8,9)]==['sparse','sparse','medium','medium','crowded']
    assert [gap_bin(i) for i in (0,5,5.01,20,20.01)]==['short_le5','short_le5','medium_gt5_le20','medium_gt5_le20','long_gt20_within_recording']
    with pytest.raises(ValueError):density_bin(-1)
    with pytest.raises(ValueError):gap_bin(float('nan'))


def facts(visible=True):
    return {'visible':visible,'available':visible,'target_box':[0,0,10,10] if visible else None,'matched':{'p':1,'q':2,'u':None},'identity':1}


def row(f,selected=None,write=None):
    return {'frame':f,'selected_candidate_uid':selected,'predicted_box_xyxy':[0,0,10,10] if selected else None,
        'rank1_candidate_uid':selected,'rank1_identity_joint_probability':.8 if selected else 0.,'candidate_available_probability':.8,
        'memory_write':write is not None,'memory_write_candidate_uid':write}


def test_UNKNOWN_write_and_zero_write_are_not_verified_wrong_or_safe_PASS():
    c=Counts();c.add(row(1,'u','u'),facts());r=c.result()
    assert r['counts']['UNKNOWN_unmatched_writes']==1
    assert r['counts'].get('verified_other_identity_writes',0)==0
    assert r['conservative_non_target_or_unverified_write_rate']==1
    assert r['verified_identity_claim_calibration']['ECE'] is None
    d=Counts();d.add(row(1,'p'),facts());r=d.result()
    assert r['conservative_non_target_or_unverified_write_rate'] is None
    assert r['strict_correct_observation_retention']==0
    assert r['zero_write_is_not_safe_memory_PASS']


def test_recovery_keeps_failed_return_and_wrong_first_claim():
    trace=[row(1,'p'),row(2),row(3,'q'),row(4,'p'),row(5),row(6,'q')]
    result=recovery_events(trace,[facts(),facts(False),facts(),facts(),facts(False),facts()],fps=20)
    a,b=result['strict_UID_reappearance_windows']
    assert a['strict_identity_recovered'] and a['strict_delay_seconds']==.05 and a['first_accept_verified_wrong_identity']
    assert not b['strict_identity_recovered'] and b['strict_delay_seconds'] is None
    assert result['strict_verified_other_identity_takeover_runs'][-1]['no_correct_recovery_before_recording_end']


def test_hard_competing_identity_not_only_self_similarity():
    crops=[{'candidate_uid':u,'feature':np.array(v,dtype=float)} for u,v in [('p',[.6,.8]),('q',[1,0]),('u',[1,0])]]
    r=frame_facts(crops,[{'identity':1,'box':[0,0,10,10]}],{'p':1,'q':2,'u':None},1,np.array([1.,0.]))
    assert r['available'] and r['raw_anchor_hard_negative_margin']==pytest.approx(-.4)
