import numpy as np
import pytest
from sam3_intermot.one_click import Candidate, OneClickRecognizer, RuntimeConfig


def candidate(uid='a', feature=(1.,0.), box=(0,0,10,10)):
    return Candidate(uid,np.asarray(feature),box,1.,uid)


def recognizer(config=RuntimeConfig()):
    r=OneClickRecognizer(config);r.initialize([1.,0.],(0,0,10,10),recording_id='a',frame=0);return r


def test_empty_distractors_and_positive_are_real_none_decisions():
    r=recognizer()
    assert r.step(1,[],fps=30)['selected_candidate_uid'] is None
    assert r.step(2,[candidate('wrong',(0,1))],fps=30)['selected_candidate_uid'] is None
    assert r.step(3,[candidate()],fps=30)['selected_candidate_uid']=='a'


def test_anchor_and_bank_are_immutable_and_bounded():
    r=recognizer(RuntimeConfig(memory_policy='P1',bank_capacity=1))
    anchor=r.anchor;anchor[:]=[0,1]
    assert np.allclose(r.anchor,[1,0])
    for f in range(1,6):r.step(f,[candidate(str(f))],fps=30)
    assert len(r.bank)==1
    with pytest.raises(ValueError):r.bank[0].embedding[0]=0


def test_independent_session_preserves_identity_and_resets_motion_confirmation():
    r=recognizer(RuntimeConfig(memory_policy='SAFE_DELAYED'))
    for f in (1,2,3):r.step(f,[candidate()],fps=30)
    assert len(r.bank)==1
    token,anchor=r.token,r.anchor
    r.start_recording('b')
    assert r.last_box is None and r.last_native_tid is None and r.last_frame is None
    assert r._pending_count==0 and len(r.bank)==1 and r.token==token and np.allclose(r.anchor,anchor)
    assert not r.step(0,[candidate()],fps=30)['memory_write']


def test_second_click_and_truth_dict_forbidden():
    r=recognizer()
    with pytest.raises(RuntimeError):r.initialize([1,0],(0,0,10,10),recording_id='b',frame=0)
    with pytest.raises(TypeError):r.step(1,[{'target_gt_id':1}],fps=30)


def test_delay_never_uses_future_and_absence_breaks_confirmation():
    r=recognizer(RuntimeConfig(memory_policy='SAFE_DELAYED'))
    assert not r.step(1,[candidate()],fps=30)['memory_write']
    assert not r.step(2,[candidate()],fps=30)['memory_write']
    r.step(3,[],fps=30)
    assert not r.step(4,[candidate()],fps=30)['memory_write']
    assert not r.step(5,[candidate()],fps=30)['memory_write']
    assert r.step(6,[candidate()],fps=30)['memory_write']


def test_same_session_reset_and_frame_reordering_forbidden():
    r=recognizer()
    with pytest.raises(ValueError):r.start_recording('a')
    with pytest.raises(ValueError):r.step(2,[candidate()],fps=30)


def test_learned_scorer_has_explicit_availability_and_write_probability():
    seen=[]
    def scorer(anchor,bank,candidates,frame,fps,last_box,camera):
        seen.append((len(bank),frame));return {'identity_scores':[.8]*len(candidates),'candidate_available_probability':.1,'target_present_probability':.9,'safe_write_probability':0.}
    r=OneClickRecognizer(scorer=scorer);r.initialize([1,0],(0,0,10,10),recording_id='a',frame=0)
    row=r.step(1,[candidate()],fps=30)
    assert row['selected_candidate_uid'] is None and row['target_present_probability']==.9
    assert not row['memory_write'] and seen==[(0,1)]
