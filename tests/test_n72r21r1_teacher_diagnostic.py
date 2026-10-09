from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import torch
from scripts.n72r21r1_teacher_diagnostic import capture_probabilities, oracle_feedback, summary


def actor():
    anchor=np.array([1.,0.],np.float32);anchor.setflags(write=False)
    return SimpleNamespace(last_frame=3,last_box=[1,1,2,2],last_accept=3,pending={'untrusted':True},
        bank=[],anchor=anchor,recording='unit',camera=None,fps=20.)


def test_copied_current_probabilities_survive_shared_future_output_mutation():
    tensor=torch.tensor([[.6,.3,.1]])
    model=SimpleNamespace(last={'joint_probabilities':tensor})
    snapshot=capture_probabilities(model,[{},{}])
    tensor[0,0]=.01
    model.last={'joint_probabilities':torch.tensor([[.1,.1,.8]])}
    assert snapshot[0]==pytest.approx(.6)
    with pytest.raises(ValueError):capture_probabilities(model,[{}])


def test_oracle_requires_completed_current_frame_and_real_current_UID():
    a=actor();rows=[{'candidate_uid':'p','feature':[1.,0.],'box_xyxy':[0,0,2,3],'conf':.9}]
    with pytest.raises(ValueError,match='after current prediction'):
        oracle_feedback(a,4,rows,'p',(None,None),write=True)
    with pytest.raises(ValueError,match='missing positive crop'):
        oracle_feedback(a,3,rows,'fabricated',(None,None),write=True)
    assert not a.bank


def test_oracle_bank_is_prior_real_positive_bounded_and_anchor_immutable():
    a=actor();digest=hashlib.sha256(a.anchor.tobytes()).hexdigest()
    rows=[{'candidate_uid':'p','feature':[0.,1.],'box_xyxy':[0,0,2,3],'conf':.9}]
    for frame in range(3,16):
        a.last_frame=frame
        oracle_feedback(a,frame,rows,'p',(None,None),write=True)
    assert len(a.bank)==8 and [e.frame for e in a.bank]==list(range(8,16))
    assert all(e.candidate_uid=='p' and not e.embedding.flags.writeable for e in a.bank)
    assert all('NOT_ONLINE' in e.source for e in a.bank)
    assert hashlib.sha256(a.anchor.tobytes()).hexdigest()==digest
    assert a.pending is None and a.last_box==[0,0,2,3]


def test_missing_positive_restores_prior_motion_without_bank_or_fabricated_box():
    a=actor();oracle_feedback(a,3,[],None,([3,3,4,4],1),write=True)
    assert a.last_box==[3,3,4,4] and a.last_accept==1 and not a.bank


def test_unknown_is_not_verified_hard_negative_and_proposal_not_commit_is_ranked():
    rows=[{'candidate_uid':u} for u in ['unknown','positive','other']]
    scores=np.array([.5,.3,.1,.1])
    decision={'selected_candidate_uid':'other','proposed_candidate_uid':'positive','candidate_available_probability':.9}
    record=summary(scores,rows,{'unknown':None,'positive':7,'other':8},7,decision,0)
    assert record['UNKNOWN_candidates']==1 and record['hard_negative_margin']==pytest.approx(.2)
    assert not record['rank1_correct'] and record['MRR']==.5 and record['rank2']
    assert record['selected_correct'] and record['selected_uid']=='positive'


def test_empty_candidate_axis_is_none_not_a_fabricated_identity():
    decision={'selected_candidate_uid':None,'candidate_available_probability':0.}
    record=summary(np.array([0.,1.]),[],{},7,decision,0)
    assert not record['candidate_available'] and record['NONE_probability']==1.
    assert record['rank1_joint_probability']==0. and not record['competitive_verified_other']
