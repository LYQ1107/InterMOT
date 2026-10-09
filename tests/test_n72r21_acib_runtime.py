"""Synthetic state-machine unit tests only; not cross-recording evidence."""
import numpy as np
import pytest
import torch
from sam3_intermot.one_click.acib_runtime import ACIBRecognizer


class AlwaysCurrent(torch.nn.Module):
    def __init__(self):super().__init__();self.marker=torch.nn.Parameter(torch.zeros(()))
    def forward(self,anchor,candidates,valid,quality,**kwargs):
        n=candidates.shape[1]
        joint=torch.zeros(1,n+1);joint[0,0 if valid.any() else n]=1.
        return {'joint_probabilities':joint,'candidate_logits':torch.ones(1,n),
                'candidate_valid_probability':torch.tensor([1. if valid.any() else 0.]),
                'write_logits':torch.ones(1,n)*10,'probability_semantics':'UNIT_TEST_NOT_PHYSICAL_PRESENCE'}


def row(uid='machine_uid'):
    return {'candidate_uid':uid,'feature':np.array([1.,0.]),'box_xyxy':[0.,0.,10.,20.],'conf':.9}


def start(policy,capacity=1):
    state=ACIBRecognizer(AlwaysCurrent(),[1,0],'anonymous_token',capacity=capacity,policy=policy)
    state.start_recording('record_A',fps=20,width=100,height=100)
    return state


def test_own_predicted_writes_bounded_and_anchor_not_overwritten():
    state=start('P1');anchor=state.anchor.copy()
    for f in range(1,5):
        output=state.step(f,[row(f'c{f}')]);assert output['memory_write']
    assert len(state.bank)==1 and state.bank[0].candidate_uid=='c4' and np.array_equal(state.anchor,anchor)
    assert state.snapshot()['bank'][0]['candidate_uid']=='c4'


def test_safe_delay_and_none_breaks_confirmation():
    state=start('SAFE_FIXED')
    assert not state.step(1,[row()])['memory_write']
    assert state.step(2,[row()])['memory_write']
    assert state.step(3,[])['selected_candidate_uid'] is None
    assert not state.step(4,[row()])['memory_write']


def test_independent_recording_resets_only_short_term_with_unknown_recency():
    state=start('P1');state.step(1,[row()]);anchor_sha=state.anchor_sha
    state.start_recording('record_B',fps=25,width=200,height=100,camera='camera_new')
    assert state.last_box is None and state.pending is None and len(state.bank)==1 and state.anchor_sha==anchor_sha
    assert state.step(1,[])['unknown_cross_recording_recency_not_concatenated_time']
    with pytest.raises(ValueError):state.step(1,[])


def test_p0_never_writes_and_invalid_fps_not_guessed():
    state=start('P0');assert not state.step(1,[row()])['memory_write'] and not state.bank
    with pytest.raises(ValueError):state.start_recording('B',fps=0,width=100,height=100)
