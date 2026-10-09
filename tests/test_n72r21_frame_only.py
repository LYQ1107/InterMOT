import numpy as np
import pytest
import torch
from sam3_intermot.one_click.frame_only_runtime import FrameOnlyRecognizer


class Spy(torch.nn.Module):
    def __init__(self):super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1));self.seen=[]
    def forward(self,anchor,candidates,valid,quality,**kwargs):
        self.seen.append((quality.clone(),kwargs['bank_metadata'].clone(),kwargs['context'].clone()))
        n=candidates.shape[1];exists=bool(valid.any())
        return {'joint_probabilities':torch.tensor([[.99, .01] if exists else [0.,1.]]),
            'candidate_logits':torch.ones(1,n),'write_logits':torch.ones(1,n)*10,
            'future_safe_logits':torch.ones(1,n)*10,'candidate_valid_probability':torch.tensor([.99 if exists else 0.])}


def test_unknown_FPS_never_becomes_synthetic_seconds_and_delayed_write_is_causal():
    model=Spy();anchor=np.ones(512,dtype=np.float32)
    r=FrameOnlyRecognizer(model,anchor,'anonymous',width=100,height=100,initial_frame=1921,initial_box=[0,0,20,40])
    row={'candidate_uid':'current','feature':anchor,'box_xyxy':[0,0,20,40],'conf':1.}
    first=r.step(1922,[row]);second=r.step(1923,[row])
    assert not first['memory_write'] and second['memory_write']
    assert second['timestamp_seconds'] is None and second['FPS'] is None
    assert r.bank[0].timestamp_seconds is None and r.bank[0].original_frame_1based==1923
    for quality,metadata,context in model.seen:
        assert not quality[...,4].any() and not metadata.any() and not context.any()
    with pytest.raises(ValueError,match='contiguous'):r.step(1925,[row])


def test_structural_empty_candidate_set_is_NONE_not_false_presence():
    r=FrameOnlyRecognizer(Spy(),np.ones(512),'anonymous',width=100,height=100,initial_frame=1,initial_box=[0,0,20,40])
    result=r.step(2,[])
    assert result['selected_candidate_uid'] is None and result['rank1_identity_joint_probability']==0.
    assert not result['memory_write'] and result['runtime_future_gt_used'] is False
