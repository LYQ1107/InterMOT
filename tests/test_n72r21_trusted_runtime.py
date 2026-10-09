import numpy as np
import torch
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer


def runtime(safe=True):
    torch.manual_seed(72101);model=ACIBMemoryNetwork();anchor=np.zeros(512,np.float32);anchor[0]=1
    with torch.no_grad():
        for head in [model.base.availability_head,model.base.write_head,model.future_safe_head]:
            for p in head.parameters():p.zero_()
        model.base.availability_head[-1].bias[0]=20;model.base.write_head[-1].bias[0]=20
        model.future_safe_head[-1].bias[0]=20 if safe else -20
    r=TrustedACIBRecognizer(model,anchor,'anonymous',policy='FULL')
    r.start_recording('a',fps=20,width=100,height=100,initial_frame=1,initial_box=[2.,3.,22.,43.])
    return r,{'candidate_uid':'u','feature':anchor,'box_xyxy':[2.,3.,22.,43.],'conf':.8}


def test_learned_future_harm_rejects_currently_high_confident_write():
    r,row=runtime(False)
    assert not r.step(2,[row])['memory_write']
    assert not r.step(3,[row])['memory_write']
    assert not r.bank and r.pending is None


def test_safe_commit_requires_real_two_observations_and_writes_current_crop():
    r,row=runtime(True)
    assert not r.step(2,[{**row,'candidate_uid':'first'}])['memory_write']
    result=r.step(3,[{**row,'candidate_uid':'current'}])
    assert result['memory_write'] and result['memory_write_candidate_uid']=='current'
    assert r.bank[0].frame==3 and r.bank[0].candidate_uid=='current'


def test_new_recording_resets_confirmation_without_new_click_or_bank_reset():
    r,row=runtime(True);r.step(2,[row]);r.step(3,[row]);sha=r.anchor_sha
    r.start_recording('independent',fps=30,width=100,height=100,camera='new')
    assert r.pending is None and r.last_box is None and len(r.bank)==1 and r.anchor_sha==sha
    result=r.step(0,[row]);assert not result['memory_write']
    assert result['unknown_cross_recording_recency_not_concatenated_time']
