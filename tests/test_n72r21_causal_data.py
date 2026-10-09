import copy
import numpy as np
import pytest
import torch
from sam3_intermot.one_click.acib import ACIBNetwork
from sam3_intermot.one_click.acib_runtime import ACIBRecognizer
from scripts.n72r21_causal_data import StateReconstructor,mixed_policy,collate,forward


def fixture():
    torch.manual_seed(72101);model=ACIBNetwork();anchor=np.zeros(512,np.float32);anchor[0]=1
    with torch.no_grad():
        for p in model.availability_head.parameters():p.zero_()
        model.availability_head[-1].bias[0]=20
    event={'sequence':'record_a','frame':1,'box_xyxy':[2.,3.,22.,43.],'fps':20.,'width':100,'height':100,'episode_uid':'token'}
    return model,anchor,event


def test_full_axis_reconstruction_equals_every_runtime_snapshot():
    model,anchor,event=fixture();runtime=ACIBRecognizer(model,anchor,'token',policy='P1',capacity=8)
    runtime.start_recording('record_a',fps=20,width=100,height=100,initial_frame=1,initial_box=event['box_xyxy'])
    state=StateReconstructor(event,anchor)
    for frame in range(2,14):
        row={'candidate_uid':str(frame),'feature':anchor,'box_xyxy':[2.,3.,22.,43.],'conf':.8};rows=[row];vectors={str(frame):anchor}
        state.check(runtime.snapshot());result=runtime.step(frame,rows);state.advance(result,rows,vectors,'P1');state.check(runtime.snapshot())
    assert len(state.state['bank'])==8
    assert state.state['bank'][0]['frame']==6


def test_none_cannot_be_reconstructed_as_oracle_write():
    model,anchor,event=fixture();state=StateReconstructor(event,anchor)
    record={'frame':2,'recording_id':'record_a','anchor_sha256':state.state['anchor_sha256'],'runtime_gt_used':False,'extra_clicks':0,
            'selected_candidate_uid':None,'memory_write':True,'machine_bank_size':1}
    with pytest.raises(ValueError,match='write semantics'):state.advance(record,[],{},'P1')


def test_future_bank_reference_and_snapshot_corruption_are_rejected():
    _,anchor,event=fixture();state=StateReconstructor(event,anchor);snapshot=copy.deepcopy(state.state);snapshot['last_accept']=2
    with pytest.raises(ValueError,match='before-state'):state.check(snapshot)
    state.state['bank']=[{'frame':10,'recording_id':'record_a'}]
    with pytest.raises(ValueError,match='future'):state.features(3,[],{})


def test_mixed_assignment_is_whole_episode_deterministic():
    assert [mixed_policy(s,['a','b','c','d']) for s in ['a','b','c','d']]==['P0','P1','P0','P1']


def test_collated_bank_is_used_by_real_network_with_finite_gradients():
    model,anchor,_=fixture();a=torch.from_numpy(anchor)
    e={'anchor':a,'candidates':a[None],'quality':torch.zeros(1,5),'context':torch.zeros(3),'target':0,'availability':0,
       'write_labels':torch.ones(1),'write_verified':torch.ones(1,dtype=torch.bool),'bank_vectors':(a,), 'bank_metadata':torch.zeros(1,2)}
    batch=collate([e]);out=forward(model,batch);out['candidate_logits'].sum().backward()
    assert batch['bank_valid'].tolist()==[[True]]
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
