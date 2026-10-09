import hashlib
import torch
import pytest
from sam3_intermot.evaluation.checkpoint_audit import verify_checkpoint


def checkpoint(tmp_path,*,wrong_schema=False,nonfinite=False,missing=False):
    model=torch.nn.Linear(2,1);state=model.state_dict()
    if nonfinite:state['weight'][0,0]=float('nan')
    if missing:state.pop('bias')
    path=tmp_path/'model.pt';schema={'FIT':['train1'],'INNER':'train2','OUTER':'train3'}
    torch.save({'schema':{} if wrong_schema else schema,'model':state,'epoch':2,'seed':72101,'optimizer':{},'rng':{}},path)
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),schema


def test_actual_SHA_schema_finite_weights_and_strict_load(tmp_path):
    p,h,s=checkpoint(tmp_path);r=verify_checkpoint(p,h,s,torch.nn.Linear(2,1))
    assert r['strict_state_dict_loaded'] and r['all_model_tensors_finite'] and r['weights_only_deserialization']
    assert r['model_parameters']==3 and r['epoch']==2 and not r['scientific_success_from_loader']


def test_SHA_is_checked_before_deserializing_arbitrary_bytes(tmp_path):
    p,h,s=checkpoint(tmp_path);p.write_bytes(b'not a pickle')
    with pytest.raises(ValueError,match='SHA mismatch'):verify_checkpoint(p,h,s,torch.nn.Linear(2,1))


@pytest.mark.parametrize('kind,exception,match',[
    ('wrong_schema',ValueError,'schema differs'),('nonfinite',ValueError,'nonfinite'),('missing',RuntimeError,'Missing key')])
def test_invalid_schema_weights_or_structure_never_passes(tmp_path,kind,exception,match):
    p,h,s=checkpoint(tmp_path,**{kind:True})
    with pytest.raises(exception,match=match):verify_checkpoint(p,h,s,torch.nn.Linear(2,1))


def test_audit_does_not_overwrite_or_retrain_checkpoint(tmp_path):
    p,h,s=checkpoint(tmp_path);before=p.read_bytes()
    r=verify_checkpoint(p,h,s,torch.nn.Linear(2,1))
    assert p.read_bytes()==before and not r['new_training']
