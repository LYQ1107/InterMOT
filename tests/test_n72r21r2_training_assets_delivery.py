from copy import deepcopy
import numpy as np
import pytest
import torch
from scripts.n72r21r2_training_assets_delivery_v1 import checkpoint_specs, tensor_changes, numeric_digest


def record():
    return {"checkpoint_path": "/exact/project/epoch2.pt", "checkpoint_sha256": "b"*64, "initial_sha256": "c"*64,
            "epochs": [{"epoch":1,"checkpoint_path":"/exact/project/epoch1.pt","checkpoint_sha256":"a"*64},
                       {"epoch":2,"checkpoint_path":"/exact/project/epoch2.pt","checkpoint_sha256":"b"*64}],
            "best_epoch":2,"seed":730101,"family":"HEAD","protocol_sha256":"p","changed_weight_elements":1}


def test_selected_epoch_alias_is_one_physical_byte_inventory_entry():
    inventory = checkpoint_specs(record())
    assert len(inventory)==3 and inventory['/exact/project/epoch2.pt']['roles']==['SELECTED','EPOCH']


@pytest.mark.parametrize("bad",["no_fit","skip","bool_epoch","wrong_selected","conflicting_bytes"])
def test_invalid_history_or_failure_cannot_be_a_weight_manifest(bad):
    r=record()
    if bad=="no_fit":r.pop('checkpoint_path')
    elif bad=="skip":r['epochs'][0]['epoch']=2
    elif bad=="bool_epoch":r['epochs'][0]['epoch']=True
    elif bad=="wrong_selected":r['best_epoch']=3
    else:r['epochs'][1]['checkpoint_sha256']='z'*64
    with pytest.raises(ValueError):checkpoint_specs(r)


def test_unsealed_initial_bytes_explicitly_missing_not_an_invented_seal():
    r=record();r.pop('initial_sha256')
    assert checkpoint_specs(r)['/exact/project/initial.pt']['expected_sha256'] is None


def tensor_fixture():
    r=record();saved={k:r[k] for k in ['seed','family','protocol_sha256']}
    saved.update(epoch=2,model={'w':torch.tensor([1.,0.])})
    return r,saved,{'w':torch.zeros(2)}


def test_actual_tensor_changes_not_file_existence_or_optimizer_step_count():
    r,s,i=tensor_fixture();assert tensor_changes(s,i,r)==1
    s['model']['w'].zero_()
    with pytest.raises(ValueError,match='change'):tensor_changes(s,i,r)


@pytest.mark.parametrize("bad",['seed','epoch','shape','dtype','nonfinite','count'])
def test_selected_tensor_identity_axis_and_delta_fail_closed(bad):
    r,s,i=tensor_fixture()
    if bad=='seed':s['seed']=730103
    elif bad=='epoch':s['epoch']=1
    elif bad=='shape':i['w']=torch.zeros(3)
    elif bad=='dtype':i['w']=torch.zeros(2,dtype=torch.float64)
    elif bad=='nonfinite':s['model']['w'][0]=float('nan')
    else:r['changed_weight_elements']=2
    with pytest.raises(ValueError):tensor_changes(s,i,r)


def test_numeric_digest_binds_shape_dtype_values_and_original_row_order():
    a=np.array([[1.,2.],[3.,4.]],np.float32)
    assert len({numeric_digest(a),numeric_digest(a[::-1]),numeric_digest(a.reshape(4)),numeric_digest(a.astype(np.int64))})==4
    with pytest.raises(ValueError):numeric_digest(a.astype(np.float64))
    a[0,0]=np.nan
    with pytest.raises(ValueError):numeric_digest(a)
