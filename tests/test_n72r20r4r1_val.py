"""Engineering checks for the frozen, train-only VAL refit boundary."""
from copy import deepcopy
import pytest
from scripts import n72r20r4r1_val as val
from sam3_intermot.association.opportunity_models import ActionValueModel


def checkpoint():
    model=ActionValueModel('C3',10)
    return {'actual_training_sequences':list(val.SEQUENCES),'forbidden_splits':['val','test'],
            'feature_names':val.RELIABILITY_FEATURES,'state_dict':model.state_dict()}


@pytest.mark.parametrize('mutation',['val_fit','missing_train','test_allowed','feature_leak'])
def test_all_train_native_rejects_lineage_leak(monkeypatch,mutation):
    c=checkpoint()
    if mutation=='val_fit':c['actual_training_sequences'].append('dancetrack0004')
    elif mutation=='missing_train':c['actual_training_sequences'].pop()
    elif mutation=='test_allowed':c['forbidden_splits']=['val']
    else:c['feature_names']=(*val.RELIABILITY_FEATURES[:-1],'future_gt_id')
    monkeypatch.setattr(val,'sha256',lambda p:'sealed')
    monkeypatch.setattr(val.torch,'load',lambda *a,**kw:deepcopy(c))
    with pytest.raises(ValueError,match='train lineage'):
        val.AllTrainNative([{'path':'engineering-only.pt','sha256':'sealed'}])


def test_all_train_native_sha_checked_before_deserialization(monkeypatch):
    monkeypatch.setattr(val,'sha256',lambda p:'changed')
    monkeypatch.setattr(val.torch,'load',lambda *a,**kw:pytest.fail('unverified checkpoint loaded'))
    with pytest.raises(ValueError,match='SHA mismatch'):
        val.AllTrainNative([{'path':'engineering-only.pt','sha256':'sealed'}])


def test_all_train_native_freezes_all_parameters(monkeypatch):
    c=checkpoint()
    monkeypatch.setattr(val,'sha256',lambda p:'sealed')
    monkeypatch.setattr(val.torch,'load',lambda *a,**kw:deepcopy(c))
    ensemble=val.AllTrainNative([{'path':'engineering-only.pt','sha256':'sealed'}])
    assert all(not p.requires_grad for m in ensemble.models for p in m.parameters())
    assert all(not m.training for m in ensemble.models)
