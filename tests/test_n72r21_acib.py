import torch
import pytest
from sam3_intermot.one_click.acib import ACIBNetwork,training_losses,AVAILABILITY_CLASSES
from scripts.n72r21_train_t0 import split,collate


def test_acib_real_gradients_probabilities_and_immutable_inputs():
    torch.manual_seed(72101);model=ACIBNetwork(input_dim=8,hidden_dim=4)
    anchor=torch.randn(2,8);before=anchor.clone();candidates=torch.randn(2,3,8)
    valid=torch.tensor([[True,True,True],[False,False,False]])
    quality=torch.zeros(2,3,5);out=model(anchor,candidates,valid,quality)
    assert torch.equal(anchor,before)
    assert torch.allclose(out['joint_probabilities'].sum(-1),torch.ones(2),atol=1e-6)
    assert out['joint_probabilities'][1,-1]==1 and not out['joint_probabilities'][1,:3].any()
    assert out['physical_presence_probability'] is None
    assert 'PHYSICAL_ABSENCE_UNRESOLVED' in AVAILABILITY_CLASSES[-1]
    parts=training_losses(out,torch.tensor([1,3]),torch.tensor([0,2]),torch.zeros(2,3),valid)
    parts['total'].backward()
    assert torch.isfinite(parts['total']) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_bank_padding_does_not_change_current_scores():
    torch.manual_seed(72102);model=ACIBNetwork(input_dim=8,hidden_dim=4).eval()
    anchor=torch.randn(1,8);candidates=torch.randn(1,2,8);valid=torch.ones(1,2,dtype=torch.bool);quality=torch.zeros(1,2,5)
    bank=torch.randn(1,1,8);metadata=torch.zeros(1,1,2)
    first=model(anchor,candidates,valid,quality,bank=bank,bank_valid=torch.ones(1,1,dtype=torch.bool),bank_metadata=metadata)
    second=model(anchor,candidates,valid,quality,bank=torch.cat([bank,torch.randn(1,2,8)],dim=1),
                 bank_valid=torch.tensor([[True,False,False]]),bank_metadata=torch.zeros(1,3,2))
    assert torch.allclose(first['joint_probabilities'],second['joint_probabilities'],atol=1e-6)


def test_exact_sequence_fold_no_inner_or_outer_fit_overlap():
    seq=[f's{i}' for i in range(8)]
    for outer in seq:
        fit,inner=split(outer,seq)
        assert len(fit)==6 and outer not in fit and inner not in fit and inner!=outer
    with pytest.raises(ValueError):split('s0',seq[:7])


def test_none_target_uses_shared_padded_batch_slot():
    def example(n,target):
        return {'anchor':torch.zeros(512),'candidates':torch.zeros(n,512),'quality':torch.zeros(n,5),
                'context':torch.zeros(3),'target':target,'availability':0 if target>=0 else 2,
                'write_labels':torch.zeros(n),'write_verified':torch.zeros(n,dtype=torch.bool)}
    batch=collate([example(0,-1),example(2,1)])
    assert batch['target_slot'].tolist()==[2,1] and not batch['valid'][0].any()
