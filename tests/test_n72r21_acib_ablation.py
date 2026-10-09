import torch
from sam3_intermot.one_click.acib import ACIBNetwork
from sam3_intermot.one_click.acib_ablation import ACIBVariantNetwork


def inputs():
    torch.manual_seed(72101);return (torch.randn(2,512),torch.randn(2,3,512),torch.tensor([[True,True,True],[False,False,False]]),torch.randn(2,3,5))


def test_full_variant_exact_AA_with_original_equations_and_parameter_names():
    args=inputs();base=ACIBNetwork();variant=ACIBVariantNetwork();variant.load_state_dict(base.state_dict(),strict=True)
    bank=torch.randn(2,4,512);valid=torch.tensor([[True,True,True,False],[False,False,False,False]]);meta=torch.randn(2,4,2);context=torch.randn(2,3)
    a=base(*args,bank=bank,bank_valid=valid,bank_metadata=meta,context=context);b=variant(*args,bank=bank,bank_valid=valid,bank_metadata=meta,context=context)
    for key in ['candidate_logits','availability_logits','joint_probabilities','write_logits','evidence']:assert torch.equal(a[key],b[key])


def test_no_availability_is_truly_closed_set_except_structural_empty_set():
    model=ACIBVariantNetwork(availability=False);out=model(*inputs())
    assert out['joint_probabilities'][0,-1]==0 and out['joint_probabilities'][1,-1]==1
    assert out['candidate_valid_probability'].tolist()==[1.,0.]


def test_bank_only_masks_all_anchor_score_consistency_features():
    model=ACIBVariantNetwork(anchor_mode='bank_only');out=model(*inputs())
    assert torch.equal(out['evidence'][:,:,[0,1,6]],torch.zeros(2,3,3))
    assert out['inference_variant']['retrained_architecture'] is False
