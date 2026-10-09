import torch
from sam3_intermot.one_click.acib import ACIBNetwork
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork,memory_training_losses


def fixture():
    torch.manual_seed(72101);model=ACIBMemoryNetwork();anchor=torch.nn.functional.normalize(torch.randn(2,512),dim=-1)
    candidates=torch.nn.functional.normalize(torch.randn(2,3,512),dim=-1);valid=torch.ones(2,3,dtype=torch.bool);quality=torch.zeros(2,3,5)
    return model,(anchor,candidates,valid,quality)


def test_base_outputs_are_exact_and_future_head_separate():
    model,inputs=fixture();base=ACIBNetwork();base.load_state_dict(model.base.state_dict(),strict=True)
    first=base(*inputs);second=model(*inputs)
    for key in ['joint_probabilities','write_logits','candidate_logits','evidence']:assert torch.equal(first[key],second[key])
    assert model.future_safe_head is not model.base.write_head


def test_future_truth_only_enters_loss_and_all_unknown_is_masked():
    model,inputs=fixture();output=model(*inputs)
    batch={'target_slot':torch.tensor([0,3]),'availability_label':torch.tensor([0,2]),'write_labels':torch.zeros(2,3),
           'write_verified':torch.ones(2,3,dtype=torch.bool),'future_safe_labels':torch.ones(2,3),'future_safe_verified':torch.zeros(2,3,dtype=torch.bool)}
    parts=memory_training_losses(output,batch);assert parts['future_safe']==0
    parts['total'].backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_verified_risk_examples_really_update_independent_head():
    model,inputs=fixture();output=model(*inputs)
    batch={'target_slot':torch.tensor([0,3]),'availability_label':torch.tensor([0,2]),'write_labels':torch.zeros(2,3),
           'write_verified':torch.ones(2,3,dtype=torch.bool),'future_safe_labels':torch.zeros(2,3),'future_safe_verified':torch.tensor([[True,False,False],[False,True,False]])}
    memory_training_losses(output,batch)['total'].backward()
    assert model.future_safe_head[-1].weight.grad.abs().sum()>0
