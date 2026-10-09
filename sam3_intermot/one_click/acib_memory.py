"""Independent current-input future-safe head; no future input at inference."""
import torch
from torch import nn
from torch.nn import functional as F
from sam3_intermot.one_click.acib import ACIBNetwork,training_losses


class ACIBMemoryNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.base=ACIBNetwork()
        self.future_safe_head=nn.Sequential(nn.Linear(18,64),nn.GELU(),nn.Linear(64,1))
        self.last_future_safe_probability=None

    def load_base(self,state):
        self.base.load_state_dict(state,strict=True)

    def forward(self,anchor,candidates,valid,quality,**kwargs):
        output=self.base(anchor,candidates,valid,quality,**kwargs)
        ranking=torch.softmax(output['candidate_logits'],dim=-1)*valid
        available=output['candidate_valid_probability'];n=candidates.shape[1]
        evidence=torch.cat([output['evidence'],ranking[:,:,None],available[:,None,None].expand(-1,n,1)],dim=-1)
        logits=self.future_safe_head(evidence).float().squeeze(-1).masked_fill(~valid,-1e4)
        output['future_safe_logits']=logits
        self.last_future_safe_probability=logits.detach().sigmoid()
        return output


def memory_training_losses(output,batch):
    parts=training_losses(output,batch['target_slot'],batch['availability_label'],batch['write_labels'],batch['write_verified'])
    mask=batch['future_safe_verified']
    risk=F.binary_cross_entropy_with_logits(output['future_safe_logits'][mask],batch['future_safe_labels'][mask]) if mask.any() else output['future_safe_logits'].sum()*0.
    parts['future_safe']=risk;parts['total']=parts['total']+.5*risk
    return parts
