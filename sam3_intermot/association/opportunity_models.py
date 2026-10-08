"""Small causal action-value models with explicit structured component heads."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .opportunity_tracker import FEATURE_NAMES


class ActionValueModel(nn.Module):
    def __init__(self,family,feature_dim=len(FEATURE_NAMES)):
        super().__init__();self.family=family;self.feature_dim=feature_dim
        if family not in ("C2","C3","C4","C5","C6"):raise ValueError("not a learned action family")
        if family=="C2":
            if feature_dim!=len(FEATURE_NAMES):raise ValueError("scalar uses the action feature schema")
            self.scalar=nn.Parameter(torch.tensor([0.,0.,0.]))
        elif family=="C3":self.network=nn.Linear(feature_dim,3)
        elif family=="C6":
            if feature_dim!=len(FEATURE_NAMES):raise ValueError("structured requires named causal action features")
            self.gain_indices=(0,1,2,3,10,11,13,14,18,19,20,21,22,23)
            self.damage_indices=(4,5,6,7,8,9,10,15,16,17)
            self.risk_indices=(0,1,2,3,10,11,12,13,14,20,21,22,23)
            def head(n):return nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Linear(32,8),nn.ReLU(),nn.Linear(8,1))
            self.gain_head=head(len(self.gain_indices));self.damage_head=head(len(self.damage_indices));self.risk_head=head(len(self.risk_indices))
            self.temperature=nn.Parameter(torch.zeros(2))
        else:
            # Match the actual structured architecture within integer-width
            # limits, rather than claiming equal capacity from equal names.
            reference=ActionValueModel("C6");budget=sum(p.numel() for p in reference.parameters())
            width=min(range(16,80),key=lambda w:abs((feature_dim*w+w+w*(w//2)+w//2+(w//2)*3+3)-budget))
            self.network=nn.Sequential(nn.Linear(feature_dim,width),nn.ReLU(),nn.Linear(width,width//2),nn.ReLU(),nn.Linear(width//2,3))
        self.register_buffer("mean",torch.zeros(feature_dim));self.register_buffer("scale",torch.ones(feature_dim))
        if self.parameter_count>=50000:raise ValueError("small model budget exceeded")

    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def forward(self,features):
        x=(features-self.mean)/self.scale
        if self.feature_dim==len(FEATURE_NAMES):
            # The representation-independent supervision bank is P0. Its
            # query/anchor agreement, confirmation count, drift and update
            # count have no identifiable state variation. Do not let random
            # coefficients on those constant fit columns become "learned"
            # reliability when a memory ablation changes them at runtime.
            x=x.clone()
            for index,expected in ((3,1.),(21,0.),(22,0.),(23,0.)):
                if abs(float(self.mean[index])-expected)<1e-6 and float(self.scale[index])<=.050001:
                    x[:,index]=0.
        if self.family=="C2":
            raw=F.softplus(self.scalar[0])*features[:,1]-F.softplus(self.scalar[1])*features[:,6]+self.scalar[2]
            return {"benefit_logit":raw,"harm_logit":-raw,"value":3*torch.tanh(raw/3)}
        if self.family=="C6":
            gain=2*torch.tanh(self.gain_head(x[:,self.gain_indices]).squeeze(-1))
            damage=5*torch.sigmoid(self.damage_head(x[:,self.damage_indices]).squeeze(-1))
            risk_logit=self.risk_head(x[:,self.risk_indices]).squeeze(-1);risk=torch.sigmoid(risk_logit)
            raw=gain-2*damage-risk
            return {"benefit_logit":raw*(1+F.softplus(self.temperature[0])),
                "harm_logit":-raw*(1+F.softplus(self.temperature[1])),"value":3*torch.tanh(raw/3),
                "gain":gain,"damage":damage,"risk_logit":risk_logit}
        out=self.network(x)
        return {"benefit_logit":out[:,0],"harm_logit":out[:,1],"value":3*torch.tanh(out[:,2]/3)}

    def predict(self,features):
        self.eval()
        f=np.asarray(features,dtype=np.float32)
        if f.shape!=(self.feature_dim,) or not np.isfinite(f).all():raise ValueError("causal feature schema mismatch")
        with torch.inference_mode():
            out=self(torch.as_tensor(f).reshape(1,-1))
            result={"beneficial":float(torch.sigmoid(out["benefit_logit"])[0]),"harmful":float(torch.sigmoid(out["harm_logit"])[0]),"value":float(out["value"][0])}
            if self.family=="C6":result.update(target_future_gain=float(out["gain"][0]),competitor_damage=float(out["damage"][0]),memory_risk=float(torch.sigmoid(out["risk_logit"])[0]))
            return result


def action_loss(model,x,labels,components,groups,loss_family,class_weights):
    out=model(x);positive,harmful,value=labels.T
    cost=1.+2.*components[:,3]+components[:,1]
    base=(F.binary_cross_entropy_with_logits(out["benefit_logit"],positive,reduction="none",pos_weight=class_weights[0])*cost).mean()
    base+=F.binary_cross_entropy_with_logits(out["harm_logit"],harmful,pos_weight=class_weights[1])
    regression=F.smooth_l1_loss(out["value"],torch.clamp(value,-3.,3.))
    if loss_family=="L0":loss=base+.1*regression
    elif loss_family=="L2":loss=regression+.1*base
    elif loss_family in ("L1","L3"):
        differences=value[:,None]-value[None,:]
        paired=(groups[:,None]==groups[None,:])&(differences>.05)
        if paired.any():ranking=F.softplus(-(out["value"][:,None]-out["value"][None,:])[paired]).mean()
        else:ranking=F.softplus(-torch.sign(value[value!=0])*out["value"][value!=0]).mean() if (value!=0).any() else out["value"].sum()*0.
        loss=ranking+.2*base+.1*regression
        if loss_family=="L3":
            # Actual harmful/component supervision penalizes approval risk,
            # not a cosmetic confidence rescaling after a generic MLP.
            approval=torch.sigmoid(out["benefit_logit"])
            loss+=3.*(approval*(harmful+components[:,1]+components[:,2])).mean()
    else:raise ValueError("unregistered loss family")
    if model.family=="C6":
        loss+=F.smooth_l1_loss(out["gain"],torch.clamp(components[:,0],-2.,2.))
        loss+=F.smooth_l1_loss(out["damage"],torch.clamp(components[:,1],0.,5.))
        loss+=F.binary_cross_entropy_with_logits(out["risk_logit"],components[:,2])
    return loss
