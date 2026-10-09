"""Matched-state correction heads with explicit target and competitor risk."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .intervention_features import FEATURE_NAMES
from .learned_authority import TrajectoryAuthorityHead


class OnPolicyAuthorityHead(nn.Module):
    def __init__(self,family):
        super().__init__();self.family=family
        if family not in ('LOGISTIC','GLOBAL_RISK','CAUSAL_TEMPORAL'):raise ValueError('unregistered correction head')
        self.base=TrajectoryAuthorityHead(family)
        if family=='GLOBAL_RISK':self.components=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,3))

    def forward(self,x):
        primary=self.base(x)
        return torch.cat((primary,self.components(x[:,-1])),dim=-1) if self.family=='GLOBAL_RISK' else primary


def decoded_predictions(output):
    p=output[:,:3].softmax(1).detach().numpy();raw=output.detach().numpy();result=[]
    for q,r in zip(p,raw,strict=True):
        other=float(1/(1+np.exp(-np.clip(r[5],-40,40)))) if len(r)==7 else None
        target=float(1/(1+np.exp(-np.clip(r[6],-40,40)))) if len(r)==7 else None
        risk=max(float(q[2]),other or 0.,target or 0.)
        result.append({'beneficial':float(q[1]),'harmful':risk,'abstain':float(q[0]),'value':float(r[3]),
            'class_harm':float(q[2]),'other_person_harm':other,'target_future_harm':target,
            'target_only_value':float(r[4]) if len(r)==7 else None})
    return result


class OnPolicyAuthorityPredictor:
    def __init__(self,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        if saved['schema']!='N72R21R1_ON_POLICY_AUTHORITY_V1' or saved['feature_names']!=list(FEATURE_NAMES):raise ValueError('correction checkpoint schema')
        self.model=OnPolicyAuthorityHead(saved['family']).eval();self.model.load_state_dict(saved['model'],strict=True)
        self.mean=np.array(saved['FIT_mean'],np.float32);self.std=np.array(saved['FIT_std'],np.float32);self.selection=saved['selection']
        if self.mean.shape!=(32,) or self.std.shape!=(32,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0):raise ValueError('FIT normalizer schema')
        if not all(torch.isfinite(v).all() for v in self.model.state_dict().values()):raise ValueError('non-finite checkpoint weights')

    @torch.inference_mode()
    def predict(self,current,history=None):
        past=np.zeros((3,32),np.float32) if history is None else np.asarray(history,np.float32)
        x=np.asarray(current,np.float32)
        if x.shape!=(32,) or past.shape!=(3,32) or not np.isfinite(x).all() or not np.isfinite(past).all():raise ValueError('current/past input schema')
        joined=np.concatenate((past,x[None]),axis=0)
        return decoded_predictions(self.model(torch.from_numpy((joined-self.mean)/self.std)[None]))[0]
