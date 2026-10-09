"""Strict current joint-commit writer; future labels exist only in TRAIN loss."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .intervention_features import FEATURE_NAMES


class JointWriteHead(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,3))

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=32:raise ValueError('exact current committed-observation feature axis')
        return self.head(x)


class JointWritePredictor:
    def __init__(self,path):
        saved=torch.load(Path(path),map_location='cpu',weights_only=True)
        if saved['schema']!='N72R21R1_JOINT_COMMIT_WRITE_V1' or saved['feature_names']!=list(FEATURE_NAMES):raise ValueError('joint writer checkpoint schema')
        self.model=JointWriteHead().eval();self.model.load_state_dict(saved['model'],strict=True)
        self.mean=np.array(saved['FIT_mean'],np.float32);self.std=np.array(saved['FIT_std'],np.float32);self.selection=saved['selection']
        if self.mean.shape!=(32,) or self.std.shape!=(32,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<=0):raise ValueError('FIT-only writer normalizer')
        if not all(torch.isfinite(t).all() for t in self.model.state_dict().values()):raise ValueError('non-finite writer weights')

    @torch.inference_mode()
    def predict(self,current):
        x=np.asarray(current,np.float32)
        if x.shape!=(32,) or not np.isfinite(x).all():raise ValueError('invalid current committed feature vector')
        correct,other,target=self.model(torch.from_numpy((x-self.mean)/self.std)[None])[0].sigmoid().tolist()
        calibrated=self.selection.get('status')=='CALIBRATED_DIAGNOSTIC'
        return {'beneficial':correct if calibrated else 0.,'harmful':max(1-correct,other,target) if calibrated else 1.,
            'abstain':1-correct,'value':correct-other-target,'current_correct_probability':correct,
            'other_person_future_harm':other,'target_future_N10_risk':target}
