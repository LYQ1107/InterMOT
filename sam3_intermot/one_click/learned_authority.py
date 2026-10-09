"""Small joint-state trajectory heads; runtime accepts only visual/state arrays."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .intervention_features import FEATURE_NAMES

FAMILIES = ('LEARNABLE_SCALAR', 'LOGISTIC', 'SMALL_MLP', 'PAIRWISE_RANKER',
            'GLOBAL_RISK', 'CAUSAL_TEMPORAL', 'DUAL_STATE', 'ACTION_VALUE')
VISUAL = tuple(FEATURE_NAMES.index(k) for k in (
    'proposed_probability', 'NONE_probability', 'candidate_margin', 'available_probability',
    'anchor_cosine', 'anchor_margin', 'anchor_advantage_vs_KEEP', 'KEEP_anchor_cosine',
    'quality', 'actor_bank_size', 'prototype_anchor_agreement', 'proposed_prototype_cosine',
    'joint_entropy', 'NONE_advantage', 'actor_probability_learned'))
GLOBAL = tuple(i for i in range(len(FEATURE_NAMES)) if i not in VISUAL)


class TrajectoryAuthorityHead(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family not in FAMILIES: raise ValueError('unregistered learned authority')
        self.family = family; d = len(FEATURE_NAMES)
        if family == 'LEARNABLE_SCALAR': self.head = nn.Linear(1, 4)
        elif family == 'LOGISTIC': self.head = nn.Linear(d, 4)
        elif family == 'CAUSAL_TEMPORAL':
            self.temporal = nn.GRU(d, 16, batch_first=True); self.head = nn.Linear(16, 4)
        elif family == 'DUAL_STATE':
            self.visual = nn.Sequential(nn.Linear(len(VISUAL), 16), nn.ReLU())
            self.global_state = nn.Sequential(nn.Linear(len(GLOBAL), 16), nn.ReLU())
            self.head = nn.Linear(32, 4)
        else: self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (4, len(FEATURE_NAMES)):
            raise ValueError('exactly3 causal previous vectors plus current action required')
        current = x[:, -1]
        if self.family == 'LEARNABLE_SCALAR': return self.head(current[:, 6:7])
        if self.family == 'CAUSAL_TEMPORAL': return self.head(self.temporal(x)[0][:, -1])
        if self.family == 'DUAL_STATE':
            return self.head(torch.cat((self.visual(current[:, VISUAL]), self.global_state(current[:, GLOBAL])), dim=-1))
        return self.head(current)


class TrajectoryAuthorityPredictor:
    """Strict load; no annotations, mutable history, or hidden future state."""
    def __init__(self, checkpoint):
        saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
        if saved['schema'] != 'N72R21R1_TRAJECTORY_AUTHORITY_V1': raise ValueError('head schema mismatch')
        if saved['feature_names'] != list(FEATURE_NAMES): raise ValueError('feature axis mismatch')
        self.model = TrajectoryAuthorityHead(saved['family']).eval()
        self.model.load_state_dict(saved['model'], strict=True)
        self.mean = np.asarray(saved['FIT_mean'], np.float32)
        self.std = np.asarray(saved['FIT_std'], np.float32)
        if self.mean.shape != (len(FEATURE_NAMES),) or np.any(self.std <= 0): raise ValueError('invalid FIT-only normalizer')
        self.selection = saved['selection']

    @torch.inference_mode()
    def predict(self, current, history=None):
        x = np.asarray(current, np.float32)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all(): raise ValueError('invalid current feature vector')
        past = np.zeros((3, len(FEATURE_NAMES)), np.float32) if history is None else np.asarray(history, np.float32)
        if past.shape != (3, len(FEATURE_NAMES)) or not np.isfinite(past).all(): raise ValueError('invalid causal history')
        joined = np.concatenate((past, x[None]), axis=0)
        logits = self.model(torch.from_numpy((joined-self.mean)/self.std)[None])[0]
        p = logits[:3].softmax(0).numpy()
        return {'beneficial': float(p[1]), 'harmful': float(p[2]), 'abstain': float(p[0]), 'value': float(logits[3])}


def eligible_prediction(prediction, selection):
    if selection.get('status') != 'CALIBRATED_DIAGNOSTIC': return False
    return bool(prediction['beneficial'] >= selection['benefit_min']
        and prediction['harmful'] <= selection['harm_max']
        and prediction['value'] > selection['value_min']
        and prediction['beneficial'] > prediction['abstain'])
