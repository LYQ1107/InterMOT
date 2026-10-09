"""Explicit B7/B8 gate, using each bridge's actual causal proposal history.

Baseline/challenger futures are independently evolved TRAIN supervision,
never online GT branches. The runtime predicts their difference from current
and past features; the original scorer and complete global solve are intact.
"""
from copy import deepcopy
import numpy as np
from .safe_mot_bridge import SafeMOTIdentityBridge
from .intervention_features import FEATURE_NAMES, feature_vector


class OwnHistoryPredictor:
    def __init__(self, owner, fitted):
        self.owner, self.fitted = owner, fitted

    def predict(self, current):
        frame = self.owner.tracker.frame + 1
        previous = dict(self.owner.causal_gate_history)
        history = np.array([previous.get(f, [0.] * len(FEATURE_NAMES)) for f in range(frame-3, frame)], np.float32)
        return self.fitted.predict(current, history)


class TrajectoryGateMOTBridge(SafeMOTIdentityBridge):
    def __init__(self, *args, fitted_predictor, **kwargs):
        self.causal_gate_history = []
        self.fitted_predictor = fitted_predictor
        super().__init__(*args, **kwargs)
        if self.policy.family not in ('two_branch', 'risk'):
            raise ValueError('explicit B7/B8 family required')
        self.predictor = OwnHistoryPredictor(self, fitted_predictor)

    def clone(self):
        result = super().clone()
        result.causal_gate_history = deepcopy(self.causal_gate_history)
        result.predictor = OwnHistoryPredictor(result, result.fitted_predictor)
        return result

    def step(self, frame, rows):
        past = deepcopy(self.causal_gate_history)
        actual = super().step(frame, rows)
        features = actual['authority']['features']
        if features is not None:
            self.causal_gate_history.append((frame, feature_vector(features).tolist()))
            self.causal_gate_history = self.causal_gate_history[-3:]
        actual['authority'].update(actual_causal_gate_history_before=past,
            baseline_challenger_future_is_TRAIN_supervision_only=True,
            borrowed_or_future_history=False)
        return actual
