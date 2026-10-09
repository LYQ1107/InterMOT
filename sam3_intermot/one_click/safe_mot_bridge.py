"""Proposal -> conservative authority -> exact full MOT -> committed feedback.

No global matcher/scorer changes, target-only stitching, history rewrites or
future inference. Two-branch value uses TRAIN-fitted predictions at runtime.
"""
from dataclasses import asdict
from copy import deepcopy
import numpy as np
from .mot_bridge import MOTIdentityBridge
from .acib_runtime import overlap
from .joint_intervention_primitives import clone_bridge, prepare_proposal, check_action, commit_action
from .intervention_features import validate_runtime_rows, opportunity_features
from .intervention_gate import GatePolicy, decide_authority
from sam3_intermot.association.opportunity_solver import AssociationAction


class SafeMOTIdentityBridge(MOTIdentityBridge):
    def __init__(self, event, identity=None, *, policy=None, predictor=None, no_human=False, frames=None):
        super().__init__(event, identity, no_human=no_human, frames=frames)
        self.policy = policy or GatePolicy()
        self.predictor = predictor
        self.authority_pending = None
        self.last_intervention = None

    def clone(self):
        result = clone_bridge(self)
        result.authority_pending = deepcopy(self.authority_pending)
        return result

    def step(self, frame, rows):
        validate_runtime_rows(rows)
        if frame != self.tracker.frame + 1: raise ValueError('one original frame one global decision')
        if self.identity is None or frame <= self.event_frame or self.policy.family == 'off':
            # Off deliberately does not evaluate actor or mutate an identity bank.
            identity = self.identity
            self.identity = None
            try: result = super().step(frame, rows)
            finally: self.identity = identity
            result['authority'] = {'family': self.policy.family, 'approved': False, 'features': None, 'reason': ['NO_INTERVENTION']}
            return result
        prepared = prepare_proposal(self, frame, rows)
        action = prepared['action']; check = check_action(self, frame, rows, prepared['preview'], action)
        selected_uid = prepared['proposal']['selected_candidate_uid']
        candidate = next((r for r in rows if str(r['candidate_uid']) == selected_uid), None)
        pending = self.authority_pending
        agreement = 0. if candidate is None or pending is None else float(np.dot(candidate['feature'], pending['feature']))
        confirmed = bool(candidate is not None and pending is not None and pending['frame'] == frame - 1 and agreement >= .9 and overlap(candidate['box_xyxy'], pending['box']) >= .3)
        count = pending['count'] + 1 if confirmed else 1 if candidate is not None else 0
        self.authority_pending = None if candidate is None else {'frame': frame, 'count': count, 'feature': np.array(candidate['feature'], copy=True), 'box': list(candidate['box_xyxy'])}
        features = opportunity_features(self, frame, rows, prepared, check, confirmations=count, previous_intervention=self.last_intervention, previous_agreement=agreement)
        approved, reasons, prediction = decide_authority(self.policy, features, feasible=check['feasible'], proposed_change=bool(check.get('assignment_changed')), predictor=self.predictor)
        actual_action = action if approved else AssociationAction('KEEP', self.tracker.target_public)
        result = commit_action(self, frame, rows, prepared, actual_action)
        if approved: self.last_intervention = frame
        result['authority'] = {'family': self.policy.family, 'configuration': asdict(self.policy), 'proposed_action': action.to_dict(),
            'approved': approved, 'effective_assignment_change': bool(approved and check['assignment_changed']),
            'features': features, 'reasons': reasons, 'prediction': prediction,
            'own_KEEP_uid': prepared['preview']['target_uid'], 'future_GT_used': False}
        return result
