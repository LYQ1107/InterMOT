"""Current candidate correctness/NONE/UNKNOWN, not future override safety.

This head does not alter the tracker, ACIB weights, or committed feedback.
The prospective all-axis data and runtime use exactly the same 32 features.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .intervention_features import FEATURE_NAMES, feature_vector, opportunity_features, validate_runtime_rows
from .joint_intervention_primitives import prepare_proposal, check_action, commit_action
from .safe_mot_bridge import SafeMOTIdentityBridge
from .acib_runtime import overlap
from sam3_intermot.association.opportunity_solver import AssociationAction, action_for_candidate


SCALAR_COLUMNS = tuple(FEATURE_NAMES.index(k) for k in
    ('proposed_probability', 'NONE_probability', 'anchor_cosine', 'proposal_NONE'))


class OpenSetVerifierHead(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        if family not in ('SCALAR', 'LOGISTIC', 'MLP'):
            raise ValueError('unknown current-axis verifier family')
        size = len(SCALAR_COLUMNS) if family == 'SCALAR' else len(FEATURE_NAMES)
        self.net = nn.Sequential(nn.Linear(size, 32), nn.Tanh(), nn.Linear(32, 4)) if family == 'MLP' else nn.Linear(size, 4)

    def forward(self, x):
        return self.net(x[:, SCALAR_COLUMNS] if self.family == 'SCALAR' else x)


def predictions(logits, temperature=1.):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive frozen temperature required')
    with torch.inference_mode():
        p = torch.softmax(logits[:, :3] / temperature, dim=1).cpu().numpy()
        available = float(torch.sigmoid(logits[:, 3] / temperature).mean())
    return [{'correct': float(r[0]), 'incorrect': float(r[1]), 'unknown': float(r[2]),
             'available': available} for r in p]


class OpenSetVerifierPredictor:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['schema'] != 'N72R21R1_CURRENT_AXIS_OPEN_SET_V1' or saved['feature_names'] != list(FEATURE_NAMES):
            raise ValueError('incompatible verifier schema')
        self.model = OpenSetVerifierHead(saved['family']).eval()
        self.model.load_state_dict(saved['model'], strict=True)
        self.mean, self.std = np.array(saved['FIT_mean'], np.float32), np.array(saved['FIT_std'], np.float32)
        if self.mean.shape != (32,) or self.std.shape != (32,) or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('invalid FIT-only normalizer')
        self.selection = saved['selection']

    def predict_axis(self, axis):
        x = np.array([r['feature_vector'] for r in axis], np.float32)
        if x.shape != (len(axis), 32) or not np.isfinite(x).all():
            raise ValueError('current-axis finite 32 features required')
        with torch.inference_mode():
            logits = self.model(torch.from_numpy((x - self.mean) / self.std))
        return predictions(logits, self.selection['temperature'])


def candidate_axis(bridge, frame, rows, prepared):
    """Identical to sealed current_axis collector, without a second actor step."""
    preview = prepared['preview']; public = bridge.tracker.target_public
    result = []
    for uid in [str(r['candidate_uid']) for r in rows] + [None]:
        action = AssociationAction('KEEP', public) if uid == preview['target_uid'] else AssociationAction('REJECT_TARGET', public) if uid is None else action_for_candidate(public, uid, preview['solver'])
        check = check_action(bridge, frame, rows, preview, action)
        candidate = next((r for r in rows if str(r['candidate_uid']) == uid), None)
        pending = bridge.authority_pending
        agreement = 0. if candidate is None or pending is None else float(np.dot(candidate['feature'], pending['feature']))
        consecutive = bool(candidate is not None and pending is not None and pending['frame'] == frame - 1 and agreement >= .9 and overlap(candidate['box_xyxy'], pending['box']) >= .3)
        count = pending['count'] + 1 if consecutive else 1 if candidate is not None else 0
        specific = {**prepared, 'proposal': {**prepared['proposal'], 'selected_candidate_uid': uid}}
        features = opportunity_features(bridge, frame, rows, specific, check, confirmations=count,
            previous_intervention=bridge.last_intervention, previous_agreement=agreement)
        result.append({'candidate_uid': uid, 'features': features, 'feature_vector': feature_vector(features).tolist(),
            'action': action.to_dict(), 'action_feasible': check['feasible'], 'action_not_executed': True,
            'current_owner_public': next((int(p) for p, u in preview['base_assignments'].items() if u == uid), None) if uid is not None else None})
    return result


def simple_predictions(axis, family, temperature):
    if family not in ('E0_FROZEN', 'E1_CALIBRATED', 'E2_RELATIVE_NONE', 'E3_ANCHOR', 'E4_BASE', 'E5_TEMPORAL'):
        raise ValueError('unknown simple availability control')
    prob = np.array([r['features']['proposed_probability'] for r in axis])
    raw = np.log(np.clip(prob, 1.e-12, 1.)) / temperature
    if family == 'E2_RELATIVE_NONE':
        # Explicit candidate-relative NONE challenger, uses only current scores.
        none = axis[-1]['features']['NONE_probability']
        raw[-1] += max(0., none - max(prob[:-1], default=0.))
    raw -= raw.max(); p = np.exp(raw); p /= p.sum()
    return [{'correct': float(s), 'incorrect': float(1-s), 'unknown': 0.,
             'available': float(1-p[-1])} for s in p]


def choose(axis, pred, selection):
    """UNKNOWN means abstain/KEEP, never automatically emit or force NONE."""
    if len(axis) != len(pred) or not axis or axis[-1]['candidate_uid'] is not None:
        raise ValueError('complete real current UID axis and explicit NONE required')
    scores = [p['correct'] for p in pred]
    order = sorted(range(len(axis)), key=lambda i: (-scores[i], str(axis[i]['candidate_uid'])))
    i = order[0]; row = axis[i]; f = row['features']; p = pred[i]
    gap = scores[i] - (scores[order[1]] if len(order) > 1 else 0.)
    reasons = []
    if selection['status'] == 'CALIBRATION_ABSTAIN': reasons.append('INNER_NO_NONVACUOUS_SAFE_POINT')
    if scores[i] < selection['probability_min']: reasons.append('LOW_CURRENT_CORRECTNESS')
    if gap < selection['margin_min']: reasons.append('AMBIGUOUS_UID_OR_NONE')
    if p['unknown'] > selection['unknown_max']: reasons.append('UNKNOWN_ABSTAIN')
    family = selection.get('simple_family')
    if row['candidate_uid'] is not None:
        if family == 'E3_ANCHOR' and f['anchor_cosine'] < .7: reasons.append('IMMUTABLE_ANCHOR_DISAGREEMENT')
        if family == 'E4_BASE' and not (f['KEEP_NONE'] or f['target_LOST'] or f['base_KEEP_margin'] <= .1 or row['action']['family'] == 'KEEP'):
            reasons.append('STRONG_BASE_DISAGREEMENT')
        if family == 'E5_TEMPORAL' and f['pending_confirmation_count'] < 2: reasons.append('CAUSAL_CONFIRMATION_PENDING')
    return {'candidate_uid': row['candidate_uid'], 'axis_index': i, 'accepted': not reasons,
            'score': scores[i], 'margin': gap, 'prediction': p, 'reasons': reasons}


class OpenSetMOTIdentityBridge(SafeMOTIdentityBridge):
    def __init__(self, *args, verifier=None, selection=None, intervene=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = verifier
        self.selection = selection if selection is not None else verifier.selection
        self.intervene = intervene

    def clone(self):
        result = super().clone()
        result.selection = deepcopy(self.selection)
        return result

    def step(self, frame, rows):
        validate_runtime_rows(rows)
        if frame != self.tracker.frame + 1: raise ValueError('one original frame one global decision')
        if self.identity is None or frame <= self.event_frame:
            return super().step(frame, rows)
        prepared = prepare_proposal(self, frame, rows)
        axis = candidate_axis(self, frame, rows, prepared)
        pred = self.verifier.predict_axis(axis) if self.verifier is not None else simple_predictions(axis, self.selection['simple_family'], self.selection['temperature'])
        chosen = choose(axis, pred, self.selection); row = axis[chosen['axis_index']]
        action = AssociationAction(**row['action']); f = row['features']
        check = check_action(self, frame, rows, prepared['preview'], action)
        protection = []
        if not check['feasible']: protection.append('INFEASIBLE_GLOBAL_ACTION')
        if f['competitor_owned'] or f['displaced_count']: protection.append('OTHER_PUBLIC_ID_PROTECTED')
        if f['global_regret'] > self.selection['global_regret_max']: protection.append('GLOBAL_REGRET_LIMIT')
        approved = bool(self.intervene and chosen['accepted'] and not protection and check.get('assignment_changed'))
        # Keep original ACIB proposal/provisional pair intact: feedback adjusts
        # only to the actual global committed UID, not a provisional challenger.
        actual = commit_action(self, frame, rows, prepared, action if approved else AssociationAction('KEEP', self.tracker.target_public))
        original_uid = prepared['proposal']['selected_candidate_uid']
        original = next((r for r in axis if r['candidate_uid'] == original_uid), None)
        candidate = next((r for r in rows if str(r['candidate_uid']) == original_uid), None)
        self.authority_pending = None if candidate is None else {'frame': frame,
            'count': int(original['features']['pending_confirmation_count']),
            'feature': np.array(candidate['feature'], copy=True), 'box': list(candidate['box_xyxy'])}
        if approved: self.last_intervention = frame
        actual['authority'] = {'family': 'CURRENT_AXIS_OPEN_SET' if self.intervene else 'CURRENT_AXIS_SHADOW',
            'approved': approved, 'effective_assignment_change': approved,
            'own_KEEP_uid': prepared['preview']['target_uid'], 'features': f,
            'selected_identity_claim': chosen, 'axis_predictions': [{'candidate_uid': r['candidate_uid'], **p} for r, p in zip(axis, pred, strict=True)],
            'calibration': self.selection, 'protection_reasons': protection, 'immutable_original_ACIB_feedback': True,
            'current_correctness_not_future_safety_guarantee': True, 'no_runtime_GT_or_future_input': True,
            'TRAIN_C0_source_to_own_policy_shift_is_not_hidden': True}
        return actual
