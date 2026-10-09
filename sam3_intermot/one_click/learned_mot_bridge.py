"""Learned multi-action authority over the original complete global MOT solve."""
from copy import deepcopy
import numpy as np
from .safe_mot_bridge import SafeMOTIdentityBridge
from .intervention_features import validate_runtime_rows, opportunity_features, feature_vector, FEATURE_NAMES
from .joint_intervention_primitives import prepare_proposal, check_action, commit_action
from .acib_runtime import overlap
from .learned_authority import eligible_prediction
from sam3_intermot.association.opportunity_solver import AssociationAction, action_for_candidate


def current_candidate_actions(bridge, rows, prepared):
    preview = prepared['preview']; public = bridge.tracker.target_public
    uids = [str(r['candidate_uid']) for r in rows]
    raw = [float(np.dot(bridge.identity.anchor, r['feature'])) for r in rows]
    joint = prepared['provisional'].model.last['joint_probabilities'][0].detach().cpu().numpy()[:len(rows)]
    raw_top = sorted(range(len(rows)), key=lambda i: (-raw[i], uids[i]))[:2]
    learned_top = sorted(range(len(rows)), key=lambda i: (-float(joint[i]), uids[i]))[:1]
    selected = list(dict.fromkeys([uids[i] for i in raw_top+learned_top if uids[i] != preview['target_uid']]))[:3]
    return [AssociationAction('KEEP', public), AssociationAction('REJECT_TARGET', public)] + [action_for_candidate(public, uid, preview['solver']) for uid in selected]


class LearnedMOTIdentityBridge(SafeMOTIdentityBridge):
    def __init__(self, *args, predictor, intervene=True, **kwargs):
        super().__init__(*args, predictor=predictor, **kwargs)
        self.intervene = intervene
        self.causal_proposal_history = []

    def clone(self):
        result = super().clone()
        result.causal_proposal_history = deepcopy(self.causal_proposal_history)
        return result

    def step(self, frame, rows):
        validate_runtime_rows(rows)
        if frame != self.tracker.frame+1: raise ValueError('one frame one global commit')
        if self.identity is None or frame <= self.event_frame:
            return super().step(frame, rows)
        prepared = prepare_proposal(self, frame, rows); pending = self.authority_pending
        previous = {f: v for f, v in self.causal_proposal_history}
        history = np.array([previous.get(f, [0.]*len(FEATURE_NAMES)) for f in range(frame-3, frame)], np.float32)
        alternatives = []
        for action in current_candidate_actions(self, rows, prepared):
            check = check_action(self, frame, rows, prepared['preview'], action)
            if not check['feasible']:
                alternatives.append({'action': action.to_dict(), 'feasible': False, 'reason': check['reason']}); continue
            uid = prepared['preview']['target_uid'] if action.family == 'KEEP' else action.candidate_uid
            candidate = next((r for r in rows if str(r['candidate_uid']) == uid), None)
            agreement = 0. if candidate is None or pending is None else float(np.dot(candidate['feature'], pending['feature']))
            consecutive = bool(candidate is not None and pending is not None and pending['frame'] == frame-1 and agreement >= .9 and overlap(candidate['box_xyxy'], pending['box']) >= .3)
            count = pending['count']+1 if consecutive else 0 if candidate is None else 1
            specific = {**prepared, 'proposal': {**prepared['proposal'], 'selected_candidate_uid': uid}}
            features = opportunity_features(self, frame, rows, specific, check, confirmations=count,
                previous_intervention=self.last_intervention, previous_agreement=agreement)
            prediction = self.predictor.predict(feature_vector(features), history)
            eligible = bool(action.family != 'KEEP' and check['assignment_changed'] and eligible_prediction(prediction, self.predictor.selection))
            alternatives.append({'action': action.to_dict(), 'feasible': True, 'eligible': eligible,
                'features': features, 'prediction': prediction, 'assignment_changed': check['assignment_changed']})
        # History/pending represents the actual causal actor proposal, exactly as
        # stored by the corpus, not an uncommitted alternative or oracle winner.
        original_uid = prepared['proposal']['selected_candidate_uid']
        original_candidate = next((r for r in rows if str(r['candidate_uid']) == original_uid), None)
        agreement = 0. if original_candidate is None or pending is None else float(np.dot(original_candidate['feature'], pending['feature']))
        consecutive = bool(original_candidate is not None and pending is not None and pending['frame'] == frame-1 and agreement >= .9 and overlap(original_candidate['box_xyxy'], pending['box']) >= .3)
        count = pending['count']+1 if consecutive else 0 if original_candidate is None else 1
        self.authority_pending = None if original_candidate is None else {'frame':frame,'count':count,
            'feature':np.array(original_candidate['feature'],copy=True),'box':list(original_candidate['box_xyxy'])}
        original_check = check_action(self, frame, rows, prepared['preview'], prepared['action'])
        original_features = opportunity_features(self,frame,rows,prepared,original_check,confirmations=count,
            previous_intervention=self.last_intervention,previous_agreement=agreement)
        choices = [a for a in alternatives if a.get('eligible')]
        winner = sorted(choices,key=lambda a:(-a['prediction']['value'],str(a['action']['candidate_uid']),a['action']['family']))[0] if choices and self.intervene else None
        action = AssociationAction(**winner['action']) if winner else AssociationAction('KEEP',self.tracker.target_public)
        actual = commit_action(self,frame,rows,prepared,action)
        if winner: self.last_intervention = frame
        self.causal_proposal_history.append((frame,feature_vector(original_features).tolist()))
        self.causal_proposal_history = self.causal_proposal_history[-3:]
        actual['authority'] = {'family':'LEARNED_MULTI_ACTION' if self.intervene else 'LEARNED_SHADOW',
            'approved':bool(winner),'effective_assignment_change':bool(winner),'features':original_features,
            'alternatives':alternatives,'own_KEEP_uid':prepared['preview']['target_uid'],
            'calibration':self.predictor.selection,'causal_previous_feature_vectors':history.tolist(),
            'reason':['APPROVED_PREDICTED_BENEFIT_LOW_RISK'] if winner else ['DEFAULT_KEEP'],
            'no_runtime_GT_or_future_input':True,'no_uncommitted_history_feedback':True}
        return actual
