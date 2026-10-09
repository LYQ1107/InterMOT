"""Current/past joint-state opportunity features. No truth annotations accepted."""
import numpy as np
from sam3_intermot.association.causal_identity_tracker import margin
from sam3_intermot.association.online_associator import predicted_iou

FEATURE_NAMES = (
    'proposed_probability', 'NONE_probability', 'candidate_margin', 'available_probability',
    'anchor_cosine', 'anchor_margin', 'anchor_advantage_vs_KEEP', 'KEEP_anchor_cosine',
    'base_KEEP_margin', 'proposed_base_score', 'global_regret', 'competitor_owned',
    'displaced_count', 'competitor_margin', 'quality', 'motion_iou',
    'native_same', 'track_gap', 'trusted_gap', 'candidate_count',
    'KEEP_NONE', 'target_LOST', 'proposal_NONE', 'actor_bank_size',
    'prototype_anchor_agreement', 'proposed_prototype_cosine', 'joint_entropy', 'NONE_advantage',
    'pending_confirmation_count', 'previous_intervention_age', 'previous_candidate_agreement', 'actor_probability_learned',
)


def validate_runtime_rows(rows):
    forbidden = {'gt', 'gt_id', 'target_gt_id', 'target_gt_identity', 'oracle', 'label', 'label_index', 'future_gt', 'future_target_uid'}
    for r in rows:
        if forbidden.intersection(r) or r.get('runtime_gt_read') or r.get('runtime_future_gt_used'):
            raise ValueError('GT/future truth cannot enter runtime opportunities')
    if len({str(r['candidate_uid']) for r in rows}) != len(rows):
        raise ValueError('duplicate current-frame candidate UID')


def opportunity_features(bridge, frame, rows, prepared, check, *, confirmations=0, previous_intervention=None, previous_agreement=0.):
    validate_runtime_rows(rows)
    proposal, preview = prepared['proposal'], prepared['preview']
    uid = proposal['selected_candidate_uid']; public = bridge.tracker.target_public
    uids = [str(r['candidate_uid']) for r in rows]; i = uids.index(uid) if uid in uids else None
    baseline_uid = preview['target_uid']; b = uids.index(baseline_uid) if baseline_uid in uids else None
    axis = preview['states_before_commit_axis']; j = axis.index(public)
    state = bridge.tracker.states[public]
    anchor = np.asarray(bridge.tracker.event['human_anchor'])
    cosines = np.array([float(np.dot(anchor, r['feature'])) for r in rows])
    model = getattr(prepared['provisional'], 'model', None)
    output = getattr(model, 'last', None)
    learned = output is not None
    if learned:
        joint = output['joint_probabilities'][0].detach().cpu().numpy()
        probabilities = joint[:len(rows)]; none = float(joint[-1])
        sorted_prob = sorted(probabilities, reverse=True)
        top_margin = float(sorted_prob[0] - (sorted_prob[1] if len(sorted_prob) > 1 else 0.)) if len(sorted_prob) else 0.
        proposed_probability = none if i is None else float(probabilities[i])
        positive_joint = joint[joint > 0]
        entropy = float(-np.sum(positive_joint * np.log(positive_joint)))
    else:
        proposed_probability = float(proposal['candidate_available_probability'])
        none = 1. - proposed_probability
        top_margin = float(proposal.get('identity_margin', 0.))
        entropy = 0.
    owner = next((p for p, u in preview['base_assignments'].items() if u == uid and int(p) != public), None)
    owner_col = axis.index(int(owner)) if owner is not None else None
    values = {
        'proposed_probability': proposed_probability, 'NONE_probability': none, 'candidate_margin': top_margin,
        'available_probability': float(proposal['candidate_available_probability']),
        'anchor_cosine': -1. if i is None else float(cosines[i]), 'anchor_margin': margin(cosines, i),
        'anchor_advantage_vs_KEEP': (-1. if i is None else float(cosines[i])) - (-1. if b is None else float(cosines[b])),
        'KEEP_anchor_cosine': -1. if b is None else float(cosines[b]),
        'base_KEEP_margin': margin(preview['base_matrix'][:, j], b),
        'proposed_base_score': 0. if i is None else float(preview['base_matrix'][i, j]),
        'global_regret': float(check['global_cost']) if check['feasible'] else 1.e6,
        'competitor_owned': float(owner is not None), 'displaced_count': len(check.get('displaced_public_ids', [])),
        'competitor_margin': margin(preview['base_matrix'][:, owner_col], i) if owner_col is not None else 0.,
        'quality': 0. if i is None else float(np.clip(rows[i].get('conf', 0.), 0, 1)),
        'motion_iou': 0. if i is None else float(predicted_iou(state, np.asarray(rows[i]['box_xyxy']), frame)),
        'native_same': float(i is not None and state.last_native_tid == int(rows[i]['native_tid']) and state.last_native_scope == rows[i].get('native_scope')),
        'track_gap': frame - state.last_seen_frame, 'trusted_gap': frame - bridge.tracker.last_trusted_frame,
        'candidate_count': len(rows), 'KEEP_NONE': float(b is None), 'target_LOST': float(state.state == 'LOST'),
        'proposal_NONE': float(uid is None), 'actor_bank_size': len(getattr(bridge.identity, 'bank', [])),
        'prototype_anchor_agreement': float(np.dot(state.prototype, anchor)),
        'proposed_prototype_cosine': -1. if i is None else float(np.dot(state.prototype, rows[i]['feature'])),
        'joint_entropy': entropy, 'NONE_advantage': none - (max(probabilities, default=0.) if learned else proposed_probability),
        'pending_confirmation_count': confirmations, 'previous_intervention_age': 1000 if previous_intervention is None else frame - previous_intervention,
        'previous_candidate_agreement': previous_agreement, 'actor_probability_learned': float(learned),
    }
    if not np.isfinite(list(values.values())).all(): raise ValueError('non-finite current-state feature')
    return values


def feature_vector(values):
    """Frozen scales; fitted normalizers must be FIT-only, not new thresholds."""
    scales = {'base_KEEP_margin': 10., 'proposed_base_score': 10., 'global_regret': 10., 'competitor_margin': 10.,
              'displaced_count': 10., 'track_gap': 100., 'trusted_gap': 100., 'candidate_count': 50.,
              'actor_bank_size': 8., 'pending_confirmation_count': 3., 'previous_intervention_age': 100.}
    return np.array([np.clip(values[k] / scales.get(k, 1.), -10., 10.) for k in FEATURE_NAMES], dtype=np.float32)
