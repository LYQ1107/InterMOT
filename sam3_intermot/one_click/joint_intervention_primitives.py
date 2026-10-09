"""GT-free proposal/assignment separation using the unchanged exact solver.

These primitives never approve a proposal automatically. Callers supply a
current-frame action (or KEEP) and feed back only the actually committed UID.
"""
from copy import copy
import numpy as np

from .mot_bridge import copy_identity, committed_feedback
from .runtime import Candidate
from sam3_intermot.association.opportunity_solver import AssociationAction, action_for_candidate, solve_counterfactual_global_assignment


def clone_bridge(bridge):
    result = copy(bridge)
    result.tracker = bridge.tracker.clone()
    result.identity = copy_identity(bridge.identity)
    assert result.tracker.states is not bridge.tracker.states
    if result.identity is not None:
        assert result.identity is not bridge.identity
        assert not np.shares_memory(result.identity.anchor, bridge.identity.anchor)
        if hasattr(result.identity, 'bank'):
            assert result.identity.bank is not bridge.identity.bank
    return result


def prepare_proposal(bridge, frame, rows):
    if frame <= bridge.event_frame or bridge.identity is None:
        raise ValueError('proposal requires a post-click identity actor')
    previous, provisional = copy_identity(bridge.identity), copy_identity(bridge.identity)
    if hasattr(provisional, 'anchor_sha'):
        proposal = provisional.step(frame, rows)
    else:
        candidates = [Candidate(str(r['candidate_uid']), r['feature'], tuple(r['box_xyxy']), float(np.clip(r.get('conf', 0.), 0, 1)), str(r['native_tid'])) for r in rows]
        proposal = provisional.step(frame, candidates, fps=bridge.fps)
    preview = bridge.tracker.clone().step(rows, frame)
    public = bridge.tracker.target_public
    if public is None:
        raise RuntimeError('sole initial click did not bind an identity')
    uid = proposal['selected_candidate_uid']
    action = AssociationAction('KEEP', public) if uid == preview['target_uid'] else AssociationAction('REJECT_TARGET', public) if uid is None else action_for_candidate(public, uid, preview['solver'])
    return {'previous': previous, 'provisional': provisional, 'proposal': proposal, 'preview': preview, 'action': action}


def check_action(bridge, frame, rows, preview, action):
    axis = preview['states_before_commit_axis']
    return solve_counterfactual_global_assignment(rows, preview['base_matrix'], [bridge.tracker.states[p].pid for p in axis], axis, action, frame=frame)


def commit_action(bridge, frame, rows, prepared, action):
    check = check_action(bridge, frame, rows, prepared['preview'], action)
    if not check['feasible']:
        raise ValueError('unsafe/infeasible action: ' + check['reason'])
    d = bridge.tracker.step(rows, frame, forced_action=action)
    bridge.identity, proposal = committed_feedback(prepared['previous'], prepared['provisional'], prepared['proposal'], rows, frame, d['target_uid'])
    uids = [o['candidate_uid'] for o in d['outputs']]
    publics = [o['public_id'] for o in d['outputs']]
    if len(set(uids)) != len(uids) or len(set(publics)) != len(publics) or set(uids) != {str(r['candidate_uid']) for r in rows}:
        raise RuntimeError('incomplete or colliding joint assignment')
    if proposal['memory_write'] and proposal['memory_write_candidate_uid'] != d['target_uid']:
        raise RuntimeError('only the committed target observation can write')
    return {k: v for k, v in d.items() if k not in ('base_matrix', 'solver', 'states_before_commit_axis', 'proposals')} | {
        'identity_decision': proposal, 'proposal_feasible': check['feasible'], 'proposal_failure_reason': None,
        'joint_identity_memory_write': bool(proposal['memory_write']), 'joint_memory_write_candidate_uid': proposal['memory_write_candidate_uid'],
        'extra_clicks': 0, 'runtime_gt_read': False, 'runtime_future_gt_used': False,
        'full_multi_object_trajectory': True, 'frozen_weight_MOT_state_distribution_shift': True,
        'executed_global_regret': check['global_cost'], 'executed_displaced_public_ids': check['displaced_public_ids']}


def step_keep(bridge, frame, rows):
    prepared = prepare_proposal(bridge, frame, rows)
    return commit_action(bridge, frame, rows, prepared, AssociationAction('KEEP', bridge.tracker.target_public))
