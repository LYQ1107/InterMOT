"""Real same-prestate A3 branches; future TRAIN truth only after runtime seals."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch

from scripts.n72r21r1_common import ROOT, OUT, TRAIN, historical, read_json, write_json, sha256, storage
from scripts.n72r21r1_reproduce_v2 import checked_frames, make_actor
from scripts.n72r21r1_event_audit import verified_seals, seal_trace
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.joint_intervention_primitives import clone_bridge, prepare_proposal, check_action, commit_action, step_keep
from sam3_intermot.association.opportunity_solver import AssociationAction, action_for_candidate
from sam3_intermot.evaluation.safe_intervention_events import assignment_map, identity_outcome, contiguous_intervals
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching


PROTOCOL = OUT / 'protocol/COUNTERFACTUAL_A3.json'


def compact(result, bridge):
    return {k: result[k] for k in ('frame', 'outputs', 'target_public_id', 'target_uid', 'births', 'deaths', 'state_before', 'state_after', 'selected_action')} | {
        'memory_write': bool(result.get('joint_identity_memory_write', False)),
        'memory_write_uid': result.get('joint_memory_write_candidate_uid'),
        'actor_state': None if bridge.identity is None else bridge.identity.snapshot(),
        'target_native_after': bridge.tracker.states[bridge.tracker.target_public].last_native_tid,
        'target_prototype_anchor_cosine_after': float(np.dot(bridge.tracker.states[bridge.tracker.target_public].prototype, bridge.tracker.event['human_anchor'])),
        'runtime_GT_read': False, 'runtime_future_GT_used': False}


def event_branch(bridge, frame, rows, name, original_action):
    if name == 'BASELINE_ASSOCIATION_NO_ACTOR':
        bridge.identity = None
        return bridge.step(frame, rows), None
    prepared = prepare_proposal(bridge, frame, rows)
    public = bridge.tracker.target_public
    if name == 'KEEP_WITH_ACTOR':
        action = AssociationAction('KEEP', public)
    elif name == 'FORCED_ORIGINAL_PROPOSAL':
        action = AssociationAction(**original_action)
        assert prepared['action'].to_dict() == action.to_dict()
    elif name == 'REJECT_TARGET':
        action = AssociationAction('REJECT_TARGET', public)
    else:
        preview = prepared['preview']; axis = preview['states_before_commit_axis']
        indices = [i for i, r in enumerate(rows) if str(r['candidate_uid']) != preview['target_uid']]
        if not indices:
            return None, {'status': 'NO_ALTERNATIVE_CURRENT_CANDIDATE'}
        scores = [float(np.dot(bridge.tracker.event['human_anchor'], r['feature'])) for r in rows] if name == 'RAW_ANCHOR_TOP_ALTERNATIVE' else preview['base_matrix'][:, axis.index(public)]
        i = min(indices, key=lambda i: (-float(scores[i]), str(rows[i]['candidate_uid'])))
        action = action_for_candidate(public, str(rows[i]['candidate_uid']), preview['solver'])
    feasibility = check_action(bridge, frame, rows, prepared['preview'], action)
    if not feasibility['feasible']:
        return None, {'status': 'HARD_INFEASIBLE_NOT_OVERRIDDEN', 'action': action.to_dict(), 'reason': feasibility['reason']}
    return commit_action(bridge, frame, rows, prepared, action), {'status': 'FEASIBLE', 'action': action.to_dict(), 'global_regret': feasibility['global_cost'], 'displaced_public_ids': feasibility['displaced_public_ids']}


def run_runtime(sequence, seed):
    torch.set_num_threads(1)
    protocol = read_json(PROTOCOL)
    source = sha256(Path(__file__)); primitives = sha256(ROOT / 'sam3_intermot/one_click/joint_intervention_primitives.py')
    old_protocol = read_json(historical('protocol/MOT_TRAIN_PILOT.json'))
    seals = verified_seals(2)
    case = 'ACIB_FULL_SEED' + str(seed); key = case + '/' + sequence
    first_path = OUT / 'diagnostics/FIRST_DIVERGENCE_AUDIT.json'
    onsets = read_json(first_path)['cases'][key]['A3_predeclared_onsets']
    original = seal_trace(seals[case, sequence])
    frames, index_sha = checked_frames(sequence)
    init = read_json(historical('mot_pilot/initialization/' + sequence + '.json')); event = init['event']
    assert sha256(init['anchor_path']) == init['anchor_sha256']
    anchor = np.array(np.load(init['anchor_path'], mmap_mode='r')[event['anchor_index']], np.float32)
    condition = {'family': 'T2', 'policy': 'FULL'}
    actor, model = make_actor(condition, seed, init, anchor, old_protocol)
    bridge = MOTIdentityBridge({'event_frame': event['frame'], 'human_anchor': anchor, 'target_candidate_uid': init['clicked_candidate_uid'], 'target_box_xyxy': event['box_xyxy']}, actor, frames=len(frames))
    bridge.configure_fps(event['fps'])
    written_seals = []
    for payload, rows in frames:
        f = int(payload['frame'])
        if f > max(onsets): break
        if f in onsets:
            prefix = 'counterfactual/runtime/' + case + '/' + sequence + '/frame' + str(f)
            seal_path = OUT / (prefix + '/seal.json')
            if seal_path.exists():
                existing = read_json(seal_path)
                assert existing['source_sha256'] == source and existing['primitives_sha256'] == primitives
                assert existing['protocol_sha256'] == sha256(PROTOCOL)
                assert all(sha256(a['path']) == a['sha256'] for a in existing['artifacts'])
            else:
                storage(16 << 20); artifacts = []; failed = {}
                for name in protocol['branches']:
                    branch = clone_bridge(bridge)
                    assert branch.tracker.frame == f - 1
                    current, info = event_branch(branch, f, rows, name, original[f]['selected_action'])
                    if current is None:
                        failed[name] = info; continue
                    trace = [compact(current, branch)]
                    for future_payload, future_rows in frames[f + 1:min(len(frames), f + 101)]:
                        future = int(future_payload['frame'])
                        d = branch.step(future, future_rows) if branch.identity is None else step_keep(branch, future, future_rows)
                        trace.append(compact(d, branch))
                    relative = prefix + '/' + name + '.json'
                    path = write_json(relative, {'name': name, 'sequence': sequence, 'seed': seed, 'event_frame': f, 'action_check': info,
                        'starting_tracker_state_sha256': trace[0]['state_before'], 'runtime_GT_used': False,
                        'runtime_future_GT_used': False, 'one_click_only': True, 'rows': trace})
                    artifacts.append({'branch': name, 'path': str(path), 'sha256': sha256(path), 'frames': len(trace)})
                # Same starting tracker state even when actors differ.
                assert len({read_json(a['path'])['starting_tracker_state_sha256'] for a in artifacts}) == 1
                baseline_rows = read_json(next(a['path'] for a in artifacts if a['branch'] == 'BASELINE_ASSOCIATION_NO_ACTOR'))['rows']
                keep_rows = read_json(next(a['path'] for a in artifacts if a['branch'] == 'KEEP_WITH_ACTOR'))['rows']
                assert all(a['outputs'] == b['outputs'] and a['state_after'] == b['state_after'] for a, b in zip(baseline_rows, keep_rows, strict=True))
                write_json(prefix + '/seal.json', {'status': 'COMPLETE_ACTUAL_RUNTIME_BRANCHES', 'event_frame': f, 'sequence': sequence, 'seed': seed,
                    'source_sha256': source, 'primitives_sha256': primitives, 'protocol_sha256': sha256(PROTOCOL),
                    'first_divergence_audit_sha256': sha256(first_path), 'original_reproduction_seal': seals[case, sequence],
                    'model_source': model, 'candidate_index_sha256': index_sha, 'artifacts': artifacts, 'failed_branch_attempts': failed,
                    'all_branches_same_prestate': True, 'baseline_KEEP_full_ownership_and_state_AA': True,
                    'GT_parsed_by_runtime': False, 'no_mutable_state_shared': True, 'future_policy': 'KEEP_ONE_SHOT_DIAGNOSTIC'})
            written_seals.append(str(seal_path))
            print(json.dumps({'A3_event_runtime_sealed': key, 'frame': f, 'branches': len(read_json(seal_path)['artifacts'])}), flush=True)
        primary = bridge.step(f, rows)
        for field in ('outputs', 'target_uid', 'selected_action', 'state_before', 'state_after', 'joint_identity_memory_write'):
            assert primary[field] == original[f][field], (key, f, field)
    return written_seals


def evaluate_all():
    # Do not parse future TRAIN GT until every registered event has all seals.
    plan = read_json(OUT / 'diagnostics/FIRST_DIVERGENCE_AUDIT.json')['cases']
    verified = {}
    for key, detail in plan.items():
        for f in detail['A3_predeclared_onsets']:
            path = OUT / 'counterfactual/runtime' / key / ('frame' + str(f)) / 'seal.json'
            seal = read_json(path)
            assert seal['source_sha256'] == sha256(Path(__file__))
            assert seal['primitives_sha256'] == sha256(ROOT / 'sam3_intermot/one_click/joint_intervention_primitives.py')
            assert seal['protocol_sha256'] == sha256(PROTOCOL)
            assert seal['all_branches_same_prestate'] and seal['baseline_KEEP_full_ownership_and_state_AA']
            assert not seal['GT_parsed_by_runtime']
            assert all(sha256(a['path']) == a['sha256'] for a in seal['artifacts'])
            verified[key, f] = (path, seal)
    labels = {r['episode_uid']: r['target_gt_identity'] for r in read_json(historical('development/INITIALIZATION_TRUTH.json'))['labels']}
    truth = {}
    for sequence in ('dancetrack0001', 'dancetrack0002'):
        frames, _ = checked_frames(sequence); gt = dancetrack_annotations(TRAIN / sequence)
        truth[sequence] = {int(p['frame']): strict_candidate_matching(rows, gt.get(int(p['frame']), [])) for p, rows in frames}
    results = {}
    for (key, f), (seal_path, seal) in verified.items():
        sequence = seal['sequence']; init = read_json(historical('mot_pilot/initialization/' + sequence + '.json'))
        target = labels[init['event']['episode_uid']]; matched = truth[sequence]
        branches = {a['branch']: read_json(a['path']) for a in seal['artifacts']}
        keep = branches['KEEP_WITH_ACTOR']['rows']; public = keep[0]['target_public_id']; origin = {}
        for r in keep:
            for o in r['outputs']:
                identity = matched[r['frame']].get(o['candidate_uid'])
                if identity is not None: origin.setdefault(o['public_id'], identity)
        event_result = {'sequence': sequence, 'seed': seal['seed'], 'event_frame': f,
            'seal_sha256': sha256(seal_path), 'branches': {}, 'failed_branch_attempts': seal['failed_branch_attempts'],
            'causal_scope': 'One-shot action, same treatment prestate, identical KEEP continuation. Not whole adaptive intervention policy.',
            'non_target_origin_proxy': 'First verified KEEP identity in this short branch; not global IDF1 or full-sequence C0 origin.'}
        for name, branch in branches.items():
            rows = branch['rows']; future = []; damage_counter = {}; own_correct_frames = []; diverged = []
            for actual, baseline in zip(rows, keep, strict=True):
                frame = actual['frame']; assert baseline['frame'] == frame
                labels_now = matched[frame]; am, bm = assignment_map(actual), assignment_map(baseline)
                right = labels_now.get(am.get(public)) == target; base_right = labels_now.get(bm.get(public)) == target
                outcome = identity_outcome(am.get(public), labels_now, target)
                affected = [p for p, identity in origin.items() if p != public and labels_now.get(bm.get(p)) == identity and labels_now.get(am.get(p)) != identity]
                affected_good = [p for p, identity in origin.items() if p != public and labels_now.get(bm.get(p)) != identity and labels_now.get(am.get(p)) == identity]
                write_label = labels_now.get(actual['memory_write_uid']) if actual['memory_write'] else None
                metrics = {'offset': frame - f, 'target_correct': int(right), 'N01': int(right and not base_right), 'N10': int(base_right and not right),
                    'verified_other_takeover': int(outcome == 'VERIFIED_OTHER'), 'UNKNOWN': int(outcome == 'UNKNOWN_UNMATCHED'),
                    'target_NONE': int(outcome == 'NONE'), 'non_target_damage': len(affected), 'non_target_benefit': len(affected_good),
                    'affected_public_ids': affected, 'ownership_changed': int(am != bm), 'births': len(actual['births']), 'deaths': len(actual['deaths']),
                    'memory_writes': int(actual['memory_write']), 'wrong_or_UNKNOWN_writes': int(actual['memory_write'] and write_label != target),
                    'prototype_diff_from_KEEP': abs(actual['target_prototype_anchor_cosine_after'] - baseline['target_prototype_anchor_cosine_after']),
                    'native_diff_from_KEEP': int(actual['target_native_after'] != baseline['target_native_after'])}
                if frame == f:
                    current = metrics
                else:
                    future.append(metrics)
                    if right: own_correct_frames.append(frame)
                    if am != bm: diverged.append(frame)
                    for p in affected: damage_counter[p] = damage_counter.get(p, 0) + 1
            horizons = {}
            for h in (1, 5, 20, 50, 100):
                window = [r for r in future if r['offset'] <= h]
                horizons['H' + str(h)] = {k: sum(r[k] for r in window) for k in ('target_correct', 'N01', 'N10', 'verified_other_takeover', 'UNKNOWN', 'target_NONE', 'non_target_damage', 'non_target_benefit', 'ownership_changed', 'births', 'deaths', 'memory_writes', 'wrong_or_UNKNOWN_writes', 'native_diff_from_KEEP')}
                horizons['H' + str(h)].update(actual_future_frames=len(window), complete=len(window) == h,
                    distinct_affected_public_ids=sorted({p for r in window for p in r['affected_public_ids']}),
                    max_prototype_diff_from_KEEP=max((r['prototype_diff_from_KEEP'] for r in window), default=0.))
            event_result['branches'][name] = {'time_zero': current, 'future': horizons,
                'first_future_target_recovery_frame': min(own_correct_frames, default=None),
                'last_ownership_divergence_offset_in_100': max((frame - f for frame in diverged), default=0),
                'observed_ownership_divergence_intervals': contiguous_intervals(diverged),
                'affected_public_frame_counts_H100': damage_counter,
                'action_check': branch['action_check']}
        results[key + '/frame' + str(f)] = event_result
        forced = event_result['branches']['FORCED_ORIGINAL_PROPOSAL']
        print(json.dumps({'A3_event_evaluated': key, 'frame': f, 'forced_H100': forced['future']['H100'], 'time_zero_N10': forced['time_zero']['N10']}), flush=True)
    write_json('diagnostics/COUNTERFACTUAL_ROLLBACK.json', {'stage': 'N72R21R1', 'status': 'COMPLETE_ACTUAL_A3_FIRST_EVENT_DIAGNOSTIC',
        'events': results, 'actual_event_count': len(verified), 'actual_branch_count': sum(len(seal['artifacts']) for _, seal in verified.values()),
        'protocol_sha256': sha256(PROTOCOL), 'source_sha256': sha256(Path(__file__)),
        'offline_GT_only_after_all_registered_runtime_branches_sealed': True, 'no_VAL_test_or_confirmation_used': True,
        'scientific_PASS': False, 'full_adaptive_policy_MOT_performance_not_measured_here': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence')
    parser.add_argument('--seed', type=int, choices=(72101, 72102, 72103))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.evaluate:
        evaluate_all()
    else:
        assert args.sequence in ('dancetrack0001', 'dancetrack0002') and args.seed is not None
        run_runtime(args.sequence, args.seed)
