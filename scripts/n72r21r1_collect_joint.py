"""Prospective actual joint-state corpus; GT-free runtime and future branches."""
import argparse
from pathlib import Path
import json
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.acib_runtime import overlap
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.intervention_features import opportunity_features, feature_vector
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal, check_action, commit_action, step_keep
from sam3_intermot.association.opportunity_solver import AssociationAction, action_for_candidate
from scripts.n72r21r1_counterfactual import compact


PROTOCOL = OUT / 'protocol/JOINT_STATE_CORPUS.json'
SOURCE_CODE = ['scripts/n72r21r1_collect_joint.py', 'scripts/n72r21r1_counterfactual.py',
    'sam3_intermot/one_click/intervention_features.py', 'sam3_intermot/one_click/safe_mot_bridge.py',
    'sam3_intermot/one_click/intervention_gate.py', 'sam3_intermot/one_click/joint_intervention_primitives.py']


def source_policy(source):
    if source == 'JOINT_BASELINE_SHADOW_P0': return 'P0', GatePolicy(family='shadow')
    if source == 'JOINT_ORIGINAL_TREATMENT_FULL': return 'FULL', GatePolicy(family='original')
    values = dict(probability_min=.7, identity_margin_min=.02, base_margin_max=4., quality_min=.2,
                  anchor_cosine_min=.4, regret_max=2., delay_frames=1)
    if source == 'JOINT_DELAYED_NO_WRITE': return 'P0', GatePolicy(family='delayed', **values)
    if source == 'JOINT_FIXED_BROAD_ON_POLICY_ROUND0': return 'P0', GatePolicy(family='confidence', **values)
    raise ValueError('unregistered causal state source')


def sample_actions(bridge, frame, rows, prepared):
    preview = prepared['preview']; public = bridge.tracker.target_public
    uids = [str(r['candidate_uid']) for r in rows]
    raw = [float(np.dot(bridge.tracker.event['human_anchor'], r['feature'])) for r in rows]
    joint = prepared['provisional'].model.last['joint_probabilities'][0].detach().cpu().numpy()[:len(rows)]
    raw_top = sorted(range(len(rows)), key=lambda i: (-raw[i], uids[i]))[:2]
    learned_top = sorted(range(len(rows)), key=lambda i: (-float(joint[i]), uids[i]))[:1]
    selected = list(dict.fromkeys([uids[i] for i in raw_top + learned_top if uids[i] != preview['target_uid']]))[:3]
    return [AssociationAction('KEEP', public), AssociationAction('REJECT_TARGET', public)] + [action_for_candidate(public, uid, preview['solver']) for uid in selected]


def collect_event(bridge, frames, frame, prefix, source):
    rows = frames[frame][1]; branch = bridge.clone(); prepared = prepare_proposal(branch, frame, rows)
    artifacts = []; attempted = []
    for slot, action in enumerate(sample_actions(branch, frame, rows, prepared)):
        arm = bridge.clone(); proposal = prepare_proposal(arm, frame, rows)
        check = check_action(arm, frame, rows, proposal['preview'], action)
        if not check['feasible']:
            attempted.append({'action': action.to_dict(), 'reason': check['reason'], 'status': 'HARD_INFEASIBLE'}); continue
        # Candidate-specific action features, not original proposal score mislabeled.
        selected_uid = proposal['preview']['target_uid'] if action.family == 'KEEP' else action.candidate_uid
        feature_proposal = {**proposal, 'proposal': {**proposal['proposal'], 'selected_candidate_uid': selected_uid}}
        pending = arm.authority_pending
        candidate = next((r for r in rows if str(r['candidate_uid']) == selected_uid), None)
        agreement = 0. if candidate is None or pending is None else float(np.dot(candidate['feature'], pending['feature']))
        consecutive = bool(candidate is not None and pending is not None and pending['frame'] == frame-1 and agreement >= .9 and overlap(candidate['box_xyxy'], pending['box']) >= .3)
        confirmations = pending['count'] + 1 if consecutive else 0 if candidate is None else 1
        features = opportunity_features(arm, frame, rows, feature_proposal, check,
            confirmations=confirmations, previous_intervention=arm.last_intervention, previous_agreement=agreement)
        current = commit_action(arm, frame, rows, proposal, action); trace = [compact(current, arm)]
        for payload, future_rows in frames[frame+1:min(len(frames), frame+101)]:
            trace.append(compact(step_keep(arm, int(payload['frame']), future_rows), arm))
        relative = prefix + '/arm' + str(slot) + '.json'
        path = write_json(relative, {'action': action.to_dict(), 'source_state_distribution': source, 'features': features,
            'feature_vector': feature_vector(features).tolist(), 'rows': trace, 'runtime_GT_input': False, 'runtime_future_GT_input': False})
        artifacts.append({'action': action.to_dict(), 'path': str(path), 'sha256': sha256(path), 'frames': len(trace)})
    assert len({read_json(a['path'])['rows'][0]['state_before'] for a in artifacts}) == 1
    seal = write_json(prefix + '/seal.json', {'status': 'COMPLETE_ACTUAL_PROSPECTIVE_SAME_PRESTATE_BRANCHES',
        'frame': frame, 'source_distribution': source, 'artifacts': artifacts, 'infeasible_attempts': attempted,
        'no_runtime_GT': True, 'no_GT_best_action': True, 'no_shared_mutable_tracker_bank_pending': True,
        'source_code_SHA': {p: sha256(ROOT / p) for p in SOURCE_CODE}, 'protocol_sha256': sha256(PROTOCOL)})
    return {'path': str(seal), 'sha256': sha256(seal), 'actual_arms': len(artifacts)}


def run(sequence):
    torch.set_num_threads(1); protocol = read_json(PROTOCOL)
    inputs_path = OUT / 'corpus/RUNTIME_INPUTS.json'; inputs = read_json(inputs_path)
    assert inputs['protocol_sha256'] == sha256(PROTOCOL)
    assert inputs['preparation_source_sha256'] == sha256(ROOT / 'scripts/n72r21r1_prepare_corpus.py')
    events = [e for e in inputs['inputs'] if e['sequence'] == sequence]; assert len(events) == 3
    assert sha256(inputs['anchor_path']) == inputs['anchor_sha256']
    anchors = np.load(inputs['anchor_path'], mmap_mode='r'); frames, index_sha = checked_frames(sequence)
    source = inputs['frozen_identity_model']
    assert sha256(source['path']) == source['sha256'] and sha256(source['fit_record_path']) == source['fit_record_sha256']
    fit = read_json(source['fit_record_path']); saved = torch.load(source['path'], map_location='cpu', weights_only=True)
    assert saved['schema'] == fit['schema']
    model = ACIBMemoryNetwork(); model.load_state_dict(saved['model'], strict=True); model.eval()
    code = {p: sha256(ROOT / p) for p in SOURCE_CODE}
    for event in events:
        assert event['candidate_index_sha256'] == index_sha
        for state_source in protocol['state_sources']:
            prefix = 'corpus/runtime/' + state_source + '/' + event['episode_uid']
            seal_path = OUT / (prefix + '/seal.json')
            if seal_path.exists():
                previous = read_json(seal_path)
                assert previous['source_code_SHA'] == code and previous['protocol_sha256'] == sha256(PROTOCOL)
                assert previous['inputs_sha256'] == sha256(inputs_path)
                assert all(sha256(a['path']) == a['sha256'] for a in previous.get('artifacts', []))
                continue
            if event['initialization_failure']:
                write_json(prefix + '/seal.json', {'status': 'INITIALIZATION_FAILURE_NOT_RUN', 'event': event,
                    'source_code_SHA': code, 'inputs_sha256': sha256(inputs_path), 'protocol_sha256': sha256(PROTOCOL), 'artifacts': []})
                continue
            storage(128 << 20)
            memory_policy, gate = source_policy(state_source); wrapper = DecisionCapture(model).eval()
            anchor = np.array(anchors[event['anchor_index']], np.float32)
            identity = TrustedACIBRecognizer(wrapper, anchor, event['episode_uid'], capacity=8, policy=memory_policy)
            identity.start_recording(sequence, fps=event['fps'], width=event['width'], height=event['height'], initial_frame=event['frame'], initial_box=event['box_xyxy'])
            bridge = SafeMOTIdentityBridge({'event_frame': event['frame'], 'human_anchor': anchor,
                'target_candidate_uid': event['clicked_candidate_uid'], 'target_box_xyxy': event['box_xyxy']}, identity, policy=gate, frames=len(frames))
            bridge.configure_fps(event['fps'])
            trace_path = ASSETS / 'corpus/traces' / state_source / (event['episode_uid'] + '.jsonl.zst')
            trajectory_path = ASSETS / 'corpus/trackers' / (state_source + '__' + event['episode_uid']) / 'data' / (sequence + '.txt')
            for path in (trace_path, trajectory_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists(): raise FileExistsError('preserve partial corpus episode; explicit versioned recovery required')
            branch_seals = []; started = time.monotonic(); sampled = 0; committed_writes = 0
            with trace_path.open('xb') as stream, trajectory_path.open('x') as trajectory:
                proc = subprocess.Popen(['zstd', '-q', '-c', '-T1', '-3'], stdin=subprocess.PIPE, stdout=stream)
                try:
                    for payload, rows in frames:
                        f = int(payload['frame'])
                        if time.monotonic() - started > 1800: raise RuntimeError('bounded causal corpus episode exceeded1800s')
                        offset = f - event['frame'] - 16
                        if offset >= 0 and offset % 128 == 0 and sampled < 16:
                            branch_seals.append(collect_event(bridge, frames, f, prefix + '/counterfactual/frame' + str(f), state_source)); sampled += 1
                        actual = bridge.step(f, rows); committed_writes += actual['joint_identity_memory_write']
                        # Current/past state only, references to immutable input tape.
                        actual.update(episode_uid=event['episode_uid'], source_state_distribution=state_source,
                            candidate_frame_reference={'sequence': sequence, 'frame': f, 'index_SHA': index_sha},
                            cached_rank1_joint_probability=float(wrapper.last['joint_probabilities'][0, :len(rows)].max()) if f > event['frame'] and rows else 0.,
                            immutable_anchor_SHA=identity.anchor_sha, identity_state_snapshot=bridge.identity.snapshot())
                        proc.stdin.write((json.dumps(actual, sort_keys=True, allow_nan=False) + '\n').encode()); trajectory.write(trajectory_text([actual]))
                    proc.stdin.close()
                    if proc.wait() != 0: raise RuntimeError('corpus compression failed')
                finally:
                    if proc.poll() is None: proc.terminate(); proc.wait()
            write_json(prefix + '/seal.json', {'status': 'COMPLETE_ACTUAL_JOINT_CORPUS_RUNTIME', 'source_code_SHA': code,
                'protocol_sha256': sha256(PROTOCOL), 'inputs_sha256': sha256(inputs_path), 'event': event, 'source_distribution': state_source,
                'model_source': source, 'source_model_fit_exposure': event['diagnostic_role'] == 'FIT',
                'seconds': time.monotonic() - started, 'frames': len(frames), 'committed_writes': committed_writes,
                'artifacts': [{'kind': kind, 'path': str(path), 'sha256': sha256(path)} for kind, path in [('trace', trace_path), ('trajectory', trajectory_path)]],
                'counterfactual_seals': branch_seals, 'prospective_sampled_events': sampled,
                'runtime_GT_input': False, 'runtime_future_GT_input': False, 'GT_history_replacement': False,
                'no_duplicate_candidate_embedding_tape': True, 'all_candidates_globally_owned_exactly_once': True})
            print(json.dumps({'actual_joint_corpus_episode_complete': event['episode_uid'], 'source': state_source, 'prospective_events': sampled,
                'actual_same_prestate_arms': sum(s['actual_arms'] for s in branch_seals), 'seconds': round(time.monotonic()-started, 2)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--sequence', required=True); args = parser.parse_args()
    run(args.sequence)
