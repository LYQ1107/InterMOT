"""Fresh GT-free joint-MOT reproduction with compact pre-commit diagnostics.

Reads only sealed initial human observations, weights and real candidates.
Every original frame is independently rerun; historical outputs are not copied.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from scripts.n72r21r1_common import ROOT, OUT, ASSETS, CANDIDATES, historical, read_json, write_json, sha256, storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r21_validation import cases, case_id
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.runtime import OneClickRecognizer, RuntimeConfig
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.association.opportunity_solver import AssociationAction, solve_counterfactual_global_assignment
from sam3_intermot.association.causal_identity_tracker import margin


def checked_frames(sequence):
    root = CANDIDATES / 'candidates' / sequence
    index = read_json(root / 'index.json')
    if index.get('runtime_gt_read') or index.get('runtime_future_gt_used'):
        raise ValueError('Unsafe upstream candidate lineage')
    for key, name in [('metadata', 'metadata.jsonl.zst'), ('embeddings', 'embeddings.f16')]:
        if sha256(root / name) != index[key + '_sha256']:
            raise ValueError('Historical candidate source changed')
    frames = [(p, [r for r in rows if valid_geometry(r)]) for p, rows in load_candidate_frames(CANDIDATES, sequence)]
    assert [int(p['frame']) for p, _ in frames] == list(range(index['frame_count']))
    return frames, sha256(root / 'index.json')


def make_actor(condition, seed, init, anchor, protocol):
    event = init['event']
    if condition['family'] == 'joint_baseline':
        return None, None
    if condition['family'] == 'rule':
        actor = OneClickRecognizer(RuntimeConfig(scoring=condition['scoring'], memory_policy=condition['memory_policy'], bank_capacity=8, **protocol['rule_operating_point']))
        actor.initialize(anchor, event['box_xyxy'], recording_id=event['sequence'], frame=event['frame'])
        return actor, None
    source = next(s for s in init['models'] if s['seed'] == seed)
    assert sha256(source['path']) == source['sha256']
    assert sha256(source['fit_record_path']) == source['fit_record_sha256']
    fitted = read_json(source['fit_record_path'])
    cfg = fitted['schema']['configuration']
    assert event['sequence'] not in cfg['fit_sequences'] and event['sequence'] != cfg['inner_sequence']
    checkpoint = torch.load(source['path'], map_location='cpu', weights_only=True)
    assert checkpoint['schema'] == fitted['schema']
    model = ACIBMemoryNetwork()
    model.load_state_dict(checkpoint['model'], strict=True)
    model = DecisionCapture(model).eval()
    actor = TrustedACIBRecognizer(model, anchor, event['episode_uid'], capacity=8, policy=condition['policy'])
    actor.start_recording(event['sequence'], fps=event['fps'], width=event['width'], height=event['height'], initial_frame=event['frame'], initial_box=event['box_xyxy'])
    return actor, source


def diagnostic_step(bridge, frame, rows):
    preview = bridge.tracker.clone().step(rows, frame)
    public = preview['target_public_id']
    before = bridge.tracker.states.get(public)
    before_native = None if before is None else before.last_native_tid
    before_gap = None if before is None else frame - before.last_seen_frame
    last_actor_accept = getattr(bridge.identity, 'last_accept', None)
    bank = getattr(bridge.identity, 'bank', [])
    last_write = max([getattr(e, 'frame', -1) for e in bank], default=None)
    result = bridge.step(frame, rows)
    proposal = result['identity_decision']
    by_uid = {str(r['candidate_uid']): r for r in rows}
    baseline_uid = preview['target_uid']
    proposed_uid = None if proposal is None else proposal['proposed_candidate_uid']
    axis = preview['states_before_commit_axis']
    j = axis.index(public) if public in axis else None
    uids = [str(r['candidate_uid']) for r in rows]
    base_index = uids.index(baseline_uid) if baseline_uid in uids else None
    selected = result['selected_action']
    forced = None
    if selected is not None:
        forced = solve_counterfactual_global_assignment(rows, preview['base_matrix'], [bridge.tracker.states[p].pid for p in axis], axis, AssociationAction(**selected), frame=frame)
        assert forced['feasible']
    model = getattr(bridge.identity, 'model', None)
    output = getattr(model, 'last', None)
    joint = output['joint_probabilities'][0].detach().cpu().numpy() if output is not None and proposal is not None else None
    best = sorted(joint[:-1], reverse=True) if joint is not None else []
    own_keep_map = {o['candidate_uid']: o['public_id'] for o in preview['outputs']}
    actual_map = {o['candidate_uid']: o['public_id'] for o in result['outputs']}
    def observation(uid):
        r = by_uid.get(uid)
        return None if r is None else {'candidate_uid': uid, 'native_tid': int(r['native_tid']), 'quality': float(r.get('conf', 0.)), 'box_xyxy': list(r['box_xyxy']), 'anchor_cosine': float(np.dot(bridge.tracker.event['human_anchor'], r['feature']))}
    result['r21r1_precommit_diagnostic'] = {'own_KEEP_target_uid': baseline_uid, 'proposed_uid': proposed_uid,
        'baseline_column_margin': None if j is None else margin(preview['base_matrix'][:, j], base_index),
        'baseline_global_objective': float(sum(o['score'] for o in preview['solver']['assignment_rows'])),
        'global_regret': None if forced is None else forced['global_cost'],
        'displaced_public_ids': [] if forced is None else forced['displaced_public_ids'],
        'candidate_previous_owner': next((o['public_id'] for o in preview['outputs'] if o['candidate_uid'] == proposed_uid), None),
        'own_KEEP_assignment_changed': own_keep_map != actual_map,
        'NONE_joint_probability': None if joint is None else float(joint[-1]),
        'top1_top2_joint_margin': None if not best else float(best[0] - (best[1] if len(best) > 1 else 0.)),
        'baseline_observation': observation(baseline_uid), 'proposed_observation': observation(proposed_uid),
        'precommit_native_tid': before_native,
        'precommit_track_gap': before_gap,
        'precommit_last_actor_accept_not_human_trust': last_actor_accept,
        'precommit_last_machine_bank_write': last_write, 'precommit_bank_size': len(bank),
        'GT_used_for_diagnostic': False}
    return result


def reproduce(sequence):
    torch.set_num_threads(1)
    protocol = read_json(historical('protocol/MOT_TRAIN_PILOT.json'))
    assert sequence in protocol['sequences']
    init_path = historical('mot_pilot/initialization/' + sequence + '.json')
    init = read_json(init_path)
    assert sha256(init['anchor_path']) == init['anchor_sha256'] and not init['initialization_failure']
    frames, index_sha = checked_frames(sequence)
    assert index_sha == init['candidate_index_sha256']
    event = init['event']
    anchor = np.array(np.load(init['anchor_path'], mmap_mode='r')[event['anchor_index']], np.float32)
    click = {'event_frame': event['frame'], 'human_anchor': anchor, 'target_candidate_uid': init['clicked_candidate_uid'], 'target_box_xyxy': event['box_xyxy'], 'interaction_source': 'SIMULATED_ONE_CLICK_FROM_GT'}
    source_sha = sha256(Path(__file__))
    for name, seed, condition in cases(protocol):
        case = case_id(name, seed)
        seal_relative = 'source_audit/reproduction_v2_seals/' + case + '/' + sequence + '.json'
        if (OUT / seal_relative).exists():
            seal = read_json(OUT / seal_relative)
            assert seal['source_sha256'] == source_sha
            assert all(sha256(a['path']) == a['sha256'] for a in seal['artifacts'])
            continue
        storage(128 << 20)
        actor, model_source = make_actor(condition, seed, init, anchor, protocol)
        bridge = MOTIdentityBridge(click, actor, no_human=condition.get('click') is False, frames=len(frames))
        bridge.configure_fps(event['fps'])
        trace_path = ASSETS / 'reproduction_v2/traces' / case / (sequence + '.jsonl.zst')
        trajectory_path = ASSETS / 'reproduction_v2/trackers' / case / 'data' / (sequence + '.txt')
        for path in (trace_path, trajectory_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError('Preserve partial reproduction; do not overwrite')
        old_seal = read_json(historical('mot_pilot/runtime_seals/' + case + '/' + sequence + '.json'))
        old_trace = read_zstd_jsonl(Path(next(a['path'] for a in old_seal['artifacts'] if a['kind'] == 'trace')))
        started = time.monotonic()
        with trace_path.open('xb') as stream, trajectory_path.open('x') as text:
            proc = subprocess.Popen(['zstd', '-q', '-c', '-T1', '-3'], stdin=subprocess.PIPE, stdout=stream)
            try:
                for payload, rows in frames:
                    frame = int(payload['frame'])
                    if time.monotonic() - started > 1800:
                        raise RuntimeError('Bounded reproduction case wallclock exceeded')
                    result = diagnostic_step(bridge, frame, rows)
                    decision = result['identity_decision']
                    if decision is not None:
                        decision['rank1_identity_joint_probability'] = float(actor.model.last['joint_probabilities'][0, :len(rows)].max()) if model_source is not None and rows else 0. if model_source is not None else decision['candidate_available_probability']
                    result.update(case=case, initialization_label='NO_CLICK_CONTROL' if condition.get('click') is False else 'SIMULATED_ONE_CLICK_FROM_GT')
                    for key in ('outputs', 'target_uid', 'selected_action', 'state_before', 'state_after', 'joint_identity_memory_write', 'joint_memory_write_candidate_uid'):
                        if result[key] != old_trace[frame][key]:
                            raise ValueError(f'A/A failed {case}/{sequence}/{frame}/{key}')
                    text.write(trajectory_text([result]))
                    proc.stdin.write((json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode())
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError('Reproduction compression failed')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
        expected = next(a['sha256'] for a in old_seal['artifacts'] if a['kind'] == 'trajectory')
        assert sha256(trajectory_path) == expected
        artifacts = [{'kind': kind, 'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size} for kind, path in [('trace', trace_path), ('trajectory', trajectory_path)]]
        write_json(seal_relative, {'stage': 'N72R21R1', 'case': case, 'sequence': sequence, 'frames': len(frames), 'source_sha256': source_sha, 'historical_seal_sha256': sha256(historical('mot_pilot/runtime_seals/' + case + '/' + sequence + '.json')), 'candidate_index_sha256': index_sha, 'initialization_sha256': sha256(init_path), 'model_source': model_source, 'trajectory_byte_identical_to_original': True, 'per_frame_committed_states_and_outputs_AA': True, 'artifacts': artifacts, 'GT_parsed_by_runtime': False, 'one_click_only': True, 'seconds': time.monotonic() - started})
        print(json.dumps({'new_joint_reproduction_complete': case, 'sequence': sequence, 'original_frames': len(frames), 'trajectory_byte_AA': True, 'seconds': round(time.monotonic() - started, 2)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    args = parser.parse_args()
    reproduce(args.sequence)
