"""Bounded full-joint fixed gates, shared immutable candidates, no future truth."""
import argparse
from pathlib import Path
import json
import resource
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, historical, read_json, write_json, sha256, storage
from scripts.n72r21r1_reproduce_v2 import checked_frames, make_actor
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy


PROTOCOL = OUT / 'protocol/FIXED_GATE_PILOT.json'


def registrations(protocol, seed):
    configs = []
    if seed == protocol['historical_weight_reproduction_seeds'][0]:
        configs.extend((name, {'family': 'joint_baseline', 'click': name != 'NO_HUMAN_C0'}, GatePolicy(family='off')) for name in ('NO_HUMAN_C0', 'CLICK_C0'))
        configs.extend((name, {'family': 'rule', 'scoring': scoring, 'memory_policy': memory}, GatePolicy(family='original')) for name, scoring, memory in [('RAW_ANCHOR', 'anchor', 'P0'), ('RAW_MEAN_P1', 'mean_prototype', 'P1')])
    # Controls first for early A/A evidence; all registered cases still run.
    ordered = sorted(protocol['configurations'], key=lambda cfg: (0 if cfg['policy']['family'] == 'shadow' else 1 if cfg['policy']['family'] == 'original' else 2, cfg['name']))
    for cfg in ordered:
        condition = {'family': 'T2', 'policy': 'FULL' if cfg['actor'] == 'ACIB_FULL_FROZEN' else 'P0'}
        configs.append((cfg['name'] + '_SEED' + str(seed), condition, GatePolicy(**cfg['policy'])))
    return configs


def run(sequence, seed):
    torch.set_num_threads(1)
    protocol = read_json(PROTOCOL)
    assert sequence in protocol['sequences'] and seed in protocol['historical_weight_reproduction_seeds']
    for source, digest in protocol['source_freeze'].items(): assert sha256(ROOT / source) == digest
    source_sha = sha256(Path(__file__))
    init_path = historical('mot_pilot/initialization/' + sequence + '.json'); init = read_json(init_path)
    frames, index_sha = checked_frames(sequence); assert index_sha == init['candidate_index_sha256']
    assert sha256(init['anchor_path']) == init['anchor_sha256']
    event = init['event']; anchor = np.array(np.load(init['anchor_path'], mmap_mode='r')[event['anchor_index']], np.float32)
    click = {'event_frame': event['frame'], 'human_anchor': anchor, 'target_candidate_uid': init['clicked_candidate_uid'], 'target_box_xyxy': event['box_xyxy']}
    old_protocol = read_json(historical('protocol/MOT_TRAIN_PILOT.json'))
    for name, condition, policy in registrations(protocol, seed):
        seal_relative = 'pilot/runtime_seals/' + name + '/' + sequence + '.json'
        if (OUT / seal_relative).exists():
            seal = read_json(OUT / seal_relative)
            assert seal['source_sha256'] == source_sha and seal['protocol_sha256'] == sha256(PROTOCOL)
            assert all(sha256(a['path']) == a['sha256'] for a in seal['artifacts'])
            continue
        storage(64 << 20)
        actor, model = make_actor(condition, seed, init, anchor, old_protocol)
        bridge = SafeMOTIdentityBridge(click, actor, policy=policy, no_human=condition.get('click') is False, frames=len(frames))
        bridge.configure_fps(event['fps'])
        trace_path = ASSETS / 'fixed_pilot/traces' / name / (sequence + '.jsonl.zst')
        trajectory_path = ASSETS / 'fixed_pilot/trackers' / name / 'data' / (sequence + '.txt')
        for path in (trace_path, trajectory_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists(): raise FileExistsError('preserve partial pilot, do not overwrite')
        count = written = 0; total_step = 0.; started = time.monotonic()
        with trace_path.open('xb') as stream, trajectory_path.open('x') as trajectory:
            proc = subprocess.Popen(['zstd', '-q', '-c', '-T1', '-3'], stdin=subprocess.PIPE, stdout=stream)
            try:
                for payload, rows in frames:
                    if time.monotonic() - started > 1800: raise RuntimeError('bounded pilot wallclock exceeded')
                    f = int(payload['frame']); step_start = time.perf_counter(); d = bridge.step(f, rows); total_step += time.perf_counter() - step_start
                    authority = d['authority']; count += bool(authority.get('effective_assignment_change')); written += d['joint_identity_memory_write']
                    d.update(case=name, frame_candidate_count=len(rows), source_distribution='TRUE_JOINT_TREATMENT_STATE', source_input='CACHED_SEALED_REAL_SAM3_OSNET')
                    proc.stdin.write((json.dumps(d, sort_keys=True, allow_nan=False) + '\n').encode())
                    trajectory.write(trajectory_text([d]))
                proc.stdin.close()
                if proc.wait() != 0: raise RuntimeError('pilot compressor failed')
            finally:
                if proc.poll() is None: proc.terminate(); proc.wait()
        expected_case = 'CLICK_C0' if policy.family in ('shadow', 'off') and name != 'NO_HUMAN_C0' else name if name in protocol['common_controls'] else 'ACIB_FULL_SEED' + str(seed) if policy.family == 'original' else None
        if expected_case is not None:
            old = read_json(historical('mot_pilot/runtime_seals/' + expected_case + '/' + sequence + '.json'))
            expected = next(a['sha256'] for a in old['artifacts'] if a['kind'] == 'trajectory')
            assert sha256(trajectory_path) == expected, ('AA trajectory mismatch', name, sequence)
        parameters = sum(p.numel() for p in actor.model.parameters()) if model is not None else 0
        write_json(seal_relative, {'status': 'COMPLETE_ACTUAL_FULL_JOINT_RUNTIME', 'case': name, 'sequence': sequence, 'seed': seed,
            'condition': condition, 'configuration': policy.__dict__, 'protocol_sha256': sha256(PROTOCOL),
            'source_sha256': source_sha, 'source_code_SHA': protocol['source_freeze'], 'candidate_index_sha256': index_sha,
            'initialization_sha256': sha256(init_path), 'model_source': model, 'frames': len(frames),
            'artifacts': [{'kind': kind, 'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size} for kind, path in [('trace', trace_path), ('trajectory', trajectory_path)]],
            'effective_interventions': count, 'committed_identity_writes': written, 'AA_reference_case': expected_case,
            'cached_joint_step_seconds': total_step, 'total_cached_replay_seconds': time.monotonic() - started,
            'cached_joint_FPS_not_pixel_to_output_FPS': len(frames) / total_step, 'added_identity_parameters': parameters,
            'process_cumulative_peak_RSS_bytes_not_per_case_peak': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'association_only_seconds': total_step if actor is None else None,
            'GT_runtime': False, 'future_GT_runtime': False, 'global_complete_unique_ownership': True,
            'one_click_only': True, 'scientific_success': None})
        print(json.dumps({'fixed_gate_runtime_complete': name, 'sequence': sequence, 'effective': count, 'writes': written, 'seconds': round(time.monotonic() - started, 2)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--sequence', required=True)
    parser.add_argument('--seed', type=int, required=True, choices=(72101, 72102, 72103)); args = parser.parse_args()
    run(args.sequence, args.seed)
