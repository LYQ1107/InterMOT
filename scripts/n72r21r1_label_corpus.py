"""Seal every registered actual runtime first, then separate TRAIN supervision."""
from pathlib import Path
from collections import Counter
import json
import subprocess
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, TRAIN, read_json, write_json, sha256, update_status, storage
from scripts.n72r21r1_collect_joint import SOURCE_CODE, PROTOCOL
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.evaluation.joint_trajectory_labels import label_branch
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector


def verify_registered():
    protocol = read_json(PROTOCOL); inputs_path = OUT / 'corpus/RUNTIME_INPUTS.json'; inputs = read_json(inputs_path)
    code = {p: sha256(ROOT / p) for p in SOURCE_CODE}; seals = []; failed_initializations = []
    for event in inputs['inputs']:
        for source in protocol['state_sources']:
            path = OUT / 'corpus/runtime' / source / event['episode_uid'] / 'seal.json'; seal = read_json(path)
            assert seal['source_code_SHA'] == code and seal['inputs_sha256'] == sha256(inputs_path)
            assert seal['protocol_sha256'] == sha256(PROTOCOL)
            if seal['status'] == 'INITIALIZATION_FAILURE_NOT_RUN':
                failed_initializations.append({'path': str(path), 'sha256': sha256(path)}); continue
            assert seal['status'] == 'COMPLETE_ACTUAL_JOINT_CORPUS_RUNTIME' and not seal['runtime_GT_input'] and not seal['runtime_future_GT_input'] and not seal['GT_history_replacement']
            assert all(sha256(a['path']) == a['sha256'] for a in seal['artifacts'])
            for ref in seal['counterfactual_seals']:
                assert sha256(ref['path']) == ref['sha256']
                branch = read_json(ref['path'])
                assert branch['source_code_SHA'] == code and branch['protocol_sha256'] == sha256(PROTOCOL)
                assert branch['no_runtime_GT'] and branch['no_GT_best_action']
                assert all(sha256(a['path']) == a['sha256'] for a in branch['artifacts'])
            seals.append((path, seal))
    assert len(seals) + len(failed_initializations) == 96
    return inputs, seals, failed_initializations


def run():
    storage(256 << 20)
    inputs, sealed, failed = verify_registered()  # MUST finish before any future GT.
    labels_path = OUT / 'corpus/INITIALIZATION_TRUTH.json'
    identities = {r['episode_uid']: r['target_gt_identity'] for r in read_json(labels_path)['labels']}
    output = ASSETS / 'corpus/supervision/trajectory_labels_v1.jsonl.zst'; output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError('retain partial/successful supervision, use explicit versioned recovery')
    count = Counter(); initial = []; records = 0; gt_sources = {}; sequences = sorted({s['event']['sequence'] for _, s in sealed})
    with output.open('xb') as stream:
        compressor = subprocess.Popen(['zstd', '-q', '-c', '-T1', '-3'], stdin=subprocess.PIPE, stdout=stream)
        try:
            for sequence in sequences:
                frames, index_sha = checked_frames(sequence); gt = dancetrack_annotations(TRAIN / sequence)
                matched = {int(p['frame']): strict_candidate_matching(rows, gt.get(int(p['frame']), [])) for p, rows in frames}
                gt_sources[sequence] = sha256(TRAIN / sequence / 'gt/gt.txt')
                for seal_path, seal in [(p, s) for p, s in sealed if s['event']['sequence'] == sequence]:
                    event = seal['event']; target = identities[event['episode_uid']]
                    initial.append({'episode_uid': event['episode_uid'], 'source': seal['source_distribution'], 'clicked_UID_strictly_verified_target': matched[event['frame']].get(event['clicked_candidate_uid']) == target})
                    trace = read_zstd_jsonl(Path(next(a['path'] for a in seal['artifacts'] if a['kind'] == 'trace')))
                    assert [r['frame'] for r in trace] == list(range(len(frames)))
                    history = {r['frame']: feature_vector(r['authority']['features']).tolist() for r in trace if r['authority']['features'] is not None}
                    prefix_origins = {}; cursor = 0
                    for ref in seal['counterfactual_seals']:
                        branch = read_json(ref['path']); f = branch['frame']
                        while cursor < f:
                            for observation in trace[cursor]['outputs']:
                                identity = matched[cursor].get(observation['candidate_uid'])
                                if identity is not None: prefix_origins.setdefault(observation['public_id'], identity)
                            cursor += 1
                        # Axis is from this generating source's pre-commit solver,
                        # not another rollout or a future branch's first GT match.
                        prestate_origins = {p: identity for p, identity in prefix_origins.items() if str(p) in trace[f]['base_assignments']}
                        arms = [read_json(a['path']) for a in branch['artifacts']]
                        keep = next(a for a in arms if a['action']['family'] == 'KEEP')['rows']
                        for arm, artifact in zip(arms, branch['artifacts'], strict=True):
                            assert arm['feature_vector'] == feature_vector(arm['features']).tolist()
                            raw = label_branch(arm['rows'], keep, matched, target, prestate_origins=prestate_origins)
                            continuation_diagnostic = label_branch(arm['rows'], keep, matched, target)
                            row = {'sequence': sequence, 'episode_uid': event['episode_uid'], 'role': event['diagnostic_role'],
                                'state_source': seal['source_distribution'], 'frame': f, 'action': arm['action'],
                                'features': arm['features'], 'feature_vector': arm['feature_vector'],
                                'causal_previous_feature_vectors': [history.get(k, [0.]*len(FEATURE_NAMES)) for k in range(f-3, f)],
                                'raw_trajectory_labels': raw, 'runtime_seal_SHA': sha256(seal_path),
                                'short_window_continuity_diagnostic_not_training_reward': continuation_diagnostic,
                                'pretest_GT_identity_origins_only_label_not_feature': prestate_origins,
                                'initial_clicked_UID_verified_target': matched[event['frame']].get(event['clicked_candidate_uid']) == target,
                                'actual_branch_path': artifact['path'], 'actual_branch_SHA': artifact['sha256'],
                                'candidate_index_SHA': index_sha, 'offline_TRAIN_future_supervision_only': True,
                                'future_GT_is_not_feature': True, 'same_prestate_and_prospective_schedule': True}
                            compressor.stdin.write((json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode()); records += 1
                            count[(row['role'], row['state_source'], 'records')] += 1
                            for h in (1, 5, 20, 50, 100):
                                metric = raw['future']['H'+str(h)]
                                for kind in ('complete', 'benefit_label', 'risk_label'):
                                    count[(row['role'], row['state_source'], 'H'+str(h)+'_'+kind)] += bool(metric['complete'] and metric[kind])
                print(json.dumps({'offline_trajectory_labels_scene_complete': sequence, 'cumulative_actual_action_records': records}), flush=True)
            compressor.stdin.close()
            if compressor.wait() != 0: raise RuntimeError('supervision compressor failed')
        finally:
            if compressor.poll() is None: compressor.terminate(); compressor.wait()
    write_json('corpus/TRAJECTORY_SUPERVISION_MANIFEST.json', {'stage': 'N72R21R1', 'status': 'COMPLETE_ACTUAL_TRAIN_TRAJECTORY_LABELS',
        'path': str(output), 'sha256': sha256(output), 'records': records, 'feature_names': list(FEATURE_NAMES),
        'actual_runtime_source_episodes': len(sealed), 'failed_initializations': failed,
        'source_code_SHA': {p: sha256(ROOT / p) for p in SOURCE_CODE},
        'evaluator_SHA': sha256(Path(__file__)), 'label_components_SHA': sha256(ROOT / 'sam3_intermot/evaluation/joint_trajectory_labels.py'),
        'protocol_SHA': sha256(PROTOCOL), 'offline_initialization_truth_SHA': sha256(labels_path), 'GT_sources_SHA': gt_sources,
        'origin_definition_SHA': sha256(OUT / 'protocol/TRAJECTORY_ORIGIN_DEFINITION.json'),
        'initial_clicked_UID_verification': initial, 'counts': {'/'.join(k): v for k, v in count.items()},
        'current_future_labels_separate': True, 'short_horizons_explicit_not_imputed': True,
        'proxy_not_actual_HOTA_AssA': True, 'no_VAL_test_confirmation_used': True, 'not_new_model_training': True})
    update_status(phase_C_D_joint_corpus_running=False, phase_C_D_joint_corpus_labels_complete=True, completed_new_fits=0,
        status='ACTIVE_PHASE_C_E_F_H_CONTROLLER_TRAINING_AND_DIAGNOSTICS_REQUIRED')


if __name__ == '__main__': run()
