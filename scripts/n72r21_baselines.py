"""N72R21 current-observation replay of actual frozen historical baselines.

The replay command has no truth-adapter imports. Its evaluate command is a
separate process and may open future annotations only after verifying seals.
No historical entry point/writer is called; only public frozen model classes
and the pure candidate loader are reused. These are exposed TRAIN diagnostics,
not independent validation or cross-recording identity evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from scripts.n72r21_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_zstd_jsonl
from sam3_intermot.one_click import Candidate, OneClickRecognizer, RuntimeConfig
from sam3_intermot.association.identity_authority import AdapterEnsemble, AuthorityConfig
from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.opportunity_tracker import OpportunityTracker, InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from scripts.n72r20r4r1_memory_fit import ReliabilityEnsemble

CASES = ('B0_RAW_ANCHOR', 'B1_R3R2_ADAPTER', 'B2_R4_CAUSAL_P1',
         'B3_R4R1_NATIVE', 'B4_P0', 'B4_P1', 'B4_P4', 'B4_P6', 'B5_ONLINE_NO_LTM')
ENCODER_SHA = '2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154'
GRU_SHA = '94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2'
SEALED_CODE = ('scripts/n72r21_baselines.py', 'sam3_intermot/one_click/runtime.py',
               'sam3_intermot/association/causal_identity_tracker.py',
               'sam3_intermot/association/opportunity_tracker.py',
               'sam3_intermot/association/opportunity_memory.py',
               'sam3_intermot/association/causal_state_commit.py')


def overlap(a, b):
    """Pure current-click geometry, not GT identity matching."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = (a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter
    return float(inter/union) if union > 0 else 0.


def click_uid(rows, box):
    eligible = [(overlap(r['box_xyxy'], box), str(r['candidate_uid'])) for r in rows]
    eligible = sorted((v for v in eligible if v[0] >= .5), key=lambda v: (-v[0], v[1]))
    return eligible[0][1] if eligible else None


def load_adapter(outer):
    protocol = read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    seq = protocol['sequences']; inner = seq[(seq.index(outer)+1) % len(seq)]
    paths = [ROOT.parent/'InterMOT_N72R20R4_assets/models'/f'adapter6__{outer}__seed{s}.pt'
             for s in (720321, 720322, 720323)]
    records = [read_json(p.with_suffix('.json')) for p in paths]
    adapter = AdapterEnsemble(paths, expected_shas=[r['sha256'] for r in records], forbidden_sequences=[outer, inner])
    expected = sorted(set(seq)-{outer, inner})
    if any(sorted(r['actual_training_sequences']) != expected for r in adapter.manifest):
        raise ValueError('actual strict six-sequence comparator fit differs')
    return adapter


def project(adapter, frames):
    offsets = np.cumsum([0]+[len(rows) for _, rows in frames])
    flat = np.stack([r['feature'] for _, rows in frames for r in rows])
    encoded = [np.empty_like(flat) for _ in adapter.models]
    for start in range(0, len(flat), 1024):
        for i, chunk in enumerate(adapter.encode_candidates(flat[start:start+1024])):
            encoded[i][start:start+len(chunk)] = chunk
    # Stateless projections may be prepared in advance; never pool temporal
    # features. A decision is handed only its exact current-frame slice.
    return [[x[offsets[f]:offsets[f+1]] for x in encoded] for f in range(len(frames))]


def fresh_bank(template):
    return LearnedIdentityMemoryBank.from_updater(template.updater,
        checkpoint_sha256=template.checkpoint_sha256, encoder_sha256=template.encoder_sha256)


def to_contract(decision, rows, event, anchor_sha, case):
    by_uid = {str(r['candidate_uid']): r for r in rows}
    uid = decision['target_uid']; selected = by_uid.get(uid)
    scores = np.asarray(decision['identity_scores'], float)
    top = int(np.argmax(scores)) if len(scores) and case != 'B5_ONLINE_NO_LTM' else None
    score = float(scores[top]) if top is not None else None
    gap = float(scores[top]-max([float(v) for i,v in enumerate(scores) if i != top], default=-1.)) if top is not None else None
    # Identity evidence is not the same as a learned availability posterior.
    probability = float(np.clip((score+1)/2, 0, 1)) if score is not None else float(selected is not None)
    commit = decision['memory']
    return {'frame': int(decision['frame']), 'recording_id': event['recording_id'],
        'selected_candidate_uid': uid, 'rank1_candidate_uid': str(rows[top]['candidate_uid']) if top is not None else None,
        'predicted_box_xyxy': list(selected['box_xyxy']) if selected else None,
        'identity_score': score, 'identity_margin': gap,
        'target_present_probability': probability, 'candidate_available_probability': probability,
        'presence_probability_semantics': 'UNCALIBRATED_IDENTITY_EVIDENCE_NOT_PHYSICAL_PRESENCE',
        'memory_write': bool(commit['accepted']), 'memory_write_candidate_uid': commit.get('candidate_uid') if commit['accepted'] else None,
        'memory_commit': commit, 'anchor_sha256': anchor_sha,
        'runtime_future_gt_used': False, 'runtime_gt_used': False}


def replay(sequences, cases):
    torch.set_num_threads(1)
    manifest = read_json(OUT/'development/RUNTIME_INPUTS.json')
    if sha256(manifest['anchor_path']) != manifest['anchor_sha256']: raise ValueError('anchor seal')
    anchors = np.load(manifest['anchor_path'], mmap_mode='r')
    allowed = set(read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'])
    if not set(sequences) <= allowed or not set(cases) <= set(CASES): raise ValueError('unregistered development cases')
    template = LearnedIdentityMemoryBank.from_checkpoint(ROOT/'outputs/N72R18/checkpoints/identity_memory_gru.pt',
        encoder_sha256=ENCODER_SHA, expected_encoder_sha256=ENCODER_SHA, expected_checkpoint_sha256=GRU_SHA)
    for sequence in sequences:
        started = time.monotonic(); storage(100 << 20)
        source = ROOT.parent/'InterMOT_N72R20R2_assets'
        index = read_json(source/'candidates'/sequence/'index.json')
        for field in ('metadata', 'embeddings'):
            if sha256(index[field]) != index[field+'_sha256']: raise ValueError('candidate tape SHA')
        if index.get('runtime_future_gt_used') or index.get('runtime_gt_read'): raise ValueError('GT-unsafe tape')
        frames = load_candidate_frames(source, sequence)
        if [int(p['frame']) for p,_ in frames] != list(range(len(frames))): raise ValueError('not a complete causal frame axis')
        frozen_path = ROOT/'outputs/N72R20R4R1/authority/frozen_outer'/f'{sequence}.json'
        frozen = read_json(frozen_path)
        adapter = load_adapter(sequence); encoded = project(adapter, frames)
        records = read_json(ROOT/'outputs/N72R20R4R1/memory/fit'/f'{sequence}.json')['models']['NATIVE_RELIABILITY']
        native = ReliabilityEnsemble(records, sequence, 'NATIVE_RELIABILITY')
        for case in cases:
            done = OUT/'baselines/runtime_seals'/case/f'{sequence}.json'
            if done.exists():
                previous = read_json(done)
                if previous['code_sha256'] != {p: sha256(ROOT/p) for p in SEALED_CODE}: raise ValueError('source changed; create a new registered run, never overwrite')
                for r in previous['artifacts']:
                    if sha256(r['path']) != r['sha256']: raise ValueError('resume artifact SHA')
                print(json.dumps({'sequence': sequence, 'case': case, 'status': 'RESUMED_IDENTICAL_SEALED_RUNTIME'}), flush=True)
                continue
            artifacts = []
            for event in [e for e in manifest['inputs'] if e['sequence'] == sequence]:
                if time.monotonic()-started > 1800: raise RuntimeError('30-minute per-sequence replay budget reached; completed sealed cases can resume')
                anchor = np.asarray(anchors[event['anchor_index']], np.float32).copy(); anchor /= np.linalg.norm(anchor)
                anchor_sha = hashlib.sha256(anchor.tobytes()).hexdigest()
                uid = click_uid(frames[event['frame']][1], event['box_xyxy'])
                human_event = {'event_frame': event['frame'], 'human_anchor': anchor,
                    'target_candidate_uid': uid, 'target_box_xyxy': event['box_xyxy'], 'interaction_source': 'SIMULATED_ONE_CLICK_FROM_GT'}
                cfg = AuthorityConfig(**frozen['calibration']); tracker = recognizer = None
                if case == 'B0_RAW_ANCHOR':
                    recognizer = OneClickRecognizer(RuntimeConfig())
                    recognizer.initialize(anchor, event['box_xyxy'], recording_id=sequence, frame=event['frame'])
                elif case != 'B1_R3R2_ADAPTER' and uid is not None:
                    memory = 'P1' if case in ('B2_R4_CAUSAL_P1','B4_P1') else case[-2:] if case in ('B4_P4','B4_P6') else 'P0'
                    cfg = AuthorityConfig(**{**cfg.to_dict(), 'memory': memory if memory in ('P0','P1') else 'P0'})
                    if case in ('B2_R4_CAUSAL_P1', 'B5_ONLINE_NO_LTM'):
                        tracker = CausalIdentityTracker(config=cfg, event=human_event,
                            adapter=None if case == 'B5_ONLINE_NO_LTM' else adapter,
                            bank=None if case == 'B5_ONLINE_NO_LTM' else fresh_bank(template))
                    else:
                        spec = frozen['selected']['NATIVE_IDENTITY' if case == 'B3_R4R1_NATIVE' else memory] if memory in ('P4','P6') or case == 'B3_R4R1_NATIVE' else None
                        policy = InterventionPolicy(**spec['policy']) if spec else InterventionPolicy()
                        if policy.family != 'C0': raise ValueError('baseline replay must not add an association intervention controller')
                        tracker = OpportunityTracker(config=cfg, event=human_event, adapter=adapter, bank=fresh_bank(template),
                            memory_policy=MemoryPolicy(**spec['memory']) if spec else MemoryPolicy(memory),
                            intervention_policy=policy, native_predictor=native if case == 'B3_R4R1_NATIVE' else None, audit_hashes=False)
                output = ASSETS/'baselines/runtime'/case/f"{event['episode_uid']}.jsonl.zst"
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists(): raise FileExistsError('unsealed partial/runtime exists: retain it and choose a new run ID, never overwrite')
                count = 0
                with output.open('xb') as handle:
                    process = subprocess.Popen(['zstd','-q','-c','-T1','-3'], stdin=subprocess.PIPE, stdout=handle)
                    try:
                        for payload, rows in frames:
                            f = int(payload['frame'])
                            actual = tracker.step(rows, f, encoded_candidates=encoded[f]) if tracker is not None else None
                            if f <= event['frame']: continue
                            if recognizer is not None:
                                candidates = [Candidate(str(r['candidate_uid']),r['feature'],tuple(r['box_xyxy']),float(np.clip(r.get('conf',0.),0,1)),str(r['native_tid'])) for r in rows]
                                row = recognizer.step(f, candidates, fps=event['fps'])
                            elif case == 'B1_R3R2_ADAPTER':
                                scores = adapter.scores(anchor, np.stack([r['feature'] for r in rows]) if rows else np.empty((0,512)), encoded[f])
                                top = int(np.argmax(scores)) if len(scores) else None
                                score = float(scores[top]) if top is not None else None
                                gap = float(scores[top]-max([float(s) for i,s in enumerate(scores) if i != top],default=-1.)) if top is not None else None
                                # Historical fit/inner-frozen parameters, not a
                                # newly selected threshold on this outer scene.
                                accepted = top is not None and score >= cfg.write_score and gap >= cfg.write_margin
                                selected = rows[top] if accepted else None
                                probability = float(np.clip((score+1)/2,0,1)) if score is not None else 0.
                                row = {'frame':f,'recording_id':sequence,'selected_candidate_uid':str(selected['candidate_uid']) if selected else None,
                                    'rank1_candidate_uid':str(rows[top]['candidate_uid']) if top is not None else None,
                                    'predicted_box_xyxy':list(selected['box_xyxy']) if selected else None,'identity_score':score,'identity_margin':gap,
                                    'candidate_available_probability':probability,'target_present_probability':probability,
                                    'presence_probability_semantics':'UNCALIBRATED_ADAPTER_SCORE_NOT_PHYSICAL_PRESENCE',
                                    'memory_write':False,'memory_write_candidate_uid':None,'anchor_sha256':anchor_sha,
                                    'runtime_future_gt_used':False,'runtime_gt_used':False}
                            elif actual is not None: row = to_contract(actual, rows, event, anchor_sha, case)
                            else:
                                row = {'frame':f,'recording_id':sequence,'selected_candidate_uid':None,'rank1_candidate_uid':None,
                                    'predicted_box_xyxy':None,'identity_score':None,'identity_margin':None,'target_present_probability':0.,
                                    'candidate_available_probability':0.,'memory_write':False,'memory_write_candidate_uid':None,
                                    'anchor_sha256':anchor_sha,'initialization_failure':'NO_CURRENT_CANDIDATE_OVERLAPS_SINGLE_CLICK_AT_IOU_0_5',
                                    'runtime_future_gt_used':False,'runtime_gt_used':False}
                            process.stdin.write((json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode()); count += 1
                        process.stdin.close()
                        if process.wait() != 0: raise RuntimeError('compression failed')
                    finally:
                        if process.poll() is None: process.terminate(); process.wait()
                artifacts.append({'episode_uid':event['episode_uid'],'path':str(output),'sha256':sha256(output),'frames':count,
                    'initialization_candidate_uid':uid,'initialization_failure':uid is None and case not in ('B0_RAW_ANCHOR','B1_R3R2_ADAPTER')})
                print(json.dumps({'sequence':sequence,'case':case,'episode':event['episode_uid'],'sealed_frames':count}), flush=True)
            write_json(f'baselines/runtime_seals/{case}/{sequence}.json', {'status':'SEALED_GT_FREE_RUNTIME','case':case,'sequence':sequence,
                'artifacts':artifacts,'runtime_truth_opened':False,'future_gt_used':False,'candidate_axis_changed':False,
                'candidate_sha256':{k:index[k+'_sha256'] for k in ('metadata','embeddings')},
                'initialization_manifest_sha256':sha256(OUT/'development/RUNTIME_INPUTS.json'),
                'historical_policy_manifest_sha256':sha256(frozen_path),'actual_adapter_models':adapter.manifest,
                'actual_gru_sha256':GRU_SHA if case not in ('B0_RAW_ANCHOR','B1_R3R2_ADAPTER','B5_ONLINE_NO_LTM') else None,
                'native_models':records if case == 'B3_R4R1_NATIVE' else [],
                'historical_native_training_state':'P1' if case == 'B3_R4R1_NATIVE' else None,
                'native_deployment_state':'P0' if case == 'B3_R4R1_NATIVE' else None,
                'thresholds':'B0 uncalibrated engineering defaults; B1 historical fit/inner-frozen write score/margin; tracker native NONE mechanism unchanged',
                'exposure':'All eight scenes repeatedly exposed historical TRAIN; not virgin final validation',
                'device':'CPU','AMP':False,'new_training':False,'code_sha256':{p:sha256(ROOT/p) for p in SEALED_CODE},
                'sequence_cumulative_seconds':time.monotonic()-started})


def evaluate(sequences, cases):
    # Deliberately imported only in the independent, post-seal command.
    from sam3_intermot.one_click.datasets import dancetrack_annotations, dancetrack_truth
    from sam3_intermot.evaluation.one_click_protocol import evaluate_episode
    inputs = {e['episode_uid']:e for e in read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']}
    labels = {e['episode_uid']:e for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    for sequence in sequences:
        seals = {case:read_json(OUT/'baselines/runtime_seals'/case/f'{sequence}.json') for case in cases}
        for seal in seals.values():
            if seal['code_sha256'] != {p:sha256(ROOT/p) for p in SEALED_CODE}: raise ValueError('runtime source seal')
            for record in seal['artifacts']:
                if sha256(record['path']) != record['sha256']: raise ValueError('sealed runtime SHA')
        # Only now may offline future annotations be loaded.
        gt = dancetrack_annotations(ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence)
        frames = load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets', sequence)
        truths = {e['episode_uid']:[dancetrack_truth(int(p['frame']),rows,gt.get(int(p['frame']),[]),labels[e['episode_uid']]['target_gt_identity'])
                   for p,rows in frames if int(p['frame']) > e['frame']] for e in inputs.values() if e['sequence']==sequence}
        for case, seal in seals.items():
            results = []
            for record in seal['artifacts']:
                runtime = read_zstd_jsonl(Path(record['path'])); event = inputs[record['episode_uid']]
                if not runtime:
                    results.append({'episode_uid':record['episode_uid'],'status':'NO_FUTURE_FRAMES'});continue
                result = evaluate_episode(runtime,truths[record['episode_uid']],fps=event['fps'],recording_id=sequence)
                result.update({'episode_uid':record['episode_uid'],'initialization_failure':record['initialization_failure'],
                    'runtime_sha256':record['sha256'],'development_only':True,'physical_absence_ground_truth_available':False})
                results.append(result)
            write_json(f'baselines/evaluations/{case}/{sequence}.json',{'case':case,'sequence':sequence,'episodes':results,
                'runtime_seal_sha256':sha256(OUT/'baselines/runtime_seals'/case/f'{sequence}.json'),
                'scientific_goal_success':False,'source_semantics':'EXPOSED_TRAIN_WITHIN_RECORDING_VISIBLE_GT_DIAGNOSTIC',
                'future_truth_used_only_after_runtime_seal':True})
            print(json.dumps({'sequence':sequence,'case':case,'posthoc_evaluated_episodes':len(results)}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['replay','evaluate'])
    parser.add_argument('--sequences',nargs='+',default=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'])
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));args=parser.parse_args()
    (replay if args.command=='replay' else evaluate)(args.sequences,args.cases)
