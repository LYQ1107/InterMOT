"""Frozen, GT-free independent-sequence replay and separate post-seal truth."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from sam3_intermot.one_click import Candidate,OneClickRecognizer,RuntimeConfig
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry

PROTOCOL=OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json'
CODE=['scripts/n72r21_validation.py','scripts/n72r21_t2_replay.py','sam3_intermot/one_click/runtime.py',
      'sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_memory.py',
      'sam3_intermot/one_click/acib_runtime.py','sam3_intermot/one_click/acib_trusted_runtime.py']


def cases(protocol):
    return [(name,seed,condition) for name,condition in protocol['cases'].items()
            for seed in (protocol['seeds'] if condition['family']=='T2' else [None])]


def case_id(name,seed):return f'{name}_SEED{seed}' if seed is not None else name


def replay(sequences):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL)
    if not set(sequences)<=set(protocol['sequences']):raise ValueError('frozen VAL scope')
    models_path=OUT/'validation/FROZEN_MODEL_MANIFEST.json';models=read_json(models_path)
    if models['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('model freeze protocol changed')
    saved={}
    for model in models['models']:
        if sha256(model['path'])!=model['sha256'] or sha256(model['actual_fit_record_path'])!=model['actual_fit_record_sha256']:raise ValueError('frozen models changed')
        fitted=read_json(model['actual_fit_record_path'])
        for p,h in fitted['schema']['configuration']['code_sha256'].items():
            if sha256(ROOT/p)!=h:raise ValueError('fitting source changed')
        saved[model['seed']]=torch.load(model['path'],map_location='cpu',weights_only=True)
    code={p:sha256(ROOT/p) for p in CODE};root=Path(protocol['candidate_root'])
    for sequence in sequences:
        initpath=OUT/'validation/initialization'/f'{sequence}.json';inputs=read_json(initpath)
        if inputs['protocol_sha256']!=sha256(PROTOCOL) or inputs['frozen_model_manifest_sha256']!=sha256(models_path) or sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('sealed click/model inputs changed')
        indexpath=root/'candidates'/sequence/'index.json';index=read_json(indexpath)
        if index.get('runtime_gt_read') or index.get('runtime_future_gt_used'):raise ValueError('unsafe candidate lineage')
        for kind,file in [('metadata','metadata.jsonl.zst'),('embeddings','embeddings.f16')]:
            if sha256(indexpath.parent/file)!=index[kind+'_sha256']:raise ValueError('candidate source changed')
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(root,sequence)]
        if len(frames)!=index['frame_count']:raise ValueError('incomplete frame axis')
        anchors=np.load(inputs['anchor_path'],mmap_mode='r')
        for name,seed,condition in cases(protocol):
            case=case_id(name,seed);done=OUT/'validation/runtime_seals'/case/f'{sequence}.json'
            if done.exists():
                seal=read_json(done)
                if seal['code_sha256']!=code or seal['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('sealed VAL runtime source changed')
                for artifact in seal['artifacts']:
                    if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('sealed VAL artifact changed')
                continue
            storage(100<<20);artifacts=[];started=time.monotonic();model=None
            if seed is not None:
                model=ACIBMemoryNetwork();model.load_state_dict(saved[seed]['model'],strict=True);model=DecisionCapture(model).eval()
            for event in inputs['inputs']:
                anchor=anchors[event['anchor_index']]
                if model is not None:
                    recognizer=TrustedACIBRecognizer(model,anchor,event['episode_uid'],capacity=condition['capacity'],policy=condition['policy'])
                    recognizer.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
                else:
                    config=RuntimeConfig(scoring=condition['scoring'],memory_policy=condition['memory_policy'],bank_capacity=condition['bank_capacity'],**protocol['rule_operating_point'])
                    recognizer=OneClickRecognizer(config);recognizer.initialize(anchor,event['box_xyxy'],recording_id=sequence,frame=event['frame'])
                path=ASSETS/'validation/runtime'/case/f"{event['episode_uid']}.jsonl.zst";path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise FileExistsError('retain unsealed partial VAL output')
                count=0
                with path.open('xb') as stream:
                    proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
                    try:
                        for p,rows in frames:
                            frame=int(p['frame'])
                            if frame<=event['frame']:continue
                            if time.monotonic()-started>1800:raise RuntimeError('30-minute per-case/sequence ceiling')
                            if model is not None:
                                result=recognizer.step(frame,rows)
                                result['rank1_identity_joint_probability']=float(model.last['joint_probabilities'][0,:len(rows)].max()) if rows else 0.
                                result['identity_claim_score_semantics']='LEARNED_RANK1_JOINT_CANDIDATE_PROBABILITY'
                            else:
                                candidates=[Candidate(str(r['candidate_uid']),r['feature'],tuple(r['box_xyxy']),float(np.clip(r.get('conf',0.),0,1)),str(r['native_tid'])) for r in rows]
                                result=recognizer.step(frame,candidates,fps=event['fps'])
                                result['rank1_identity_joint_probability']=result['candidate_available_probability']
                                result['identity_claim_score_semantics']='UNCALIBRATED_COSINE_MAPPED_TO_UNIT_INTERVAL_NOT_LEARNED_POSTERIOR'
                            result.update(initialization_label='SIMULATED_ONE_CLICK_FROM_GT',extra_clicks=0,frozen_validation=True)
                            proc.stdin.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                        proc.stdin.close()
                        if proc.wait()!=0:raise RuntimeError('VAL compression')
                    finally:
                        if proc.poll() is None:proc.terminate();proc.wait()
                artifacts.append({'episode_uid':event['episode_uid'],'path':str(path),'sha256':sha256(path),'frames':count})
            write_json(f'validation/runtime_seals/{case}/{sequence}.json',{'case':case,'sequence':sequence,'condition':condition,'seed':seed,
                'artifacts':artifacts,'code_sha256':code,'protocol_sha256':sha256(PROTOCOL),
                'initialization_manifest_sha256':sha256(initpath),'frozen_models_manifest_sha256':sha256(models_path),
                'candidate_index_sha256':sha256(indexpath),'GT_opened_by_runtime':False,'no_VAL_fitting_or_selection':True,'next_stage_authorized':False})
            print(json.dumps({'frozen_VAL_runtime_sealed':case,'sequence':sequence,'targets':len(artifacts),'seconds':round(time.monotonic()-started,1)}),flush=True)


def evaluate(sequences):
    from sam3_intermot.one_click.datasets import dancetrack_annotations,dancetrack_truth,strict_candidate_matching
    from sam3_intermot.evaluation.one_click_protocol import evaluate_episode
    from sam3_intermot.evaluation.one_click_curves import open_set_metrics_fast as open_set_metrics
    from sam3_intermot.evaluation.one_click_identity import strict_identity_claim_metrics
    protocol=read_json(PROTOCOL);root=Path(protocol['candidate_root'])
    for sequence in sequences:
        inputs={e['episode_uid']:e for e in read_json(OUT/'validation/initialization'/f'{sequence}.json')['inputs']}
        seals={case_id(name,seed):read_json(OUT/'validation/runtime_seals'/case_id(name,seed)/f'{sequence}.json') for name,seed,_ in cases(protocol)}
        for seal in seals.values():
            if seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE} or seal['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('runtime changed before GT')
            for artifact in seal['artifacts']:
                if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('artifact changed before GT')
        labels=read_json(OUT/'validation/initialization_truth'/f'{sequence}.json')
        gtroot=Path(protocol['dataset_root'])/'val'/sequence
        if sha256(gtroot/'gt/gt.txt')!=labels['GT_sha256']:raise ValueError('GT changed')
        identities={e['episode_uid']:e['target_gt_identity'] for e in labels['labels']};gt=dancetrack_annotations(gtroot)
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(root,sequence)]
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        for case,seal in seals.items():
            results=[];scores=[];available=[];rank_correct=[];claim_scores=[];claim_labels=[]
            for artifact in seal['artifacts']:
                event=inputs[artifact['episode_uid']];identity=identities[artifact['episode_uid']]
                truth=[dancetrack_truth(int(p['frame']),rows,gt.get(int(p['frame']),[]),identity) for p,rows in frames if int(p['frame'])>event['frame']]
                trace=read_zstd_jsonl(Path(artifact['path']));verified=[]
                for r,t in zip(trace,truth,strict=True):
                    uid=r.get('rank1_candidate_uid');axis=matched[int(r['frame'])]
                    if uid is not None and uid not in axis:raise ValueError('rank UID outside candidate axis')
                    label=axis.get(uid);known=False if uid is None else None if label is None else bool(label==identity)
                    verified.append(known);scores.append(r['candidate_available_probability']);available.append(bool(t.valid_target_candidate_uids));rank_correct.append(uid in t.valid_target_candidate_uids)
                    if known is not None:claim_scores.append(r['rank1_identity_joint_probability']);claim_labels.append(known)
                result=evaluate_episode(trace,truth,fps=event['fps'],recording_id=sequence)
                result['secondary_strict_identity_claim']=strict_identity_claim_metrics(trace,truth,verified_rank1_labels=verified)
                if seal['condition']['family']=='rule':result['secondary_strict_identity_claim']['score']='UNCALIBRATED_COSINE_MAPPED_TO_UNIT_INTERVAL_NOT_LEARNED_POSTERIOR'
                result.update(episode_uid=artifact['episode_uid'],runtime_sha256=artifact['sha256'],initialization_failure=False);results.append(result)
            write_json(f'validation/evaluations/{case}/{sequence}.json',{'case':case,'sequence':sequence,'episodes':results,
                'pooled_availability_open_set':open_set_metrics(scores,available,rank_correct),
                'pooled_verified_secondary_identity_claim':open_set_metrics(claim_scores,claim_labels,claim_labels),
                'runtime_seal_sha256':sha256(OUT/'validation/runtime_seals'/case/f'{sequence}.json'),
                'GT_sha256':sha256(gtroot/'gt/gt.txt'),'protocol_sha256':sha256(PROTOCOL),
                'secondary_evaluator_sha256':sha256(ROOT/'sam3_intermot/evaluation/one_click_identity.py'),
                'no_threshold_or_model_selection_from_VAL':True,'historical_benchmark_exposure_disclosed':True,
                'physical_absence_or_cross_recording_truth':False,'scientific_success':False,'next_stage_authorized':False})
            print(json.dumps({'frozen_VAL_posthoc_evaluated':case,'sequence':sequence}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['replay','evaluate']);parser.add_argument('--sequences',nargs='+',required=True)
    args=parser.parse_args();(replay if args.action=='replay' else evaluate)(args.sequences)
