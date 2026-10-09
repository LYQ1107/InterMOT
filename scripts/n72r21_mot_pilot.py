"""TRAIN-only full-MOT frozen-weight transfer, runtime then separate truth."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21_validation import cases,case_id
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.runtime import OneClickRecognizer,RuntimeConfig
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.evaluation.one_click_protocol import iou

PROTOCOL=OUT/'protocol/MOT_TRAIN_PILOT.json'
CODE=['scripts/n72r21_mot_pilot.py','sam3_intermot/one_click/mot_bridge.py',
    'sam3_intermot/one_click/runtime.py','sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_memory.py',
    'sam3_intermot/one_click/acib_runtime.py','sam3_intermot/one_click/acib_trusted_runtime.py','scripts/n72r21_t2_replay.py',
    'sam3_intermot/association/opportunity_tracker.py','sam3_intermot/association/opportunity_solver.py',
    'sam3_intermot/association/causal_identity_tracker.py','sam3_intermot/association/opportunity_memory.py',
    'sam3_intermot/association/causal_state_commit.py','sam3_intermot/association/online_associator.py',
    'sam3_intermot/association/identity_state.py','sam3_intermot/association/state_manager.py',
    'sam3_intermot/association/global_assignment_adapter.py','sam3_intermot/association/public_assignment.py',
    'scripts/n72r21_baselines_geometry_repair.py','scripts/n72r20r4_run_causal_tracker.py']
CANDIDATES=ROOT.parent/'InterMOT_N72R20R2_assets'
DATASET=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'


def checked_frames(sequence):
    indexpath=CANDIDATES/'candidates'/sequence/'index.json';index=read_json(indexpath)
    if index.get('runtime_gt_read') or index.get('runtime_future_gt_used'):raise ValueError('candidate runtime truth boundary')
    for kind,file in [('metadata','metadata.jsonl.zst'),('embeddings','embeddings.f16')]:
        if sha256(indexpath.parent/file)!=index[kind+'_sha256']:raise ValueError('candidate source changed')
    frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(CANDIDATES,sequence)]
    if [int(p['frame']) for p,_ in frames]!=list(range(index['frame_count'])):raise ValueError('complete original frame axis')
    return frames,indexpath


def prepare():
    protocol=read_json(PROTOCOL);sourcepath=OUT/'development/RUNTIME_INPUTS.json';source=read_json(sourcepath)
    if sha256(source['anchor_path'])!=source['anchor_sha256']:raise ValueError('existing sole anchors changed')
    code={p:sha256(ROOT/p) for p in CODE};anchors=np.load(source['anchor_path'],mmap_mode='r')
    for sequence in protocol['sequences']:
        done=OUT/'mot_pilot/initialization'/f'{sequence}.json'
        if done.exists():
            old=read_json(done)
            if old['protocol_sha256']!=sha256(PROTOCOL) or old['preparation_code_sha256']!=code:raise ValueError('prepared protocol/source changed')
            continue
        event=sorted([e for e in source['inputs'] if e['sequence']==sequence],key=lambda e:e['episode_uid'])[0]
        frames,indexpath=checked_frames(sequence);rows=frames[event['frame']][1]
        chosen=sorted(rows,key=lambda r:(-iou(event['box_xyxy'],r['box_xyxy']),str(r['candidate_uid'])))
        chosen=chosen[0] if chosen and iou(event['box_xyxy'],chosen[0]['box_xyxy'])>=.5 else None
        models=[]
        for seed in protocol['seeds']:
            path=OUT/'training/T2_COUPLED_V1'/f'{sequence}__seed{seed}.json';fit=read_json(path);cfg=fit['schema']['configuration']
            if not fit['completed'] or cfg['outer_sequence']!=sequence or sequence in cfg['fit_sequences'] or sequence==cfg['inner_sequence']:raise ValueError('outer exclusion')
            if sha256(fit['best_checkpoint_path'])!=fit['best_checkpoint_sha256']:raise ValueError('existing fitted model changed')
            for p,h in cfg['code_sha256'].items():
                if sha256(ROOT/p)!=h:raise ValueError('fitted source changed')
            models.append({'seed':seed,'path':fit['best_checkpoint_path'],'sha256':fit['best_checkpoint_sha256'],
                'fit_record_path':str(path),'fit_record_sha256':sha256(path),'fit_sequences':cfg['fit_sequences'],'inner_sequence':cfg['inner_sequence']})
        write_json(f'mot_pilot/initialization/{sequence}.json',{'sequence':sequence,'event':event,
            'initialization_failure':chosen is None,'clicked_candidate_uid':None if chosen is None else str(chosen['candidate_uid']),
            'clicked_candidate_box_IoU':None if chosen is None else iou(event['box_xyxy'],chosen['box_xyxy']),
            'anchor_path':source['anchor_path'],'anchor_sha256':source['anchor_sha256'],'models':models,
            'source_click_manifest_sha256':sha256(sourcepath),'candidate_index_sha256':sha256(indexpath),
            'initialization_truth_file_for_separate_evaluation_only':'outputs/N72R21/development/INITIALIZATION_TRUTH.json',
            'protocol_sha256':sha256(PROTOCOL),'preparation_code_sha256':code,
            'future_GT_parsed_by_this_preparation':False,'new_pixels_weights_or_datasets_copied':False,'next_stage_authorized':False})
        print(json.dumps({'MOT_pilot_initialization_prepared':sequence,'failed':chosen is None,'registered_cases':16}),flush=True)


def actor_for(condition,seed,init,anchor):
    event=init['event']
    if condition['family']=='joint_baseline':return None,None
    if condition['family']=='rule':
        protocol=read_json(PROTOCOL);actor=OneClickRecognizer(RuntimeConfig(scoring=condition['scoring'],memory_policy=condition['memory_policy'],bank_capacity=8,**protocol['rule_operating_point']))
        actor.initialize(anchor,event['box_xyxy'],recording_id=event['sequence'],frame=event['frame']);return actor,None
    source=next(s for s in init['models'] if s['seed']==seed)
    if sha256(source['path'])!=source['sha256'] or sha256(source['fit_record_path'])!=source['fit_record_sha256']:raise ValueError('frozen outer-fold weights changed')
    model=ACIBMemoryNetwork();checkpoint=torch.load(source['path'],map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model'],strict=True)
    model=DecisionCapture(model).eval();actor=TrustedACIBRecognizer(model,anchor,event['episode_uid'],capacity=8,policy=condition['policy'])
    actor.start_recording(event['sequence'],fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
    return actor,source


def replay(sequence):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL)
    if sequence not in protocol['sequences']:raise ValueError('only two frozen TRAIN pilot sequences')
    initpath=OUT/'mot_pilot/initialization'/f'{sequence}.json';init=read_json(initpath)
    if init['protocol_sha256']!=sha256(PROTOCOL) or sha256(init['anchor_path'])!=init['anchor_sha256']:raise ValueError('sealed inputs changed')
    if init['initialization_failure']:raise RuntimeError('initial candidate missing; preserve explicit failure, do not replace target')
    frames,indexpath=checked_frames(sequence)
    if sha256(indexpath)!=init['candidate_index_sha256']:raise ValueError('prepared candidate axis changed')
    event=init['event'];anchor=np.array(np.load(init['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
    click={'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':init['clicked_candidate_uid'],
        'target_box_xyxy':event['box_xyxy'],'interaction_source':'SIMULATED_ONE_CLICK_FROM_GT'}
    code={p:sha256(ROOT/p) for p in CODE}
    if code!=init['preparation_code_sha256']:raise ValueError('source changed after pilot input freeze')
    for name,seed,condition in cases(protocol):
        case=case_id(name,seed);done=OUT/'mot_pilot/runtime_seals'/case/f'{sequence}.json'
        if done.exists():
            seal=read_json(done)
            if seal['code_sha256']!=code or seal['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('sealed MOT source changed')
            for a in seal['artifacts']:
                if sha256(a['path'])!=a['sha256']:raise ValueError('sealed MOT artifact changed')
            continue
        storage(protocol['max_new_artifact_MiB']<<20);identity,model=actor_for(condition,seed,init,anchor)
        bridge=MOTIdentityBridge(click,identity,no_human=condition.get('click') is False,frames=len(frames));bridge.configure_fps(event['fps'])
        tracepath=ASSETS/'mot_pilot/traces'/case/f'{sequence}.jsonl.zst';motpath=ASSETS/'mot_pilot/trackers'/case/'data'/f'{sequence}.txt'
        for p in (tracepath,motpath):
            p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():raise FileExistsError('retain partial stage artifacts, never overwrite')
        started=time.monotonic();count=motrows=0;writes=blocked=0
        with tracepath.open('xb') as handle,motpath.open('x') as text:
            proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=handle)
            try:
                for p,rows in frames:
                    if time.monotonic()-started>protocol['max_runtime_seconds_per_scene_case']:raise RuntimeError('bounded pilot scene-case wallclock')
                    frame=int(p['frame']);result=bridge.step(frame,rows);decision=result['identity_decision']
                    if model is not None and decision is not None:
                        decision['rank1_identity_joint_probability']=float(identity.model.last['joint_probabilities'][0,:len(rows)].max()) if rows else 0.
                    elif decision is not None:decision['rank1_identity_joint_probability']=decision['candidate_available_probability']
                    result.update(case=case,initialization_label='NO_CLICK_CONTROL' if condition.get('click') is False else 'SIMULATED_ONE_CLICK_FROM_GT')
                    payload=trajectory_text([result]);text.write(payload);motrows+=len(result['outputs'])
                    proc.stdin.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                    writes+=result['joint_identity_memory_write'];blocked+=result['proposal_feasible'] is False
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError('MOT trace compression failed')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
        artifacts=[{'kind':kind,'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size} for kind,p in [('trace',tracepath),('trajectory',motpath)]]
        write_json(f'mot_pilot/runtime_seals/{case}/{sequence}.json',{'case':case,'seed':seed,'condition':condition,'sequence':sequence,
            'artifacts':artifacts,'frames':count,'trajectory_rows':motrows,'actual_joint_writes':writes,'hard_infeasible_proposals':blocked,
            'code_sha256':code,'protocol_sha256':sha256(PROTOCOL),'initialization_manifest_sha256':sha256(initpath),
            'source_model':model,'seconds':time.monotonic()-started,'GT_parsed_by_runtime':False,
            'one_to_one_complete_candidate_ownership_checked_every_frame':True,'new_training':False,
            'frozen_weight_MOT_state_shift_not_retrained_architecture':True,'no_SOT_work':True,'next_stage_authorized':False})
        print(json.dumps({'full_MOT_pilot_runtime_sealed':case,'sequence':sequence,'frames':count,'writes':writes,'blocked':blocked,'seconds':round(time.monotonic()-started,1)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','replay']);parser.add_argument('--sequence');args=parser.parse_args()
    (prepare() if args.action=='prepare' else replay(args.sequence))
