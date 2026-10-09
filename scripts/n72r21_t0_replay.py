"""Frozen T0 P0 replay. Future truth is opened only by separate evaluate CLI.

This empty-machine-bank reference is not T1/T2/full ACIB or physical presence.
Each current frame is scored once from its real candidates and sole anchor.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch
from sam3_intermot.one_click.acib import ACIBNetwork
from scripts.n72r21_common import ROOT,ASSETS,OUT,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry

TRAIN_RUN='T0_AMP_R1'
SEALED=['scripts/n72r21_t0_replay.py','sam3_intermot/one_click/acib.py']


def current_input(anchor,rows,event,frame):
    count=max(1,len(rows));candidates=torch.zeros(1,count,512);quality=torch.zeros(1,count,5);valid=torch.zeros(1,count,dtype=torch.bool)
    for i,r in enumerate(rows):
        feature=np.asarray(r['feature'],np.float32);feature=feature/max(np.linalg.norm(feature),1e-8)
        candidates[0,i]=torch.from_numpy(feature);valid[0,i]=True
        x,y,right,bottom=r['box_xyxy'];w,h=right-x,bottom-y
        quality[0,i]=torch.tensor([np.clip(r.get('conf',0.),0,1),np.clip(w*h/(event['width']*event['height']),0,1),np.clip(w/h,0,4)/4,0.,0.])
    context=torch.tensor([[math.log1p((frame-event['frame'])/event['fps'])/5,0.,0.]])
    return anchor[None],candidates,valid,quality,context


def replay(sequences,seeds):
    torch.set_num_threads(1);storage(50<<20)
    protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    if not set(sequences)<=set(protocol['sequences']) or not set(seeds)<=set(protocol['new_training']['seeds']):raise ValueError('frozen scope')
    inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
    if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor seal')
    anchors=np.load(inputs['anchor_path'],mmap_mode='r');code={p:sha256(ROOT/p) for p in SEALED}
    for sequence in sequences:
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
        for seed in seeds:
            case=f'{TRAIN_RUN}_P0_SEED{seed}';done=OUT/'experiments/T0/runtime_seals'/case/f'{sequence}.json'
            if done.exists():
                old=read_json(done)
                if old['code_sha256']!=code:raise ValueError('sealed runtime code changed; new namespace required')
                for r in old['artifacts']:
                    if sha256(r['path'])!=r['sha256']:raise ValueError('runtime seal changed')
                continue
            fitted=OUT/'training'/TRAIN_RUN/f'{sequence}__seed{seed}.json';record=read_json(fitted)
            if not record['completed'] or record['schema']['outer_sequence']!=sequence:raise ValueError('fit incomplete or outer mismatched')
            if sequence in record['schema']['fit_sequences'] or sequence==record['schema']['inner_sequence']:raise ValueError('outer leakage')
            if sha256(record['best_checkpoint_path'])!=record['best_checkpoint_sha256']:raise ValueError('best checkpoint SHA')
            saved=torch.load(record['best_checkpoint_path'],map_location='cpu',weights_only=True)
            if saved['schema']!=record['schema']:raise ValueError('checkpoint schema seal')
            for p,h in saved['schema']['code_sha256'].items():
                if sha256(ROOT/p)!=h:raise ValueError('training source changed; verify source revision explicitly')
            model=ACIBNetwork();model.load_state_dict(saved['model'],strict=True);model.eval();artifacts=[]
            for event in [e for e in inputs['inputs'] if e['sequence']==sequence]:
                anchor=np.array(anchors[event['anchor_index']],copy=True);anchor/=np.linalg.norm(anchor)
                anchor_sha=hashlib.sha256(anchor.tobytes()).hexdigest();query=torch.from_numpy(anchor)
                path=ASSETS/'experiments/T0/runtime'/case/f"{event['episode_uid']}.jsonl.zst";path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise FileExistsError('preserve unsealed partial; no overwrite')
                count=0
                with path.open('xb') as stream:
                    proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
                    try:
                        for p,rows in frames:
                            frame=int(p['frame'])
                            if frame<=event['frame']:continue
                            a,c,v,q,context=current_input(query,rows,event,frame)
                            with torch.inference_mode():result=model(a,c,v,q,context=context)
                            joint=result['joint_probabilities'][0].numpy();n=c.shape[1];selection=int(joint.argmax())
                            ranking=int(result['candidate_logits'][0].argmax()) if rows else None
                            selected=rows[selection] if selection<n and selection<len(rows) else None
                            probability=float(result['candidate_valid_probability'][0]);rankuid=str(rows[ranking]['candidate_uid']) if ranking is not None else None
                            row={'frame':frame,'recording_id':sequence,'selected_candidate_uid':str(selected['candidate_uid']) if selected else None,
                                 'rank1_candidate_uid':rankuid,'predicted_box_xyxy':list(selected['box_xyxy']) if selected else None,
                                 'candidate_available_probability':probability,'target_present_probability':probability,
                                 'physical_presence_probability':None,'presence_probability_semantics':result['probability_semantics'],
                                 'availability_probabilities':result['availability_probabilities'][0].tolist(),
                                 'identity_score':float(joint[ranking]) if ranking is not None else None,
                                 'memory_write':False,'memory_write_candidate_uid':None,'machine_bank_size':0,
                                 'anchor_sha256':anchor_sha,'runtime_gt_used':False,'runtime_future_gt_used':False,
                                 'initialization_label':'SIMULATED_ONE_CLICK_FROM_GT','extra_clicks':0,'training_stage':'T0_ONLY_EMPTY_MACHINE_BANK'}
                            proc.stdin.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                        proc.stdin.close()
                        if proc.wait()!=0:raise RuntimeError('trace compression')
                    finally:
                        if proc.poll() is None:proc.terminate();proc.wait()
                if hashlib.sha256(query.numpy().tobytes()).hexdigest()!=anchor_sha:raise ValueError('anchor mutated')
                artifacts.append({'episode_uid':event['episode_uid'],'path':str(path),'sha256':sha256(path),'frames':count})
            write_json(f'experiments/T0/runtime_seals/{case}/{sequence}.json',{'case':case,'sequence':sequence,'artifacts':artifacts,
                'code_sha256':code,'actual_fit_record_sha256':sha256(fitted),'actual_best_checkpoint_sha256':record['best_checkpoint_sha256'],
                'selected_best_epoch':saved['epoch'],'checkpoint_selection':'Inner loss only, no outer selection',
                'no_runtime_GT_or_future_frames':True,'actual_trained_T0':True,'T1_T2_T3_or_full_memory_claim':False,
                'physical_absence_head':False,'zero_write_memory_PASS':False})
            print(json.dumps({'T0_real_replay_sealed':case,'sequence':sequence,'episodes':len(artifacts)}),flush=True)


def evaluate(sequences,seeds):
    from sam3_intermot.one_click.datasets import dancetrack_annotations,dancetrack_truth
    from sam3_intermot.evaluation.one_click_protocol import evaluate_episode
    inputs={e['episode_uid']:e for e in read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']}
    labels={e['episode_uid']:e['target_gt_identity'] for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    for sequence in sequences:
        cases=[f'{TRAIN_RUN}_P0_SEED{seed}' for seed in seeds]
        seals={case:read_json(OUT/'experiments/T0/runtime_seals'/case/f'{sequence}.json') for case in cases}
        for seal in seals.values():
            if seal['code_sha256']!={p:sha256(ROOT/p) for p in SEALED}:raise ValueError('runtime code seal')
            for r in seal['artifacts']:
                if sha256(r['path'])!=r['sha256']:raise ValueError('runtime SHA before GT read')
        gt=dancetrack_annotations(ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence)
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
        for case,seal in seals.items():
            result=[]
            for r in seal['artifacts']:
                event=inputs[r['episode_uid']];identity=labels[r['episode_uid']]
                truth=[dancetrack_truth(int(p['frame']),rows,gt.get(int(p['frame']),[]),identity) for p,rows in frames if int(p['frame'])>event['frame']]
                metrics=evaluate_episode(read_zstd_jsonl(Path(r['path'])),truth,fps=event['fps'],recording_id=sequence)
                metrics.update(episode_uid=r['episode_uid'],development_only=True,runtime_sha256=r['sha256'],initialization_failure=False)
                result.append(metrics)
            write_json(f'experiments/T0/evaluations/{case}/{sequence}.json',{'case':case,'sequence':sequence,'episodes':result,
                'runtime_seal_sha256':sha256(OUT/'experiments/T0/runtime_seals'/case/f'{sequence}.json'),
                'physical_absence_truth_or_head':False,'F1_visible_GT_gap_availability_diagnostic':True,
                'T0_is_not_T1_T2_T3':True,'scientific_success':False,'next_stage_authorized':False})
            print(json.dumps({'T0_posthoc_evaluated':case,'sequence':sequence,'episodes':len(result)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['replay','evaluate']);parser.add_argument('--sequences',nargs='+',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103]);args=parser.parse_args()
    (replay if args.action=='replay' else evaluate)(args.sequences,args.seeds)
