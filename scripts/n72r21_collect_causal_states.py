"""Actual model-produced TRAIN/inner P0/P1 states; no future labels/features.

Every frame is run. Only state-reference snapshots at the frozen fit stride
are cached, avoiding duplicate embedding/video stores. Labels are a later
post-seal operation, never an oracle memory update during this process.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from sam3_intermot.one_click.acib import ACIBNetwork
from sam3_intermot.one_click.acib_runtime import ACIBRecognizer
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from scripts.n72r21_baselines_geometry_repair import valid_geometry

CODE=['scripts/n72r21_collect_causal_states.py','sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_runtime.py']


def run(outer,seeds,scope,policies,capacity):
    torch.set_num_threads(1)
    source_protocol=OUT/'protocol/CAUSAL_STATE_ROLLOUT.json';protocol=read_json(source_protocol)
    if scope not in ('pilot','fit','inner') or not set(policies)<= {'P0','P1'} or capacity not in protocol['capacities']:raise ValueError('frozen state-development scope')
    sequence_protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    if outer not in sequence_protocol['sequences'] or not set(seeds)<=set(sequence_protocol['new_training']['seeds']):raise ValueError('registered outer/seeds')
    inputs=read_json(OUT/'development/RUNTIME_INPUTS.json');anchors=np.load(inputs['anchor_path'],mmap_mode='r')
    if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor seal')
    code={p:sha256(ROOT/p) for p in CODE};storage(200<<20)
    for seed in seeds:
        fitted=OUT/'training/T0_AMP_R1'/f'{outer}__seed{seed}.json';record=read_json(fitted)
        if not record['completed'] or record['schema']['outer_sequence']!=outer:raise ValueError('actual completed matching source fit')
        sequences=record['schema']['fit_sequences'][:1] if scope=='pilot' else record['schema']['fit_sequences'] if scope=='fit' else [record['schema']['inner_sequence']]
        if outer in sequences:raise ValueError('outer cannot enter TRAIN/inner state collector')
        if sha256(record['best_checkpoint_path'])!=record['best_checkpoint_sha256']:raise ValueError('checkpoint SHA')
        saved=torch.load(record['best_checkpoint_path'],map_location='cpu',weights_only=True)
        model=ACIBNetwork();model.load_state_dict(saved['model'],strict=True);model.eval()
        for sequence in sequences:
            frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
            for policy in policies:
                case=f'{outer}__seed{seed}__{policy}__K{capacity}'
                done=OUT/'training/causal_states/seals'/case/f'{sequence}.json'
                if done.exists():
                    previous=read_json(done)
                    if previous['code_sha256']!=code or previous['source_protocol_sha256']!=sha256(source_protocol):raise ValueError('sealed source changed: new state version needed')
                    for r in previous['artifacts']:
                        if sha256(r['path'])!=r['sha256']:raise ValueError('state tape changed')
                    print({'reused_sealed_state_case':case,'sequence':sequence},flush=True);continue
                started=time.monotonic();artifacts=[]
                for event in [e for e in inputs['inputs'] if e['sequence']==sequence]:
                    recognizer=ACIBRecognizer(model,np.asarray(anchors[event['anchor_index']]),event['episode_uid'],capacity=capacity,policy=policy)
                    recognizer.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],
                                               initial_frame=event['frame'],initial_box=event['box_xyxy'])
                    destination=ASSETS/'training/causal_states'/case/f"{event['episode_uid']}.jsonl.zst";destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists():raise FileExistsError('preserve unsealed partial, no overwrite')
                    count=snapshots=writes=0
                    with destination.open('xb') as stream:
                        proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
                        try:
                            for p,rows in frames:
                                frame=int(p['frame'])
                                if frame<=event['frame']:continue
                                if time.monotonic()-started>1800:raise RuntimeError('30-minute per-sequence state-job ceiling')
                                before=recognizer.snapshot() if (frame-event['frame']-1)%5==0 else None
                                result=recognizer.step(frame,rows)
                                result['TRAIN_state_before']=before;result['TRAIN_state_snapshot_only_reference_not_GT']=before is not None
                                count+=1;snapshots+=before is not None;writes+=result['memory_write']
                                proc.stdin.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode())
                            proc.stdin.close()
                            if proc.wait()!=0:raise RuntimeError('state compression')
                        finally:
                            if proc.poll() is None:proc.terminate();proc.wait()
                    artifacts.append({'episode_uid':event['episode_uid'],'path':str(destination),'sha256':sha256(destination),
                                      'bytes':destination.stat().st_size,'frames':count,'fit_stride_snapshots':snapshots,'own_machine_writes':writes})
                    print(json.dumps({'causal_state':case,'sequence':sequence,'episode':event['episode_uid'],'frames':count,'own_writes':writes,'seconds':round(time.monotonic()-started,1)}),flush=True)
                write_json(f'training/causal_states/seals/{case}/{sequence}.json',{
                    'case':case,'outer':outer,'sequence':sequence,'scope_at_first_collection':scope,'seed':seed,'policy':policy,'capacity':capacity,
                    'code_sha256':code,'source_protocol_sha256':sha256(source_protocol),'source_fit_record_sha256':sha256(fitted),
                    'source_checkpoint_SHA256':record['best_checkpoint_sha256'],'artifacts':artifacts,
                    'all_actual_future_frames_run':True,'raw_identity_embeddings_not_duplicated_into_state_snapshots':True,
                    'GT_read_by_state_collection':False,'P1_is_own_prediction_not_oracle_teacher_forcing':True,
                    'physical_absence_or_cross_recording_science_claim':False,'T1_fit_yet_completed':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103])
    parser.add_argument('--scope',choices=['pilot','fit','inner'],default='pilot');parser.add_argument('--policies',nargs='+',default=['P0','P1']);parser.add_argument('--capacity',type=int,default=8)
    args=parser.parse_args();run(args.outer,args.seeds,args.scope,args.policies,args.capacity)
