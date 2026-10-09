"""Controlled matched/mismatched/mixed T1 deployment; GT only post-seal."""
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
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry

COMPARISONS={'P0_TO_P0':('P0','P0'),'P1_TO_P1':('P1','P1'),'P1_TO_P0':('P1','P0'),
             'MIXED_TO_P1':('MIXED','P1'),'MIXED_TO_P0':('MIXED','P0')}
CODE=['scripts/n72r21_t1_replay.py','sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_runtime.py']


def replay(sequences,seeds,comparisons):
    torch.set_num_threads(1);storage(100<<20)
    inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
    if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('initial anchor changed')
    anchors=np.load(inputs['anchor_path'],mmap_mode='r');code={p:sha256(ROOT/p) for p in CODE}
    registered=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    if not set(sequences)<=set(registered['sequences']) or not set(seeds)<=set(registered['new_training']['seeds']):raise ValueError('scope')
    for sequence in sequences:
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
        for seed in seeds:
            for comparison in comparisons:
                condition,policy=COMPARISONS[comparison];case=f'T1_{comparison}_K8_SEED{seed}'
                destination=OUT/'experiments/T1/runtime_seals'/case/f'{sequence}.json'
                if destination.exists():
                    previous=read_json(destination)
                    if previous['code_sha256']!=code:raise ValueError('sealed T1 runtime source changed')
                    for artifact in previous['artifacts']:
                        if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('runtime changed')
                    continue
                fitpath=OUT/'training/T1_CAUSAL_V1'/f'{sequence}__seed{seed}__{condition}.json';fitted=read_json(fitpath)
                configuration=fitted['schema']['configuration']
                if not fitted['completed'] or configuration['outer_sequence']!=sequence or sequence in configuration['fit_sequences'] or sequence==configuration['inner_sequence']:raise ValueError('split or fit')
                if sha256(fitted['best_checkpoint_path'])!=fitted['best_checkpoint_sha256']:raise ValueError('checkpoint SHA')
                for p,h in configuration['code_sha256'].items():
                    if sha256(ROOT/p)!=h:raise ValueError('training source changed')
                saved=torch.load(fitted['best_checkpoint_path'],map_location='cpu',weights_only=True)
                if saved['schema']!=fitted['schema']:raise ValueError('checkpoint schema')
                model=ACIBNetwork();model.load_state_dict(saved['model'],strict=True);model.eval();artifacts=[];started=time.monotonic()
                for event in [e for e in inputs['inputs'] if e['sequence']==sequence]:
                    recognizer=ACIBRecognizer(model,anchors[event['anchor_index']],event['episode_uid'],capacity=8,policy=policy)
                    recognizer.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],
                                               initial_frame=event['frame'],initial_box=event['box_xyxy'])
                    path=ASSETS/'experiments/T1/runtime'/case/f"{event['episode_uid']}.jsonl.zst";path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():raise FileExistsError('preserve unsealed partial')
                    count=0
                    with path.open('xb') as stream:
                        proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
                        try:
                            for p,rows in frames:
                                frame=int(p['frame'])
                                if frame<=event['frame']:continue
                                if time.monotonic()-started>1800:raise RuntimeError('30-minute case/scene ceiling')
                                record=recognizer.step(frame,rows);record.update(initialization_label='SIMULATED_ONE_CLICK_FROM_GT',training_stage='T1_FROZEN_BEHAVIOR_CAUSAL_STATES')
                                proc.stdin.write((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                            proc.stdin.close()
                            if proc.wait()!=0:raise RuntimeError('runtime compression')
                        finally:
                            if proc.poll() is None:proc.terminate();proc.wait()
                    artifacts.append({'episode_uid':event['episode_uid'],'path':str(path),'sha256':sha256(path),'frames':count})
                write_json(f'experiments/T1/runtime_seals/{case}/{sequence}.json',{'case':case,'sequence':sequence,'comparison':comparison,'seed':seed,
                    'capacity':8,'train_condition':condition,'deploy_policy':policy,'code_sha256':code,'artifacts':artifacts,
                    'fit_record_sha256':sha256(fitpath),'best_checkpoint_sha256':fitted['best_checkpoint_sha256'],'best_epoch':saved['epoch'],
                    'training_deployment_state_shift':comparison=='P1_TO_P0','state_shift_control_not_chosen_by_outer_performance':True,
                    'GT_read_by_runtime':False,'future_gt_or_extra_click':False,'physical_absence_or_cross_recording_claim':False,
                    'T2_memory_coupled_training_complete':False,'next_stage_authorized':False})
                print(json.dumps({'T1_runtime_sealed':case,'sequence':sequence,'seconds':round(time.monotonic()-started,1)}),flush=True)


def evaluate(sequences,seeds,comparisons):
    from sam3_intermot.one_click.datasets import dancetrack_annotations,dancetrack_truth
    from sam3_intermot.evaluation.one_click_protocol import evaluate_episode
    inputs={e['episode_uid']:e for e in read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']}
    labels={e['episode_uid']:e['target_gt_identity'] for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    for sequence in sequences:
        cases=[f'T1_{comparison}_K8_SEED{seed}' for seed in seeds for comparison in comparisons]
        seals={case:read_json(OUT/'experiments/T1/runtime_seals'/case/f'{sequence}.json') for case in cases}
        for seal in seals.values():
            if seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE}:raise ValueError('runtime source changed')
            for artifact in seal['artifacts']:
                if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('runtime changed before GT opened')
        gtroot=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence;gt=dancetrack_annotations(gtroot)
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
        for case,seal in seals.items():
            metrics=[]
            for artifact in seal['artifacts']:
                event=inputs[artifact['episode_uid']];identity=labels[artifact['episode_uid']]
                truth=[dancetrack_truth(int(p['frame']),rows,gt.get(int(p['frame']),[]),identity) for p,rows in frames if int(p['frame'])>event['frame']]
                result=evaluate_episode(read_zstd_jsonl(Path(artifact['path'])),truth,fps=event['fps'],recording_id=sequence)
                result.update(episode_uid=artifact['episode_uid'],runtime_sha256=artifact['sha256'],development_only=True);metrics.append(result)
            write_json(f'experiments/T1/evaluations/{case}/{sequence}.json',{'case':case,'sequence':sequence,'episodes':metrics,
                'runtime_seal_sha256':sha256(OUT/'experiments/T1/runtime_seals'/case/f'{sequence}.json'),'GT_sha256':sha256(gtroot/'gt/gt.txt'),
                'physical_absence_truth_available':False,'scientific_success':False,'next_stage_authorized':False})
            print(json.dumps({'T1_posthoc_evaluated':case,'sequence':sequence}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['replay','evaluate']);parser.add_argument('--sequences',nargs='+',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103]);parser.add_argument('--comparisons',nargs='+',choices=list(COMPARISONS),default=list(COMPARISONS))
    args=parser.parse_args();(replay if args.action=='replay' else evaluate)(args.sequences,args.seeds,args.comparisons)
