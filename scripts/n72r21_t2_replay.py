"""Frozen T2 learned-write and explicitly labelled inference-state controls."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_ablation import ACIBVariantNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry

CODE=['scripts/n72r21_t2_replay.py','sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_memory.py',
      'sam3_intermot/one_click/acib_ablation.py','sam3_intermot/one_click/acib_runtime.py','sam3_intermot/one_click/acib_trusted_runtime.py']


class DecisionCapture(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.last=None
    @property
    def last_future_safe_probability(self):return self.model.last_future_safe_probability
    def forward(self,*args,**kwargs):
        self.last=self.model(*args,**kwargs);return self.last


def replay(sequences,seeds,names):
    torch.set_num_threads(1);storage(100<<20);protocol_path=OUT/'protocol/T2_DEPLOYMENT_ABLATIONS.json';protocol=read_json(protocol_path)
    registered=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    if not set(sequences)<=set(registered['sequences']) or not set(seeds)<=set(protocol['seeds']) or not set(names)<=set(protocol['cases']):raise ValueError('frozen T2 replay scope')
    inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
    if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor SHA')
    anchors=np.load(inputs['anchor_path'],mmap_mode='r');code={p:sha256(ROOT/p) for p in CODE}
    for sequence in sequences:
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
        for seed in seeds:
            fitpath=OUT/'training/T2_COUPLED_V1'/f'{sequence}__seed{seed}.json';fitted=read_json(fitpath);configuration=fitted['schema']['configuration']
            if not fitted['completed'] or configuration['outer_sequence']!=sequence or sequence in configuration['fit_sequences'] or sequence==configuration['inner_sequence']:raise ValueError('T2 fit/split')
            if not fitted['future_safe_head_trained_on_verified_paired_supervision'] or fitted['schema']['fit_data']['known_risk']<1:raise ValueError('actual future-risk supervision required')
            if sha256(fitted['best_checkpoint_path'])!=fitted['best_checkpoint_sha256']:raise ValueError('T2 checkpoint SHA')
            for p,h in configuration['code_sha256'].items():
                if sha256(ROOT/p)!=h:raise ValueError('T2 fitting source changed; version explicitly')
            saved=torch.load(fitted['best_checkpoint_path'],map_location='cpu',weights_only=True)
            if saved['schema']!=fitted['schema']:raise ValueError('T2 checkpoint schema')
            for name in names:
                condition=protocol['cases'][name];case=f'T2_{name}_SEED{seed}';done=OUT/'experiments/T2/runtime_seals'/case/f'{sequence}.json'
                if done.exists():
                    old=read_json(done)
                    if old['code_sha256']!=code or old['deployment_protocol_sha256']!=sha256(protocol_path):raise ValueError('sealed T2 runtime source changed')
                    for artifact in old['artifacts']:
                        if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('T2 runtime changed')
                    continue
                model=ACIBMemoryNetwork()
                model.base=ACIBVariantNetwork(anchor_mode=condition.get('anchor_mode','full'),selection=condition.get('selection','learned'),availability=condition.get('availability',True))
                model.load_state_dict(saved['model'],strict=True);model=DecisionCapture(model).eval();artifacts=[];started=time.monotonic()
                for event in [e for e in inputs['inputs'] if e['sequence']==sequence]:
                    runtime=TrustedACIBRecognizer(model,anchors[event['anchor_index']],event['episode_uid'],capacity=condition['capacity'],policy=condition['policy'])
                    runtime.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
                    path=ASSETS/'experiments/T2/runtime'/case/f"{event['episode_uid']}.jsonl.zst";path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():raise FileExistsError('preserve unsealed partial T2 episode')
                    count=0
                    with path.open('xb') as stream:
                        proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
                        try:
                            for p,rows in frames:
                                frame=int(p['frame'])
                                if frame<=event['frame']:continue
                                if time.monotonic()-started>1800:raise RuntimeError('30-minute T2 case/scene cap')
                                result=runtime.step(frame,rows);result.update(initialization_label='SIMULATED_ONE_CLICK_FROM_GT',
                                    training_stage='T2_FIXED_BEHAVIOR_PAIRED_MEMORY_SUPERVISION',inference_control=name,
                                    inference_deletion_not_retrained_architecture=name in ('BANK_ONLY_IDENTITY_P1','UNIFORM_BANK_ATTENTION','WITHOUT_AVAILABILITY'))
                                result['rank1_identity_joint_probability']=float(model.last['joint_probabilities'][0,:len(rows)].max()) if rows else 0.
                                result['NONE_joint_probability']=float(model.last['joint_probabilities'][0,-1])
                                proc.stdin.write((json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                            proc.stdin.close()
                            if proc.wait()!=0:raise RuntimeError('T2 compression')
                        finally:
                            if proc.poll() is None:proc.terminate();proc.wait()
                    artifacts.append({'episode_uid':event['episode_uid'],'path':str(path),'sha256':sha256(path),'frames':count})
                write_json(f'experiments/T2/runtime_seals/{case}/{sequence}.json',{'case':case,'sequence':sequence,'seed':seed,'condition':condition,'artifacts':artifacts,
                    'code_sha256':code,'deployment_protocol_sha256':sha256(protocol_path),'actual_T2_fit_record_sha256':sha256(fitpath),
                    'actual_T2_best_checkpoint_sha256':fitted['best_checkpoint_sha256'],'INNER_selected_epoch':saved['epoch'],
                    'actual_learned_future_safe_head':True,'GT_future_or_extra_click_used_by_runtime':False,
                    'inference_controls_not_retrained_architectures':True,'no_cross_recording_training_claim':True,'next_stage_authorized':False})
                print(json.dumps({'T2_runtime_sealed':case,'sequence':sequence,'episodes':len(artifacts),'seconds':round(time.monotonic()-started,1)}),flush=True)


def evaluate(sequences,seeds,names):
    from sam3_intermot.one_click.datasets import dancetrack_annotations,dancetrack_truth,strict_candidate_matching
    from sam3_intermot.evaluation.one_click_protocol import evaluate_episode
    from sam3_intermot.evaluation.one_click_identity import strict_identity_claim_metrics
    inputs={e['episode_uid']:e for e in read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']}
    labels={e['episode_uid']:e['target_gt_identity'] for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    for sequence in sequences:
        cases=[f'T2_{name}_SEED{seed}' for seed in seeds for name in names]
        seals={case:read_json(OUT/'experiments/T2/runtime_seals'/case/f'{sequence}.json') for case in cases}
        for seal in seals.values():
            if seal['code_sha256']!={p:sha256(ROOT/p) for p in CODE}:raise ValueError('T2 runtime code changed before labels')
            for artifact in seal['artifacts']:
                if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('T2 runtime artifact changed before labels')
        gtroot=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence;gt=dancetrack_annotations(gtroot)
        frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        for case,seal in seals.items():
            results=[]
            for artifact in seal['artifacts']:
                event=inputs[artifact['episode_uid']];identity=labels[artifact['episode_uid']]
                truth=[dancetrack_truth(int(p['frame']),rows,gt.get(int(p['frame']),[]),identity) for p,rows in frames if int(p['frame'])>event['frame']]
                trace=read_zstd_jsonl(Path(artifact['path']))
                result=evaluate_episode(trace,truth,fps=event['fps'],recording_id=sequence)
                verified=[]
                for r in trace:
                    uid=r.get('rank1_candidate_uid');axis=matched[int(r['frame'])]
                    if uid is not None and uid not in axis:raise ValueError('rank1 UID outside sealed candidate axis')
                    label=axis.get(uid)
                    verified.append(False if uid is None else None if label is None else bool(label==identity))
                result['secondary_strict_identity_claim']=strict_identity_claim_metrics(trace,truth,verified_rank1_labels=verified)
                result.update(episode_uid=artifact['episode_uid'],runtime_sha256=artifact['sha256'],initialization_failure=False,development_only=True);results.append(result)
            write_json(f'experiments/T2/evaluations/{case}/{sequence}.json',{'case':case,'sequence':sequence,'episodes':results,
                'runtime_seal_sha256':sha256(OUT/'experiments/T2/runtime_seals'/case/f'{sequence}.json'),'GT_sha256':sha256(gtroot/'gt/gt.txt'),
                'visible_GT_gap_not_physical_absence_truth':True,'inference_controls_not_retrained_architectures':True,
                'evaluator_source_sha256':sha256(ROOT/'sam3_intermot/evaluation/one_click_identity.py'),
                'secondary_metric_protocol_sha256':sha256(OUT/'protocol/T2_STRICT_IDENTITY_CLAIM_METRICS.json'),
                'secondary_UNKNOWN_repair_protocol_sha256':sha256(OUT/'protocol/T2_VERIFIED_CLAIM_CALIBRATION_REPAIR.json'),
                'independent_final_validation':False,'scientific_success':False,'next_stage_authorized':False})
            print(json.dumps({'T2_posthoc_evaluated':case,'sequence':sequence}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['replay','evaluate']);parser.add_argument('--sequences',nargs='+',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103]);parser.add_argument('--cases',nargs='+')
    args=parser.parse_args();names=args.cases or list(read_json(OUT/'protocol/T2_DEPLOYMENT_ABLATIONS.json')['cases'])
    (replay if args.action=='replay' else evaluate)(args.sequences,args.seeds,names)
