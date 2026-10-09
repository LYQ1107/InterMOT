"""T1-produced causal states plus true paired TRAIN-only memory rollouts.

No annotations are opened here. Both branches use their own future decisions
and memory writes, on the same next up-to-ten real candidate frames. Offline
labels are added only after these artifacts are SHA-sealed.
"""
from __future__ import annotations
import argparse
import copy
import json
import subprocess
import time
import numpy as np
import torch
from sam3_intermot.one_click.acib import ACIBNetwork
from sam3_intermot.one_click.acib_runtime import ACIBRecognizer
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from scripts.n72r21_baselines_geometry_repair import valid_geometry

CODE=['scripts/n72r21_collect_coupled_states.py','sam3_intermot/one_click/acib.py','sam3_intermot/one_click/acib_runtime.py']


class Capture(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.last_logits=None
    def forward(self,*args,**kwargs):
        result=self.base(*args,**kwargs);self.last_logits=result['candidate_logits'][0].detach().cpu().tolist();return result


def clone_state(runtime):
    result=copy.copy(runtime);result.bank=list(runtime.bank)
    result.last_box=copy.copy(runtime.last_box);result.pending=copy.deepcopy(runtime.pending)
    # Model/immutable embeddings are shared read-only. All mutable state is
    # separate; creating counterfactual branches cannot modify main rollout.
    return result


def paired_rollout(before,frame,rows,future_frames):
    omitted=clone_state(before);committed=clone_state(before)
    omitted.policy='P0';first_without=omitted.step(frame,rows);omitted.policy='P1'
    committed.policy='P1';first_with=committed.step(frame,rows)
    if first_without['selected_candidate_uid']!=first_with['selected_candidate_uid']:raise ValueError('identical pre-write candidate decision must match')
    if first_with['selected_candidate_uid'] is None:raise ValueError('no own proposal')
    branches={'without_current_write':[],'with_current_write':[]}
    for p,candidates in future_frames:
        for name,runtime in [('without_current_write',omitted),('with_current_write',committed)]:
            record=runtime.step(int(p['frame']),candidates)
            record['current_candidate_logits']={str(r['candidate_uid']):float(value) for r,value in zip(candidates,runtime.model.last_logits[:len(candidates)],strict=True)}
            branches[name].append(record)
    return {'proposal_frame':frame,'proposed_candidate_uid':first_with['selected_candidate_uid'],
            'current_decision_identical_before_write':True,'current_write_is_only_intervention':True,
            'both_future_policies':'P1_OWN_PREDICTIONS_NOT_ORACLE','branches':branches,
            'GT_read':False,'future_features_online_input_at_proposal':False}


def run(outer,seeds,scope):
    torch.set_num_threads(1);protocol_path=OUT/'protocol/T2_MEMORY_COUPLED_TRAINING.json';protocol=read_json(protocol_path)
    registered=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    if outer not in registered['sequences'] or not set(seeds)<=set(protocol['seeds']) or scope not in ('fit','inner','pilot'):raise ValueError('registered T2 collection')
    inputs=read_json(OUT/'development/RUNTIME_INPUTS.json')
    if sha256(inputs['anchor_path'])!=inputs['anchor_sha256']:raise ValueError('anchor SHA')
    anchors=np.load(inputs['anchor_path'],mmap_mode='r');code={p:sha256(ROOT/p) for p in CODE};storage(200<<20)
    for seed in seeds:
        fitpath=OUT/'training/T1_CAUSAL_V1'/f'{outer}__seed{seed}__MIXED.json';fitted=read_json(fitpath);configuration=fitted['schema']['configuration']
        if not fitted['completed'] or configuration['outer_sequence']!=outer:raise ValueError('source completed matching T1 fit')
        for p,h in configuration['code_sha256'].items():
            if sha256(ROOT/p)!=h:raise ValueError('source fitting code changed')
        if sha256(fitted['best_checkpoint_path'])!=fitted['best_checkpoint_sha256']:raise ValueError('source checkpoint changed')
        saved=torch.load(fitted['best_checkpoint_path'],map_location='cpu',weights_only=True)
        model=ACIBNetwork();model.load_state_dict(saved['model'],strict=True);model=Capture(model).eval()
        sequences=configuration['fit_sequences'][:1] if scope=='pilot' else configuration['fit_sequences'] if scope=='fit' else [configuration['inner_sequence']]
        if outer in sequences:raise ValueError('outer state collection forbidden')
        for sequence in sequences:
            case=f'{outer}__seed{seed}__T1MIXED_P1_K8';done=OUT/'training/coupled_states/seals'/case/f'{sequence}.json'
            if done.exists():
                old=read_json(done)
                if old['code_sha256']!=code or old['protocol_sha256']!=sha256(protocol_path) or old['source_T1_fit_record_sha256']!=sha256(fitpath):raise ValueError('sealed coupled source changed')
                for a in old['artifacts']:
                    for kind in ('states','branches'):
                        if sha256(a[kind+'_path'])!=a[kind+'_sha256']:raise ValueError('coupled artifact changed')
                print(json.dumps({'reused_coupled':case,'sequence':sequence}),flush=True);continue
            frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
            artifacts=[];started=time.monotonic()
            for event in [e for e in inputs['inputs'] if e['sequence']==sequence]:
                runtime=ACIBRecognizer(model,anchors[event['anchor_index']],event['episode_uid'],capacity=8,policy='P1')
                runtime.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
                directory=ASSETS/'training/coupled_states'/case;directory.mkdir(parents=True,exist_ok=True)
                paths={kind:directory/f"{event['episode_uid']}.{kind}.jsonl.zst" for kind in ('states','branches')}
                if any(path.exists() for path in paths.values()):raise FileExistsError('preserve unsealed partial coupled rollout')
                frame_count=snapshots=proposals=none_proposals=0
                with paths['states'].open('xb') as state_stream,paths['branches'].open('xb') as branch_stream:
                    state_proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=state_stream)
                    branch_proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=branch_stream)
                    try:
                        for position,(p,rows) in enumerate(frames):
                            frame=int(p['frame'])
                            if frame<=event['frame']:continue
                            if time.monotonic()-started>1800:raise RuntimeError('30-minute coupled case/sequence cap')
                            snapshot=runtime.snapshot() if (frame-event['frame']-1)%5==0 else None
                            propose=(frame-event['frame']-1)%protocol['risk_proposal_stride']==0
                            before=clone_state(runtime) if propose else None
                            record=runtime.step(frame,rows);record['TRAIN_state_before']=snapshot
                            state_proc.stdin.write((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode());frame_count+=1;snapshots+=snapshot is not None
                            if propose:
                                if record['selected_candidate_uid'] is None:none_proposals+=1;continue
                                branches=paired_rollout(before,frame,rows,frames[position+1:position+1+protocol['future_branch_horizon_frames']])
                                if branches['proposed_candidate_uid']!=record['selected_candidate_uid']:raise ValueError('main/branch proposed UID mismatch')
                                branches['main_source_state_SHA256_anchor']=record['anchor_sha256']
                                branch_proc.stdin.write((json.dumps(branches,sort_keys=True,allow_nan=False)+'\n').encode());proposals+=1
                        for proc in (state_proc,branch_proc):
                            proc.stdin.close()
                            if proc.wait()!=0:raise RuntimeError('coupled compression failed')
                    finally:
                        for proc in (state_proc,branch_proc):
                            if proc.poll() is None:proc.terminate();proc.wait()
                artifact={'episode_uid':event['episode_uid'],'frames':frame_count,'state_snapshots':snapshots,'paired_proposals':proposals,'NONE_proposals_excluded':none_proposals}
                for kind,path in paths.items():artifact.update({kind+'_path':str(path),kind+'_sha256':sha256(path),kind+'_bytes':path.stat().st_size})
                artifacts.append(artifact)
                print(json.dumps({'coupled_case':case,'sequence':sequence,'episode':event['episode_uid'],'frames':frame_count,'paired_proposals':proposals,'seconds':round(time.monotonic()-started,1)}),flush=True)
            write_json(f'training/coupled_states/seals/{case}/{sequence}.json',{'case':case,'outer':outer,'sequence':sequence,'seed':seed,
                'scope_at_collection':scope,'code_sha256':code,'protocol_sha256':sha256(protocol_path),'source_T1_fit_record_sha256':sha256(fitpath),
                'source_T1_checkpoint_sha256':fitted['best_checkpoint_sha256'],'source_T1_best_epoch':saved['epoch'],'artifacts':artifacts,
                'all_actual_future_frames_run':True,'runtime_GT_used':False,'oracle_positive_memory_used':False,
                'paired_future_trajectories_current_write_intervention_only':True,'T2_fit_complete':False,
                'physical_absence_or_cross_recording_supervision_available':False,'next_stage_authorized':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--outer',required=True);parser.add_argument('--seeds',nargs='+',type=int,default=[72101,72102,72103])
    parser.add_argument('--scope',choices=['pilot','fit','inner'],default='pilot');args=parser.parse_args();run(args.outer,args.seeds,args.scope)
