"""Committed-only memory controls and capacity/aggregation in full joint MOT."""
import argparse
from dataclasses import asdict
import json
import resource
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r21r1_collect_joint import source_policy
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.committed_identity_memory import CommittedIdentityMemory,MemoryCommitPolicy
from sam3_intermot.one_click.committed_memory_bridge import CommittedMemoryMOTBridge
from sam3_intermot.one_click.joint_write_authority import JointWritePredictor
from sam3_intermot.one_click.intervention_gate import GatePolicy

PROTOCOL=OUT/'protocol/COMMITTED_MEMORY_FULL_PILOT_V1.json'


def freeze():
    cases=[];seen=set();fit_plan=read_json(OUT/'protocol/JOINT_WRITE_HEADS_V1.json')
    def add(family,mode,capacity=8,aggregation='attention',seed=None,variant='DEFAULT',**kwargs):
        name='__'.join((family.upper(),mode,'K'+str(capacity),aggregation,variant,'seed'+str(seed)))
        if name in seen:return
        seen.add(name);policy=MemoryCommitPolicy(family=family,capacity=capacity,aggregation=aggregation,**kwargs)
        c={'case':name,'kind':'COMMITTED_MEMORY','write_policy':asdict(policy),'mode':mode,'seed':seed,
            'family':'__'.join((family.upper(),mode,'K'+str(capacity),aggregation,variant)),
            'state_contrast':'ACTUAL_COMMITTED_MEMORY','reward':'CURRENT_AND_PAIRED_FUTURE_WRITE_RISK'}
        if family=='risk':
            path=OUT/'training/joint_write'/('JOINT_COMMITTED_CURRENT_AND_FUTURE_RISK_MLP__seed'+str(seed)+'.json');r=read_json(path)
            assert sha256(r['checkpoint_path'])==r['checkpoint_SHA'] and all(sha256(ROOT/p)==s for p,s in r['source_code_SHA'].items())
            c.update(fit_record_path=str(path),fit_record_SHA=sha256(path),checkpoint_path=r['checkpoint_path'],checkpoint_SHA=r['checkpoint_SHA'],selection=r['selection'])
        cases.append(c)
    for family in ('frozen','unsafe','consensus','delayed','diverse','risk','rollback'):
        for mode in ('SHADOW','COUPLED'):
            seeds=fit_plan['seeds'] if family=='risk' else [None]
            for seed in seeds:add(family,mode,seed=seed)
    # Nontrivial unsafe and fixed consensus axes retained, even when safe writes
    # are zero. A degenerate safe-bank ablation must be labeled as such.
    for family in ('unsafe','consensus'):
        for capacity in (1,4,8):
            for aggregation in ('attention','mean','confidence'):add(family,'COUPLED',capacity,aggregation)
    for family in ('consensus','delayed'):
        add(family,'COUPLED',variant='LOW_PROBABILITY_DIAGNOSTIC',probability_min=.1)
    for delay in (1,3,5):add('delayed','COUPLED',variant='DELAY'+str(delay),delay_frames=delay)
    # Looser cue control is predeclared, not selected from its outer effects.
    add('diverse','COUPLED',variant='LOW_PROBABILITY_DIAGNOSTIC',probability_min=.1)
    baseline=next(c for c in cases if c['write_policy']['family']=='frozen' and c['mode']=='SHADOW');baseline['case']='CLICK_C0'
    del baseline['family'];del baseline['state_contrast'];del baseline['reward']
    assert len(cases)==40
    files=['scripts/n72r21r1_memory_pilot.py','sam3_intermot/one_click/committed_identity_memory.py',
        'sam3_intermot/one_click/committed_memory_bridge.py','sam3_intermot/one_click/joint_write_authority.py',
        'sam3_intermot/one_click/intervention_features.py','sam3_intermot/one_click/intervention_gate.py',
        'sam3_intermot/one_click/joint_intervention_primitives.py','sam3_intermot/one_click/safe_mot_bridge.py',
        'scripts/n72r21r1_collect_joint.py']
    write_json('protocol/COMMITTED_MEMORY_FULL_PILOT_V1.json',{'stage':'N72R21R1','final_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_full_memory_policy_effects':True,'cases':cases,'sequences':['dancetrack0001','dancetrack0002'],
        'planned_actual_rollouts':80,'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),
        'initialization_verification_SHA':sha256(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json'),
        'source_code_SHA':{p:sha256(ROOT/p) for p in files},'capacity_aggregation_axes':[1,4,8],
        'aggregation_modes':['attention','mean','confidence'],'attention':'Frozen existing identity-model attention, not newly trained bank attention.',
        'SHADOW':'Bank may write; complete tracker output/native/prototype state must remain C0-AA, not MOT success.',
        'COUPLED':'Same predeclared broad confidence authority used by paired-write branches; compare against frozen-bank coupled baseline and C0.',
        'cue_consensus_not_independent_statistical_confirmation':True,'unsafe_control_not_deployable':True,
        'all_cases_and_seeds_retained_no_best_outer_effect_selection':True,'all_abstaining_risk_heads_run_as_explicit_controls':True,
        'memory_GATE':'Nonzero accepted, wrong-or-UNKNOWN<=2%, correct retention>=60%; both actual correct-commit and candidate-available denominators reported. Zero writes cannot PASS.',
        'future_GT_triggered_rollback':False,'no_VAL_TEST_confirmation_tuning':True,
        'historical_development_not_virgin_confirmation':True,'max_CPU_workers':3,'threads_per_worker':1,
        'max_seconds_per_case':1800,'reserve_GiB':60,'next_stage_authorized':False})


def run(sequence,case_name):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL);condition=next(c for c in protocol['cases'] if c['case']==case_name)
    assert sequence in protocol['sequences'] and all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    inputs_path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(inputs_path)==protocol['input_SHA'];inputs=read_json(inputs_path)
    event=min((e for e in inputs['inputs'] if e['sequence']==sequence),key=lambda e:(e['frame'],e['episode_uid']))
    prefix='memory/pilot/runtime_seals/'+case_name+'/'+sequence+'.json'
    if (OUT/prefix).exists():
        s=read_json(OUT/prefix);assert s['protocol_SHA']==sha256(PROTOCOL)
        assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts']);return
    flags=read_json(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json')['initial_clicked_UID_verification']
    assert all(r['clicked_UID_strictly_verified_target'] for r in flags if r['episode_uid']==event['episode_uid'])
    storage(64<<20);frames,index_SHA=checked_frames(sequence);assert index_SHA==event['candidate_index_sha256']
    assert sha256(inputs['anchor_path'])==inputs['anchor_sha256'];source=inputs['frozen_identity_model'];assert sha256(source['path'])==source['sha256']
    anchor=np.array(np.load(inputs['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
    saved=torch.load(source['path'],map_location='cpu',weights_only=True);model=DecisionCapture(ACIBMemoryNetwork()).eval();model.model.load_state_dict(saved['model'],strict=True)
    predictor=None
    if condition['write_policy']['family']=='risk':
        assert sha256(condition['checkpoint_path'])==condition['checkpoint_SHA'] and sha256(condition['fit_record_path'])==condition['fit_record_SHA']
        predictor=JointWritePredictor(condition['checkpoint_path'])
    actor=CommittedIdentityMemory(model,anchor,event['episode_uid'],write_policy=MemoryCommitPolicy(**condition['write_policy']),write_predictor=predictor)
    actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
    gate=GatePolicy(family='shadow') if condition['mode']=='SHADOW' else source_policy('JOINT_FIXED_BROAD_ON_POLICY_ROUND0')[1]
    bridge=CommittedMemoryMOTBridge({'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],
        'target_box_xyxy':event['box_xyxy']},actor,policy=gate,frames=len(frames));bridge.configure_fps(event['fps'])
    trace_path=ASSETS/'memory/pilot/traces'/case_name/(sequence+'.jsonl.zst');trajectory_path=ASSETS/'memory/pilot/trackers'/case_name/'data'/(sequence+'.txt')
    for path in (trace_path,trajectory_path):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('preserve partial full memory replay')
    began=time.monotonic();step_seconds=0.;effective=0;writes=0;maximum_bank=0
    with trace_path.open('xb') as stream,trajectory_path.open('x') as trajectory:
        compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for payload,rows in frames:
                if time.monotonic()-began>protocol['max_seconds_per_case']:raise RuntimeError('full memory replay time cap')
                f=int(payload['frame']);start=time.perf_counter();actual=bridge.step(f,rows);step_seconds+=time.perf_counter()-start
                effective+=bool(actual['authority'].get('effective_assignment_change'));writes+=actual['joint_identity_memory_write']
                bank=bridge.identity.bank;maximum_bank=max(maximum_bank,len(bank))
                actual.update(case=case_name,episode_uid=event['episode_uid'],candidate_index_SHA=index_SHA,
                    cached_rank1_joint_probability=float(model.last['joint_probabilities'][0,:len(rows)].max()) if f>event['frame'] and rows else 0.,
                    trusted_bank_mean_anchor_cosine_after=float(np.mean([e.anchor_consistency for e in bank])) if bank else None,
                    rollback_count_after=bridge.identity.rollback_count,
                    target_prototype_anchor_cosine_after=float(np.dot(bridge.tracker.states[bridge.tracker.target_public].prototype,anchor)) if bridge.tracker.target_public is not None else None)
                compressor.stdin.write((json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode());trajectory.write(trajectory_text([actual]))
            compressor.stdin.close()
            if compressor.wait():raise RuntimeError('full memory replay compression failure')
        finally:
            if compressor.poll() is None:compressor.terminate();compressor.wait()
    reference=None
    if condition['mode']=='SHADOW':
        baseline_path=OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json';baseline=read_json(baseline_path)
        expected=next(a['sha256'] for a in baseline['artifacts'] if a['kind']=='trajectory');assert sha256(trajectory_path)==expected
        reference={'path':str(baseline_path),'trajectory_SHA':expected}
    write_json(prefix,{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_COMMITTED_MEMORY_FULL_JOINT_RUNTIME','case':case_name,'sequence':sequence,
        'event':event,'condition':condition,'frames':len(frames),'effective_interventions':effective,'committed_writes':writes,'maximum_bank_size':maximum_bank,
        'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],'inputs_SHA':protocol['input_SHA'],
        'artifacts':[{'kind':kind,'path':str(path),'sha256':sha256(path)} for kind,path in [('trace',trace_path),('trajectory',trajectory_path)]],
        'AA_reference':reference,'runtime_GT_input':False,'runtime_future_GT_input':False,'history_rewritten':False,
        'full_global_exact_unique_candidate_ownership':True,'extra_clicks':0,'cached_joint_step_seconds':step_seconds,
        'cached_total_seconds':time.monotonic()-began,'cached_joint_FPS_not_pixel_FPS':len(frames)/step_seconds,
        'identity_parameter_count':sum(p.numel() for p in model.parameters()),
        'controller_parameter_count':sum(p.numel() for p in predictor.model.parameters()) if predictor is not None else 0,
        'association_only_seconds':None,'cumulative_process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'not_independent_confirmation':True,'no_VAL_TEST_used':True,'scientific_success':None})
    print(json.dumps({'actual_memory_full_MOT_complete':case_name,'sequence':sequence,'writes':writes,'effective':effective,
        'max_bank':maximum_bank,'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--sequence');parser.add_argument('--case');args=parser.parse_args()
    if args.freeze:freeze()
    else:run(args.sequence,args.case)
