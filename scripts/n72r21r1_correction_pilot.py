"""Frozen on-policy correction heads in actual own-history complete joint MOT."""
import argparse
import itertools
import json
import os
import resource
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.learned_mot_bridge import LearnedMOTIdentityBridge
from sam3_intermot.one_click.on_policy_authority import OnPolicyAuthorityPredictor
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy

PROTOCOL=OUT/'protocol/ON_POLICY_CORRECTION_PILOT_V1.json'


def freeze():
    plan=read_json(OUT/'protocol/ON_POLICY_CORRECTION_HEADS_V1.json');cases=[]
    for family,contrast,seed in itertools.product(plan['families'],plan['data_contrasts'],plan['seeds']):
        uid='__'.join((family,contrast,'seed'+str(seed)));path=OUT/'training/on_policy_correction'/(uid+'.json');r=read_json(path)
        assert sha256(r['checkpoint_path'])==r['checkpoint_SHA'] and all(sha256(ROOT/p)==s for p,s in r['source_code_SHA'].items())
        cases.append({'case':uid,'kind':'LEARNED_MULTI_ACTION','family':family,'state_contrast':contrast,
            'reward':'H100_GLOBAL_RISK','seed':seed,'fit_record_path':str(path),'fit_record_SHA':sha256(path),
            'checkpoint_path':r['checkpoint_path'],'checkpoint_SHA':r['checkpoint_SHA'],'selection':r['selection']})
    cases=[{'case':'CLICK_C0','kind':'CLICK_C0'},{'case':'LEARNED_SHADOW','kind':'LEARNED_SHADOW'}]+cases
    files=['scripts/n72r21r1_correction_pilot.py','sam3_intermot/one_click/on_policy_authority.py',
        'sam3_intermot/one_click/learned_mot_bridge.py','sam3_intermot/one_click/learned_authority.py',
        'sam3_intermot/one_click/safe_mot_bridge.py','sam3_intermot/one_click/joint_intervention_primitives.py',
        'sam3_intermot/one_click/intervention_features.py']
    write_json('protocol/ON_POLICY_CORRECTION_PILOT_V1.json',{'stage':'N72R21R1','final_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'cases':cases,'sequences':['dancetrack0001','dancetrack0002'],'planned_actual_rollouts':40,
        'shadow_head':cases[2]['case'],'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),
        'source_code_SHA':{p:sha256(ROOT/p) for p in files},'frozen_before_full_policy_effects':True,
        'retain_all_registered_contrasts_seeds_and_abstaining_fits':True,'max_seconds_per_case':1800,
        'episode_selection':'First frozen corpus click per sequence, no observed-effect selection or replacement.',
        'historical_development_not_virgin_confirmation':True,'no_VAL_test_confirmation_tuning':True,
        'memory':'P0 isolate on-policy authority correction','next_stage_authorized':False})


def run(sequence,case_name):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL);condition=next(c for c in protocol['cases'] if c['case']==case_name)
    assert sequence in protocol['sequences'] and all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    inputs_path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(inputs_path)==protocol['input_SHA'];inputs=read_json(inputs_path)
    event=min((e for e in inputs['inputs'] if e['sequence']==sequence),key=lambda e:(e['frame'],e['episode_uid']))
    prefix='on_policy/correction_pilot/runtime_seals/'+case_name+'/'+sequence+'.json'
    if (OUT/prefix).exists():
        seal=read_json(OUT/prefix);assert seal['protocol_SHA']==sha256(PROTOCOL)
        assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);return
    storage(64<<20);frames,index_SHA=checked_frames(sequence);assert index_SHA==event['candidate_index_sha256']
    assert sha256(inputs['anchor_path'])==inputs['anchor_sha256']
    anchor=np.array(np.load(inputs['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
    click={'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']}
    source=inputs['frozen_identity_model'];model=None;predictor=None
    if condition['kind']=='CLICK_C0':bridge=SafeMOTIdentityBridge(click,policy=GatePolicy(family='off'),frames=len(frames))
    else:
        assert sha256(source['path'])==source['sha256'];saved=torch.load(source['path'],map_location='cpu',weights_only=True)
        model=DecisionCapture(ACIBMemoryNetwork()).eval();model.model.load_state_dict(saved['model'],strict=True)
        actor=TrustedACIBRecognizer(model,anchor,event['episode_uid'],policy='P0',capacity=8)
        actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
        head=next(c for c in protocol['cases'] if c['case']==protocol['shadow_head']) if condition['kind']=='LEARNED_SHADOW' else condition
        assert sha256(head['checkpoint_path'])==head['checkpoint_SHA'] and sha256(head['fit_record_path'])==head['fit_record_SHA']
        predictor=OnPolicyAuthorityPredictor(head['checkpoint_path'])
        bridge=LearnedMOTIdentityBridge(click,actor,predictor=predictor,intervene=condition['kind']!='LEARNED_SHADOW',frames=len(frames))
    bridge.configure_fps(event['fps']);trace_path=ASSETS/'on_policy/correction_pilot/traces'/case_name/(sequence+'.jsonl.zst')
    trajectory_path=ASSETS/'on_policy/correction_pilot/trackers'/case_name/'data'/(sequence+'.txt')
    for path in (trace_path,trajectory_path):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('preserve partial correction replay')
    began=time.monotonic();step_seconds=0.;effective=0
    with trace_path.open('xb') as stream,trajectory_path.open('x') as trajectory:
        compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for payload,rows in frames:
                if time.monotonic()-began>protocol['max_seconds_per_case']:raise RuntimeError('correction replay time cap')
                f=int(payload['frame']);start=time.perf_counter();actual=bridge.step(f,rows);step_seconds+=time.perf_counter()-start
                effective+=bool(actual['authority'].get('effective_assignment_change'))
                actual.update(case=case_name,episode_uid=event['episode_uid'],candidate_index_SHA=index_SHA,
                    cached_rank1_joint_probability=float(model.last['joint_probabilities'][0,:len(rows)].max()) if model is not None and f>event['frame'] and rows else 0.)
                compressor.stdin.write((json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode());trajectory.write(trajectory_text([actual]))
            compressor.stdin.close()
            if compressor.wait():raise RuntimeError('correction compressor failure')
        finally:
            if compressor.poll() is None:compressor.terminate();compressor.wait()
    reference=None
    if condition['kind'] in ('CLICK_C0','LEARNED_SHADOW') or predictor.selection['status']=='CALIBRATION_ABSTAIN':
        baseline_path=OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json';baseline=read_json(baseline_path)
        expected=next(a['sha256'] for a in baseline['artifacts'] if a['kind']=='trajectory');assert sha256(trajectory_path)==expected
        reference={'path':str(baseline_path),'trajectory_SHA':expected}
    write_json(prefix,{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_ON_POLICY_CORRECTION_FULL_JOINT_RUNTIME',
        'case':case_name,'sequence':sequence,'event':event,'condition':condition,'frames':len(frames),'effective_interventions':effective,
        'committed_writes':0,'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],'inputs_SHA':protocol['input_SHA'],
        'artifacts':[{'kind':kind,'path':str(path),'sha256':sha256(path)} for kind,path in [('trace',trace_path),('trajectory',trajectory_path)]],
        'AA_reference':reference,'runtime_GT_input':False,'runtime_future_GT_input':False,'history_rewritten':False,
        'full_global_exact_unique_candidate_ownership':True,'extra_clicks':0,'cached_joint_step_seconds':step_seconds,
        'cached_total_seconds':time.monotonic()-began,'cached_joint_FPS_not_pixel_FPS':len(frames)/step_seconds,
        'identity_parameter_count':sum(p.numel() for p in model.parameters()) if model is not None else 0,
        'controller_parameter_count':sum(p.numel() for p in predictor.model.parameters()) if predictor is not None else 0,
        'association_only_seconds':step_seconds if model is None else None,'cumulative_process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'not_independent_confirmation':True,'no_VAL_TEST_used':True,'scientific_success':None})
    print(json.dumps({'correction_replay_complete':case_name,'sequence':sequence,'effective':effective,'seconds':round(time.monotonic()-began,2)}),flush=True)


def driver():
    protocol=read_json(PROTOCOL);pending=[(c['case'],s) for c in protocol['cases'] for s in protocol['sequences']];active=[];finished=[]
    logs=ASSETS/'on_policy/correction_pilot/logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    python=str(ROOT.parent/'InterMOT/.venv/bin/python');update_status(phase_C5_correction_full_MOT_running=True)
    while pending or active:
        while pending and len(active)<2:
            case,sequence=pending.pop(0);seal_path=OUT/'on_policy/correction_pilot/runtime_seals'/case/(sequence+'.json')
            if seal_path.exists():
                seal=read_json(seal_path);assert seal['protocol_SHA']==sha256(PROTOCOL)
                assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);finished.append({'case':case,'sequence':sequence,'returncode':0,'verified_existing':True});continue
            path=logs/(case+'__'+sequence+'.log')
            if path.exists():raise FileExistsError('preserve partial correction worker log')
            storage(64<<20);handle=path.open('x');command=[python,str(ROOT/'scripts/n72r21r1_correction_pilot.py'),'--case',case,'--sequence',sequence]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);active.append((process,handle,path,case,sequence))
        for entry in list(active):
            process,handle,path,case,sequence=entry;rc=process.poll()
            if rc is None:continue
            handle.close();active.remove(entry);finished.append({'case':case,'sequence':sequence,'returncode':rc,'log_path':str(path),'log_SHA':sha256(path)})
            print(json.dumps({'correction_replay_terminal':case,'sequence':sequence,'returncode':rc,'pending':len(pending)}),flush=True)
        if active:time.sleep(.25)
    write_json('on_policy/correction_pilot/SCHEDULER_RESULT.json',{'planned_runs':40,'workers':finished,'CPU_workers_max':2,'no_GPU_use':True})
    update_status(phase_C5_correction_full_MOT_running=False,actual_correction_full_MOT_runs=sum(r['returncode']==0 for r in finished))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--driver',action='store_true')
    parser.add_argument('--sequence');parser.add_argument('--case');args=parser.parse_args()
    if args.freeze:freeze()
    elif args.driver:driver()
    else:run(args.sequence,args.case)
