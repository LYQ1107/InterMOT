"""Prospective explicit B7/B8 gates with independent own-history runtimes."""
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
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.learned_authority import TrajectoryAuthorityPredictor
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.trajectory_gate_bridge import TrajectoryGateMOTBridge

PROTOCOL=OUT/'protocol/EXPLICIT_B7_B8_JOINT_PILOT_V1.json'


def freeze():
    cases=[{'case':'CLICK_C0','kind':'REUSED_ACTUAL_BASELINE_NOT_NEW_RUN'}]
    for gate,head,reward in [('two_branch','CAUSAL_TEMPORAL','H100_GLOBAL_RISK'),('risk','GLOBAL_RISK','H100_GLOBAL_RISK'),
                             ('two_branch','LOGISTIC','H5'),('risk','LOGISTIC','H5')]:
        for seed in (72111,72112,72113):
            uid='__'.join((head,'MIXED',reward,'seed'+str(seed)));path=OUT/'training/authority'/(uid+'.json');r=read_json(path)
            assert sha256(r['checkpoint_path'])==r['checkpoint_SHA']
            assert all(sha256(ROOT/k)==v for k,v in r['source_code_SHA'].items())
            s=r['selection'];active=s['status']=='CALIBRATED_DIAGNOSTIC'
            policy=GatePolicy(family=gate,probability_min=0.,identity_margin_min=-1.,base_margin_max=100.,quality_min=0.,
                anchor_cosine_min=-1.,regret_max=.2,displaced_max=0,delay_frames=2,allow_reject=True,
                benefit_min=s['benefit_min'] if active else 1.01,harm_max=s['harm_max'] if active else -.01,
                value_min=s['value_min'] if active else 1.e9)
            cases.append({'case':'__'.join((gate.upper(),head,reward,'seed'+str(seed))),'kind':'EXPLICIT_TRAJECTORY_GATE',
                'family':gate.upper()+'_'+head,'state_contrast':'OWN_CAUSAL_PROPOSAL_HISTORY_P0','reward':reward,'seed':seed,
                'policy':asdict(policy),'selection':s,'fit_record_path':str(path),'fit_record_SHA':sha256(path),
                'checkpoint_path':r['checkpoint_path'],'checkpoint_SHA':r['checkpoint_SHA']})
    assert len(cases)==13
    files=['scripts/n72r21r1_trajectory_gate_pilot.py','sam3_intermot/one_click/trajectory_gate_bridge.py',
        'sam3_intermot/one_click/safe_mot_bridge.py','sam3_intermot/one_click/intervention_gate.py',
        'sam3_intermot/one_click/joint_intervention_primitives.py','sam3_intermot/one_click/intervention_features.py',
        'sam3_intermot/one_click/learned_authority.py']
    write_json('protocol/EXPLICIT_B7_B8_JOINT_PILOT_V1.json',{'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_new_gate_effects':True,'cases':cases,'sequences':['dancetrack0001','dancetrack0002'],
        'new_actual_full_joint_runs_planned':24,'reused_baseline_runs':2,'new_optimization':False,
        'head_selection':'Lexical preregistered H100 primary temporal/globalrisk plus all3seeds H5 linear reward diagnostic; not best outer policy',
        'TRAIN_counterfactual_branches':'Actual prior paired same-prestate KEEP/challenger independent futures supervise raw benefit/harm; no online oracle branch',
        'runtime_causal_history':'Own gate original ACIB proposal features at real preceding3 frames, zero only where no preceding postclick history; clone-local adapter',
        'runtime_fixed_structural_gate':'Original gate family, .2 global-regret ceiling, zero displaced owners; B7 causal delay2, frozen FIT head INNER cutoffs; all0 calibrated heads retained',
        'source_code_SHA':{k:sha256(ROOT/k) for k in files},'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),
        'max_CPU_workers':2,'threads_per_worker':1,'max_seconds_per_case':1800,'reserve_GiB':60,
        'historically_exposed_TRAIN_not_generalization':True,'no_VAL_TEST_confirmation':True,'next_stage_authorized':False})


def run(sequence,case_name):
    torch.set_num_threads(1);p=read_json(PROTOCOL);assert sequence in p['sequences']
    case=next(c for c in p['cases'] if c['case']==case_name);assert case_name!='CLICK_C0'
    assert all(sha256(ROOT/k)==v for k,v in p['source_code_SHA'].items())
    path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(path)==p['input_SHA'];inputs=read_json(path)
    event=sorted([e for e in inputs['inputs'] if e['sequence']==sequence],key=lambda e:(e['frame'],e['episode_uid']))[0]
    assert not event['initialization_failure'];prefix='trajectory_gate/runtime_seals/'+case_name+'/'+sequence+'.json'
    if (OUT/prefix).exists():
        s=read_json(OUT/prefix);assert s['protocol_SHA']==sha256(PROTOCOL) and all(sha256(a['path'])==a['sha256'] for a in s['artifacts']);return
    storage(64<<20);frames,index_SHA=checked_frames(sequence);assert index_SHA==event['candidate_index_sha256']
    assert sha256(inputs['anchor_path'])==inputs['anchor_sha256'];anchor=np.array(np.load(inputs['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
    source=inputs['frozen_identity_model'];assert sha256(source['path'])==source['sha256'] and sha256(source['fit_record_path'])==source['fit_record_sha256']
    saved=torch.load(source['path'],map_location='cpu',weights_only=True);model=DecisionCapture(ACIBMemoryNetwork()).eval();model.model.load_state_dict(saved['model'],strict=True)
    assert sha256(case['fit_record_path'])==case['fit_record_SHA'] and sha256(case['checkpoint_path'])==case['checkpoint_SHA']
    fitted=TrajectoryAuthorityPredictor(case['checkpoint_path']);actor=TrustedACIBRecognizer(model,anchor,event['episode_uid'],policy='P0',capacity=8)
    actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
    click={'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']}
    bridge=TrajectoryGateMOTBridge(click,actor,policy=GatePolicy(**case['policy']),fitted_predictor=fitted,frames=len(frames));bridge.configure_fps(event['fps'])
    trace=ASSETS/'trajectory_gate/traces'/case_name/(sequence+'.jsonl.zst');trajectory=ASSETS/'trajectory_gate/trackers'/case_name/'data'/(sequence+'.txt')
    for dest in (trace,trajectory):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise FileExistsError('preserve partial B7/B8 runtime')
    began=time.monotonic();step_seconds=0.;effective=0
    with trace.open('xb') as stream,trajectory.open('x') as text:
        proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for payload,rows in frames:
                if time.monotonic()-began>p['max_seconds_per_case']:raise RuntimeError('B7/B8 time cap')
                f=int(payload['frame']);start=time.perf_counter();r=bridge.step(f,rows);step_seconds+=time.perf_counter()-start
                effective+=bool(r['authority'].get('effective_assignment_change'));assert not r['joint_identity_memory_write'] and not bridge.identity.bank
                r.update(case=case_name,episode_uid=event['episode_uid'],candidate_index_SHA=index_SHA,
                    cached_rank1_joint_probability=float(model.last['joint_probabilities'][0,:len(rows)].max()) if f>event['frame'] and rows else 0.,
                    runtime_GT_read=False,runtime_future_GT_input=False,source_distribution='ACTUAL_OWN_B7_B8_GATE_COMMITTED_HISTORY')
                proc.stdin.write((json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode());text.write(trajectory_text([r]))
            proc.stdin.close()
            if proc.wait():raise RuntimeError('B7/B8 compressor failed')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
    reference=None
    if case['selection']['status']=='CALIBRATION_ABSTAIN':
        old=OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json';baseline=read_json(old)
        expected=next(a['sha256'] for a in baseline['artifacts'] if a['kind']=='trajectory');assert sha256(trajectory)==expected
        reference={'path':str(old),'trajectory_SHA':expected}
    write_json(prefix,{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_B7_B8_FULL_JOINT_RUNTIME','case':case_name,'sequence':sequence,
        'condition':case,'event':event,'frames':len(frames),'effective_interventions':effective,'committed_writes':0,
        'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':p['source_code_SHA'],'inputs_SHA':p['input_SHA'],'candidate_index_SHA':index_SHA,
        'identity_model_source':source,'artifacts':[{'kind':k,'path':str(a),'sha256':sha256(a),'bytes':a.stat().st_size} for k,a in [('trace',trace),('trajectory',trajectory)]],
        'AA_reference':reference,'actual_complete_original_frame_axis':True,'one_click':True,'extra_clicks':0,
        'full_global_exact_unique_candidate_ownership':True,'runtime_GT_input':False,'runtime_future_GT_input':False,'history_rewritten':False,
        'cached_joint_step_seconds':step_seconds,'cached_total_seconds':time.monotonic()-began,'cached_joint_FPS_not_pixel_FPS':len(frames)/step_seconds,
        'identity_parameter_count':sum(v.numel() for v in model.parameters()),'controller_parameter_count':sum(v.numel() for v in fitted.model.parameters()),
        'association_only_seconds':None,'cumulative_process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'not_independent_confirmation':True,'no_VAL_TEST_used':True,'scientific_success':None})
    print(json.dumps({'actual_B7_B8_complete':case_name,'sequence':sequence,'effective':effective}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--sequence');parser.add_argument('--case');a=parser.parse_args()
    if a.freeze:freeze()
    else:run(a.sequence,a.case)
