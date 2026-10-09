"""All E0-E6 policies on their own complete joint online histories."""
import argparse
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
from scripts.n72r21r1_train_open_set import PROTOCOL as FIT_PROTOCOL
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.open_set_verifier import OpenSetVerifierPredictor,OpenSetMOTIdentityBridge

PROTOCOL=OUT/'protocol/CURRENT_AXIS_FULL_JOINT_PILOT_V1.json'


def freeze():
    p=read_json(FIT_PROTOCOL);simple_path=OUT/'availability/current_axis_verifier/SIMPLE_CALIBRATION_V1.json'
    simple=read_json(simple_path);assert simple['protocol_SHA']==sha256(FIT_PROTOCOL)
    cases=[{'case':'CLICK_C0','kind':'REUSED_ACTUAL_SEALED_C0_NOT_NEW_RUN'}]
    for family in p['families']:
        for seed in p['seeds']:
            path=OUT/'training/current_axis_verifier'/(family+'__seed'+str(seed)+'.json');r=read_json(path)
            assert r['protocol_SHA']==sha256(FIT_PROTOCOL) and sha256(r['checkpoint_path'])==r['checkpoint_SHA']
            assert all(sha256(ROOT/k)==v for k,v in r['source_code_SHA'].items())
            case={'case':family+'__CALIBRATED__seed'+str(seed),'kind':'LEARNED_CURRENT_AXIS','family':family,
                'state_contrast':'FULL_C0_SOURCE_TO_OWN_POLICY_P0','reward':'CURRENT_UID_NONE_UNKNOWN',
                'seed':seed,'fit_record_path':str(path),'fit_record_SHA':sha256(path),
                'checkpoint_path':r['checkpoint_path'],'checkpoint_SHA':r['checkpoint_SHA'],'selection':r['selection']}
            cases.append(case)
            if family=='LOGISTIC':
                cases.append({**case,'case':family+'__UNCONSTRAINED_DIAGNOSTIC__seed'+str(seed),
                    'reward':'UNCONSTRAINED_CURRENT_UID_DIAGNOSTIC','selection':{**r['selection'],**p['diagnostic_operating_point'],
                        'global_regret_max':p['runtime_global_regret_max']},'not_INNER_selected_or_deployable':True})
    for family in p['simple_controls']:
        cases.append({'case':family,'kind':'SIMPLE_CURRENT_AXIS','selection':simple['controls'][family]['selection']})
    shadow=next(c for c in cases if c['case']=='LOGISTIC__CALIBRATED__seed72111')
    cases.append({**shadow,'case':'CURRENT_AXIS_SHADOW','kind':'LEARNED_SHADOW'})
    assert len(cases)==20
    files=['scripts/n72r21r1_open_set_pilot.py','sam3_intermot/one_click/open_set_verifier.py',
        'sam3_intermot/one_click/safe_mot_bridge.py','sam3_intermot/one_click/joint_intervention_primitives.py',
        'sam3_intermot/one_click/intervention_features.py','sam3_intermot/one_click/acib_trusted_runtime.py']
    write_json('protocol/CURRENT_AXIS_FULL_JOINT_PILOT_V1.json',{'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_full_policy_effects':True,'cases':cases,'sequences':p['runtime_full_scope'],
        'new_actual_full_joint_runs_planned':38,'reused_baseline_runs':2,'all9_heads_all3_unconstrained_linear_all6_simple_retained':True,
        'simple_calibration_SHA':sha256(simple_path),'fit_protocol_SHA':sha256(FIT_PROTOCOL),
        'episode_selection':'First prospectively valid episode each sequence, same prior cases, never chosen by effect',
        'source_code_SHA':{f:sha256(ROOT/f) for f in files},'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),
        'E0_semantics':'Frozen ACIB joint argmax under same conservative owner/regret protection, not original permissive ACIB controller',
        'trained_availability_head':'Separate candidate-availability diagnostic, correctness3class chooses UID/NONE; physical presence not available truth',
        'claim_safety_not_future_trajectory_safety':True,'global_solver_scorer_unchanged':True,'no_new_candidate_generation':True,
        'historical_TRAIN_0002_INNER_and_0001_exposed_not_generalization':True,'max_CPU_workers':2,'threads_per_worker':1,
        'max_seconds_per_case':1800,'reserve_GiB':60,'no_VAL_TEST_confirmation':True,'next_stage_authorized':False})


def run(sequence,case_name):
    torch.set_num_threads(1);p=read_json(PROTOCOL);assert sequence in p['sequences']
    case=next(c for c in p['cases'] if c['case']==case_name);assert case_name!='CLICK_C0'
    assert all(sha256(ROOT/k)==v for k,v in p['source_code_SHA'].items())
    path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(path)==p['input_SHA'];inputs=read_json(path)
    event=sorted([e for e in inputs['inputs'] if e['sequence']==sequence],key=lambda e:(e['frame'],e['episode_uid']))[0]
    assert not event['initialization_failure']
    prefix='availability/current_axis_pilot/runtime_seals/'+case_name+'/'+sequence+'.json'
    if (OUT/prefix).exists():
        r=read_json(OUT/prefix);assert r['protocol_SHA']==sha256(PROTOCOL)
        assert all(sha256(a['path'])==a['sha256'] for a in r['artifacts']);return
    storage(64<<20);frames,index_SHA=checked_frames(sequence);assert index_SHA==event['candidate_index_sha256']
    assert sha256(inputs['anchor_path'])==inputs['anchor_sha256']
    anchor=np.array(np.load(inputs['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
    source=inputs['frozen_identity_model'];assert sha256(source['path'])==source['sha256'] and sha256(source['fit_record_path'])==source['fit_record_sha256']
    saved=torch.load(source['path'],map_location='cpu',weights_only=True);model=DecisionCapture(ACIBMemoryNetwork()).eval()
    model.model.load_state_dict(saved['model'],strict=True)
    actor=TrustedACIBRecognizer(model,anchor,event['episode_uid'],policy='P0',capacity=8)
    actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
    predictor=None
    if case['kind']!='SIMPLE_CURRENT_AXIS':
        assert sha256(case['fit_record_path'])==case['fit_record_SHA'] and sha256(case['checkpoint_path'])==case['checkpoint_SHA']
        predictor=OpenSetVerifierPredictor(case['checkpoint_path'])
    click={'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']}
    bridge=OpenSetMOTIdentityBridge(click,actor,verifier=predictor,selection=case['selection'],intervene=case['kind']!='LEARNED_SHADOW',frames=len(frames));bridge.configure_fps(event['fps'])
    trace_path=ASSETS/'availability/current_axis_pilot/traces'/case_name/(sequence+'.jsonl.zst')
    trajectory_path=ASSETS/'availability/current_axis_pilot/trackers'/case_name/'data'/(sequence+'.txt')
    for dest in (trace_path,trajectory_path):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise FileExistsError('preserve partial current-axis own-policy replay')
    began=time.monotonic();step_seconds=0.;effective=0
    with trace_path.open('xb') as stream,trajectory_path.open('x') as trajectory:
        proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for payload,rows in frames:
                if time.monotonic()-began>p['max_seconds_per_case']:raise RuntimeError('current-axis own-policy time cap')
                f=int(payload['frame']);start=time.perf_counter();actual=bridge.step(f,rows);step_seconds+=time.perf_counter()-start
                effective+=bool(actual['authority'].get('effective_assignment_change'))
                assert not actual['joint_identity_memory_write'] and not bridge.identity.bank
                actual.update(case=case_name,episode_uid=event['episode_uid'],candidate_index_SHA=index_SHA,
                    cached_rank1_joint_probability=float(model.last['joint_probabilities'][0,:len(rows)].max()) if f>event['frame'] and rows else 0.,
                    runtime_GT_read=False,runtime_future_GT_input=False,source_distribution='ACTUAL_OWN_OPEN_SET_POLICY_COMMITTED_FULL_JOINT_HISTORY')
                proc.stdin.write((json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode());trajectory.write(trajectory_text([actual]))
            proc.stdin.close()
            if proc.wait():raise RuntimeError('current-axis trace compressor failed')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
    reference=None
    if case['kind']=='LEARNED_SHADOW' or case['selection']['status']=='CALIBRATION_ABSTAIN':
        old=OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json';baseline=read_json(old)
        expected=next(a['sha256'] for a in baseline['artifacts'] if a['kind']=='trajectory')
        assert sha256(trajectory_path)==expected,'C0-preserving shadow/abstain AA failed'
        reference={'path':str(old),'trajectory_SHA':expected}
    write_json(prefix,{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_CURRENT_AXIS_FULL_JOINT_RUNTIME','case':case_name,
        'sequence':sequence,'event':event,'condition':case,'frames':len(frames),'effective_interventions':effective,'committed_writes':0,
        'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':p['source_code_SHA'],'inputs_SHA':p['input_SHA'],'candidate_index_SHA':index_SHA,
        'identity_model_source':source,'artifacts':[{'kind':k,'path':str(a),'sha256':sha256(a),'bytes':a.stat().st_size} for k,a in [('trace',trace_path),('trajectory',trajectory_path)]],
        'AA_reference':reference,'actual_complete_original_frame_axis':True,'one_click':True,'extra_clicks':0,
        'full_global_exact_unique_candidate_ownership':True,'runtime_GT_input':False,'runtime_future_GT_input':False,'history_rewritten':False,
        'cached_joint_step_seconds':step_seconds,'cached_total_seconds':time.monotonic()-began,'cached_joint_FPS_not_pixel_FPS':len(frames)/step_seconds,
        'identity_parameter_count':sum(v.numel() for v in model.parameters()),'controller_parameter_count':sum(v.numel() for v in predictor.model.parameters()) if predictor else 0,
        'association_only_seconds':None,'cumulative_process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'not_independent_confirmation':True,'no_VAL_TEST_used':True,'current_identity_claim_not_future_override_safety':True,'scientific_success':None})
    print(json.dumps({'actual_open_set_full_joint':case_name,'sequence':sequence,'effective':effective,'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--sequence');parser.add_argument('--case');a=parser.parse_args()
    if a.freeze:freeze()
    else:run(a.sequence,a.case)
