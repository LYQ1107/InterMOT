"""Frozen-head actual complete original-frame global MOT, no runtime GT."""
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
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.learned_mot_bridge import LearnedMOTIdentityBridge
from sam3_intermot.one_click.learned_authority import TrajectoryAuthorityPredictor
from sam3_intermot.one_click.intervention_gate import GatePolicy

PROTOCOL=OUT/'protocol/LEARNED_FULL_MOT_PILOT_V1.json'


def run(sequence,case_name):
    torch.set_num_threads(1); protocol=read_json(PROTOCOL)
    assert sequence in protocol['sequences']; condition=next(c for c in protocol['cases'] if c['case']==case_name)
    assert all(sha256(ROOT/p)==digest for p,digest in protocol['source_code_SHA'].items())
    inputs_path=OUT/'corpus/RUNTIME_INPUTS.json'; assert sha256(inputs_path)==protocol['input_SHA']
    inputs=read_json(inputs_path)
    event=sorted([e for e in inputs['inputs'] if e['sequence']==sequence],key=lambda e:(e['frame'],e['episode_uid']))[0]
    assert not event['initialization_failure']
    prefix='learned_pilot/runtime_seals/'+case_name+'/'+sequence+'.json'
    if (OUT/prefix).exists():
        seal=read_json(OUT/prefix); assert seal['protocol_SHA']==sha256(PROTOCOL)
        assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts'])
        print(json.dumps({'verified_existing_completed_learned_rollout':case_name,'sequence':sequence}),flush=True); return
    storage(64<<20); frames,index_SHA=checked_frames(sequence)
    assert index_SHA==event['candidate_index_sha256'] and sha256(inputs['anchor_path'])==inputs['anchor_sha256']
    anchor=np.array(np.load(inputs['anchor_path'],mmap_mode='r')[event['anchor_index']],np.float32)
    source=inputs['frozen_identity_model']; model=None; actor=None; predictor=None
    if condition['kind'] not in ('CLICK_C0','NO_HUMAN_C0'):
        assert sha256(source['path'])==source['sha256'] and sha256(source['fit_record_path'])==source['fit_record_sha256']
        saved=torch.load(source['path'],map_location='cpu',weights_only=True); fit=read_json(source['fit_record_path'])
        assert saved['schema']==fit['schema']; model=DecisionCapture(ACIBMemoryNetwork()).eval()
        model.model.load_state_dict(saved['model'],strict=True)
        actor=TrustedACIBRecognizer(model,anchor,event['episode_uid'],policy='P0',capacity=8)
        actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
        if condition['kind']=='LEARNED_SHADOW':
            head=next(c for c in protocol['cases'] if c['case']==protocol['shadow_head'])
        else: head=condition
        assert sha256(head['fit_record_path'])==head['fit_record_SHA'] and sha256(head['checkpoint_path'])==head['checkpoint_SHA']
        predictor=TrajectoryAuthorityPredictor(head['checkpoint_path'])
    click={'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']}
    bridge=SafeMOTIdentityBridge(click,policy=GatePolicy(family='off'),no_human=condition['kind']=='NO_HUMAN_C0',frames=len(frames)) if actor is None else LearnedMOTIdentityBridge(click,actor,predictor=predictor,intervene=condition['kind']!='LEARNED_SHADOW',frames=len(frames))
    bridge.configure_fps(event['fps'])
    trace_path=ASSETS/'learned_pilot/traces'/case_name/(sequence+'.jsonl.zst')
    trajectory_path=ASSETS/'learned_pilot/trackers'/case_name/'data'/(sequence+'.txt')
    for path in (trace_path,trajectory_path):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError('preserve partial learned replay, explicit versioned repair required')
    began=time.monotonic(); step_seconds=0.; effective=0; writes=0
    with trace_path.open('xb') as stream,trajectory_path.open('x') as trajectory:
        compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for payload,rows in frames:
                if time.monotonic()-began>protocol['max_seconds_per_case']: raise RuntimeError('learned full rollout time cap')
                frame=int(payload['frame']); started=time.perf_counter(); actual=bridge.step(frame,rows); step_seconds+=time.perf_counter()-started
                effective+=bool(actual['authority'].get('effective_assignment_change')); writes+=actual['joint_identity_memory_write']
                actual.update(case=case_name,episode_uid=event['episode_uid'],candidate_index_SHA=index_SHA,
                    cached_rank1_joint_probability=float(model.last['joint_probabilities'][0,:len(rows)].max()) if model is not None and frame>event['frame'] and rows else 0.,
                    runtime_GT_read=False,runtime_future_GT_input=False,source_distribution='ACTUAL_OWN_LEARNED_POLICY_COMMITTED_HISTORY')
                compressor.stdin.write((json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode()); trajectory.write(trajectory_text([actual]))
            compressor.stdin.close()
            if compressor.wait(): raise RuntimeError('learned rollout compressor failed')
        finally:
            if compressor.poll() is None: compressor.terminate(); compressor.wait()
    reference=None
    if case_name in ('CLICK_C0','LEARNED_SHADOW') or (predictor is not None and predictor.selection['status']=='CALIBRATION_ABSTAIN'):
        baseline=read_json(OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json')
        expected=next(a['sha256'] for a in baseline['artifacts'] if a['kind']=='trajectory')
        assert sha256(trajectory_path)==expected, 'actual no-intervention full trajectory AA failed'
        reference={'path':str(OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json'),'trajectory_SHA':expected}
    write_json(prefix,{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_LEARNED_FULL_JOINT_RUNTIME','case':case_name,'sequence':sequence,
        'condition':condition,'event':event,'frames':len(frames),'effective_interventions':effective,'committed_writes':writes,
        'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],'inputs_SHA':sha256(inputs_path),
        'candidate_index_SHA':index_SHA,'identity_model_source':source if actor is not None else None,
        'artifacts':[{'kind':kind,'path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size} for kind,path in [('trace',trace_path),('trajectory',trajectory_path)]],
        'AA_reference':reference,'actual_complete_original_frame_axis':True,'one_click':case_name!='NO_HUMAN_C0','extra_clicks':0,
        'full_global_exact_unique_candidate_ownership':True,'runtime_GT_input':False,'runtime_future_GT_input':False,'history_rewritten':False,
        'cached_joint_step_seconds':step_seconds,'cached_total_seconds':time.monotonic()-began,'cached_joint_FPS_not_pixel_FPS':len(frames)/step_seconds,
        'identity_parameter_count':sum(p.numel() for p in model.parameters()) if model is not None else 0,
        'controller_parameter_count':sum(p.numel() for p in predictor.model.parameters()) if predictor is not None else 0,
        'association_only_seconds':step_seconds if model is None else None,'cumulative_process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'not_independent_confirmation':True,'no_VAL_TEST_used':True,'scientific_success':None})
    print(json.dumps({'actual_learned_MOT_complete':case_name,'sequence':sequence,'effective':effective,'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--sequence',required=True);parser.add_argument('--case',required=True)
    args=parser.parse_args();run(args.sequence,args.case)
