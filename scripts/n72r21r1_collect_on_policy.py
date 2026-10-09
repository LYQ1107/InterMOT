"""Actual learned-policy joint history and prospective independent branches."""
import argparse
import json
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r21r1_collect_joint import collect_event
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.learned_authority import TrajectoryAuthorityPredictor
from sam3_intermot.one_click.learned_mot_bridge import LearnedMOTIdentityBridge

PROTOCOL=OUT/'protocol/ON_POLICY_ROUND1.json'


def run(sequence):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL)
    assert all(sha256(ROOT/p)==h for p,h in protocol['source_code_SHA'].items())
    assert sha256(protocol['checkpoint_path'])==protocol['checkpoint_SHA']
    assert sha256(protocol['policy_fit_path'])==protocol['policy_fit_SHA']
    inputs_path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(inputs_path)==protocol['inputs_SHA']
    inputs=read_json(inputs_path);source=protocol['identity_model_source']
    assert sha256(inputs['anchor_path'])==inputs['anchor_sha256'] and sha256(source['path'])==source['sha256']
    saved=torch.load(source['path'],map_location='cpu',weights_only=True)
    assert saved['schema']==read_json(source['fit_record_path'])['schema']
    base=ACIBMemoryNetwork();base.load_state_dict(saved['model'],strict=True);base.eval()
    anchors=np.load(inputs['anchor_path'],mmap_mode='r');frames,index_SHA=checked_frames(sequence)
    events=[e for e in protocol['events'] if e['sequence']==sequence]
    if not events:raise ValueError('sequence not registered in this round')
    for event in events:
        prefix='on_policy/round1/runtime/'+event['episode_uid'];seal_path=OUT/prefix/'seal.json'
        if seal_path.exists():
            seal=read_json(seal_path);assert seal['protocol_SHA']==sha256(PROTOCOL)
            assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);continue
        storage(96<<20);assert index_SHA==event['candidate_index_sha256']
        anchor=np.array(anchors[event['anchor_index']],np.float32);model=DecisionCapture(base).eval()
        actor=TrustedACIBRecognizer(model,anchor,event['episode_uid'],capacity=8,policy='P0')
        actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
        predictor=TrajectoryAuthorityPredictor(protocol['checkpoint_path'])
        bridge=LearnedMOTIdentityBridge({'event_frame':event['frame'],'human_anchor':anchor,
            'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']},actor,predictor=predictor,frames=len(frames))
        bridge.configure_fps(event['fps'])
        trace_path=ASSETS/'on_policy/round1/traces'/(event['episode_uid']+'.jsonl.zst')
        trajectory_path=ASSETS/'on_policy/round1/trackers'/event['episode_uid']/'data'/(sequence+'.txt')
        for path in (trace_path,trajectory_path):
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('preserve partial on-policy episode, explicit versioned recovery required')
        began=time.monotonic();branches=[];effective=0
        with trace_path.open('xb') as stream,trajectory_path.open('x') as trajectory:
            compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
            try:
                for payload,rows in frames:
                    f=int(payload['frame'])
                    if time.monotonic()-began>protocol['max_seconds_per_episode']:raise RuntimeError('on-policy episode time cap')
                    offset=f-event['frame']-16
                    if offset>=0 and offset%128==0 and len(branches)<16:
                        branches.append(collect_event(bridge,frames,f,prefix+'/counterfactual/frame'+str(f),protocol['state_source']))
                    actual=bridge.step(f,rows);effective+=bool(actual['authority'].get('effective_assignment_change'))
                    actual.update(episode_uid=event['episode_uid'],source_state_distribution=protocol['state_source'],
                        cached_rank1_joint_probability=float(model.last['joint_probabilities'][0,:len(rows)].max()) if f>event['frame'] and rows else 0.,
                        candidate_index_SHA=index_SHA,identity_state_snapshot=bridge.identity.snapshot())
                    compressor.stdin.write((json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode());trajectory.write(trajectory_text([actual]))
                compressor.stdin.close()
                if compressor.wait():raise RuntimeError('on-policy compressor failed')
            finally:
                if compressor.poll() is None:compressor.terminate();compressor.wait()
        write_json(prefix+'/seal.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_LEARNED_ON_POLICY_JOINT_RUNTIME',
            'event':event,'state_source':protocol['state_source'],'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],
            'inputs_SHA':protocol['inputs_SHA'],'checkpoint_SHA':protocol['checkpoint_SHA'],'frames':len(frames),
            'effective_interventions':effective,'seconds':time.monotonic()-began,'counterfactual_seals':branches,
            'artifacts':[{'kind':kind,'path':str(path),'sha256':sha256(path)} for kind,path in [('trace',trace_path),('trajectory',trajectory_path)]],
            'runtime_GT_input':False,'runtime_future_GT_input':False,'GT_history_replacement':False,
            'extra_clicks':0,'original_exact_solver_unchanged':True,'not_independent_confirmation':True})
        print(json.dumps({'actual_on_policy_episode_complete':event['episode_uid'],'effective':effective,'sampled_events':len(branches),'actual_arms':sum(b['actual_arms'] for b in branches),'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True);args=parser.parse_args();run(args.sequence)
