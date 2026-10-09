"""Actual committed-observation curriculum and paired single-write diagnostics.

WRITE_CURRENT is an explicit TRAIN counterfactual, never a deployed GT policy.
Both future branches use the same predeclared confidence authority and no more
writes, so a bank-induced joint-assignment effect can be measured causally.
"""
import argparse
import json
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r21r1_collect_joint import source_policy
from scripts.n72r21r1_counterfactual import compact
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_runtime import Evidence,unit
from sam3_intermot.one_click.committed_identity_memory import CommittedIdentityMemory,MemoryCommitPolicy
from sam3_intermot.one_click.committed_memory_bridge import CommittedMemoryMOTBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy

PROTOCOL=OUT/'protocol/COMMITTED_MEMORY_CURRICULUM_V1.json'


def freeze():
    inputs=read_json(OUT/'corpus/RUNTIME_INPUTS.json');flags={}
    for r in read_json(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json')['initial_clicked_UID_verification']:
        old=flags.setdefault(r['episode_uid'],r['clicked_UID_strictly_verified_target']);assert old==r['clicked_UID_strictly_verified_target']
    events=[e for e in inputs['inputs'] if flags[e['episode_uid']]];assert len(events)==22
    files=['scripts/n72r21r1_memory_collect.py','sam3_intermot/one_click/committed_identity_memory.py',
        'sam3_intermot/one_click/committed_memory_bridge.py','sam3_intermot/one_click/intervention_features.py',
        'sam3_intermot/one_click/joint_intervention_primitives.py','sam3_intermot/one_click/safe_mot_bridge.py',
        'scripts/n72r21r1_collect_joint.py','scripts/n72r21r1_counterfactual.py']
    write_json('protocol/COMMITTED_MEMORY_CURRICULUM_V1.json',{'stage':'N72R21R1','final_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_memory_effects_or_write_fits':True,'events':events,'planned_full_joint_rollouts':22,
        'excluded_unverified_initial_clicks_no_replacement':[e['episode_uid'] for e in inputs['inputs'] if not flags[e['episode_uid']]],
        'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),'source_code_SHA':{p:sha256(ROOT/p) for p in files},
        'source_policy':'C0 shadow authority plus frozen immutable-anchor writer, own complete global commit history.',
        'prospective_pair_positions':'event+16,+144,+272; NONE is logged skipped, never replaced or GT-selected.',
        'paired_treatment':'FORCE_CURRENT_ACTUAL_COMMITTED_CROP_WRITE versus DENY_WRITE, then same fixed broad-confidence authority for H100; no subsequent writes.',
        'paired_intervention_not_deployed_policy':True,'never_write_actor_proposal_if_not_actual_commit':True,
        'future_gate_source':'Existing JOINT_FIXED_BROAD_ON_POLICY_ROUND0, fixed before this branch.',
        'supervision':'Only after all22 sources sealed; exact committed UID features each frame, canonical origins and paired future H1/5/20/50/100. FIT/INNER only fits; exposed0001 diagnostic only.',
        'FIT_heads':'Joint-state current correct/wrong-or-UNKNOWN and paired future other-person/N10 risk; three seeds, FIT-only normalizer, INNER risk/retention curve.',
        'capacity_note':'Source bank empty and one-shot capacity8 only; capacities1/4/8 and aggregation are independent later full memory pilot, not fabricated from this pair.',
        'max_CPU_workers':2,'threads_per_worker':1,'max_seconds_per_episode':1800,'reserve_GiB':60,
        'fresh_confirmation_VAL_TEST_untouched':True,'next_stage_authorized':False})


def force_current_write(bridge,frame,rows,actual,features):
    uid=actual['target_uid'];candidate=next((r for r in rows if str(r['candidate_uid'])==uid),None)
    if candidate is None or bridge.tracker.frame!=frame or bridge.identity.last_frame!=frame:
        raise ValueError('diagnostic write requires actual current committed crop')
    if uid!=actual['base_assignments'].get(str(actual['target_public_id'])):
        raise ValueError('curriculum source must be own C0 commit')
    actor=bridge.identity;vector=unit(candidate['feature']);vector.setflags(write=False)
    actor.bank.append(Evidence(vector,actor.recording,frame,actor.camera,frame/actor.fps,
        float(features['proposed_probability']),'TRAIN_COUNTERFACTUAL_CURRENT_COMMIT_NOT_DEPLOYED_WRITER',
        float(features['quality']),float(vector@actor.anchor),uid));actor.bank=actor.bank[-actor.capacity:]
    return {**actual,'joint_identity_memory_write':True,'joint_memory_write_candidate_uid':uid}


def paired_write(bridge,frames,frame,actual,prefix):
    rows=frames[frame][1]
    if actual['target_uid'] is None:
        return {'frame':frame,'status':'SKIPPED_COMMITTED_NONE_NO_REPLACEMENT','actual_arms':0}
    features=actual['committed_observation_features'];artifacts=[]
    for name in ('DENY_WRITE','WRITE_CURRENT_COMMITTED'):
        arm=bridge.clone();current=actual
        if name=='WRITE_CURRENT_COMMITTED':current=force_current_write(arm,frame,rows,actual,features)
        arm.policy=source_policy('JOINT_FIXED_BROAD_ON_POLICY_ROUND0')[1]
        trace=[compact(current,arm)]
        for payload,future_rows in frames[frame+1:min(len(frames),frame+101)]:
            step=arm.step(int(payload['frame']),future_rows);row=compact(step,arm)
            row.update(rank1_candidate_uid=step['identity_decision']['rank1_candidate_uid'],
                identity_candidate_available_probability=step['identity_decision']['candidate_available_probability'])
            trace.append(row)
        path=write_json(prefix+'/'+name+'.json',{'name':name,'features':features,'rows':trace,
            'only_actual_current_commit_written':True,'not_deployed_write_policy':True,'runtime_GT_input':False,'runtime_future_GT_input':False})
        artifacts.append({'name':name,'path':str(path),'sha256':sha256(path),'frames':len(trace)})
    first=[read_json(a['path'])['rows'][0] for a in artifacts]
    assert first[0]['outputs']==first[1]['outputs'] and first[0]['state_after']==first[1]['state_after']
    assert not bridge.identity.bank
    path=write_json(prefix+'/seal.json',{'status':'COMPLETE_ACTUAL_SAME_POSTCOMMIT_WRITE_PAIR','frame':frame,
        'protocol_SHA':sha256(PROTOCOL),'artifacts':artifacts,'same_current_global_assignment':True,
        'no_runtime_GT':True,'no_shared_mutable_bank_tracker_or_pending':True})
    return {'frame':frame,'status':'COMPLETE_PAIR','path':str(path),'sha256':sha256(path),'actual_arms':2}


def run(sequence):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL)
    assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    inputs_path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(inputs_path)==protocol['input_SHA'];inputs=read_json(inputs_path)
    source=inputs['frozen_identity_model'];assert sha256(source['path'])==source['sha256']
    assert sha256(inputs['anchor_path'])==inputs['anchor_sha256']
    saved=torch.load(source['path'],map_location='cpu',weights_only=True);base=ACIBMemoryNetwork().eval();base.load_state_dict(saved['model'],strict=True)
    anchors=np.load(inputs['anchor_path'],mmap_mode='r');frames,index_SHA=checked_frames(sequence)
    events=[e for e in protocol['events'] if e['sequence']==sequence]
    if not events:raise ValueError('sequence outside frozen curriculum')
    for event in events:
        prefix='memory/curriculum/runtime/'+event['episode_uid'];seal_path=OUT/prefix/'seal.json'
        if seal_path.exists():
            seal=read_json(seal_path);assert seal['protocol_SHA']==sha256(PROTOCOL)
            assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);continue
        storage(96<<20);assert index_SHA==event['candidate_index_sha256']
        anchor=np.array(anchors[event['anchor_index']],np.float32);model=DecisionCapture(base).eval()
        actor=CommittedIdentityMemory(model,anchor,event['episode_uid'],write_policy=MemoryCommitPolicy(family='frozen'))
        actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
        bridge=CommittedMemoryMOTBridge({'event_frame':event['frame'],'human_anchor':anchor,
            'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']},actor,policy=GatePolicy(family='shadow'),frames=len(frames));bridge.configure_fps(event['fps'])
        trace_path=ASSETS/'memory/curriculum/traces'/(event['episode_uid']+'.jsonl.zst')
        trajectory_path=ASSETS/'memory/curriculum/trackers'/event['episode_uid']/'data'/(sequence+'.txt')
        for path in (trace_path,trajectory_path):
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError('preserve partial committed memory curriculum')
        began=time.monotonic();pairs=[];positions={event['frame']+k for k in (16,144,272)}
        with trace_path.open('xb') as stream,trajectory_path.open('x') as trajectory:
            compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
            try:
                for payload,rows in frames:
                    if time.monotonic()-began>protocol['max_seconds_per_episode']:raise RuntimeError('memory curriculum time cap')
                    f=int(payload['frame']);actual=bridge.step(f,rows)
                    assert not actual['joint_identity_memory_write'] and not bridge.identity.bank
                    if f in positions:pairs.append(paired_write(bridge,frames,f,actual,prefix+'/paired_write/frame'+str(f)))
                    actual.update(episode_uid=event['episode_uid'],candidate_index_SHA=index_SHA,
                        cached_rank1_joint_probability=float(model.last['joint_probabilities'][0,:len(rows)].max()) if f>event['frame'] and rows else 0.)
                    compressor.stdin.write((json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode());trajectory.write(trajectory_text([actual]))
                compressor.stdin.close()
                if compressor.wait():raise RuntimeError('memory curriculum compressor failure')
            finally:
                if compressor.poll() is None:compressor.terminate();compressor.wait()
        baseline_path=OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json';baseline=read_json(baseline_path)
        expected=next(a['sha256'] for a in baseline['artifacts'] if a['kind']=='trajectory');assert sha256(trajectory_path)==expected
        write_json(prefix+'/seal.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_COMMITTED_MEMORY_CURRICULUM',
            'event':event,'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],'inputs_SHA':protocol['input_SHA'],
            'frames':len(frames),'paired_write_seals':pairs,'seconds':time.monotonic()-began,
            'artifacts':[{'kind':kind,'path':str(path),'sha256':sha256(path)} for kind,path in [('trace',trace_path),('trajectory',trajectory_path)]],
            'C0_full_trajectory_AA':True,'baseline_reference_SHA':sha256(baseline_path),'runtime_GT_input':False,
            'runtime_future_GT_input':False,'GT_history_replacement':False,'extra_clicks':0})
        print(json.dumps({'actual_committed_memory_curriculum_complete':event['episode_uid'],
            'actual_pair_arms':sum(p['actual_arms'] for p in pairs),'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--sequence');args=parser.parse_args()
    if args.freeze:freeze()
    else:run(args.sequence)
