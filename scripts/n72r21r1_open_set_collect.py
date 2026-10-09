"""Prospective complete current candidate/NONE axis in actual joint C0 history."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r21_t2_replay import DecisionCapture
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.intervention_features import opportunity_features,feature_vector
from sam3_intermot.one_click.acib_runtime import overlap
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal,check_action
from sam3_intermot.association.opportunity_solver import AssociationAction,action_for_candidate

PROTOCOL=OUT/'protocol/JOINT_OPEN_SET_CURRENT_AXIS_V1.json'


def freeze():
    curriculum=read_json(OUT/'protocol/COMMITTED_MEMORY_CURRICULUM_V1.json')
    files=['scripts/n72r21r1_open_set_collect.py','sam3_intermot/one_click/safe_mot_bridge.py',
        'sam3_intermot/one_click/intervention_features.py','sam3_intermot/one_click/joint_intervention_primitives.py',
        'sam3_intermot/one_click/acib_trusted_runtime.py','sam3_intermot/one_click/acib_runtime.py']
    write_json('protocol/JOINT_OPEN_SET_CURRENT_AXIS_V1.json',{'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_new_open_set_data_or_fits':True,'events':curriculum['events'],'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),
        'source_code_SHA':{p:sha256(ROOT/p) for p in files},'actual_scene_scope':'same8 historical TRAIN,22 verified initial sole clicks',
        'source_history':'Actual full joint C0 shadow P0 committed history, not oracle/target-only history',
        'sample_rule':'Every32 original frames starting event+1, all real current candidates plus explicit NONE; no replacement/outcome sampling',
        'sample_period_frames':32,'all_current_candidate_UIDs_and_NONE':True,'full_global_assignment_recomputed_for_each_proposal':True,
        'labels':'Separate process only after all22 runtime/source/index/input SHA seals. TARGET versus verified OTHER versus UNKNOWN; NONE available truth separate. UNKNOWN never becomes a verified hard negative.',
        'taxonomy':'current anchor look-alikes; verified-other native continuity; box overlap wrong UID; current public owner; false/unmatched crop kept UNKNOWN; target unavailable/absent; no future or GT-best candidate features',
        'planned_training':'Fixed scalar/logistic/MLP candidate-correctness verifier; FIT6/INNER0002, historical0001 diagnostic only; epoch and cutoff calibration use INNER only; no deployment of uncalibrated epoch',
        'max_CPU_workers':2,'threads_per_worker':1,'max_seconds_per_episode':1800,'reserve_GiB':60,
        'new_pixels_weights_or_tape':False,'no_VAL_TEST_confirmation':True,'next_stage_authorized':False})


def current_axis(bridge,frame,rows):
    prepared=prepare_proposal(bridge,frame,rows);preview=prepared['preview'];public=bridge.tracker.target_public
    axis=[str(r['candidate_uid']) for r in rows]+[None];result=[]
    for uid in axis:
        action=AssociationAction('KEEP',public) if uid==preview['target_uid'] else AssociationAction('REJECT_TARGET',public) if uid is None else action_for_candidate(public,uid,preview['solver'])
        check=check_action(bridge,frame,rows,preview,action)
        candidate=next((r for r in rows if str(r['candidate_uid'])==uid),None);pending=bridge.authority_pending
        agreement=0. if candidate is None or pending is None else float(np.dot(candidate['feature'],pending['feature']))
        consecutive=bool(candidate is not None and pending is not None and pending['frame']==frame-1 and agreement>=.9 and overlap(candidate['box_xyxy'],pending['box'])>=.3)
        count=pending['count']+1 if consecutive else 1 if candidate is not None else 0
        fake={**prepared,'proposal':{**prepared['proposal'],'selected_candidate_uid':uid}}
        features=opportunity_features(bridge,frame,rows,fake,check,confirmations=count,
            previous_intervention=bridge.last_intervention,previous_agreement=agreement)
        result.append({'candidate_uid':uid,'features':features,'feature_vector':feature_vector(features).tolist(),
            'action':action.to_dict(),'action_feasible':check['feasible'],'action_not_executed':True,
            'current_owner_public':next((int(p) for p,u in preview['base_assignments'].items() if u==uid),None) if uid is not None else None})
    return {'frame':frame,'own_current_KEEP_UID':preview['target_uid'],'target_public_id':public,
        'candidate_count':len(rows),'axis':result,'all_current_candidates_and_explicit_NONE':True,
        'provisional_state_only_no_candidate_branch_committed':True}


def run(sequence):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL)
    assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    inputs_path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(inputs_path)==protocol['input_SHA'];inputs=read_json(inputs_path)
    identity=inputs['frozen_identity_model'];assert sha256(identity['path'])==identity['sha256'] and sha256(inputs['anchor_path'])==inputs['anchor_sha256']
    saved=torch.load(identity['path'],map_location='cpu',weights_only=True);base=ACIBMemoryNetwork().eval();base.load_state_dict(saved['model'],strict=True)
    frames,index_SHA=checked_frames(sequence);anchors=np.load(inputs['anchor_path'],mmap_mode='r')
    events=[e for e in protocol['events'] if e['sequence']==sequence]
    if not events:raise ValueError('outside frozen historical TRAIN scope')
    for event in events:
        prefix='availability/current_axis_v1/'+event['episode_uid'];seal_path=OUT/prefix/'seal.json'
        if seal_path.exists():
            seal=read_json(seal_path);assert seal['protocol_SHA']==sha256(PROTOCOL)
            assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);continue
        storage(32<<20);assert index_SHA==event['candidate_index_sha256'];anchor=np.array(anchors[event['anchor_index']],np.float32)
        actor=TrustedACIBRecognizer(DecisionCapture(base).eval(),anchor,event['episode_uid'],policy='P0',capacity=8)
        actor.start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
        bridge=SafeMOTIdentityBridge({'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],'target_box_xyxy':event['box_xyxy']},actor,policy=GatePolicy(family='shadow'),frames=len(frames));bridge.configure_fps(event['fps'])
        old_path=OUT/'corpus/runtime/JOINT_BASELINE_SHADOW_P0'/event['episode_uid']/'seal.json';old_seal=read_json(old_path)
        assert all(sha256(a['path'])==a['sha256'] for a in old_seal['artifacts'])
        old=read_zstd_jsonl(Path(next(a['path'] for a in old_seal['artifacts'] if a['kind']=='trace')))
        destination=ASSETS/'availability/current_axis_v1'/(event['episode_uid']+'.jsonl.zst');destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise FileExistsError('preserve partial current-axis collector')
        began=time.monotonic();groups=0;candidate_rows=0
        with destination.open('xb') as stream:
            proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
            try:
                for payload,rows in frames:
                    if time.monotonic()-began>protocol['max_seconds_per_episode']:raise RuntimeError('current axis episode cap')
                    f=int(payload['frame']);sample=None
                    if f>event['frame'] and (f-event['frame']-1)%protocol['sample_period_frames']==0:sample=current_axis(bridge,f,rows)
                    actual=bridge.step(f,rows)
                    for field in ('outputs','target_uid','state_before','state_after','selected_action'):
                        assert actual[field]==old[f][field],(event['episode_uid'],f,field)
                    assert not actual['joint_identity_memory_write'] and not bridge.identity.bank
                    if sample is not None:
                        sample.update(sequence=sequence,episode_uid=event['episode_uid'],role=event['diagnostic_role'],
                            actual_committed_UID=actual['target_uid'],source_original_C0_state_before_SHA=actual['state_before'],
                            current_axis_feature_not_GT_or_future=True)
                        assert sample['own_current_KEEP_UID']==actual['target_uid']
                        proc.stdin.write((json.dumps(sample,sort_keys=True,allow_nan=False)+'\n').encode());groups+=1;candidate_rows+=len(sample['axis'])
                proc.stdin.close()
                if proc.wait():raise RuntimeError('current axis compressor failure')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
        write_json(prefix+'/seal.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_CURRENT_AXIS_JOINT_C0_RUNTIME','event':event,
            'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],'input_SHA':protocol['input_SHA'],
            'candidate_index_SHA':index_SHA,'reference_full_C0_seal_SHA':sha256(old_path),'full_C0_states_outputs_exactly_reproduced':True,
            'full_original_frames':len(frames),'sampled_current_groups':groups,'candidate_plus_NONE_rows':candidate_rows,
            'artifacts':[{'path':str(destination),'sha256':sha256(destination),'kind':'current_axis'}],
            'runtime_GT_input':False,'runtime_future_GT_input':False,'GT_history_replacement':False,
            'initial_single_click_only':True,'seconds':time.monotonic()-began,'not_model_fit_or_online_identity_success':True})
        print(json.dumps({'actual_current_axis_episode':event['episode_uid'],'groups':groups,'candidate_NONE_rows':candidate_rows,
            'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--sequence');a=p.parse_args()
    if a.freeze:freeze()
    else:run(a.sequence)
