"""C1/C6 posthoc oracle-state contrast and immutable current-cache repair.

Oracle GT feedback is explicitly diagnostic and cannot enter an online MOT
result. It occurs AFTER current prediction, uses only a real strictly matched
current candidate, and never fabricates a missing crop or rewrites trajectories.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,TRAIN,read_json,write_json,sha256,storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r21r1_memory_label import verify_all
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21_t2_replay import DecisionCapture
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from sam3_intermot.one_click.acib_runtime import Evidence,unit
from sam3_intermot.one_click.committed_identity_memory import CommittedIdentityMemory,MemoryCommitPolicy
from sam3_intermot.one_click.committed_memory_bridge import CommittedMemoryMOTBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.evaluation.one_click_protocol import open_set_metrics
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching

PROTOCOL=OUT/'protocol/TEACHER_STATE_DIAGNOSTIC_V1.json'
MODES=('ACTUAL_JOINT_C0','TARGET_ONLY_P0_OWN','POSTHOC_ORACLE_GT_MOTION_ONLY','POSTHOC_ORACLE_GT_MOTION_AND_POSITIVE_BANK')


def freeze():
    inputs=read_json(OUT/'corpus/RUNTIME_INPUTS.json');curriculum=read_json(OUT/'protocol/COMMITTED_MEMORY_CURRICULUM_V1.json')
    files=['scripts/n72r21r1_teacher_diagnostic.py','scripts/n72r21r1_memory_label.py',
        'sam3_intermot/one_click/committed_identity_memory.py','sam3_intermot/one_click/committed_memory_bridge.py',
        'sam3_intermot/one_click/acib_trusted_runtime.py','sam3_intermot/one_click/acib_runtime.py']
    write_json('protocol/TEACHER_STATE_DIAGNOSTIC_V1.json',{'stage':'N72R21R1','final_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'frozen_before_new_diagnostic_effects':True,'modes':list(MODES),'events':curriculum['events'],'planned_episodes':22,
        'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),'identity_model_source':inputs['frozen_identity_model'],
        'reference_curriculum_SHA':sha256(OUT/'protocol/COMMITTED_MEMORY_CURRICULUM_V1.json'),
        'source_code_SHA':{p:sha256(ROOT/p) for p in files},'oracle_state_label':'POSTHOC_ORACLE_DIAGNOSTIC_NOT_ONLINE_MOT',
        'oracle_rule':'After scoring frame f, update next-frame motion and optionally bank with the current real strictly GT-matched positive; no current/future GT input to scoring, no GT-created box, no online result or deployment claim.',
        'teacher_bank_capacity':8,'no_GT_feedback_to_OWN_or_actual_C0_state':True,
        'metadata_repair':'Capture C0 model probabilities immediately after current step, before any diagnostic branch. Compare original22 source trace outputs/committed features byte values and write a separate current-probability correction sidecar; original source SHA never changed.',
        'metrics':'Same-person versus verified-other hard-negative margin, rank1, candidate-relative availability/NONE and uncertainty. UNKNOWN remains distinct. Not TrackEval or a mathematical representation ceiling.',
        'max_seconds_per_sequence':1800,'threads_per_worker':1,'max_CPU_workers':2,
        'no_VAL_TEST_confirmation':True,'new_training':False,'next_stage_authorized':False})


def capture_probabilities(model,rows):
    result=np.array(model.last['joint_probabilities'][0].detach().cpu().numpy(),copy=True)
    if len(result)!=max(1,len(rows))+1 or not np.isfinite(result).all():raise ValueError('current candidate/NONE axis')
    return result


def oracle_feedback(actor,frame,rows,positive_uid,prior_motion,*,write):
    if actor.last_frame!=frame:raise ValueError('oracle diagnostic feedback only after current prediction')
    row=next((r for r in rows if str(r['candidate_uid'])==positive_uid),None)
    if positive_uid is not None and row is None:raise ValueError('cannot fabricate missing positive crop')
    actor.pending=None
    if row is None:actor.last_box,actor.last_accept=prior_motion;return
    actor.last_box=list(row['box_xyxy']);actor.last_accept=frame
    if write:
        vector=unit(row['feature']);vector.setflags(write=False)
        actor.bank.append(Evidence(vector,actor.recording,frame,actor.camera,frame/actor.fps,1.,
            'POSTHOC_ORACLE_PREVIOUS_GT_POSITIVE_NOT_ONLINE',float(row['conf']),float(vector@actor.anchor),str(positive_uid)))
        actor.bank=actor.bank[-8:]


def summary(probabilities,rows,matched,target,decision,bank_size):
    scores=probabilities[:len(rows)];uids=[str(r['candidate_uid']) for r in rows]
    ordered=sorted(range(len(rows)),key=lambda i:(-float(scores[i]),uids[i]));positive=[i for i,u in enumerate(uids) if matched.get(u)==target]
    negative=[i for i,u in enumerate(uids) if matched.get(u) not in (None,target)]
    rank=next((j+1 for j,i in enumerate(ordered) if i in positive),None)
    margin=max((float(scores[i]) for i in positive),default=0.)-max((float(scores[i]) for i in negative),default=0.)
    positive_uid=uids[positive[0]] if positive else None
    selected_uid=decision.get('proposed_candidate_uid',decision['selected_candidate_uid'])
    return {'candidate_available':bool(positive),'positive_uid':positive_uid,'rank1_correct':rank==1,
        'MRR':0. if rank is None else 1./rank,'rank2':rank is not None and rank<=2,'rank3':rank is not None and rank<=3,
        'competitive_verified_other':bool(positive and negative),'hard_negative_margin':margin if positive and negative else None,
        'UNKNOWN_candidates':sum(matched.get(u) is None for u in uids),'NONE_probability':float(probabilities[-1]),
        'candidate_available_probability':float(decision['candidate_available_probability']),
        'rank1_joint_probability':float(max(scores,default=0.)),
        'top1_top2_joint_margin':float(scores[ordered[0]]-scores[ordered[1]]) if len(ordered)>1 else None,
        'joint_entropy':float(-np.sum(probabilities[probabilities>0]*np.log(probabilities[probabilities>0]))),
        'selected_uid':selected_uid,'bank_size_before_prediction':bank_size,
        'selected_correct':bool(selected_uid is not None and matched.get(selected_uid)==target),
        'no_target_GT_created_candidate':True}


def run(sequence):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL);assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    _,sources=verify_all();source_by_uid={s['event']['episode_uid']:(p,s) for p,s in sources}
    inputs_path=OUT/'corpus/RUNTIME_INPUTS.json';assert sha256(inputs_path)==protocol['input_SHA'];inputs=read_json(inputs_path)
    source=inputs['frozen_identity_model'];assert sha256(source['path'])==source['sha256'] and sha256(inputs['anchor_path'])==inputs['anchor_sha256']
    saved=torch.load(source['path'],map_location='cpu',weights_only=True);base=ACIBMemoryNetwork().eval();base.load_state_dict(saved['model'],strict=True)
    anchors=np.load(inputs['anchor_path'],mmap_mode='r');frames,index_SHA=checked_frames(sequence)
    # This process is explicitly GT-aware posthoc C1, never a deployable result.
    gt=dancetrack_annotations(TRAIN/sequence);truth={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
    targets={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']}
    for event in [e for e in protocol['events'] if e['sequence']==sequence]:
        prefix='diagnostics/teacher_state_v1/'+event['episode_uid'];seal_path=OUT/prefix/'seal.json'
        if seal_path.exists():
            seal=read_json(seal_path);assert seal['protocol_SHA']==sha256(PROTOCOL)
            assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);continue
        storage(64<<20);assert index_SHA==event['candidate_index_sha256'];target=targets[event['episode_uid']]
        anchor=np.array(anchors[event['anchor_index']],np.float32);assert truth[event['frame']].get(event['clicked_candidate_uid'])==target
        actors={};models={}
        for mode in MODES:
            models[mode]=DecisionCapture(base).eval()
            actors[mode]=CommittedIdentityMemory(models[mode],anchor,event['episode_uid'],write_policy=MemoryCommitPolicy(family='frozen')) if mode=='ACTUAL_JOINT_C0' else TrustedACIBRecognizer(models[mode],anchor,event['episode_uid'],policy='P0',capacity=8)
            actors[mode].start_recording(sequence,fps=event['fps'],width=event['width'],height=event['height'],initial_frame=event['frame'],initial_box=event['box_xyxy'])
        bridge=CommittedMemoryMOTBridge({'event_frame':event['frame'],'human_anchor':anchor,'target_candidate_uid':event['clicked_candidate_uid'],
            'target_box_xyxy':event['box_xyxy']},actors['ACTUAL_JOINT_C0'],policy=GatePolicy(family='shadow'),frames=len(frames));bridge.configure_fps(event['fps'])
        source_path,reference=source_by_uid[event['episode_uid']]
        old=read_zstd_jsonl(Path(next(a['path'] for a in reference['artifacts'] if a['kind']=='trace')))
        assert [r['frame'] for r in old]==list(range(len(frames)))
        destination=ASSETS/'diagnostics/teacher_state_v1'/(event['episode_uid']+'.jsonl.zst');destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise FileExistsError('preserve partial teacher diagnostic')
        began=time.monotonic();corrections=[];count=0;all_committed_features_equal=True
        with destination.open('xb') as stream:
            compressor=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
            try:
                for payload,rows in frames:
                    if time.monotonic()-began>protocol['max_seconds_per_sequence']:raise RuntimeError('teacher diagnostic time cap')
                    f=int(payload['frame']);actual=bridge.step(f,rows)
                    for field in ('outputs','target_uid','state_before','state_after','selected_action'):
                        assert actual[field]==old[f][field],(event['episode_uid'],f,field)
                    if f<=event['frame']:continue
                    assert actual['committed_observation_features']==old[f]['committed_observation_features'],(event['episode_uid'],f,'writer_features')
                    # Copy NOW: subsequent diagnostic calls must not reinterpret
                    # a shared mutable last-output cache as the current frame.
                    current_probabilities=capture_probabilities(models['ACTUAL_JOINT_C0'],rows)
                    current=summary(current_probabilities,rows,truth[f],target,actual['identity_decision'],0)
                    current['actual_committed_uid']=actual['target_uid'];current['joint_effective_actions']=0
                    modes={'ACTUAL_JOINT_C0':current}
                    if old[f]['cached_rank1_joint_probability']!=current['rank1_joint_probability']:
                        corrections.append({'frame':f,'old_cached_rank1_joint_probability':old[f]['cached_rank1_joint_probability'],
                            'corrected_current_rank1_joint_probability':current['rank1_joint_probability'],
                            'source_fields_outputs_states_and_training_features_unchanged':True})
                    positive=current['positive_uid']
                    for mode in MODES[1:]:
                        actor=actors[mode];prior_motion=(None if actor.last_box is None else list(actor.last_box),actor.last_accept)
                        bank_size=len(actor.bank);decision=actor.step(f,rows);probabilities=capture_probabilities(models[mode],rows)
                        modes[mode]=summary(probabilities,rows,truth[f],target,decision,bank_size)
                        if mode.startswith('POSTHOC_ORACLE'):
                            oracle_feedback(actor,f,rows,positive,prior_motion,write=mode.endswith('POSITIVE_BANK'))
                        else:assert not actor.bank
                    record={'sequence':sequence,'episode_uid':event['episode_uid'],'role':event['diagnostic_role'],'frame':f,
                        'time_gap_original_frames':f-event['frame'],'candidate_count':len(rows),
                        'target_visible_GT_diagnostic_only':any(r['identity']==target for r in gt.get(f,[])),
                        'modes':modes,'oracle_modes_not_online_MOT_results':True,
                        'all_original_C0_outputs_and_committed_training_features_exactly_reproduced':True}
                    compressor.stdin.write((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                compressor.stdin.close()
                if compressor.wait():raise RuntimeError('teacher diagnostic compressor failed')
            finally:
                if compressor.poll() is None:compressor.terminate();compressor.wait()
        correction_path=write_json(prefix+'/CURRENT_CACHE_CORRECTIONS.json',{'original_source_seal_SHA':sha256(source_path),
            'diagnostic_only_cache_field':True,'corrections':corrections,'original_seals_traces_and_fits_not_modified':True,
            'all_current_committed_feature_vectors_states_and_outputs_equal':all_committed_features_equal,
            'old_field_not_read_by_memory_label_or_joint_write_train':True})
        write_json(prefix+'/seal.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_POSTHOC_TEACHER_CONTRAST_AND_CACHE_REPLAY',
            'event':event,'protocol_SHA':sha256(PROTOCOL),'source_code_SHA':protocol['source_code_SHA'],'candidate_index_SHA':index_SHA,
            'original_source_seal_SHA':sha256(source_path),'diagnostic_frames':count,'corrected_cache_fields':len(corrections),
            'artifacts':[{'kind':kind,'path':str(path),'sha256':sha256(path)} for kind,path in [('diagnostic',destination),('cache_corrections',correction_path)]],
            'teacher_feedback_uses_previous_real_positive_only':True,'teacher_not_GT_free_online_result':True,
            'C0_full_outputs_states_and_actual_write_training_features_equal':True,'no_VAL_TEST_confirmation_used':True,
            'seconds':time.monotonic()-began,'not_scientific_success':True})
        print(json.dumps({'teacher_diagnostic_episode_complete':event['episode_uid'],'frames':count,
            'cache_metadata_corrections':len(corrections),'seconds':round(time.monotonic()-began,2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--sequence');args=parser.parse_args()
    if args.freeze:freeze()
    else:run(args.sequence)
