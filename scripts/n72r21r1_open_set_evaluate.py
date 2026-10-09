"""All actual own-policy E0-E6 seals before GT, then common full TrackEval."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,HISTORY,TRAIN,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_open_set_pilot import PROTOCOL
from scripts.n72r21r1_learned_evaluate import offline,aggregate,artifact
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.one_click_protocol import open_set_metrics,ratio
from sam3_intermot.evaluation.safe_intervention_events import contiguous_intervals


def verify():
    p=read_json(PROTOCOL);seals={}
    assert all(sha256(ROOT/k)==v for k,v in p['source_code_SHA'].items())
    for case in p['cases']:
        for sequence in p['sequences']:
            path=OUT/'learned_pilot/runtime_seals/CLICK_C0'/(sequence+'.json') if case['case']=='CLICK_C0' else OUT/'availability/current_axis_pilot/runtime_seals'/case['case']/(sequence+'.json')
            r=read_json(path)
            if case['case']!='CLICK_C0':
                assert r['protocol_SHA']==sha256(PROTOCOL) and r['source_code_SHA']==p['source_code_SHA'] and r['inputs_SHA']==p['input_SHA']
            assert all(sha256(a['path'])==a['sha256'] for a in r['artifacts'])
            assert not r['runtime_GT_input'] and not r['runtime_future_GT_input'] and not r['history_rewritten']
            assert r['full_global_exact_unique_candidate_ownership'] and r['extra_clicks']==0
            seals[case['case'],sequence]=r
    assert len(seals)==40
    return p,seals


def claim_diagnostics(p,seals):
    truth={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']};result={}
    for sequence in p['sequences']:
        frames,_=checked_frames(sequence);gt=dancetrack_annotations(TRAIN/sequence)
        matched={int(payload['frame']):strict_candidate_matching(rows,gt.get(int(payload['frame']),[])) for payload,rows in frames}
        for case in p['cases']:
            if case['case']=='CLICK_C0':continue
            seal=seals[case['case'],sequence];target=truth[seal['event']['episode_uid']]
            trace=read_zstd_jsonl(artifact(seal,'trace'));counts=Counter();records=[];availability=[];available_truth=[];rank=[];claims=[];claim_frames=[];claim_risks=[];takeovers=[]
            for r in trace:
                if r['frame']<=seal['event']['frame']:continue
                authority=r['authority'];claim=authority['selected_identity_claim'];uid=claim['candidate_uid'];axis=matched[r['frame']]
                available=target in axis.values();outcome='CORRECT_NONE' if uid is None and not available else 'INCORRECT_NONE' if uid is None else 'UNKNOWN' if axis.get(uid) is None else 'TARGET' if axis[uid]==target else 'VERIFIED_OTHER'
                correct=outcome in ('TARGET','CORRECT_NONE');counts['frames']+=1;counts['available_frames']+=available
                counts['accepted_identity_claims']+=claim['accepted'];counts['correct_target_identity_claims']+=claim['accepted'] and outcome=='TARGET'
                counts['wrong_or_UNKNOWN_identity_claims']+=claim['accepted'] and not correct;counts['accepted_'+outcome]+=claim['accepted']
                counts['identity_claims_blocked_by_authority']+=claim['accepted'] and bool(authority['protection_reasons'])
                counts['actually_committed_new_actions']+=authority['approved']
                availability.append(claim['prediction']['available']);available_truth.append(available);rank.append(outcome=='TARGET')
                claims.append(claim['score']);claim_frames.append(r['frame']);claim_risks.append(not correct)
                if claim['accepted'] and outcome=='VERIFIED_OTHER':takeovers.append(r['frame'])
                records.append((claim['score'],not correct,claim['accepted'],outcome))
            curve=[]
            for cutoff in (.5,.8,.9,.95,.98):
                selected=[r for r in records if r[0]>=cutoff]
                curve.append({'current_score_cutoff':cutoff,'selected_frames':len(selected),'coverage':len(selected)/len(records),
                    'wrong_or_UNKNOWN_risk':ratio(sum(r[1] for r in selected),len(selected)),
                    'target_recall_given_available':ratio(sum(r[3]=='TARGET' for r in selected),counts['available_frames']),
                    'score_only_diagnostic_ignores_margin_UNKNOWN_and_authority_filters':True})
            # Claim ranking curve is distinct from actual emitted MOT target UID.
            result[case['case']+'/'+sequence]={'counts':dict(counts),'accepted_claim_risk':ratio(counts['wrong_or_UNKNOWN_identity_claims'],counts['accepted_identity_claims']),
                'accepted_target_claim_recall_given_available':ratio(counts['correct_target_identity_claims'],counts['available_frames']),
                'separate_candidate_availability_head':open_set_metrics(availability,available_truth,rank),
                'identity_score_curve':open_set_metrics(claims,available_truth,rank),'all5_score_risk_coverage_points':curve,
                'accepted_verified_other_claim_intervals':contiguous_intervals(takeovers),
                'uncommitted_identity_claim_not_emitted_MOT_output':True,'curves_diagnostic_not_outer_threshold_selection':True,
                'UNKNOWN_counted_in_unverified_risk_not_verified_other_FPR':True}
    return result


def run():
    p,seals=verify();stats,detection=offline(p,seals);claims=claim_diagnostics(p,seals)
    directory=ASSETS/'availability/current_axis_pilot/trackers/CLICK_C0/data';directory.mkdir(parents=True,exist_ok=True)
    for sequence in p['sequences']:
        source=artifact(seals['CLICK_C0',sequence],'trajectory');link=directory/(sequence+'.txt')
        if link.exists():assert link.is_symlink() and link.resolve()==source.resolve()
        else:link.symlink_to(source)
    storage(64<<20);destination=ASSETS/'availability/current_axis_pilot/trackeval_v1'
    if destination.exists():raise FileExistsError('preserve current-axis full TrackEval attempt')
    destination.mkdir(parents=True);seqmap=destination/'seqmap.txt'
    with seqmap.open('x') as f:f.write('name\n'+'\n'.join(p['sequences'])+'\n')
    pinned=HISTORY/'third_party/MOTIP/TrackEval';commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()
    assert commit=='12c8791b303e0a0b50f753af204249e622d0281a'
    cases=[c['case'] for c in p['cases']];command=_trackeval_command(ASSETS/'availability/current_axis_pilot/trackers',destination,cases,seqmap,gt_split='train',gt_folder=TRAIN)
    command[2]=str(pinned/'scripts/run_mot_challenge.py');log=destination/'trackeval.log';began=time.monotonic()
    with log.open('x') as stream:actual=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    receipt={'command':command,'returncode':actual.returncode,'seconds':time.monotonic()-began,'log_path':str(log),'log_SHA':sha256(log),
        'pinned_TrackEval_commit':commit,'same_settings_all20_cases':True,'wrapper_SHA':sha256(ROOT/'scripts/n72r20r3r2r3_trackeval_entry.py'),'third_party_modified':False}
    write_json('availability/current_axis_pilot/TRACKEVAL_INVOCATION.json',receipt)
    if actual.returncode:raise RuntimeError('current-axis full TrackEval failed; retained log')
    metrics={c:trackeval_summary(parse_trackeval(destination,c,p['sequences'])) for c in cases}
    assert all(all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')) for m in metrics.values())
    groups=aggregate(p,stats,metrics)
    result={'stage':'N72R21R1','status':'COMPLETE_ACTUAL_CURRENT_AXIS_FULL_JOINT_OPEN_SET_EVALUATION',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','new_actual_full_joint_runs':38,'reused_actual_baseline_runs':2,
        'protocol_SHA':sha256(PROTOCOL),'statistics':stats,'TrackEval':metrics,'all_seed_groups':groups,'identity_claim_diagnostics':claims,
        'detection_multiset_audit':detection,'invocation':receipt,'evaluator_source_SHA':sha256(__file__),
        'all9_heads_3diagnostics_6simple_retained':True,'no_VAL_TEST_confirmation':True,'next_stage_authorized':False,
        'current_identity_calibration_not_future_safety_guarantee':True,'whole_goal_scientific_decision':'PENDING'}
    write_json('availability/current_axis_pilot/RESULT.json',result)
    update_status(current_axis_full_joint_evaluation_complete=True,actual_current_axis_new_joint_runs=38,
        current_axis_direction_passing_groups=sum(r['direction_pass'] for r in groups.values()),next_stage_authorized=False)
    print(json.dumps({'actual_E0_E6_TrackEval_new_runs':38,'groups_direction_pass':sum(r['direction_pass'] for r in groups.values())}),flush=True)


if __name__=='__main__':run()
