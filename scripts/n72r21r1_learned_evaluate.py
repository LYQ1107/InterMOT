"""Seal all full joint policies, then actual GT/TrackEval and paired diagnostics."""
from collections import Counter
from pathlib import Path
import subprocess
import time
import json
import numpy as np
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,TRAIN,HISTORY,read_json,write_json,sha256,update_status,storage
from scripts.n72r21r1_learned_pilot import PROTOCOL
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.safe_intervention_events import assignment_map,identity_outcome,analyze_target_events,contiguous_intervals
from sam3_intermot.evaluation.joint_trajectory_labels import non_target_damage,memory_safety
from sam3_intermot.evaluation.mot_detection_audit import audit_unchanged_detections
from sam3_intermot.evaluation.one_click_protocol import ratio,open_set_metrics


def verify_all():
    protocol=read_json(PROTOCOL);seals={}
    assert all(sha256(ROOT/p)==digest for p,digest in protocol['source_code_SHA'].items())
    for case in protocol['cases']:
        for sequence in protocol['sequences']:
            path=OUT/'learned_pilot/runtime_seals'/case['case']/(sequence+'.json');seal=read_json(path)
            assert seal['status']=='COMPLETE_ACTUAL_LEARNED_FULL_JOINT_RUNTIME' and seal['protocol_SHA']==sha256(PROTOCOL)
            assert seal['source_code_SHA']==protocol['source_code_SHA'] and seal['inputs_SHA']==protocol['input_SHA']
            assert not seal['runtime_GT_input'] and not seal['runtime_future_GT_input'] and not seal['history_rewritten']
            assert seal['full_global_exact_unique_candidate_ownership'] and seal['extra_clicks']==0
            assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts'])
            if case['case']=='LEARNED_SHADOW':
                repair=read_json(OUT/'learned_pilot/shadow_metadata_repair'/(sequence+'_COMPLETE.json'))
                assert repair['actual_runtime_seal_SHA']==sha256(path)
                receipt_path=OUT/'learned_pilot/shadow_metadata_repair'/(sequence+'.json')
                assert repair['repair_receipt_SHA']==sha256(receipt_path)
                receipt=read_json(receipt_path)
                assert sha256(ROOT/'scripts/n72r21r1_learned_shadow_repair.py')==receipt['repair_source_SHA']
                assert not receipt['checkpoint_selection_changed']
            seals[case['case'],sequence]=seal
    assert len(seals)==120
    return protocol,seals


def artifact(seal,kind):return Path(next(a['path'] for a in seal['artifacts'] if a['kind']==kind))


def offline(protocol,seals):
    truth={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']}
    stats={};detections={}
    for sequence in protocol['sequences']:
        frames,index_SHA=checked_frames(sequence);gt=dancetrack_annotations(TRAIN/sequence)
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        baseline_seal=seals['CLICK_C0',sequence];baseline=read_zstd_jsonl(artifact(baseline_seal,'trace'))
        event=baseline_seal['event'];target=truth[event['episode_uid']];origins={}
        base_public=baseline[event['frame']]['target_public_id']
        for r in baseline:
            for o in r['outputs']:
                identity=matched[r['frame']].get(o['candidate_uid'])
                if identity is not None:origins.setdefault(o['public_id'],identity)
        baseline_text=artifact(baseline_seal,'trajectory').read_text()
        for condition in protocol['cases']:
            case=condition['case'];seal=seals[case,sequence];trace=read_zstd_jsonl(artifact(seal,'trace'))
            assert [r['frame'] for r in trace]==list(range(len(frames)))
            detections[case+'/'+sequence]=audit_unchanged_detections(baseline_text,artifact(seal,'trajectory').read_text())
            if case=='NO_HUMAN_C0':
                stats[case+'/'+sequence]={'target_identity_metrics':'NOT_APPLICABLE_NO_CLICK','cached_joint_seconds':seal['cached_joint_step_seconds']};continue
            public=trace[event['frame']]['target_public_id'];counts=Counter();affected=Counter();observations=[]
            available=[];rank=[];prob=[];known=[];claim=[];correct={};base_correct={};harmful_action=Counter()
            for actual,base in zip(trace,baseline,strict=True):
                f=actual['frame'];am,bm=assignment_map(actual),assignment_map(base);axis=matched[f]
                assert set(am.values())=={str(r['candidate_uid']) for r in frames[f][1]}
                if f<=event['frame']:assert am==bm;continue
                uid=am.get(public);outcome=identity_outcome(uid,axis,target);right=outcome=='TARGET';br=axis.get(bm.get(base_public))==target
                authority=actual['authority'];own_uid=authority.get('own_KEEP_uid',actual['base_assignments'].get(str(public)))
                kr=axis.get(own_uid)==target
                observations.append({'frame':f,'C0_correct':br,'actual_correct':right,'own_KEEP_correct':kr,'effective_action':bool(authority.get('effective_assignment_change'))})
                correct[f]=right;base_correct[f]=br;counts['frames']+=1;counts['candidate_available']+=target in axis.values()
                counts['target_visible']+=any(a['identity']==target for a in gt.get(f,[]));counts['selected']+=uid is not None
                counts[outcome]+=1;counts['unavailable_false_accept']+=target not in axis.values() and uid is not None
                counts['unavailable_frames']+=target not in axis.values()
                damage=non_target_damage(actual,base,axis,origins,base_public,target)
                for p in damage['harmed_public_ids']:affected[p]+=1
                counts['same_target_fragment_displacements']+=len(damage['same_target_fragment_displaced_public_ids'])
                if actual['joint_identity_memory_write']:
                    assert actual['joint_memory_write_candidate_uid']==uid;counts['write_'+outcome]+=1
                decision=actual['identity_decision']
                if decision is not None:
                    ruid=decision['rank1_candidate_uid'];routcome=identity_outcome(ruid,axis,target)
                    available.append(target in axis.values());rank.append(routcome=='TARGET');prob.append(float(decision['candidate_available_probability']))
                    if routcome!='UNKNOWN_UNMATCHED':known.append(routcome=='TARGET');claim.append(actual['cached_rank1_joint_probability'])
                if authority.get('effective_assignment_change'):harmful_action['current_harm']+=kr and not right;harmful_action['current_benefit']+=right and not kr
            events=analyze_target_events(observations);persistence={}
            for h in (5,20,50,100):
                eligible=[f for f in correct if correct[f] and not base_correct[f] and f+h<len(frames)]
                persistence['H'+str(h)]={'overlapping_benefit_frames':len(eligible),'fully_persisted_future_benefit_frames':sum(all(correct[f+k] and not base_correct[f+k] for k in range(1,h+1)) for f in eligible),'not_independent_onsets':True}
            stats[case+'/'+sequence]={'counts':dict(counts),'target_events':events,'effective_interventions':seal['effective_interventions'],
                'current_direct_effects':dict(harmful_action),'strict_UID_recall_visible':ratio(counts['TARGET'],counts['target_visible']),
                'strict_UID_recall_given_candidate':ratio(counts['TARGET'],counts['candidate_available']),
                'wrong_person_takeover_frames':counts['VERIFIED_OTHER'],'UNKNOWN_selected_frames':counts['UNKNOWN_UNMATCHED'],
                'other_person_C0_origin_damage_frames':sum(affected.values()),'distinct_affected_other_public_IDs':sorted(affected),
                'same_target_fragment_displacements':counts['same_target_fragment_displacements'],'non_target_origin_proxy_not_global_IDF1':True,
                'candidate_unavailable_FPR':ratio(counts['unavailable_false_accept'],counts['unavailable_frames']),
                'candidate_relative_availability_curve':open_set_metrics(prob,available,rank) if prob else None,
                'known_identity_claim_curve':open_set_metrics(claim,known,known) if claim else None,
                'correction_persistence':persistence,'memory':memory_safety(counts['write_TARGET'],counts['write_VERIFIED_OTHER'],counts['write_UNKNOWN_UNMATCHED'],counts['TARGET']),
                'cached_efficiency':{k:seal[k] for k in ('cached_joint_step_seconds','cached_total_seconds','cached_joint_FPS_not_pixel_FPS','association_only_seconds','controller_parameter_count','identity_parameter_count','cumulative_process_peak_RSS_bytes')}}
    return stats,detections


def trackeval(protocol):
    pinned=HISTORY/'third_party/MOTIP/TrackEval'
    commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()
    assert commit=='12c8791b303e0a0b50f753af204249e622d0281a'
    root=ASSETS/'learned_pilot/trackeval_v1'
    if root.exists():raise FileExistsError('retain prior TrackEval attempt; explicit versioned repair required')
    storage(64<<20);root.mkdir(parents=True);seqmap=root/'seqmap.txt'
    with seqmap.open('x') as f:f.write('name\n'+'\n'.join(protocol['sequences'])+'\n')
    cases=[c['case'] for c in protocol['cases']]
    command=_trackeval_command(ASSETS/'learned_pilot/trackers',root,cases,seqmap,gt_split='train',gt_folder=TRAIN)
    command[2]=str(pinned/'scripts/run_mot_challenge.py');began=time.monotonic();log=root/'trackeval.log'
    with log.open('x') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    receipt={'command':command,'returncode':result.returncode,'seconds':time.monotonic()-began,
        'log_path':str(log),'log_SHA':sha256(log),'pinned_TrackEval_commit':commit,'same_settings_all60_cases':True,
        'wrapper_SHA':sha256(ROOT/'scripts/n72r20r3r2r3_trackeval_entry.py'),'third_party_modified':False}
    write_json('learned_pilot/TRACKEVAL_INVOCATION.json',receipt)
    if result.returncode:raise RuntimeError('actual full-policy TrackEval failed; log retained')
    metrics={c:trackeval_summary(parse_trackeval(root,c,protocol['sequences'])) for c in cases}
    assert all(all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')) for m in metrics.values())
    return metrics


def aggregate(protocol,stats,metrics):
    groups={}
    for c in protocol['cases']:
        key='__'.join((c['family'],c['state_contrast'],c['reward'])) if 'family' in c else c['case']
        groups.setdefault(key,[]).append(c['case'])
    results={};rng=np.random.default_rng(72114)
    for group,cases in groups.items():
        deltas=[];counts=Counter()
        for seq in protocol['sequences']:
            deltas.append({k:float(np.mean([metrics[c]['per_sequence'][seq][k]-metrics['CLICK_C0']['per_sequence'][seq][k] for c in cases])) for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')})
            for c in cases:
                s=stats[c+'/'+seq]
                if 'target_events' not in s:continue
                counts['N01']+=s['target_events']['N01_frames']/len(cases);counts['N10']+=s['target_events']['N10_frames']/len(cases)
                counts['effective_interventions']+=s['effective_interventions']/len(cases)
                counts['other_person_damage_frames']+=s['other_person_C0_origin_damage_frames']/len(cases)
                counts['wrong_takeover_frames']+=s['wrong_person_takeover_frames']/len(cases)
        bootstrap=np.array([np.mean([deltas[i]['HOTA'] for i in rng.integers(0,len(deltas),len(deltas))]) for _ in range(2000)])
        mean={k:float(np.mean([d[k] for d in deltas])) for k in deltas[0]}
        results[group]={'cases_all_seeds':cases,'mean_seed_summed_scene_counts':dict(counts),
            'macro_sequence_paired_delta':mean,'HOTA_sequence_cluster_95CI':np.quantile(bootstrap,[.025,.975]).tolist(),
            'bootstrap_sequence_clusters':2,'bootstrap_replicates':2000,'bootstrap_seed':72114,
            'repeated_seeds_not_independent_sequences':True,
            'nonvacuity_pass':counts['effective_interventions']>0,
            'direction_pass':counts['effective_interventions']>0 and counts['N01']>counts['N10'] and counts['other_person_damage_frames']==0,
            'numerical_MOT_gate':mean['HOTA']>=.005 and mean['AssA']>0 and mean['DetA']>=-.005 and min(d['HOTA'] for d in deltas)>=-.02,
            'not_generalization_or_final_scientific_PASS':True}
    return results


def run():
    protocol,seals=verify_all()  # No missing jobs can silently disappear.
    stats,detection=offline(protocol,seals);metrics=trackeval(protocol);groups=aggregate(protocol,stats,metrics)
    write_json('learned_pilot/RESULT.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_LEARNED_FULL_MOT_PILOT',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','source_evaluator_SHA':sha256(Path(__file__)),
        'protocol_SHA':sha256(PROTOCOL),'actual_sealed_runs':120,'actual_cases':60,'independent_sequences':2,
        'statistics':stats,'TrackEval':metrics,'all_seed_groups':groups,'detection_multiset_audit':detection,
        'historical_development_NOT_fresh_confirmation':True,'all_comparisons_retained_not_best_effect_selection':True,
        'zero_write_not_memory_usefulness_PASS':True,'next_stage_authorized':False,'whole_goal_scientific_decision':'PENDING'})
    update_status(completed_learned_full_MOT_runs=120,phase_H_learned_full_MOT_evaluation_complete=True,
        next_stage_authorized=False,scientific_decision=None,status='ACTIVE_REQUIRES_ON_POLICY_MEMORY_MULTI_PERSON_AND_FINAL_CONTRACTS')
    print(json.dumps({'actual_full_TrackEval_and_diagnostics_complete':120,'groups':len(groups),
        'groups_direction_pass':sum(g['direction_pass'] for g in groups.values()),'Goal_still_active':True}),flush=True)


if __name__=='__main__':run()
