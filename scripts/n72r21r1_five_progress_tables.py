"""Five required tables from sealed actual results, with unresolved cells explicit.

No best-outer-policy selection, no inferred causal-root counts or missing MOT
scores. These progress tables do not close the scientific Goal.
"""
from collections import Counter
import json
import numpy as np
from scripts.n72r21r1_common import OUT,read_json,write_json,sha256,update_status


def source(relative):
    path=OUT/relative
    return read_json(path),{'path':str(path),'sha256':sha256(path)}


def count_row(label,groups,stats,metrics):
    seq=['dancetrack0001','dancetrack0002'];counts=Counter();right=visible=0;IDs=set();harm_episodes=0
    for s in seq:
        for c in groups:
            r=stats[c+'/'+s];events=r.get('target_events',{})
            counts['ChangedEvents_frame_decisions']+=r['effective_interventions']/len(groups)
            counts['N01_frames']+=events.get('N01_frames',r.get('N01',0))/len(groups)
            counts['N10_frames']+=events.get('N10_frames',r.get('N10',0))/len(groups)
            harm=events.get('direct_harmful_decisions_against_own_KEEP',r.get('direct_harmful_own_KEEP_decisions',0))
            harm_episodes+=bool(harm)/len(groups)
            if 'counts' in r:right+=r['counts'].get('TARGET',0)/len(groups);visible+=r['counts'].get('target_visible',0)/len(groups)
            else:right+=r['strict_correct_UID_frames']/len(groups);visible+=r['visible_frames']/len(groups)
            for pid in r.get('distinct_affected_other_public_IDs',r.get('distinct_affected_non_target_public_IDs',[])):IDs.add((s,pid))
    return {'Method':label,**dict(counts),'HarmEpisodes_mean_seed':harm_episodes,'DistinctAffectedIDs_sequence_public_union':len(IDs),
        'Recall_strict_UID_visible':right/visible if visible else None,
        'counts_unit':'Mean seeds inside scene then sum2 exposed TRAIN scenes; ChangedEvents are actual changed frame decisions, not independent causal roots',
        'all9_combined_TrackEval_mean_seed':{k:float(np.mean([metrics[c][k] for c in groups])) for k in ('HOTA','AssA','IDF1','DetA','IDSW','LocA','MOTA','FP','FN')}}


def run():
    fixed,fsha=source('pilot/FIXED_GATE_RESULT.json');learned,lsha=source('learned_pilot/RESULT.json');memory,msha=source('memory/pilot/RESULT.json')
    open_set,esha=source('availability/current_axis_pilot/RESULT.json');gates,bsha=source('trajectory_gate/RESULT.json')
    frames,asha=source('diagnostics/POSTHOC_FRAME_EVENTS.json');branches,csha=source('diagnostics/COUNTERFACTUAL_ROLLBACK.json')
    whole,wsha=source('data/MULTITARGET_FULL_TRACKEVAL_RESULT_V1.json');density,dsha=source('data/DENSITY_MASKED_TRACKEVAL_RESULT_V1.json')
    def is_N10(r):return r['C0_correct'] and not r['actual_correct']
    def direct(r):return r['effective_action'] and r['own_KEEP_correct'] and not r['actual_correct']
    filters=[('WrongCandidate',lambda r:r['actual_uid'] is not None and r['actual_identity_outcome']!='TARGET'),
        ('IncorrectNONE',lambda r:r['actual_uid'] is None and bool(r['positive_candidate_uids'])),
        ('Takeover_verified_OTHER_subset',lambda r:r['actual_identity_outcome']=='VERIFIED_OTHER'),
        ('StateDrift_observed_wrong_own_KEEP_not_new_current_direct_harm',lambda r:is_N10(r) and not r['own_KEEP_correct']),
        ('BirthDeath_cooccurrence_not_causal_attribution',lambda r:bool(r['births'] or r['deaths'])),
        ('CandidateMissing',lambda r:not r['positive_candidate_uids'])]
    table1=[]
    for label,predicate in filters:
        chosen=[(case,r) for case,rs in frames['cases'].items() for r in rs if predicate(r)];ids=set()
        for case,r in chosen:
            for pid in r['affected_non_target_public_ids']:ids.add((case.split('/')[1],pid))
        table1.append({'FailureType':label,'DirectEvents_frame_decisions_not_independent_roots':sum(direct(r) for _,r in chosen),
            'Observed_N10_frames':sum(is_N10(r) for _,r in chosen),'PropagatedFrames_observed_N10_wrong_own_KEEP':sum(is_N10(r) and not r['own_KEEP_correct'] for _,r in chosen),
            'AffectedIDs_unique_sequence_public_union':len(ids),'matching_postclick_frames':len(chosen),
            'Cause':'Selected first-onset paired branches below prove forced wrong edges propagate; row membership is observational and overlaps other categories.'})
    selected_proofs=[]
    for key,e in branches['events'].items():
        b=e['branches'].get('FORCED_ORIGINAL_PROPOSAL');k=e['branches'].get('KEEP_WITH_ACTOR')
        if b and k:selected_proofs.append({'event':key,'time_zero':b['time_zero'],'H100':b['future']['H100'],
            'KEEP_H100_target_correct':k['future']['H100']['target_correct'],'causal_scope':e['causal_scope']})
    table2=[];fixedmetrics=fixed['all_actual_TrackEval_metrics_0to1'];fixedstats=fixed['offline_target_event_statistics']
    for name,prefix in [('C0','CLICK_C0'),('Confidence','CONFIDENCE_P0_BROAD'),('Recovery','RECOVERY_P0_BROAD'),('GlobalRegret','GLOBAL_REGRET_P0_BROAD')]:
        cases=[c for c in fixedmetrics if c==prefix or c.startswith(prefix+'_SEED')]
        table2.append(count_row(name,cases,fixedstats,fixedmetrics))
    for name,result,match in [('RiskLearned_H100',gates,lambda c:c.startswith('RISK__GLOBAL_RISK')),
                              ('TwoBranch_H5_diagnostic',gates,lambda c:c.startswith('TWO_BRANCH__LOGISTIC__H5')),
                              ('OpenSet_LOGISTIC_unconstrained_diagnostic',open_set,lambda c:c.startswith('LOGISTIC__UNCONSTRAINED'))]:
        cases=[c for c in result['TrackEval'] if match(c)];assert len(cases)==3
        table2.append(count_row(name,cases,result['statistics'],result['TrackEval']))
    table3=[{'Method':'CLICK_C0','metrics':fixedmetrics['CLICK_C0'],'scope':'Full original2 TRAIN sequences combined'}]
    for name,result,cases in [('OriginalACIB',fixed,[c for c in fixedmetrics if c.startswith('ORIGINAL_ACIB_FULL_SEED')]),
        ('JointTrained_H100_LOGISTIC',learned,[c for c in learned['TrackEval'] if c.startswith('LOGISTIC__MIXED__H100_GLOBAL_RISK')]),
        ('SafeMemoryOnly_consensus_shadow',memory,['CONSENSUS__SHADOW__K8__attention__DEFAULT__seedNone'])]:
        metrics=result.get('TrackEval',result.get('all_actual_TrackEval_metrics_0to1'));assert cases
        table3.append({'Method':name,'metrics':{k:float(np.mean([metrics[c][k] for c in cases])) for k in ('HOTA','AssA','IDF1','DetA','IDSW','LocA','MOTA','FP','FN')},
            'scope':'Full original2 TRAIN sequences combined, mean seeds; no winner selected by outer effect'})
    for label in ('BestSimple','BestFull'):
        table3.append({'Method':label,'metrics':None,'status':'PENDING_EXPLICIT_INNER_ONLY_SELECTION_NOT_BEST_OUTER_EFFECT'})
    table4=[]
    for family in ('FROZEN','CONSENSUS','DELAYED','RISK','ROLLBACK'):
        cases=[c for c in memory['TrackEval'] if c.startswith(family+'__COUPLED__K8__attention__DEFAULT')];assert cases
        row=count_row(family,cases,memory['statistics'],memory['TrackEval']);writes=wrong=correct=eligible=0
        for c in cases:
            for sequence in ('dancetrack0001','dancetrack0002'):
                m=memory['statistics'][c+'/'+sequence]['memory'];writes+=m['accepted']/len(cases)
                wrong+=(m['verified_other']+m['UNKNOWN'])/len(cases);correct+=m['correct']/len(cases);eligible+=m['eligible_available_target_candidates']/len(cases)
        table4.append({'Policy':family,'Writes_mean_seed_sum_scene':writes,'WrongRate_with_UNKNOWN':wrong/writes if writes else None,
            'Retention_candidate_available':correct/eligible if eligible else None,'N10_frames':row['N10_frames'],
            'HOTA_combined_mean_seed':row['all9_combined_TrackEval_mean_seed']['HOTA'],
            'zero_writes_fail_usefulness_and_have_undefined_risk':not writes,'frozen_authority_coupled_not_safe_memory_only':True})
    baseline='JOINT_BASELINE_SHADOW_P0';treatment='JOINT_ORIGINAL_TREATMENT_FULL';table5=[]
    table5.append({'Split':'Historical2_TRAIN_full_combined','baselineHOTA':fixedmetrics['CLICK_C0']['HOTA'],
        'treatmentHOTA':float(np.mean([fixedmetrics[c]['HOTA'] for c in fixedmetrics if c.startswith('ORIGINAL_ACIB_FULL_SEED')])),'scope':'Combined2 full sequences, mean3 seeds'})
    # Whole8 source macro is supplied separately because it is not the same
    # estimator as the2-sequence combined number. No unnamed averaging units.
    table5.append({'Split':'Additional_historical8_TRAIN_multi_target_full_macro','metrics':whole.get('aggregate',whole.get('aggregate_by_source')),
        'scope':'22 valid clicks averaged inside each of8 historical scenes, then macro scenes; not fresh8 confirmation'})
    for stratum in ('sparse','medium','crowded'):
        d=density['strata'][stratum];bm=d['aggregate'][baseline]['macro_supported_sequence_mean'];tm=d['aggregate'][treatment]['macro_supported_sequence_mean']
        table5.append({'Split':stratum,'baselineHOTA':bm['HOTA'],'treatmentHOTA':tm['HOTA'],'deltaAssA':tm['AssA']-bm['AssA'],'deltaIDF1':tm['IDF1']-bm['IDF1'],
            'takeover_and_other_actual_frame_counts':d['actual_density_frame_target_statistics'],'supported_sequences':d['supported_sequences'],
            'scope':'POSTHOC_DENSITY_MASKED_DIAGNOSTIC_NOT_FULL_POLICY_SCORE; original frame numbers and public IDs kept; empty-stratum scenes excluded explicitly'})
    table5.append({'Split':'FrozenFreshConfirmation8','baselineHOTA':None,'treatmentHOTA':None,'deltaAssA':None,'deltaIDF1':None,'takeover':None,
        'status':'NOT_RUN_GATED_NO_GENERALIZATION_CLAIM'})
    result={'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','status':'COMPLETE_FIVE_PROGRESS_TABLES_WITH_EXPLICIT_PENDING_INNER_SELECTION',
        'Table1_FAILURE_DECOMPOSITION':table1,'Table1_SELECTED_COUNTERFACTUAL_PROOFS':selected_proofs,
        'Table2_INTERVENTION_SAFETY':table2,'Table3_GLOBAL_MOT_EFFECT':table3,'Table4_MEMORY_SAFETY':table4,'Table5_GENERALIZATION':table5,
        'source_results':[fsha,lsha,msha,esha,bsha,asha,csha,wsha,dsha],
        'all6_scene_seed_original_postclick_N10_frames':sum(is_N10(r) for rs in frames['cases'].values() for r in rs),
        'all6_scene_seed_direct_harm_frame_decisions':sum(direct(r) for rs in frames['cases'].values() for r in rs),
        'failure_categories_overlap_not_a_partition':True,'causal_scope':'15 selected actual paired events, not a causal decomposition of every103 direct decisions or all1949 N10 frames',
        'best_method_selection_complete':False,'scientific_decision':'PENDING','Goal_complete':False,'next_stage_authorized':False,'source_builder_SHA':sha256(__file__)}
    write_json('tables/FIVE_PROGRESS_TABLES_V1.json',result)
    update_status(five_progress_tables_written=True,best_method_INNER_selection_complete=False)
    print(json.dumps({'five_progress_tables':True,'direct_frame_decisions':result['all6_scene_seed_direct_harm_frame_decisions'],
        'N10_frames':result['all6_scene_seed_original_postclick_N10_frames'],'pending_best_selection_explicit':True}),flush=True)


if __name__=='__main__':run()
