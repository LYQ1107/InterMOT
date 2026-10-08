"""Read-only scientific closure analyses of already frozen DEV/VAL execution."""
from collections import Counter,defaultdict
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_fit import causal_features,decode_sequence
from scripts.n72r20r4r1_evaluate import baseline_trace
from scripts.n72r20r4r1_supervision import match_frame
from scripts.n72r20r4r1_analytics import memory_aggregate
from scripts.n72r20r4_train_authority import target_truth
from scripts.n72r20r3_common import gt_by_frame


def opportunity_table():
    path=OUT/'opportunity/FINAL_EVENT_TABLE.json'
    if path.exists():return read_json(path)
    counts=defaultdict(Counter);coverage={};window=defaultdict(list)
    mapping={'T1_RECOVERABLE_WRONG_MATCH':'Wrong Match','T2_RECOVERABLE_NONE':'NONE Recovery',
        'T4_CANDIDATE_MISSING':'Missing Candidate','T5_VALID_REJECTION':'Valid Rejection'}
    for s in SEQUENCES:
        frozen=read_json(OUT/'authority/frozen_outer'/f'{s}.json')
        runtime,audit=causal_features(s,strict_ensemble(s),k=frozen['proposal_K'],heldout=s,role='posthoc_outer')
        er,actions,_=decode_sequence(s);proposed=defaultdict(list)
        for r in runtime:proposed[(r['identity_key'][1],r['frame'])].append(r)
        coverage[s]=audit
        for key,e in er.items():
            families=([mapping[e['kind']]] if e['kind'] in mapping else [])+(['Competitor Owned'] if e['competitor_owned'] else [])
            for family in families:
                c=counts[family];c['total']+=1;c['feasible']+=int(bool(actions[key]));c['runtime_proposed']+=int(bool(proposed[key]))
                c['beneficial']+=int(any(r['H5']['beneficial'] for r in actions[key]));c['harmful']+=int(any(r['H5']['harmful'] for r in actions[key]))
                c['oracle_actions']+=len(actions[key]);c['runtime_actions']+=len(proposed[key])
                c['beneficial_actions']+=sum(r['H5']['beneficial'] for r in actions[key]);c['harmful_actions']+=sum(r['H5']['harmful'] for r in actions[key])
                c['runtime_beneficial_actions']+=sum(r['H5']['beneficial'] for r in proposed[key])
        w=read_json(OUT/'counterfactual/window_trackeval'/f'{s}.json')
        for name,meta in w['windows'].items():
            b=w['metrics']['WINDOW_BASE']['per_sequence'][name];t=w['metrics']['WINDOW_FORCED']['per_sequence'][name]
            window[meta['stratum']].append({k:t[k]-b[k] for k in ('HOTA','AssA','DetA')})
        print(json.dumps({'final_posthoc_event_table':s}),flush=True)
    result={'status':'COMPLETE_POSTHOC_NOT_POLICY_SELECTION','unit':'sampled sequence-identity-frame events; beneficial/harmful may overlap',
        'competitor_owned_is_nonexclusive_subset':True,'feasible_means_any_legal_action_not_guaranteed_correct_recovery':True,
        'missing_candidate_correct_recovery_edges':0,'rows':dict(counts),'selected_inner_K_runtime_coverage':coverage,
        'window_proxy_concordance':{s:{'windows':len(rows),'positive_HOTA_windows':sum(r['HOTA']>0 for r in rows),
            'negative_HOTA_windows':sum(r['HOTA']<0 for r in rows),'mean_window_delta_HOTA':float(np.mean([r['HOTA'] for r in rows]))} for s,rows in window.items()},
        'window_strata_are_H5_proxy_labels_not_population_sample':True,'window_deltas_not_additive_full_MOT':True}
    write_json(path,result);return result


def val_ownership():
    path=OUT/'val/OWNERSHIP_AND_PERSISTENCE.json'
    if path.exists():return read_json(path)
    evaluated=read_json(OUT/'evaluations/VAL_FROZEN.json');formal=read_json(OUT/'val/RESULT.json');result={}
    for s in sorted(events('val')):
        frames=load_frames(s,'val');gt=gt_by_frame(DATASET/'val'/s/'gt/gt.txt');event=events('val')[s];truth=target_truth(event,gt)
        base,_=baseline_trace(s,'val');matches=[match_frame(rows,gt.get(f,[])) for f,(_,rows) in enumerate(frames)];result[s]={}
        for name in ('BASELINE','TREATMENT'):
            manifest=evaluated['manifests'][name][s];delta=manifest['delta']
            if sha256(Path(delta['path']))!=delta['sha256']:raise RuntimeError('VAL ownership SHA changed')
            changes={r['frame']:r['changed_ownership'] for r in read_zstd_jsonl(Path(delta['path']))}
            af=event['event_frame'];ownership={r['candidate_uid']:r['public_id'] for r in base[af]['outputs']};ownership.update(changes.get(af,{}))
            target=ownership[base[af]['target_uid']];observations={};conflicts=preclick=changed=0
            for f,b in enumerate(base):
                owners={r['candidate_uid']:r['public_id'] for r in b['outputs']};owners.update(changes.get(f,{}))
                conflicts+=len(owners)-len(set(owners.values()));changed+=int(bool(changes.get(f)));preclick+=int(bool(changes.get(f)) and f<=af)
                if f<=af:continue
                uid=next((u for u,p in owners.items() if p==target),None)
                observations[f]=(matches[f].get(uid)==truth,matches[f].get(b['target_uid'])==truth)
            N01=sum(t and not b for t,b in observations.values());N10=sum(b and not t for t,b in observations.values())
            if (N01,N10)!=(formal['statistics'][s][name]['target']['N01'],formal['statistics'][s][name]['target']['N10']):raise RuntimeError('VAL independent target counts differ')
            onset=[f for f,(t,b) in observations.items() if t and not b and not (observations.get(f-1,(False,False))[0] and not observations.get(f-1,(False,False))[1])]
            persistent={}
            for h in (5,10,30):
                eligible=[f for f in onset if f+h<len(frames)]
                success=sum(all(observations[f+k][0] and not observations[f+k][1] for k in range(1,h+1)) for f in eligible)
                persistent[f'H{h}']={'episode_onsets_with_full_horizon':len(eligible),'all_future_N01_frames':success,'rate':success/len(eligible) if eligible else None}
            if conflicts or preclick:raise RuntimeError('VAL ownership/preclick invariant failed')
            result[s][name]={'N01':N01,'N10':N10,'global_conflicts':conflicts,'preclick_or_click_changed':preclick,'globally_changed_output_frames':changed,
                'persistence_including_native_only':persistent,'CPU_cached_replay_seconds':manifest['seconds'],'frames':manifest['frames']}
        print(json.dumps({'final_VAL_independent_ownership':s}),flush=True)
    payload={'status':'COMPLETE_POSTHOC_AFTER_FROZEN_ONE_PASS','policy_sha256':formal['policy_sha256'],'sequences':result,'runtime_GT_used':False,
        'CPU_cached_replay_latency_not_live_SAM3_FPS':True,'VAL_tuning':False}
    write_json(path,payload);return payload


def run():
    table=opportunity_table();val_audit=val_ownership();all_dev=read_json(OUT/'AGGREGATED_SCIENTIFIC_EVIDENCE.json')['variants']
    final=all_dev['FINAL_SELECTED'];baseline=all_dev['BASELINE'];v=read_json(OUT/'val/RESULT.json')
    if v['policy_sha256']!=sha256(OUT/'val/FROZEN_POLICY.json'):raise RuntimeError('VAL freeze mismatch')
    val_summary={}
    for name in ('BASELINE','TREATMENT'):
        stats=[r[name] for r in v['statistics'].values()];ownership=[r[name] for r in val_audit['sequences'].values()]
        val_summary[name]={'metrics':{k:x for k,x in v['metrics'][name].items() if k not in ('per_sequence','tracker')},'memory':memory_aggregate(stats),
            'N01':sum(r['target']['N01'] for r in stats),'N10':sum(r['target']['N10'] for r in stats),
            'non_target_damage_proxy':sum(r['target']['non_target_correct_frame_damage'] for r in stats),
            'global_conflicts':sum(r['global_conflicts'] for r in ownership),'globally_changed_output_frames':sum(r['globally_changed_output_frames'] for r in ownership),
            'CPU_cached_replay_seconds':sum(r['CPU_cached_replay_seconds'] for r in ownership),'frames':sum(r['frames'] for r in ownership)}
    try:resource=check_storage()
    except RuntimeError as error:
        # Do not lower the experiment reservation guard. Closing already-run
        # evidence is not permission to start another experiment or cache.
        resource={'experimental_guard_status':'HARD_FLOOR_REACHED_NO_NEW_EXPERIMENTS','guard_error':str(error),
            'free_bytes':shutil.disk_usage(ROOT).free,'own_stage_bytes':sum(p.stat().st_size for p in ASSETS.rglob('*') if p.is_file()),
            'heavy_cache_forbidden':True,'experimental_guard_unchanged':True,'only_existing_evidence_report_and_Git_delivery':True}
    result={'stage':STAGE,'goal':read_json(OUT/'FINAL_GOAL.json')['goal'],'goal_file':'outputs/N72R20R4R1/FINAL_GOAL.json',
        'status':'SCIENTIFIC_EXECUTION_COMPLETE_GIT_DELIVERY_PENDING','final_decision':'FAIL_VAL_GENERALIZATION',
        'secondary_bottlenecks':['FAIL_CONTROLLER_VALUE_GENERALIZATION','FAIL_MEMORY_CONTAMINATION','FAIL_IDENTITY_EVIDENCE_DISCRIMINATION',
            'FAIL_TARGET_CORRECTION_PERSISTENCE','NATIVE_RELIABILITY_P1_TO_P0_STATE_SHIFT','COUNTERFACTUAL_PROXY_NOT_FULL_MOT_OBJECTIVE'],
        'scientific_success':False,'next_stage_authorized':False,'runtime_GT_used':False,'test_accessed':False,'VAL_tuning':False,
        'DEV':{'baseline':baseline,'final_selected':final,'gate':read_json(OUT/'dev_trackeval/FINAL_SELECTED.json')['gate']},
        'VAL':{'summary':val_summary,'paired_bootstrap':v['paired_bootstrap'],'historically_exposed_not_virgin':True,'policy_sha256':v['policy_sha256']},
        'natural_training_events':read_json(OUT/'counterfactual/MULTI_HORIZON_VALUE.json'),
        'checkpoint_manifest_sha256':sha256(OUT/'checkpoints/CHECKPOINT_MANIFEST.json'),'all_R0_R8_completed':True,
        'not_run_branches':[{'branch':'LEARNED_POSITIVE_HISTORY_RELIABILITY','reason':'Zero actual source activation; cannot identify or fabricate supervised samples.',
            'substitute':'Actual positive-soft-zero equivalence and engineering hard-negative preservation tests.'},
            {'branch':'BOTH_LEARNED_NATIVE_AND_POSITIVE_RELIABILITY','reason':'Positive component has no natural active observations; two identified learned components do not exist.',
             'substitute':'Actual native-only/full-joint learned controls plus positive-zero equivalence; do not claim a learned positive component.'}],
        'regression':read_json(OUT/'audit/REGRESSION_RESULTS.json'),'resource_snapshot':resource,
        'opportunity_event_table_sha256':sha256(OUT/'opportunity/FINAL_EVENT_TABLE.json'),
        'VAL_ownership_audit_sha256':sha256(OUT/'val/OWNERSHIP_AND_PERSISTENCE.json')}
    write_json(OUT/'FINAL_RESULT.json',result)
    funnel_report={}
    for name,r in all_dev.items():
        c=r['funnel'];eligible=c.get('eligible_frames',0);proposed=c.get('proposed_actions',0);approved=c.get('authority_approved_frames',0)
        changed=c.get('assignment_changed_frames',0)
        failure=('NO_PROPOSALS' if not proposed else 'NO_GLOBALLY_FEASIBLE_ACTIONS' if not c.get('feasible_actions',0) else
            'NO_CONTROLLER_APPROVALS' if not approved else 'CHANGES_NOT_CORRECTIVE' if r['target']['N01']<=r['target']['N10'] else
            'LOCAL_GAIN_NOT_GLOBAL_HOTA_GAIN' if r['metrics']['HOTA']<=baseline['metrics']['HOTA'] else 'SOME_DEV_GAIN_NOT_RELIABLE_GENERALIZATION')
        funnel_report[name]={'counts':c,'proportions':{'feasible_per_proposed_action':c.get('feasible_actions',0)/proposed if proposed else None,
            'predicted_beneficial_per_proposed_action':c.get('predicted_beneficial_actions',0)/proposed if proposed else None,
            'approved_frames_per_eligible_frame':approved/eligible if eligible else None,'changed_per_approved_frame':changed/approved if approved else None,
            'immediate_recovery_per_changed_action_frame':c.get('correct_target_recovered_frames',0)/changed if changed else None,
            'other_tracks_preserved_per_changed_action_frame':c.get('other_tracks_preserved_frames',0)/changed if changed else None},
            'failure_reason':failure,'independent_future_N01_N10':{k:r['target'][k] for k in ('N01','N10')},'episode_persistence':r['persistence'],
            'all_global_changed_output_frames_including_native_only':r['target']['globally_changed_output_frames'],
            'per_sequence':{s:read_json(OUT/'authority/outer'/f'{s}.json')['statistics'][name]['funnel'] for s in SEQUENCES}}
    write_json(OUT/'authority/FINAL_FUNNEL_DIAGNOSTICS.json',{'units_not_a_single_monotonic_denominator':True,
        'native_reliability_can_change_assignments_without_an_authority_action':True,'variants':funnel_report})
    status=read_json(OUT/'stage_status.json');status.update(status=result['status'],final_decision=result['final_decision'],scientific_success=False,
        scientific_execution_complete=True,goal_complete=False,completed_outer_sequences=list(SEQUENCES),new_VAL_access=True,
        val_accessed_this_stage=True,test_accessed_this_stage=False,next_stage_authorized=False,git_delivery_pending=True)
    status['milestones']={f'M{i}':'COMPLETE_ACTUAL_EXECUTION' for i in range(14)}
    status['milestone_exceptions']={'M7_positive_learned_reliability':'NOT_RUN_ZERO_ACTIVATION_WITH_EXECUTED_EQUIVALENCE_CONTROL',
        'M7_both_learned_reliability':'NOT_RUN_POSITIVE_COMPONENT_UNIDENTIFIABLE_WITH_NATIVE_AND_EQUIVALENCE_CONTROLS'}
    write_json(OUT/'stage_status.json',status)
    print(json.dumps({'final_decision':result['final_decision'],'VAL_N01':val_summary['TREATMENT']['N01'],'VAL_N10':val_summary['TREATMENT']['N10'],'science_success':False}),flush=True)
    return result


if __name__=='__main__':torch.set_num_threads(1);run()
