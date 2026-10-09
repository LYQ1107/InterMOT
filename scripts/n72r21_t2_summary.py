"""Fixed-cohort T2 summaries; incomplete cohorts never become selected models."""
import json
import numpy as np
from scripts.n72r21_common import OUT,read_json,write_json,sha256
from scripts.n72r21_baseline_summary import summarize


def extended(episodes):
    result=summarize(episodes)
    strict=[e['secondary_strict_identity_claim'] for e in episodes]
    correct=sum(e['actual_strict_correct_selected_frames'] for e in strict)
    result['strict_UID_identity_recall_all_visible']=correct/result['visible_frames'] if result['visible_frames'] else None
    result['secondary_calibration_UNKNOWN_frames']=sum(e['calibration_UNKNOWN_excluded_frames'] for e in strict)
    result['secondary_calibration_verified_frames']=sum(e['calibration_verified_frames'] for e in strict)
    result['non_target_or_unverified_write_upper_bound']=result['wrong_writes']
    result['unmatched_write_is_not_verified_other_identity']=True
    result['target_unavailable_false_accept_frames']=sum(round(e['target']['negative_fpr']*(e['target']['frames']-e['target']['candidate_available_frames'])) for e in episodes if e['target']['negative_fpr'] is not None)
    negative=result['frames']-result['candidate_available_frames']
    result['deployed_negative_FPR']=result['target_unavailable_false_accept_frames']/negative if negative else None
    delays=[r['delay_seconds'] for e in episodes for r in e['target']['reacquisition_episodes'] if r['reacquired']]
    result['reacquired_only_median_delay_seconds']=float(np.median(delays)) if delays else None
    result['unsuccessful_reappearances_not_removed_from_reacquisition_recall']=True
    return result


def run():
    protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json');expected=protocol['sequences']
    deploy=read_json(OUT/'protocol/T2_DEPLOYMENT_ABLATIONS.json');results={}
    for name in deploy['cases']:
        per_seed={}
        for seed in deploy['seeds']:
            records={};sources={};case=f'T2_{name}_SEED{seed}'
            for sequence in expected:
                path=OUT/'experiments/T2/evaluations'/case/f'{sequence}.json'
                if not path.exists():continue
                record=read_json(path);sealpath=OUT/'experiments/T2/runtime_seals'/case/f'{sequence}.json'
                if record['runtime_seal_sha256']!=sha256(sealpath):raise ValueError('evaluation source seal changed')
                records[sequence]=record['episodes'];sources[sequence]=sha256(path)
            if records:per_seed[str(seed)]={'per_sequence':{s:extended(e) for s,e in records.items()},
                'pooled':extended([e for episodes in records.values() for e in episodes]),
                'source_evaluation_sha256':sources,'cohort_complete':set(records)==set(expected)}
        if not per_seed:continue
        complete=[s for s in expected if all(str(seed) in per_seed and s in per_seed[str(seed)]['per_sequence'] for seed in deploy['seeds'])]
        metrics={}
        for metric in ('target_recall_all_visible','strict_UID_identity_recall_all_visible','deployed_negative_FPR','reacquisition_recall','wrong_write_rate','correct_observation_retention'):
            eligible=[s for s in complete if all(per_seed[str(seed)]['per_sequence'][s][metric] is not None for seed in deploy['seeds'])]
            values=np.asarray([np.mean([per_seed[str(seed)]['per_sequence'][s][metric] for seed in deploy['seeds']]) for s in eligible])
            ci=None
            if len(values)>1:
                rng=np.random.default_rng(protocol['bootstrap']['seed']);draw=rng.integers(0,len(values),(protocol['bootstrap']['replicates'],len(values)))
                ci=np.quantile(values[draw].mean(1),[.025,.975]).tolist()
            metrics[metric]={'seed_mean_sequence_macro':float(values.mean()) if len(values) else None,
                'sequence_cluster_95pct_CI':ci,'eligible_sequences':eligible,'not_frame_IID':True}
        results[name]={'per_seed':per_seed,'sequences_with_all_three_seeds':complete,'macro_metrics':metrics,
            'complete_registered_cohort':len(complete)==len(expected),'deployment_condition':deploy['cases'][name]}
    write_json('experiments/T2/DEVELOPMENT_SUMMARY.json',{'final_goal_file':'outputs/N72R21/FINAL_GOAL.json',
        'deployment_controls':results,'scope':'Eight historically exposed F1 TRAIN scenes; incomplete cohorts explicit',
        'inference_deletion_and_capacity_state_shift_not_retrained_architecture':True,
        'zero_write_memory_PASS_allowed':False,'no_model_selected_from_outer':True,
        'independent_final_validation':False,'T3_complete':False,'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps({k:{'three_seed_sequences':len(r['sequences_with_all_three_seeds']),
        'macro_recall':r['macro_metrics']['target_recall_all_visible']['seed_mean_sequence_macro']} for k,r in results.items()}))


if __name__=='__main__':run()
