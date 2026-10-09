"""Three actual seeds, paired sequence clusters; not frame-IID or final PASS."""
import json
import numpy as np
from scripts.n72r21_common import OUT,read_json,write_json,sha256
from scripts.n72r21_baseline_summary import summarize


def run():
    protocol=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json');sequences=protocol['sequences'];seeds=protocol['new_training']['seeds']
    by_seed={};sources={};training={}
    for seed in seeds:
        case=f'T0_AMP_R1_P0_SEED{seed}';per_seq={};episodes=[]
        for sequence in sequences:
            source=OUT/'experiments/T0/evaluations'/case/f'{sequence}.json'
            record=read_json(source);per_seq[sequence]=summarize(record['episodes']);episodes+=record['episodes'];sources[f'{case}/{sequence}']=sha256(source)
            fit=read_json(OUT/'training/T0_AMP_R1'/f'{sequence}__seed{seed}.json')
            if not fit['completed'] or sha256(fit['best_checkpoint_path'])!=fit['best_checkpoint_sha256']:raise ValueError('actual completed fit checkpoint')
            selected=next(r for r in fit['logs'] if r['best_selected_on_inner_loss'] and r['epoch']==read_json(OUT/'experiments/T0/runtime_seals'/case/f'{sequence}.json')['selected_best_epoch'])
            training[f'{sequence}__seed{seed}']={'selected_epoch':selected['epoch'],'best_SHA256':fit['best_checkpoint_sha256'],
                    'inner_at_selected':selected['inner'],'model_parameters':fit['model_parameters'],'actual_fit_epochs':len(fit['logs']),
                    'AMP_overflow_events':sum(len(r['fit']['AMP_overflow_events']) for r in fit['logs']),
                    'accepted_gradients_all_finite':all(r['fit']['accepted_optimizer_gradients_finite'] for r in fit['logs'])}
        by_seed[str(seed)]={'pooled':summarize(episodes),'per_sequence':per_seq}
    base=read_json(OUT/'baselines/DEVELOPMENT_SUMMARY.json')['baseline_results']['B0_RAW_ANCHOR']['per_sequence']
    sequence_rates=np.asarray([np.mean([by_seed[str(seed)]['per_sequence'][s]['target_recall_all_visible'] for seed in seeds]) for s in sequences])
    differences=sequence_rates-np.asarray([base[s]['target_recall_all_visible'] for s in sequences])
    rng=np.random.default_rng(protocol['bootstrap']['seed']);draw=rng.integers(0,len(sequences),(protocol['bootstrap']['replicates'],len(sequences)))
    report={'status':'COMPLETE_T0_WITHIN_RECORDING_TRAIN_DIAGNOSTIC_T1_T3_PENDING','final_goal_file':'outputs/N72R21/FINAL_GOAL.json',
            'actual_fold_seed_fits':len(training),'unique_original_target_episodes':52,'repeated_seed_runtime_episodes':156,
            'three_actual_seeds':seeds,'per_seed':by_seed,'actual_training_and_inner_selection':training,'source_evaluation_SHA256':sources,
            'seed_mean_sequence_macro_target_recall':float(sequence_rates.mean()),
            'seed_mean_sequence_cluster_95pct_CI':np.quantile(sequence_rates[draw].mean(axis=1),[.025,.975]).tolist(),
            'paired_sequence_macro_delta_vs_uncalibrated_raw_anchor':float(differences.mean()),
            'paired_sequence_delta_95pct_CI':np.quantile(differences[draw].mean(axis=1),[.025,.975]).tolist(),
            'raw_anchor_and_T0_operating_points_not_equivalently_calibrated':True,
            'no_outer_threshold_architecture_or_checkpoint_selection':True,'sequence_seeds_not_independent_frame_samples':True,
            'historical_TRAIN_exposure_not_virgin_generalization':True,'physical_absence_or_cross_recording_test':False,
            'zero_write_memory_PASS_allowed':False,'T0_only_not_full_ACIB_memory_training':True,
            'scientific_success':False,'next_stage_authorized':False}
    write_json('experiments/T0/DEVELOPMENT_SUMMARY.json',report)
    print(json.dumps({'actual_fits':len(training),'pooled_recall_by_seed':{s:v['pooled']['target_recall_all_visible'] for s,v in by_seed.items()},
                      'sequence_macro_seed_mean_recall':report['seed_mean_sequence_macro_target_recall'],'scientific_success':False}),flush=True)


if __name__=='__main__':run()
