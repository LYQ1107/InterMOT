"""Assemble complete frozen evidence; never fit, infer, select or rewrite seals.

This script intentionally refuses partial validation. A negative scientific
decision is distinct from Goal completion, which also requires final audits and
an exact, clean Git publication. SOT remains deferred by the latest user.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.n72r21_common import ROOT, OUT, read_json, write_json, sha256, utcnow
from scripts.n72r21_delivery_metadata import reference, view
from scripts.n72r21_validation import cases, case_id
from sam3_intermot.evaluation.delivery_tables import (
    count_metrics, recovery_summary, sequence_interval, seed_average_sequence_interval,
    require_complete_validation, unavailable_cross_recording,
)


def verified_summary_sources(summary, base):
    for case, record in summary.items():
        for sequence, digest in record['source_evaluation_sha256'].items():
            path = OUT / base / case / f'{sequence}.json'
            if sha256(path) != digest: raise ValueError('summary evaluation source changed')
            evaluation = read_json(path)
            seal = OUT / base.replace('evaluations', 'runtime_seals') / case / f'{sequence}.json'
            if sha256(seal) != evaluation['runtime_seal_sha256']: raise ValueError('summary runtime seal changed')


def breakdowns(scope, sequences, registered, bootstrap):
    by_case = {c: {'counts': defaultdict(Counter), 'per_sequence': {}, 'episodes': []} for c in registered}
    sources = {}
    for sequence in sequences:
        path = OUT / 'failure_analysis' / scope / f'{sequence}.json'; record = read_json(path)
        if set(record['cases']) != set(registered): raise ValueError('complete compared breakdown cases required')
        if not record['all_compared_runtimes_verified_before_truth']: raise ValueError('unverified posthoc input')
        for code, digest in record['code_sha256'].items():
            if sha256(ROOT / code) != digest: raise ValueError('frozen breakdown source changed')
        if sha256(OUT / 'protocol/POSTHOC_FAILURE_BREAKDOWNS.json') != record['protocol_sha256']:
            raise ValueError('frozen breakdown protocol changed')
        base = 'validation' if scope == 'VAL' else 'experiments/T2'
        for case, case_record in record['cases'].items():
            if sha256(OUT / base / 'runtime_seals' / case / f'{sequence}.json') != record['runtime_seal_sha256'][case]:
                raise ValueError('breakdown runtime seal changed')
            if sha256(OUT / base / 'evaluations' / case / f'{sequence}.json') != case_record['original_evaluation_sha256']:
                raise ValueError('breakdown evaluated source changed')
            dest = by_case[case]
            dest['per_sequence'][sequence] = case_record['strata']
            for group, r in case_record['strata'].items(): dest['counts'][group].update(r['counts'])
            dest['episodes'].extend(case_record['episodes'])
        sources[sequence] = sha256(path)
    result = {}
    for case, r in by_case.items():
        strata = {}
        for group, counts in r['counts'].items():
            per_sequence = {s: count_metrics(v[group]['counts']) for s, v in r['per_sequence'].items() if group in v}
            metrics = count_metrics(counts)
            metrics['sequence_cluster_uncertainty'] = {
                key: sequence_interval({s: v[key] for s, v in per_sequence.items()}, bootstrap)
                for key in ('target_recall_box_IoU0_5','strict_UID_target_recall','candidate_coverage_given_visible',
                            'strict_recognition_recall_given_available','target_unavailable_FPR_at_frozen_deployment')}
            strata[group] = metrics
        result[case] = {'strata': strata, 'strict_recovery_and_takeover': recovery_summary(r['episodes'])}
    return {'scope': scope, 'cases': result, 'source_breakdown_sha256': sources,
            'stratum_PR_AUC_or_ECE_is_not_obtained_by_averaging_scene_curves': True,
            'full_cohort_pooled_calibration_is_reported_separately': True}


def family_ids(protocol, family):
    return [case_id(family, seed) for seed in (protocol['seeds'] if protocol['cases'][family]['family'] == 'T2' else [None])]


def paired_family_metric(summary, ids, baseline, metric, bootstrap):
    return seed_average_sequence_interval({c: {s: r[metric] - baseline[s][metric] if r[metric] is not None and baseline[s][metric] is not None else None
                                                  for s, r in summary['cases'][c]['per_sequence'].items()} for c in ids}, bootstrap)


def gates(summary, curves, protocol):
    spec = read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    limits = spec['development_effect_gate']; ids = family_ids(protocol, 'ACIB_FULL')
    baseline = summary['cases']['RAW_OSNET_ANCHOR']['per_sequence']
    recall = paired_family_metric(summary, ids, baseline, 'target_recall_all_visible', protocol['bootstrap'])
    reacq = paired_family_metric(summary, ids, baseline, 'reacquisition_recall', protocol['bootstrap'])
    takeover = seed_average_sequence_interval({c: {s: r['verified_wrong_person_takeover_frames']/r['frames'] - baseline[s]['verified_wrong_person_takeover_frames']/baseline[s]['frames']
                                                  for s,r in summary['cases'][c]['per_sequence'].items()} for c in ids}, protocol['bootstrap'])
    macro = summary['families']['ACIB_FULL']['macro_metrics']
    legal_recall = [curves['cases'][c]['primary_candidate_availability']['recall_at_fpr_2pct'] for c in ids]
    memory = {c: {k: summary['cases'][c]['pooled_fixed_operating_point'][k] for k in ('writes','wrong_writes','wrong_write_rate','correct_observation_retention','memory_joint_gate')} for c in ids}
    checks = {
        'box_recall_delta': recall['sequence_macro_mean'] >= limits['target_recall_improvement_vs_frozen_anchor'],
        'paired_sequence_recall_CI_lower': recall['sequence_cluster_95pct_CI'][0] > limits['paired_sequence_ci_lower_gt'],
        'reacquisition_delta': reacq['sequence_macro_mean'] is not None and reacq['sequence_macro_mean'] >= limits['reacquisition_recall_improvement'],
        'maximum_takeover_rate_increase': takeover['sequence_macro_mean'] <= limits['maximum_takeover_rate_increase'],
        'deployed_negative_FPR': macro['deployed_negative_FPR']['sequence_macro_mean'] <= limits['maximum_negative_fpr'],
        'minimum_all_visible_usefulness': macro['target_recall_all_visible']['sequence_macro_mean'] >= spec['minimum_usefulness']['all_visible_target_recall'],
    }
    return {
        'Gate0': {'status':'NOT_PROVEN_END_TO_END_ONLINE_PIXEL_CAUSALITY',
                  'current_past_GT_free_one_click_replay_contract':True, 'additional_runtime_clicks':0,
                  'reason':'Cached SAM3 candidate tapes do not prove upstream future-pixel causality or online latency'},
        'Gate1': {'status':'PASS' if all(checks.values()) else 'FAIL', 'checks':checks,
                  'paired_box_recall_delta':recall, 'paired_reacquisition_delta':reacq,
                  'paired_verified_other_takeover_rate_delta_all_future_frames':takeover,
                  'strict_UID_identity_recall_reported_separately':macro['strict_UID_identity_recall_all_visible'],
                  'frozen_reference':reference('protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')},
        'Gate2': {'status':'PASS' if all(x is not None and x >= spec['minimum_usefulness']['correct_id_recall_at_negative_fpr_2pct'] for x in legal_recall) else 'FAIL',
                  'correct_identity_recall_at_primary_target_unavailable_FPR2_by_seed':dict(zip(ids,legal_recall)),
                  'required_recall':spec['minimum_usefulness']['correct_id_recall_at_negative_fpr_2pct'],
                  'availability_calibration_is_not_correct_identity_calibration':True,
                  'VAL_curve_does_not_change_deployment_threshold':True},
        'Gate3': {'status':'NOT_EVALUABLE_NO_LAWFUL_LOCAL_CROSS_RECORDING_MEDIA', 'actual_episodes':0, 'PASS':False},
        'Gate4': {'status':'PASS' if all(r['memory_joint_gate'] for r in memory.values()) else 'FAIL',
                  'all_three_seed_actual_pooled_write_rates':memory, 'zero_write_PASS_allowed':False,
                  'frozen_limits':spec['memory_joint_gate']},
    }


def run():
    direction = read_json(OUT/'RESEARCH_DIRECTION_OVERRIDE.json')
    if direction['research_primary_track'] != 'INTERACTIVE_MULTI_OBJECT_TRACKING' or direction['new_SOT_download_inference_training_or_resource_retry'] is not False:
        raise ValueError('MOT primary / SOT deferred required')
    protocol = read_json(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json')
    registered = [case_id(n,s) for n,s,_ in cases(protocol)]
    summary = read_json(OUT/'validation/FROZEN_SUMMARY.json')
    require_complete_validation(summary, registered, protocol['sequences'])
    curves = read_json(OUT/'validation/POOLED_CURVES.json')
    if not curves['cohort_complete'] or set(curves['cases']) != set(registered): raise ValueError('complete pooled VAL curves required')
    if summary['protocol_sha256'] != sha256(OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json') or curves['protocol_sha256'] != summary['protocol_sha256']:
        raise ValueError('VAL protocol changed')
    verified_summary_sources(summary['cases'], 'validation/evaluations')
    for axis, digest in curves['runtime_seal_sha256'].items():
        if sha256(OUT/'validation/runtime_seals'/f'{axis}.json') != digest: raise ValueError('pooled curve source changed')
    catalog = read_json(OUT/'checkpoints/SHA256_MANIFEST.json')
    if catalog['actual_checkpoints_verified'] != 248: raise ValueError('actual248 checkpoint audit required')
    for r in catalog['fitted_checkpoints']:
        if not r['strict_state_dict_loaded'] or sha256(r['path']) != r['sha256']: raise ValueError('verified checkpoint changed')
    for r in catalog['fits']:
        if sha256(ROOT/r['fit_record']) != r['sha256']: raise ValueError('actual fit source changed')
    mot = read_json(OUT/'mot_pilot/RESULT.json'); verification = read_json(OUT/'mot_pilot/RESULT_VERIFICATION.json')
    if verification['RESULT_sha256'] != sha256(OUT/'mot_pilot/RESULT.json') or len(verification['all32_sealed_runtime_sources_and_artifacts_verified']) != 32:
        raise ValueError('all32 full MOT verification required')
    val_breakdown = breakdowns('VAL', protocol['sequences'], registered, protocol['bootstrap'])
    development = read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')
    deploy = read_json(OUT/'protocol/T2_DEPLOYMENT_ABLATIONS.json')
    t2 = read_json(OUT/'experiments/T2/DEVELOPMENT_SUMMARY.json')
    if len(t2['deployment_controls']) != 11 or not all(r['complete_registered_cohort'] for r in t2['deployment_controls'].values()):
        raise ValueError('all264 T2 controls required')
    t2_rows = {}
    for name, record in t2['deployment_controls'].items():
        verified_summary_sources({f'T2_{name}_SEED{s}':r for s,r in record['per_seed'].items()}, 'experiments/T2/evaluations')
        t2_rows[name] = {'per_seed_pooled':{s:r['pooled'] for s,r in record['per_seed'].items()},
                         'macro_metrics':record['macro_metrics'], 'deployment_condition':record['deployment_condition']}
    principal = [f'T2_{n}_SEED{s}' for s in deploy['seeds'] for n in ('FULL_K8','ANCHOR_ONLY_P0','WITHOUT_SAFE_WRITE_P1','MEAN_PROTOTYPE_P1')]
    t2_breakdown = breakdowns('T2', development['sequences'], principal, development['bootstrap'])
    baseline = read_json(OUT/'baselines/DEVELOPMENT_SUMMARY.json')
    if len(baseline['baseline_results']) != 9 or not all(r['cohort_complete'] for r in baseline['baseline_results'].values()):
        raise ValueError('all9 historical baseline cohorts required')
    verified_summary_sources(baseline['baseline_results'],'baselines/evaluations')
    baseline_rows = {c:{'pooled':r['pooled'],'sequence_macro_target_recall':r['sequence_macro_target_recall']} for c,r in baseline['baseline_results'].items()}
    val_rows = {c:{'pooled_fixed_operating_point':r['pooled_fixed_operating_point'],
                   'recovered_only_delay_must_not_exclude_failed_returns_from_recall':True,
                   'strict_recovery_and_takeover':val_breakdown['cases'][c]['strict_recovery_and_takeover']} for c,r in summary['cases'].items()}
    sources = ['validation/FROZEN_SUMMARY.json','validation/POOLED_CURVES.json','experiments/T2/DEVELOPMENT_SUMMARY.json',
               'experiments/T2/POOLED_PRINCIPAL_CURVES.json','baselines/DEVELOPMENT_SUMMARY.json','mot_pilot/RESULT.json',
               'mot_pilot/RESULT_VERIFICATION.json','checkpoints/SHA256_MANIFEST.json','RESEARCH_DIRECTION_OVERRIDE.json']
    table_a = {'F1_historically_exposed_eight_TRAIN_baselines':baseline_rows,
               'T2_eight_TRAIN_three_seed_learned_controls':t2_rows,
               'independent_sequence_frozen25_VAL_identity_subsystem':val_rows,
               'VAL_families_and_sequence_cluster_CI':summary['families'],
               'VAL_R3R2_adapter_and_historical_memory':'NOT_REGISTERED_OR_RUN_ON_THIS_FROZEN_VAL; do not transplant TRAIN values',
               'separate_one_target_episodes_are_full_MOT':False}
    table_b = {'frozen25_VAL_actual_pooled_curves_all15_cases':curves['cases'],
               'T2_principal_pooled_curves':read_json(OUT/'experiments/T2/POOLED_PRINCIPAL_CURVES.json')['cases'],
               'availability_AP_ECE_is_not_identity_claim_AP_ECE':True,
               'UNKNOWN_excluded_only_secondary_calibration_not_primary_recall':True,
               'deployed_FPR_is_in_Table_A_and_D_not_replaced_by_diagnostic_curve':True}
    table_c = {'within_video_VAL':val_breakdown,'historically_exposed_TRAIN':t2_breakdown,
               'cross_recording':[unavailable_cross_recording(s) for s in ('CROSS_CAMERA','CROSS_DAY','CROSS_CAMERA_AND_DAY','CROSS_SESSION')],
               'density_and_gt_overlap_are_fixed_posthoc_strata_not_ground_truth_physical_occlusion':True}
    table_d = {'T2_all11_controls_three_seeds':t2_rows,
               'VAL_all15_actual_write_policy_results':{c:r['pooled_fixed_operating_point'] for c,r in summary['cases'].items()},
               'frozen_consensus_controls':{c:r for c,r in baseline_rows.items() if c in ('B4_P0','B4_P1','B4_P4','B4_P6')},
               'zero_write_wrong_rate_is_null_retention_is_zero_and_joint_gate_FAIL':True,
               'unsafe_control_is_not_current_deployable_safe_memory':True}
    table_e = {'research_primary_track':'MOT','scope':'Two registered TRAIN scenes;32 joint full multi-object trajectories;16cases;three learned seeds; NOT VAL generalization',
               'official_metrics_all16_cases':mot['actual_TrackEval_metrics'],
               'deltas_vs_CLICK_C0':mot['deltas_vs_clicked_C0'],
               'target_effect_and_collateral_damage':verification['target_aggregates_all16_cases'],
               'same_numeric_detection_multisets_verified':True,
               'metric_matching_can_change_DetA_LocA_FP_FN_despite_unchanged_detection_boxes':True,
               'diagnostic_findings':verification['diagnostic_findings'],
               'SOT_current_status':'DEFERRED_BY_USER_NO_NEW_WORK',
               'historical_SOT_auxiliary_only':reference('baselines/SOT_DEVELOPMENT_SUMMARY.json')}
    tables = {'Table_A':table_a,'Table_B':table_b,'Table_C':table_c,'Table_D':table_d,'Table_E':table_e}
    view('evaluation/TABLES_A_TO_E.json',{'tables':tables,'assembler_source_sha256':sha256(Path(__file__))},sources)
    for relative, value in {
        'experiments/WITHIN_VIDEO.json':table_a,
        'experiments/REAPPEARANCE.json':{'VAL':{c:r['strict_recovery_and_takeover'] for c,r in val_breakdown['cases'].items()}},
        'experiments/CROSS_CAMERA.json':unavailable_cross_recording('CROSS_CAMERA'),
        'experiments/CROSS_DAY.json':unavailable_cross_recording('CROSS_DAY'),
        'experiments/CROSS_SESSION.json':unavailable_cross_recording('CROSS_SESSION'),
        'experiments/LOW_DENSITY.json':{'VAL':{c:r['strata'].get('DENSITY_sparse') for c,r in val_breakdown['cases'].items()}},
        'experiments/HIGH_DENSITY.json':{'VAL':{c:r['strata'].get('DENSITY_crowded') for c,r in val_breakdown['cases'].items()}},
        'ablations/ANCHOR_VS_BANK.json':{'T2_controls':{k:v for k,v in t2_rows.items() if k in ('FULL_K1','FULL_K4','FULL_K8','ANCHOR_ONLY_P0','MEAN_PROTOTYPE_P1','BANK_ONLY_IDENTITY_P1','UNIFORM_BANK_ATTENTION')},'inference_state_shift_not_separately_retrained_architecture':True},
        'ablations/NONE.json':{'with_vs_without_availability':{k:t2_rows[k] for k in ('FULL_K8','WITHOUT_AVAILABILITY')},'retraining_or_deployment_threshold_change':False},
        'ablations/SAFE_WRITE.json':table_d,
        'ablations/STATE_MATCHING.json':read_json(OUT/'experiments/T1/PAIRED_STATE_CONDITION_EFFECTS.json'),
        'ablations/VISUAL_ENCODER.json':read_json(OUT/'evaluation/FROZEN_ENCODER_CONTROLS.json'),
        'evaluation/TARGET_METRICS.json':table_a,
        'evaluation/OPEN_SET_METRICS.json':table_b,
        'evaluation/LONG_TERM_METRICS.json':table_c,
        'evaluation/MEMORY_SAFETY.json':table_d,
        'evaluation/MOT_SUPPLEMENTARY.json':table_e,
        'evaluation/UNCERTAINTY.json':{'VAL_families':summary['families'],'VAL_paired_recall_vs_anchor':summary['paired_recall_deltas_vs_raw_anchor'],
                                     'bootstrap':protocol['bootstrap'],'unit':'sequence; average three seeds within each sequence',
                                     'MOT_two_scene_formal_generalization_CI':None,'no_frame_IID_CI':True},
    }.items():
        extra = ['experiments/T1/PAIRED_STATE_CONDITION_EFFECTS.json'] if relative.endswith('STATE_MATCHING.json') else ['evaluation/FROZEN_ENCODER_CONTROLS.json'] if relative.endswith('VISUAL_ENCODER.json') else []
        view(relative, {'results':value,'assembler_source_sha256':sha256(Path(__file__))},sources+extra)
    gate_results = gates(summary,curves,protocol)
    if any(gate_results[k]['status']=='FAIL' for k in ('Gate1','Gate2','Gate4')): decision = 'FAIL'
    else: decision = 'AMBIGUOUS_NOT_PROVEN_ALL_REQUIRED_GATES'
    result = {'stage':'N72R21','final_goal_file':'outputs/N72R21/FINAL_GOAL.json',
              'research_direction_file':'outputs/N72R21/RESEARCH_DIRECTION_OVERRIDE.json','utc':utcnow(),
              'scientific_decision':decision,'scientific_success':False,'gates':gate_results,
              'research_primary_track':'MOT','SOT_work_status':'DEFERRED_BY_USER',
              'current_frozen_system_reliable_long_term_identity_not_established':True,
              'full_MOT_scientific_generalization_established':False,
              'MOT_all12_learned_cases_worse_than_clicked_C0':verification['diagnostic_findings'],
              'complete_frozen_VAL_scene_cases':375,'actual_VAL_one_click_targets':273,
              'actual_checkpoint_files_strictly_loaded_and_SHA_verified':248,
              'source_evidence':[reference(p) for p in sources],
              'tables_file':'outputs/N72R21/evaluation/TABLES_A_TO_E.json',
              'Goal_completion_proven_by_this_result':False,
              'final_history_storage_regression_semantic_and_Git_audits_still_required':True,
              'automatic_next_stage_allowed':False,'next_stage_authorized':False}
    write_json('FINAL_RESULT.json',result)
    print(json.dumps({'complete_VAL':375,'complete_T2':264,'complete_joint_MOT':32,'tables':list(tables),
                      'scientific_decision':decision,'gates':{k:r['status'] for k,r in gate_results.items()},'Goal_complete':False}))


if __name__ == '__main__': run()
