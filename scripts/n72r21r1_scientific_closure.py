"""Bounded negative closure after all registered inexpensive branches, not PASS.

Fresh confirmation is conditional on development gates, so NOT_RUN cannot
be converted into either successful or failed independent generalization.
"""
from copy import deepcopy
import json
import numpy as np
from scripts.n72r21r1_common import OUT,read_json,write_json,sha256,update_status,storage


def metric_reference(relative,cases):
    d=read_json(OUT/relative);metrics=d.get('TrackEval',d.get('all_actual_TrackEval_metrics_0to1'))
    assert cases and all(c in metrics for c in cases)
    return {'source_result_path':str(OUT/relative),'source_result_SHA':sha256(OUT/relative),'actual_source_cases':cases,
        'new_rollouts_claimed_for_alias':0,'full_original2_TRAIN_combined_mean_seed':{k:float(np.mean([metrics[c][k] for c in cases]))
            for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')}}


def run():
    selection_path=OUT/'selection/INNER_ONLY_METHOD_SELECTION_V1.json';selection=read_json(selection_path)
    assert selection['qualified_simple_candidates']==selection['qualified_new_candidates']==0
    results=['learned_pilot/RESULT.json','on_policy/correction_pilot/RESULT.json','memory/pilot/RESULT.json',
             'availability/current_axis_pilot/RESULT.json','trajectory_gate/RESULT.json']
    groups=[]
    for relative in results:
        d=read_json(OUT/relative)
        for name,g in d['all_seed_groups'].items():
            assert not g['direction_pass']
            groups.append({'namespace':relative,'group':name,'direction_pass':False,'nonvacuity_pass':g['nonvacuity_pass'],
                'numerical_MOT_gate':g['numerical_MOT_gate'],'source_result_SHA':sha256(OUT/relative)})
    fixed=read_json(OUT/'pilot/FIXED_GATE_RESULT.json');fm=fixed['all_actual_TrackEval_metrics_0to1']
    cases=lambda prefix:[c for c in fm if c==prefix or c.startswith(prefix+'_SEED')]
    primary=lambda prefix:[c for c in read_json(OUT/'trajectory_gate/RESULT.json')['TrackEval'] if c.startswith(prefix)]
    memory_cases=[c for c in read_json(OUT/'memory/pilot/RESULT.json')['TrackEval'] if c.startswith('RISK__COUPLED__K8__attention__DEFAULT')]
    new=selection['BEST_NEW_METHOD_DIAGNOSTIC_REFERENCE'];new_relative=str(new['source_result_path']).split('/outputs/N72R21R1/',1)[1]
    definitions=[('NO_HUMAN_C0','pilot/FIXED_GATE_RESULT.json',['NO_HUMAN_C0']),
        ('CLICK_C0','pilot/FIXED_GATE_RESULT.json',['CLICK_C0']),('RAW_OSNET','pilot/FIXED_GATE_RESULT.json',['RAW_ANCHOR']),
        ('R3R2_ADAPTER','frozen_controls/RESULT.json',['R3R2_ADAPTER']),('R4R1_NATIVE','frozen_controls/RESULT.json',['R4R1_NATIVE']),
        ('R21_ACIB_FROZEN','pilot/FIXED_GATE_RESULT.json',cases('ORIGINAL_ACIB_FULL')),
        ('ACIB_SHADOW_ONLY','pilot/FIXED_GATE_RESULT.json',cases('SHADOW_ACIB_P0')),
        ('ACIB_GATED','pilot/FIXED_GATE_RESULT.json',cases('CONFIDENCE_P0_BROAD')),
        ('ACIB_RECOVERY_ONLY','pilot/FIXED_GATE_RESULT.json',cases('RECOVERY_P0_BROAD')),
        ('ACIB_GLOBAL_RISK','trajectory_gate/RESULT.json',primary('RISK__GLOBAL_RISK__H100_GLOBAL_RISK')),
        ('JOINT_STATE_RETRAINED','learned_pilot/RESULT.json',['LOGISTIC__MIXED__H100_GLOBAL_RISK__seed'+str(s) for s in (72111,72112,72113)]),
        ('JOINT_STATE_PLUS_SAFE_MEMORY','memory/pilot/RESULT.json',memory_cases),
        ('SHUFFLED_IDENTITY','frozen_controls/RESULT.json',['SHUFFLED_IDENTITY_ORIGINAL']),
        ('ORIGINAL_CONTROLLER','pilot/FIXED_GATE_RESULT.json',cases('ORIGINAL_ACIB_FULL')),
        ('BEST_SIMPLE_BASELINE','pilot/FIXED_GATE_RESULT.json',['CLICK_C0']),
        ('BEST_NEW_METHOD',new_relative,[new['case']])]
    controls={name:metric_reference(path,c) for name,path,c in definitions};assert len(controls)==16
    controls['R4R1_NATIVE']['historical_native_scorer_differs_not_new_unchanged_C0_controller']=True
    controls['BEST_SIMPLE_BASELINE']['selection']='C0 fallback, zero qualified useful simple method'
    controls['BEST_NEW_METHOD']['selection']='No qualified useful new method; lexical H100 abstain diagnostic representative, not a winning algorithm'
    controls['NO_HUMAN_C0']['target_metrics']='NOT_APPLICABLE_NO_CLICK'
    controls['R21_ACIB_FROZEN']['aliases_same_actual_runs_as']='ORIGINAL_CONTROLLER'
    tables=deepcopy(read_json(OUT/'tables/FIVE_PROGRESS_TABLES_V1.json'))
    for r in tables['Table3_GLOBAL_MOT_EFFECT']:
        if r['Method'] in ('BestSimple','BestFull'):
            ref=controls['BEST_SIMPLE_BASELINE' if r['Method']=='BestSimple' else 'BEST_NEW_METHOD']
            r.update(metrics=ref['full_original2_TRAIN_combined_mean_seed'],status=ref['selection'],source_reference=ref)
    tables.update(status='COMPLETE_FIVE_FINAL_TABLES_BOUNDED_HISTORICAL_NEGATIVE',best_method_selection_complete=True,
        scientific_decision='FAIL_GLOBAL_MOT_TRANSFER',Goal_complete=False,next_stage_authorized=False,
        prior_progress_table_SHA=sha256(OUT/'tables/FIVE_PROGRESS_TABLES_V1.json'),INNER_selection_SHA=sha256(selection_path))
    write_json('tables/FIVE_FINAL_TABLES_V1.json',tables)
    write_json('controls/MANDATORY_16_ACTUAL_CONTROL_REFERENCES_V1.json',{'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'status':'COMPLETE_16_REFERENCES_TO_REAL_SEALED_EVALUATIONS','controls':controls,'aliases_not_new_experiments':True,
        'all_original9_metrics_present':True,'INNER_selection_SHA':sha256(selection_path),'next_stage_authorized':False})
    evidence=read_json(OUT/'tests/CONTRACT_REFRESH_CURRENT_AXIS_V1.json')
    assert evidence['passed']==923 and evidence['failures']==5 and evidence['all123_new_strict_loader_receipts_verified']
    record={'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json',
        'status':'COMPLETE_BOUNDED_SCIENTIFIC_NEGATIVE_PENDING_CODE_DELIVERY','scientific_decision':'FAIL_GLOBAL_MOT_TRANSFER',
        'secondary_bottlenecks':['FAIL_OVERRIDE_SAFETY','FAIL_TRAJECTORY_VALUE_LEARNING','FAIL_MEMORY_CONTAMINATION'],
        'answer':'No demonstrated safe useful one-click full-joint-MOT improvement under the tested frozen candidate inputs, historical8TRAIN scope and bounded controllers. Loose interventions damage correct trajectories; protected policies abstain. Not a universal impossibility claim.',
        'gates':{'Gate0':'ENGINEERING_CONTRACTS_AND_ACTUAL_AA_PASS_FOR_TESTED_RUNTIME_SCHEMAS_NOT_ALL_TESTS_PASS',
                 'Gate1':'FAIL_NONVACUITY_OR_BENEFIT_HARM_AND_COLLATERAL','Gate2':'FAIL_GLOBAL_MOT_IMPROVEMENT',
                 'Gate3':'NOT_RUN_DEVELOPMENT_GATE_FAILED_NO_GENERALIZATION_CLASSIFICATION',
                 'Memory':'FAIL_WRONG_OR_UNKNOWN_WRITE_RISK_OR_NONVACUOUS_CORRECT_RETENTION'},
        'all_retained_learned_correction_memory_open_set_gate_groups':groups,
        'actual_counts':{'new_fitted_heads':123,'nonzero_gradient_optimizer_steps':29000,'strict_new_loader_count':123,
            'old_inherited_weight_SHA_catalog':248,'new_epoch_selected_weight_SHA_audit':2804,
            'A_reproduction_runs':64,'A_selected_same_prestate_events':15,'A_actual_same_prestate_branches':90,
            'B_fixed_complete_joint_runs':164,'C_D_actual_joint_source_rollouts':96,'C_D_raw_action_labels':2942,
            'C5_on_policy_rollouts':19,'C5_paired_action_labels':598,'C5_correction_complete_joint_runs':40,
            'H_learned_complete_joint_runs':120,'F_committed_curriculum_rollouts':22,'F_actual_paired_write_branches':128,
            'F_complete_memory_joint_runs':80,'C1_teacher_diagnostic_episodes':22,'E6_all_axis_source_rollouts':22,
            'E6_all_axis_sampled_frame_groups':713,'E6_candidate_NONE_rows':4874,'E0_E6_new_complete_joint_runs':38,
            'B7_B8_new_complete_joint_runs':24,'new_missing_frozen_control_runs':10,
            'G_existing_multi_target_full_TrackEval_records':88,'G_existing_multi_target_valid_clicks':22,'G_independent_exposed_sequences':8},
        'counts_not_to_be_added_as_independent_episodes':'Separate optimization, replay, branch, teacher diagnostic and reused evaluation units; no pooled statistical sample count.',
        'selected_model_and_threshold_freeze':str(selection_path),'INNER_selection_SHA':sha256(selection_path),
        'scientifically_qualified_new_policy':None,'BEST_SIMPLE_BASELINE':'CLICK_C0',
        'retained_new_diagnostic_reference':new['case'],'not_best_outer_effect_selection':True,
        'final_tables_SHA':sha256(OUT/'tables/FIVE_FINAL_TABLES_V1.json'),
        'mandatory16_control_index_SHA':sha256(OUT/'controls/MANDATORY_16_ACTUAL_CONTROL_REFERENCES_V1.json'),
        'regression_evidence':{'full_passed':923,'historical_failed':5,'additional_selection_focused_passed':2,
            'contract_refresh_SHA':sha256(OUT/'tests/CONTRACT_REFRESH_CURRENT_AXIS_V1.json')},
        'conditional_not_run':{'fresh_FIT16_INNER8_CONFIRM8':'No qualifying nonvacuous safe development policy; frozen split remains unchanged and no extra candidate generation',
            'C5_round2':'At-most2 allowance not requirement; round1 own-policy correction still failed every full-joint group, other registered families subsequently failed too',
            'relational_or_backbone_expansion':'Current lower-cost direction/safety and learned-value gates never passed; adding capacity without a trustworthy useful operating point not justified',
            'additional_datasets':'No asset deficit; actual40TRAIN reused. Candidate changes cannot be used to fix a wrong override when a correct candidate already existed',
            'full_SAM3_pixels_VAL_TEST_SOT':'Not authorized by stage gate and no need to establish the bounded negative result'},
        'limitations':['Historical8TRAIN exposed, FIT6 and INNER0002 reused; no virgin generalization evidence',
            'Only2 scenes in full learned-policy pilots, paired bootstrap cluster count2; seeds are not independent scenes',
            '15 selected paired events prove limited roots, not an exhaustive causal decomposition of all103 direct harm decisions or1949 N10 frames',
            'Sparse/crowded TrackEval masks are posthoc diagnostics, not full-policy scores',
            'Teacher GT-motion/bank is POSTHOC_ORACLE diagnostic, not deployable online memory or a strict representational upper bound',
            'UNKNOWN crop is not verified other identity; zero writes have undefined risk and zero usefulness',
            'Cached replay timing does not include SAM3/OSNet pixels-to-output inference'],
        'resource_snapshot':storage(),'scientific_success':False,'next_stage_authorized':False,
        'source_closure_SHA':sha256(__file__),'application_Goal_complete_before_verified_delivery':False}
    write_json('FINAL_RESULT.json',record)
    update_status(scientific_decision='FAIL_GLOBAL_MOT_TRANSFER',scientific_success=False,next_stage_authorized=False,
        scientific_bound_branches_closed=True,five_final_tables_written=True,mandatory16_actual_controls_index_complete=True,
        status='SCIENTIFIC_NEGATIVE_COMPLETE_PENDING_CODE_ONLY_DELIVERY',application_goal_status='active')
    print(json.dumps({'scientific_decision':record['scientific_decision'],'mandatory_controls':16,'final_tables':5,'next_stage_authorized':False,
        'Goal_still_active_until_verified_code_delivery':True}),flush=True)


if __name__=='__main__':run()
