"""25 individually mapped engineering contracts, not a scientific PASS stamp."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from scripts.n72r21r1_common import ROOT,OUT,HISTORY,read_json,write_json,sha256,utcnow,update_status


CONTRACTS=[
    (1,'Preclick equals C0',['test_preclick_identity_proposals_cannot_change_any_public_assignment']),
    (2,'Off and shadow full C0 equality',['test_full_joint_off_and_shadow_match_C0','test_learned_shadow_is_full_joint_C0_including_prototype_and_native_history']),
    (3,'Only one human click',['test_second_click_and_truth_dict_forbidden']),
    (4,'No current/future truth enters runtime',['test_gt_rejected_before_model_forward','test_posthoc_oracle_state_never_becomes_online_runtime_truth']),
    (5,'Only real current candidate UID',['test_mask_shape_stale_candidate_and_recovery_preconditions','test_oracle_requires_completed_current_frame_and_real_current_UID']),
    (6,'Globally unique candidate/public ownership',['test_approved_alternative_is_exact_global_commit_and_p0_never_writes']),
    (7,'Forced-edge hard constraints',['test_forced_edge_changes_all_ownership_and_is_exact_constrained_optimum','test_hard_negative_is_infeasible_even_with_huge_identity_score']),
    (8,'Candidate-relative NONE not omitted other tracks',['test_reject_is_public_none_not_omitted_other_assignments','test_empty_candidate_axis_is_none_not_a_fabricated_identity']),
    (9,'Rejected proposal never writes its crop',['test_rejected_proposal_crop_never_written_even_unsafe_control']),
    (10,'Actor receives actual global commit',['test_approved_alternative_is_exact_global_commit_and_p0_never_writes']),
    (11,'Uncommitted proposal cannot contaminate bank',['test_uncommitted_write_fails_closed','test_rejected_proposal_never_writes_challenger_crop']),
    (12,'Branches own mutable states and banks',['test_clones_isolate_joint_state_anchor_banks_and_pending','test_cloned_pending_trusted_and_rollback_snapshots_are_isolated']),
    (13,'Wrong treatment state preserved and traceable',['test_precommit_scalars_do_not_follow_mutable_state','test_n10_run_may_start_when_baseline_recovers_without_action']),
    (14,'Event counts not frame counts',['test_harm_frames_are_not_independent_initiating_actions']),
    (15,'Other-person damage not target fragmentation',['test_non_target_damage_counts_ids_not_target_and_preserves_UNKNOWN','test_recovering_same_target_fragment_is_not_other_person_damage']),
    (16,'Same pinned TrackEval configuration',['test_pinned_CLEAR_FP_FN_can_change_with_ID_continuity_without_new_detections']),
    (17,'VAL never trains',['test_new_fit_reader_ignores_VAL_TEST_confirmation_and_exposed_outer','test_val_is_gated_and_test_never_authorized']),
    (18,'TEST never calibrates',['test_new_fit_reader_ignores_VAL_TEST_confirmation_and_exposed_outer','test_val_is_gated_and_test_never_authorized']),
    (19,'Matched own-history train/deploy feature and action axes',['test_runtime_action_axis_matches_real_corpus_action_axis','test_temporal_clone_and_current_past_input_are_isolated']),
    (20,'Verified negatives distinct from UNKNOWN',['test_open_set_unknown_and_physical_absence_never_become_verified_negative_UID','test_unknown_is_not_known_other_or_false_negative_presence']),
    (21,'Zero writes cannot pass memory usefulness',['test_zero_writes_are_undefined_risk_and_never_usefulness','test_zero_writes_never_pass_memory_safety']),
    (22,'Missing candidate cannot fabricate GT box',['test_missing_positive_restores_prior_motion_without_bank_or_fabricated_box','test_oracle_requires_completed_current_frame_and_real_current_UID']),
    (23,'No runtime GT-best action selection',['test_posthoc_oracle_state_never_becomes_online_runtime_truth','test_donor_is_lexical_current_UID_not_truth_or_future_best']),
    (24,'Historical R21 artifacts immutable',['test_history_escape_is_rejected']),
    (25,'Checkpoint SHA/strict schema/FIT lineage',['test_strict_checkpoint_and_history_have_no_mutable_predictor_state','test_runtime_rejects_noncausal_schema']),
]


def run():
    path=OUT/'tests/FULL_REGRESSION_TEACHER_OPEN_SET_CONTROLS_V1.xml';xml=ET.parse(path);tests=list(xml.iter('testcase'))
    failing=[r for r in tests if r.find('failure') is not None or r.find('error') is not None]
    previous=ET.parse(OUT/'tests/FULL_REGRESSION_ON_POLICY_MEMORY_PUBLICATION_V2.xml')
    old={(r.get('classname'),r.get('name')) for r in previous.iter('testcase') if r.find('failure') is not None or r.find('error') is not None}
    actual={(r.get('classname'),r.get('name')) for r in failing};assert len(actual)==5 and actual==old
    history=read_json(OUT/'source_audit/HISTORY_SHA_MANIFEST.json')
    assert all(sha256(p)==s for p,s in history['files'].items())
    assert not subprocess.check_output(['git','-c','core.fsmonitor=false','status','--porcelain'],cwd=HISTORY,text=True).strip()
    checkpoints=read_json(OUT/'checkpoints/SEALED_MANIFEST_V1.json');assert checkpoints['actual_new_selected_checkpoint_strict_loads']==114
    invocation=read_json(OUT/'frozen_controls/TRACKEVAL_INVOCATION.json')
    assert invocation['returncode']==0 and invocation['same_settings_all6_cases'] and not invocation['third_party_modified']
    assert sha256(invocation['log_path'])==invocation['log_SHA']
    teacher=read_json(OUT/'diagnostics/teacher_state_v1/RESULT.json')
    assert teacher['actual_episodes']==22 and teacher['all_original_full_C0_states_outputs_and_32_committed_training_features_equal']
    open_set=read_json(OUT/'availability/current_axis_v1/SUPERVISION_MANIFEST.json')
    assert open_set['all22_source_seals_verified_before_truth_opened'] and sha256(open_set['path'])==open_set['sha256']
    mapping=[]
    for number,description,names in CONTRACTS:
        matched=[r for r in tests if any(r.get('name','').split('[')[0]==name for name in names)]
        assert matched and all(any(r.get('name','').split('[')[0]==name for r in matched) for name in names)
        assert all(r.find('failure') is None and r.find('error') is None and r.find('skipped') is None for r in matched)
        record={'contract':number,'requirement':description,'actual_passing_regression_nodes':[
            r.get('classname')+'::'+r.get('name') for r in matched],'evidence_scope':'Engineering behavior for tested schemas, not a proof of scientific usefulness/generalization'}
        if number==16:record.update(actual_common_TrackEval_invocation_SHA=sha256(OUT/'frozen_controls/TRACKEVAL_INVOCATION.json'),
            FP_FN_metric_ID_dependency_does_not_imply_changed_exported_detections=True)
        if number==19:record.update(scientific_train_deploy_distribution_shift_solved=False,
            scope='Matching runtime/source feature and action wiring is tested; new policy empirical state distribution can still differ. E6 fits/own-policy evaluation remain required.')
        if number==24:record.update(actual_history_file_SHA_rechecks=len(history['files']),original_Git_worktree_clean=True)
        if number==25:record.update(actual_new_strict_loader_count=114,checkpoint_manifest_SHA=sha256(OUT/'checkpoints/SEALED_MANIFEST_V1.json'))
        mapping.append(record)
    write_json('tests/CONTRACT_EVIDENCE_V1.json',{'stage':'N72R21R1','status':'COMPLETE_25_ENGINEERING_CONTRACT_EVIDENCE_MAP_NOT_SCIENTIFIC_CLOSURE',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','utc':utcnow(),'contracts':mapping,'individual_mapped_contracts':25,
        'actual_full_regression_XML_path':str(path),'actual_full_regression_XML_SHA':sha256(path),'passed':len(tests)-len(failing),'failures':len(failing),
        'unchanged_historical_failure_nodes':sorted('::'.join(r) for r in actual),'no_test_or_third_party_source_modified_to_mask_failures':True,
        'source_auditor_SHA':sha256(Path(__file__)),'scientific_PASS':False,'Goal_complete':False,'next_stage_authorized':False})
    update_status(engineering_25_contract_map_complete=True,latest_full_regression_passed=len(tests)-len(failing),latest_full_regression_failed=len(failing))
    print(json.dumps({'engineering_contracts':25,'regression_passed':len(tests)-len(failing),'historical_failures':5,
        'reverified_old_artifacts':len(history['files']),'whole_Goal_complete':False}),flush=True)


if __name__=='__main__':run()
