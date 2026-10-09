"""Freeze real learned-policy histories before new effects or correction fits."""
from scripts.n72r21r1_common import ROOT,OUT,read_json,write_json,sha256,update_status


def run():
    fit_path=OUT/'training/authority/LOGISTIC__MIXED__H5__seed72111.json'
    fit=read_json(fit_path)
    assert fit['status']=='COMPLETE_ACTUAL_CONTROLLER_OPTIMIZATION'
    assert sha256(fit['checkpoint_path'])==fit['checkpoint_SHA']
    inputs=read_json(OUT/'corpus/RUNTIME_INPUTS.json')
    verification=read_json(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json')['initial_clicked_UID_verification']
    flags={}
    for r in verification:
        previous=flags.setdefault(r['episode_uid'],r['clicked_UID_strictly_verified_target'])
        assert previous==r['clicked_UID_strictly_verified_target']
    events=[e for e in inputs['inputs'] if e['diagnostic_role'] in ('FIT','INNER') and flags[e['episode_uid']]]
    excluded=[e['episode_uid'] for e in inputs['inputs'] if e['diagnostic_role'] in ('FIT','INNER') and not flags[e['episode_uid']]]
    assert len(events)==19 and len(excluded)==2
    files=['scripts/n72r21r1_collect_on_policy.py','scripts/n72r21r1_collect_joint.py',
        'sam3_intermot/one_click/learned_mot_bridge.py','sam3_intermot/one_click/learned_authority.py',
        'sam3_intermot/one_click/joint_intervention_primitives.py','sam3_intermot/one_click/intervention_features.py',
        'scripts/n72r21r1_counterfactual.py']
    write_json('protocol/ON_POLICY_ROUND1.json',{'stage':'N72R21R1','version':'C5_REAL_LEARNED_ERROR_HISTORY_ROUND1',
        'final_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','frozen_before_round1_effects_or_correction_fits':True,
        'purpose':'Actual learned-policy own-history error correction curriculum, not best model or success claim.',
        'policy_choice':'First preregistered short-horizon H5 logistic/mixed seed72111. Deliberate error-producing policy retained; no best outer-effect selection.',
        'policy_fit_path':str(fit_path),'policy_fit_SHA':sha256(fit_path),'checkpoint_path':fit['checkpoint_path'],'checkpoint_SHA':fit['checkpoint_SHA'],
        'identity_model_source':inputs['frozen_identity_model'],'inputs_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),
        'events':events,'excluded_unverified_initial_clicks_no_replacement':excluded,
        'state_source':'ACTUAL_LEARNED_H5_ON_POLICY_ROUND1','planned_full_joint_rollouts':19,
        'memory_policy':'P0 fixed, isolate authority-state correction from write changes.',
        'sampling':'event+16, every128 original frames, max16 positions, no effect or GT correctness sampling.',
        'counterfactuals':'Original frozen corpus raw-top2/ACIB-top1/KEEP/REJECT primitive, same own prestate, single action then future KEEP. Envelope additionally pins learned history source/checkpoint.',
        'branch_method_protocol_SHA':sha256(OUT/'protocol/JOINT_STATE_CORPUS.json'),
        'branch_primitive_code_SHA':{p:sha256(ROOT/p) for p in read_json(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json')['source_code_SHA']},
        'supervision':'Read future TRAIN GT only after all19 source and branch artifacts sealed; previous3 proposal vectors from actual own history. No GT replacing past state.',
        'correction_fit_plan':'Bounded LOGISTIC/GLOBAL_RISK/CAUSAL_TEMPORAL, seeds72111/72112/72113, ON_POLICY_ONLY and ORIGINAL_MIXED_PLUS_ON_POLICY; H100 global-risk labels, existing epoch/INNER calibration rules. All contrasts retained.',
        'future_rounds_max_total':2,'new_global_FIT_INNER_confirmation_and_VAL_TEST_untouched':True,
        'max_CPU_workers':2,'threads_per_worker':1,'max_seconds_per_episode':1800,'reserve_GiB':60,
        'source_code_SHA':{p:sha256(ROOT/p) for p in files},'next_stage_authorized':False})
    update_status(phase_C5_on_policy_protocol_frozen=True,phase_C5_on_policy_running=False,git_publication_clean=False)


if __name__=='__main__':run()
