"""Freeze all registered head comparisons before their full joint-MOT effects."""
from pathlib import Path
from scripts.n72r21r1_common import ROOT, OUT, read_json, write_json, sha256


def run():
    protocol=read_json(OUT/'protocol/LEARNED_AUTHORITY_V1.json')
    scheduler=read_json(OUT/'training/AUTHORITY_FIT_SCHEDULER_RESULT.json')
    assert scheduler['registered_fits']==scheduler['successful_workers']==93
    contrasts=[(family,'MIXED',protocol['primary_supervision']) for family in protocol['families']]
    contrasts += [(family,state,protocol['primary_supervision']) for family in ('LOGISTIC','CAUSAL_TEMPORAL') for state in ('BASELINE_ONLY','TREATMENT_ONLY')]
    contrasts += [('LOGISTIC','MIXED',reward) for reward in protocol['LOGISTIC_MIXED_reward_contrasts']]
    cases=[{'case':name,'kind':name} for name in ('CLICK_C0','NO_HUMAN_C0','LEARNED_SHADOW')]
    for family,state,reward in contrasts:
        for seed in protocol['seeds']:
            uid='__'.join((family,state,reward,'seed'+str(seed)))
            path=OUT/'training/authority'/(uid+'.json'); record=read_json(path)
            assert record['status']=='COMPLETE_ACTUAL_CONTROLLER_OPTIMIZATION'
            assert sha256(record['checkpoint_path'])==record['checkpoint_SHA']
            cases.append({'case':uid,'kind':'LEARNED_MULTI_ACTION','fit_record_path':str(path),
                'fit_record_SHA':sha256(path),'checkpoint_path':record['checkpoint_path'],'checkpoint_SHA':record['checkpoint_SHA'],
                'family':family,'state_contrast':state,'reward':reward,'seed':seed,'selection':record['selection']})
    assert len(cases)==60 and len({c['case'] for c in cases})==60
    source_files=['scripts/n72r21r1_learned_pilot.py','sam3_intermot/one_click/learned_mot_bridge.py',
        'sam3_intermot/one_click/learned_authority.py','sam3_intermot/one_click/joint_intervention_primitives.py',
        'sam3_intermot/one_click/intervention_features.py','sam3_intermot/one_click/safe_mot_bridge.py']
    write_json('protocol/LEARNED_FULL_MOT_PILOT_V1.json',{'stage':'N72R21R1','version':'H_CLOSED_LOOP_HISTORICAL_V1',
        'final_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','frozen_before_full_policy_effects':True,
        'sequences':['dancetrack0001','dancetrack0002'],'cases':cases,'actual_sequence_cases_planned':120,
        'case_selection':'All preregistered families MIXED, LOGISTIC/CAUSAL BASELINE/TREATMENT, all LOGISTIC reward contrasts, all3seeds; no outer effect selection.',
        'episode_selection':'First prospectively initialized corpus episode per sequence, no replacement by observed effects. Each case is an independent complete original-frame full global MOT rollout.',
        'identity_model':'Same corpus lexicographically first historical0001seed72101 frozen identity model; P0 no-write only. Not new identity-backbone training.',
        'shadow_head':'LOGISTIC__MIXED__H100_GLOBAL_RISK__seed72111 fixed by lexical family convention, not best effects.',
        'default_KEEP_and_nonvacuity':'All abstaining fits still run as actual controls; zero interventions not PASS.',
        'calibration_warning':'Historical INNER0002 was used for epoch/authority calibration. Full0002 is development, not held-out evaluation. 0001 also historically exposed.',
        'metrics':'All nine pinned TrackEval metrics on full sequences under one identical configuration. Actual N01/N10, propagation, other-person damage, candidate availability/NONE/UNKNOWN and memory writes reported separately.',
        'source_code_SHA':{p:sha256(ROOT/p) for p in source_files},
        'input_SHA':sha256(OUT/'corpus/RUNTIME_INPUTS.json'),'supervision_SHA':sha256(OUT/'corpus/TRAJECTORY_SUPERVISION_MANIFEST.json'),
        'max_CPU_workers':4,'threads_per_worker':1,'max_seconds_per_case':1800,'reserve_GiB':60,
        'no_VAL_TEST_confirmation_tuning':True,'scientific_success':None,'next_stage_authorized':False})


if __name__=='__main__': run()
