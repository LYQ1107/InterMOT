"""Derived mutable stage progress; frozen Goal/protocols never rewritten."""
from collections import Counter
import json
import xml.etree.ElementTree as ET
from scripts.n72r21_common import ROOT,OUT,read_json,write_json,storage,utcnow


def run():
    status=read_json(OUT/'stage_status.json')
    fits=[read_json(p) for p in (OUT/'training/T1_CAUSAL_V1').glob('*.json')]
    t2=[read_json(p) for p in (OUT/'training/T2_COUPLED_V1').glob('*.json')]
    regression=read_json(OUT/'tests/REGRESSION_CURRENT.json')
    status.update(status='ACTIVE_T2_FROZEN_VALIDATION_AND_LAWFUL_DOMAIN_DIAGNOSTICS',progress_snapshot_utc=utcnow(),
        T1_completed_fits=sum(r['completed'] for r in fits),T1_expected_fits=72,
        T1_completed_fits_by_outer=dict(Counter(r['schema']['configuration']['outer_sequence'] for r in fits if r['completed'])),
        T1_all_fitting_completed=len(fits)==72 and all(r['completed'] for r in fits),
        T1_currently_partial_fits=sum(not r['completed'] for r in fits),
        T1_outer_runtime_cases_completed=len(list((OUT/'experiments/T1/runtime_seals').glob('*/*.json'))),
        T1_outer_runtime_cases_expected=120,
        T2_own_state_and_paired_branch_scene_seals=len(list((OUT/'training/coupled_states/seals').glob('*/*.json'))),
        T2_completed_fits=sum(r['completed'] for r in t2),T2_expected_fits=24,
        T2_outer_runtime_cases_completed=len(list((OUT/'experiments/T2/runtime_seals').glob('*/*.json'))),
        T2_outer_runtime_cases_expected=264,
        frozen_VAL_initialized_sequences=len(list((OUT/'validation/initialization').glob('*.json'))),
        frozen_VAL_initialized_targets=sum(len(read_json(p)['inputs']) for p in (OUT/'validation/initialization').glob('*.json')),
        frozen_VAL_runtime_scene_cases_completed=len(list((OUT/'validation/runtime_seals').glob('*/*.json'))),
        frozen_VAL_runtime_scene_cases_expected=375,
        frozen_VAL_evaluated_scene_cases=len(list((OUT/'validation/evaluations').glob('*/*.json'))),
        LaSOT_frame_only_candidate_windows_sealed=len(list((OUT/'experiments/LaSOT_identity/candidate_seals').glob('*.json'))),
        LaSOT_frame_only_identity_windows_evaluated=len(list((OUT/'experiments/LaSOT_identity/evaluations').glob('*.json'))),
        T2_paired_real_data_gradient_preflight_passed=(OUT/'training/T2_REAL_DATA_PREFLIGHT.json').exists(),
        T2_T3_complete=False,scientific_success=None,next_stage_authorized=False)
    focused_path=regression['focused_actual_XML_path'];tests=list(ET.parse(focused_path).getroot().iter('testcase'))
    failed=sum(r.find('failure') is not None or r.find('error') is not None for r in tests)
    skipped=sum(r.find('skipped') is not None for r in tests)
    status['latest_regression_summary_file']='outputs/N72R21/tests/REGRESSION_CURRENT.json'
    status['complete_repo_regression']={'passed':regression['passed'],'failed':regression['failed'],
        'scope':regression['actual_XML_path'],'source_XML_sha256':regression['actual_XML_sha256'],
        'focused_and_full_revision_may_differ':True}
    status['focused_tests']={'passed':len(tests)-failed-skipped,'failed':failed,'skipped':skipped,'scope':focused_path}
    status['T1_fitting_completed']=status['T1_all_fitting_completed']
    status['T2_paired_real_data_gradient_preflight_passed']=(OUT/'training/T2_REAL_DATA_PREFLIGHT.json').exists() and read_json(OUT/'training/T2_REAL_DATA_PREFLIGHT.json')['all_gradients_finite']
    status['milestones']['M5_M13']='T0_24_T1_72_FITS_AND_T1_120_REPLAYS_COMPLETE_T2_ACTUAL_FITTING_AND_FROZEN_VAL_ACTIVE_T3_AND_FINAL_SCIENCE_PENDING'
    write_json('stage_status.json',status);write_json('storage/INTERMEDIATE_STORAGE.json',storage())
    print(json.dumps({k:status[k] for k in ('T1_completed_fits','T1_outer_runtime_cases_completed','T2_own_state_and_paired_branch_scene_seals','T2_completed_fits','progress_snapshot_utc')}))


if __name__=='__main__':run()
