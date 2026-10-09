"""Exact clean delivery and sealed negative evidence; no new experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from scripts.n72r21r1_common import ROOT,OUT,HISTORY,read_json,write_json,sha256,storage,utcnow,update_status


def run(remote):
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    assert git('branch','--show-current')=='codex/n72r21r1-safe-joint-mot-intervention'
    assert git('rev-parse','HEAD')==remote
    assert git('rev-parse','refs/remotes/origin/codex/n72r21r1-safe-joint-mot-intervention')==remote
    assert not git('-c','core.fsmonitor=false','status','--porcelain')
    publication=read_json(OUT/'git_delivery'/('PUBLISH_VERIFICATION__'+remote+'.json'))
    assert publication['local_HEAD']==publication['fresh_remote_HEAD']==remote and publication['clean_worktree_fsmonitor_disabled']
    assert not publication['force_push'] and publication['no_assets_or_credentials_published']
    catalog=read_json(ROOT/'docs/N72R21R1_LOCAL_EVIDENCE_SHA.json')
    assert all(sha256(OUT/p)==s for p,s in catalog['local_artifact_SHA256'].items())
    result=read_json(OUT/'FINAL_RESULT.json');assert result['scientific_decision']=='FAIL_GLOBAL_MOT_TRANSFER'
    assert not result['scientific_success'] and not result['next_stage_authorized'] and result['scientifically_qualified_new_policy'] is None
    table=read_json(OUT/'tables/FIVE_FINAL_TABLES_V1.json');assert table['best_method_selection_complete']
    assert len(read_json(OUT/'controls/MANDATORY_16_ACTUAL_CONTROL_REFERENCES_V1.json')['controls'])==16
    path=OUT/'tests/FULL_REGRESSION_FINAL_DELIVERY_V1.xml';tests=list(ET.parse(path).iter('testcase'))
    failed=[r for r in tests if r.find('failure') is not None or r.find('error') is not None]
    old=read_json(OUT/'tests/CONTRACT_REFRESH_CURRENT_AXIS_V1.json')
    assert sorted(r.get('classname')+'::'+r.get('name') for r in failed)==old['unchanged_historical_failure_nodes']
    assert len(tests)-len(failed)==925 and len(failed)==5
    for name in ('TEACHER_STATE_DIAGNOSTIC_V1.json','JOINT_OPEN_SET_CURRENT_AXIS_V1.json','CURRENT_AXIS_VERIFIER_FITS_V1.json',
                 'CURRENT_AXIS_FULL_JOINT_PILOT_V1.json','FROZEN_COMPARATOR_JOINT_PILOT_V1.json','EXPLICIT_B7_B8_JOINT_PILOT_V1.json'):
        p=read_json(OUT/'protocol'/name);assert all(sha256(ROOT/k)==v for k,v in p['source_code_SHA'].items())
    history=read_json(OUT/'source_audit/HISTORY_SHA_MANIFEST.json');assert all(sha256(p)==s for p,s in history['files'].items())
    assert not subprocess.check_output(['git','-c','core.fsmonitor=false','status','--porcelain'],cwd=HISTORY,text=True).strip()
    record={'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','utc':utcnow(),
        'status':'COMPLETE_SCIENTIFIC_BOUND_SCOPE_AND_VERIFIED_CODE_ONLY_DELIVERY_READY_TO_MARK_GOAL_COMPLETE',
        'scientific_decision':'FAIL_GLOBAL_MOT_TRANSFER','scientific_success':False,'next_stage_authorized':False,
        'local_HEAD':remote,'fresh_remote_HEAD_supplied_from_actual_GitHub_ref_read':remote,'worktree_clean':True,
        'publication_receipt_SHA':sha256(OUT/'git_delivery'/('PUBLISH_VERIFICATION__'+remote+'.json')),
        'FINAL_RESULT_SHA':sha256(OUT/'FINAL_RESULT.json'),'final_tables_SHA':sha256(OUT/'tables/FIVE_FINAL_TABLES_V1.json'),
        'final_regression_XML_SHA':sha256(path),'actual_full_regression_passed':925,'historical_failures':5,
        'original371_SHAs_reverified':len(history['files']),'original_worktree_clean':True,'all_frozen_final_code_SHAs_equal':True,
        'resource_snapshot':storage(),'source_verifier_SHA':sha256(__file__),
        'no_future_stage_or_training_or_confirmation_started':True,'actual_application_Goal_completion_tool_still_required':True}
    write_json('delivery/FINAL_VERIFICATION_V1.json',record)
    update_status(status='COMPLETE_BOUNDED_SCIENTIFIC_NEGATIVE_CODE_DELIVERY_VERIFIED',scientific_decision='FAIL_GLOBAL_MOT_TRANSFER',
        scientific_success=False,next_stage_authorized=False,git_publication_clean=True,git_local_equals_fresh_remote=True,
        git_publication_HEAD=remote,last_published_commit=remote,git_publication_local_remote_identical=True,
        completed_full_regression_passed=925,latest_full_regression_passed=925,completed_full_regression_failed=5,latest_full_regression_failed=5,
        full_regression_latest={'passed':925,'failed':5,'receipt':'outputs/N72R21R1/delivery/FINAL_VERIFICATION_V1.json'},
        resource_snapshot=record['resource_snapshot'],current_status_explanation='All bounded A-H branches,123 fits, five tables,16 controls,25 contracts and code-only delivery completed; scientific FAIL_GLOBAL_MOT_TRANSFER, no useful new method, no fresh confirmation/VAL/TEST or next stage. Ready to mark application Goal complete.',
        application_goal_status='active_ready_to_mark_complete')
    print(json.dumps({'scientific_decision':'FAIL_GLOBAL_MOT_TRANSFER','verified_local_remote_HEAD':remote,'clean':True,
        'regression_passed':925,'historical_failed':5,'next_stage_authorized':False,'ready_for_Goal_completion':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fresh-remote-head',required=True);a=p.parse_args();run(a.fresh_remote_head)
