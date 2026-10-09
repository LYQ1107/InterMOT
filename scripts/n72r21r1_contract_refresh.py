"""Preserve911/25-contract receipt, refresh actual923 tests and123 loaders."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from scripts.n72r21r1_common import ROOT,OUT,HISTORY,read_json,write_json,sha256,utcnow,storage,update_status
from scripts.n72r21r1_contract_audit import CONTRACTS


def run():
    path=OUT/'tests/FULL_REGRESSION_EXPLICIT_GATES_V1.xml';tests=list(ET.parse(path).iter('testcase'))
    failures=[r for r in tests if r.find('failure') is not None or r.find('error') is not None]
    actual={(r.get('classname'),r.get('name')) for r in failures}
    prior=OUT/'tests/CONTRACT_EVIDENCE_V1.json';old=read_json(prior)
    assert sorted('::'.join(r) for r in actual)==old['unchanged_historical_failure_nodes'] and len(actual)==5
    assert len(tests)-len(failures)==923
    mapping=[]
    for number,description,names in CONTRACTS:
        nodes=[r for r in tests if r.get('name','').split('[')[0] in names]
        assert all(any(r.get('name','').split('[')[0]==n for r in nodes) for n in names)
        assert all(r.find('failure') is None and r.find('error') is None and r.find('skipped') is None for r in nodes)
        mapping.append({'contract':number,'requirement':description,'passing_nodes':[r.get('classname')+'::'+r.get('name') for r in nodes],
            'engineering_not_empirical_scientific_PASS':True})
    additional=[r for r in tests if r.get('classname','') in ('tests.test_n72r21r1_open_set_verifier','tests.test_n72r21r1_trajectory_gate_bridge')]
    assert len(additional)==11 and all(r.find('failure') is None and r.find('error') is None for r in additional)
    checkpoint=read_json(OUT/'checkpoints/SEALED_MANIFEST_V1.json');extension=read_json(OUT/'checkpoints/CURRENT_AXIS_EXTENSION_V1.json')
    assert checkpoint['actual_new_selected_checkpoint_strict_loads']==114 and extension['additional_strict_loads']==9
    weights={**checkpoint['weight_files_SHA'],**extension['additional_weights_SHA']}
    assert all(sha256(p)==s for p,s in weights.items())
    history=read_json(OUT/'source_audit/HISTORY_SHA_MANIFEST.json');assert all(sha256(p)==s for p,s in history['files'].items())
    assert not subprocess.check_output(['git','-c','core.fsmonitor=false','status','--porcelain'],cwd=HISTORY,text=True).strip()
    receipts=[]
    for relative,field in [('frozen_controls/TRACKEVAL_INVOCATION.json','same_settings_all6_cases'),
                           ('availability/current_axis_pilot/TRACKEVAL_INVOCATION.json','same_settings_all20_cases'),
                           ('trajectory_gate/TRACKEVAL_INVOCATION.json','same_settings_all13_cases')]:
        r=read_json(OUT/relative);assert r['returncode']==0 and r[field] and not r['third_party_modified']
        assert r['pinned_TrackEval_commit']=='12c8791b303e0a0b50f753af204249e622d0281a' and sha256(r['log_path'])==r['log_SHA']
        receipts.append({'path':str(OUT/relative),'sha256':sha256(OUT/relative)})
    record={'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','status':'COMPLETE_REFRESHED_25_ENGINEERING_CONTRACTS_NOT_SCIENTIFIC_CLOSURE',
        'utc':utcnow(),'preserved_prior25_map_SHA':sha256(prior),'contracts':mapping,'additional11_open_set_and_own_history_nodes':[r.get('classname')+'::'+r.get('name') for r in additional],
        'actual_latest_regression_XML_path':str(path),'XML_SHA':sha256(path),'passed':923,'failures':5,
        'unchanged_historical_failure_nodes':sorted('::'.join(r) for r in actual),'all123_new_strict_loader_receipts_verified':True,
        'all_epoch_selected_weights_rechecked':len(weights),'original371_artifact_SHAs_rechecked':len(history['files']),
        'original_worktree_clean':True,'actual_common_TrackEval_receipts':receipts,'resource_snapshot':storage(),
        'source_auditor_SHA':sha256(__file__),'empirical_train_deploy_distribution_shift_not_declared_solved':True,
        'scientific_PASS':False,'Goal_complete':False,'next_stage_authorized':False}
    write_json('tests/CONTRACT_REFRESH_CURRENT_AXIS_V1.json',record)
    update_status(latest_full_regression_passed=923,latest_full_regression_failed=5,completed_full_regression_passed=923,completed_full_regression_failed=5,
        full_regression_latest={'passed':923,'failed':5,'receipt':'outputs/N72R21R1/tests/CONTRACT_REFRESH_CURRENT_AXIS_V1.json'},
        actual_new_head_strict_load_count=123,actual_total_nonzero_gradient_optimizer_steps=29000,resource_snapshot=record['resource_snapshot'])
    print(json.dumps({'contracts':25,'actual_regression':{'passed':923,'historical_failed':5},'weight_SHAs':len(weights),'historySHAs':len(history['files'])}),flush=True)


if __name__=='__main__':run()
