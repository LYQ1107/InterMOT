"""Common full-MOT metrics for actual explicit B7/B8 branch-value gates."""
import json
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,HISTORY,TRAIN,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_trajectory_gate_pilot import PROTOCOL
from scripts.n72r21r1_learned_evaluate import offline,aggregate,artifact
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary


def run():
    p=read_json(PROTOCOL);seals={};assert all(sha256(ROOT/k)==v for k,v in p['source_code_SHA'].items())
    for case in p['cases']:
        for sequence in p['sequences']:
            path=OUT/'learned_pilot/runtime_seals/CLICK_C0'/(sequence+'.json') if case['case']=='CLICK_C0' else OUT/'trajectory_gate/runtime_seals'/case['case']/(sequence+'.json')
            r=read_json(path)
            if case['case']!='CLICK_C0':assert r['protocol_SHA']==sha256(PROTOCOL) and r['source_code_SHA']==p['source_code_SHA']
            assert all(sha256(a['path'])==a['sha256'] for a in r['artifacts'])
            assert not r['runtime_GT_input'] and not r['runtime_future_GT_input'] and not r['history_rewritten']
            assert r['full_global_exact_unique_candidate_ownership'] and r['extra_clicks']==0;seals[case['case'],sequence]=r
    assert len(seals)==26;stats,detection=offline(p,seals)
    directory=ASSETS/'trajectory_gate/trackers/CLICK_C0/data';directory.mkdir(parents=True,exist_ok=True)
    for sequence in p['sequences']:
        source=artifact(seals['CLICK_C0',sequence],'trajectory');link=directory/(sequence+'.txt')
        if link.exists():assert link.is_symlink() and link.resolve()==source.resolve()
        else:link.symlink_to(source)
    storage(64<<20);destination=ASSETS/'trajectory_gate/trackeval_v1'
    if destination.exists():raise FileExistsError('preserve explicit B7/B8 TrackEval attempt')
    destination.mkdir(parents=True);seqmap=destination/'seqmap.txt'
    with seqmap.open('x') as f:f.write('name\n'+'\n'.join(p['sequences'])+'\n')
    pinned=HISTORY/'third_party/MOTIP/TrackEval';commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()
    assert commit=='12c8791b303e0a0b50f753af204249e622d0281a';cases=[c['case'] for c in p['cases']]
    command=_trackeval_command(ASSETS/'trajectory_gate/trackers',destination,cases,seqmap,gt_split='train',gt_folder=TRAIN)
    command[2]=str(pinned/'scripts/run_mot_challenge.py');log=destination/'trackeval.log';began=time.monotonic()
    with log.open('x') as stream:actual=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    receipt={'command':command,'returncode':actual.returncode,'seconds':time.monotonic()-began,'log_path':str(log),'log_SHA':sha256(log),
        'pinned_TrackEval_commit':commit,'same_settings_all13_cases':True,'wrapper_SHA':sha256(ROOT/'scripts/n72r20r3r2r3_trackeval_entry.py'),'third_party_modified':False}
    write_json('trajectory_gate/TRACKEVAL_INVOCATION.json',receipt)
    if actual.returncode:raise RuntimeError('explicit B7/B8 TrackEval failed, log preserved')
    metrics={c:trackeval_summary(parse_trackeval(destination,c,p['sequences'])) for c in cases}
    assert all(all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')) for m in metrics.values())
    groups=aggregate(p,stats,metrics)
    write_json('trajectory_gate/RESULT.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_EXPLICIT_B7_B8_FULL_JOINT_EVALUATION',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','new_actual_full_joint_runs':24,'reused_baseline_runs':2,
        'protocol_SHA':sha256(PROTOCOL),'statistics':stats,'TrackEval':metrics,'all_seed_groups':groups,'detection_multiset_audit':detection,
        'invocation':receipt,'evaluator_source_SHA':sha256(__file__),'not_generalization_or_best_method_selection':True,
        'no_VAL_TEST_confirmation':True,'next_stage_authorized':False,'whole_goal_scientific_decision':'PENDING'})
    update_status(explicit_B7_B8_actual_joint_evaluation_complete=True,actual_explicit_B7_B8_new_joint_runs=24,
        explicit_B7_B8_direction_passing_groups=sum(r['direction_pass'] for r in groups.values()))
    print(json.dumps({'actual_B7_B8_TrackEval_new_runs':24,'groups_direction_pass':sum(r['direction_pass'] for r in groups.values())}),flush=True)


if __name__=='__main__':run()
