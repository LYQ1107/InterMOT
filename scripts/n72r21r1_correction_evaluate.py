"""Post-seal correction pilot evaluation, all fits and seeds retained."""
from pathlib import Path
import json
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,TRAIN,HISTORY,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_correction_pilot import PROTOCOL
from scripts.n72r21r1_learned_evaluate import offline,aggregate
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary


def run():
    protocol=read_json(PROTOCOL);seals={}
    assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items())
    for c in protocol['cases']:
        for sequence in protocol['sequences']:
            s=read_json(OUT/'on_policy/correction_pilot/runtime_seals'/c['case']/(sequence+'.json'))
            assert s['status']=='COMPLETE_ACTUAL_ON_POLICY_CORRECTION_FULL_JOINT_RUNTIME'
            assert s['protocol_SHA']==sha256(PROTOCOL) and s['source_code_SHA']==protocol['source_code_SHA']
            assert not s['runtime_GT_input'] and not s['runtime_future_GT_input'] and not s['history_rewritten']
            assert s['full_global_exact_unique_candidate_ownership'] and s['extra_clicks']==0
            assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts']);seals[c['case'],sequence]=s
    assert len(seals)==40
    stats,detection=offline(protocol,seals)
    pinned=HISTORY/'third_party/MOTIP/TrackEval';commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()
    assert commit=='12c8791b303e0a0b50f753af204249e622d0281a'
    root=ASSETS/'on_policy/correction_pilot/trackeval_v1'
    if root.exists():raise FileExistsError('preserve previous correction TrackEval attempt')
    storage(64<<20);root.mkdir(parents=True);seqmap=root/'seqmap.txt'
    with seqmap.open('x') as f:f.write('name\n'+'\n'.join(protocol['sequences'])+'\n')
    cases=[c['case'] for c in protocol['cases']]
    command=_trackeval_command(ASSETS/'on_policy/correction_pilot/trackers',root,cases,seqmap,gt_split='train',gt_folder=TRAIN)
    command[2]=str(pinned/'scripts/run_mot_challenge.py');log=root/'trackeval.log';began=time.monotonic()
    with log.open('x') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    write_json('on_policy/correction_pilot/TRACKEVAL_INVOCATION.json',{'command':command,'returncode':result.returncode,
        'seconds':time.monotonic()-began,'log_path':str(log),'log_SHA':sha256(log),'pinned_TrackEval_commit':commit,
        'same_settings_all20_cases':True,'third_party_modified':False})
    if result.returncode:raise RuntimeError('correction TrackEval failed, original log preserved')
    metrics={c:trackeval_summary(parse_trackeval(root,c,protocol['sequences'])) for c in cases}
    assert all(all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')) for m in metrics.values())
    groups=aggregate(protocol,stats,metrics)
    write_json('on_policy/correction_pilot/RESULT.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_ON_POLICY_CORRECTION_FULL_MOT_EVALUATION',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','actual_sealed_runs':40,'actual_cases':20,'independent_sequences':2,
        'protocol_SHA':sha256(PROTOCOL),'statistics':stats,'TrackEval':metrics,'all_seed_groups':groups,'detection_multiset_audit':detection,
        'all_contrasts_seeds_and_abstaining_fits_retained':True,'historical_development_not_fresh_confirmation':True,
        'source_evaluator_SHA':sha256(Path(__file__)),'shared_evaluator_SHA':sha256(ROOT/'scripts/n72r21r1_learned_evaluate.py'),
        'zero_intervention_not_PASS':True,'next_stage_authorized':False,'whole_goal_scientific_decision':'PENDING'})
    update_status(phase_C5_correction_full_MOT_evaluation_complete=True,actual_correction_full_MOT_runs=40)
    print(json.dumps({'actual_correction_TrackEval_complete':40,'groups':len(groups),'direction_pass':sum(g['direction_pass'] for g in groups.values()),'Goal_still_active':True}),flush=True)


if __name__=='__main__':run()
