"""Current controls sealed first; common pinned TrackEval and strict GT labels."""
import json
from pathlib import Path
import subprocess
import time
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,HISTORY,TRAIN,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_frozen_control_pilot import PROTOCOL,CASES
from scripts.n72r21r1_learned_evaluate import offline,aggregate
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary


def run():
    protocol=read_json(PROTOCOL);assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items());seals={}
    for sequence in protocol['sequences']:
        path=OUT/'learned_pilot/runtime_seals/CLICK_C0'/(sequence+'.json');seal=read_json(path)
        assert all(sha256(a['path'])==a['sha256'] for a in seal['artifacts']);seals['CLICK_C0',sequence]=seal
        # Read-only link to the actually run, sealed C0 trajectory; not a new
        # runtime claimed by copying previously emitted results.
        directory=ASSETS/'frozen_controls/trackers/CLICK_C0/data';directory.mkdir(parents=True,exist_ok=True)
        source=Path(next(a['path'] for a in seal['artifacts'] if a['kind']=='trajectory'));link=directory/(sequence+'.txt')
        if link.exists():assert link.is_symlink() and link.resolve()==source.resolve()
        else:link.symlink_to(source)
        for case in CASES:
            s=read_json(OUT/'frozen_controls/runtime_seals'/case/(sequence+'.json'))
            assert s['protocol_SHA']==sha256(PROTOCOL) and s['source_code_SHA']==protocol['source_code_SHA']
            assert not s['runtime_GT_input'] and not s['runtime_future_GT_input'] and not s['history_rewritten']
            assert s['full_global_exact_unique_candidate_ownership'] and s['extra_clicks']==0
            assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts']);seals[case,sequence]=s
    assert len(seals)==12
    statistics,detections=offline(protocol,seals)
    pinned=HISTORY/'third_party/MOTIP/TrackEval';commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()
    assert commit=='12c8791b303e0a0b50f753af204249e622d0281a'
    storage(64<<20);destination=ASSETS/'frozen_controls/trackeval_v1'
    if destination.exists():raise FileExistsError('preserve frozen comparator TrackEval attempt')
    destination.mkdir(parents=True);seqmap=destination/'seqmap.txt'
    with seqmap.open('x') as f:f.write('name\n'+'\n'.join(protocol['sequences'])+'\n')
    cases=[c['case'] for c in protocol['cases']]
    command=_trackeval_command(ASSETS/'frozen_controls/trackers',destination,cases,seqmap,gt_split='train',gt_folder=TRAIN)
    command[2]=str(pinned/'scripts/run_mot_challenge.py');log=destination/'trackeval.log';began=time.monotonic()
    with log.open('x') as stream:actual=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    receipt={'command':command,'returncode':actual.returncode,'seconds':time.monotonic()-began,
        'log_path':str(log),'log_SHA':sha256(log),'pinned_TrackEval_commit':commit,
        'wrapper_SHA':sha256(ROOT/'scripts/n72r20r3r2r3_trackeval_entry.py'),'same_settings_all6_cases':True,'third_party_modified':False}
    write_json('frozen_controls/TRACKEVAL_INVOCATION.json',receipt)
    if actual.returncode:raise RuntimeError('frozen comparator TrackEval failed, attempt retained')
    metrics={c:trackeval_summary(parse_trackeval(destination,c,protocol['sequences'])) for c in cases}
    assert all(all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')) for m in metrics.values())
    result={'stage':'N72R21R1','status':'COMPLETE_ACTUAL_FROZEN_COMPARATOR_JOINT_MOT_EVALUATION',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','new_actual_full_joint_runs':10,'reused_C0_runs':2,
        'protocol_SHA':sha256(PROTOCOL),'statistics':statistics,'TrackEval':metrics,'groups':aggregate(protocol,statistics,metrics),
        'detection_multiset_audit':detections,'invocation':receipt,'historical_exposed_TRAIN_not_confirmation':True,
        'native_historical_scorer_difference_not_new_C0_preserving_controller':True,
        'adapter_and_native_identity_claim_scores_are_uncalibrated':True,'all_negative_controls_retained':True,
        'next_stage_authorized':False,'whole_goal_scientific_decision':'PENDING'}
    write_json('frozen_controls/RESULT.json',result)
    update_status(missing_frozen_comparator_runs_complete=True,actual_new_frozen_comparator_joint_runs=10)
    print(json.dumps({'frozen_comparator_actual_TrackEval_complete':10,
        'effects':{c:{'HOTA':metrics[c]['HOTA'],'N01':result['groups'][c]['mean_seed_summed_scene_counts'].get('N01',0),
            'N10':result['groups'][c]['mean_seed_summed_scene_counts'].get('N10',0)} for c in cases}}),flush=True)


if __name__=='__main__':run()
