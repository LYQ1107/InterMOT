"""Full pinned metrics for all22 valid historical joint source episodes.

These88 trajectories were actually generated earlier; read-only links do not
count as new MOT rollouts. This is exposed multi-target development, not fresh
scene confirmation, and target representation/oracle outputs are never scored.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,HISTORY,TRAIN,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_label_corpus import verify_registered
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.safe_intervention_events import assignment_map,analyze_target_events,identity_outcome
from sam3_intermot.evaluation.mot_detection_audit import audit_unchanged_detections


def run():
    inputs,original_seals,_=verify_registered();events=read_json(OUT/'protocol/COMMITTED_MEMORY_CURRICULUM_V1.json')['events']
    valid={e['episode_uid'] for e in events};sources=read_json(OUT/'protocol/JOINT_STATE_CORPUS.json')['state_sources']
    lookup={(s['source_distribution'],s['event']['episode_uid']):(p,s) for p,s in original_seals if s['event']['episode_uid'] in valid}
    assert len(lookup)==88
    sequence_order=sorted({e['sequence'] for e in inputs['inputs']});episode_slot={}
    for sequence in sequence_order:
        for slot,e in enumerate(sorted([r for r in inputs['inputs'] if r['sequence']==sequence],key=lambda r:r['episode_uid'])):episode_slot[e['episode_uid']]=slot
    plan={'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','all22_valid_clicks_and4_original_generating_sources':True,
        'actual_existing_full_joint_trajectories':88,'new_runtime_rollouts':0,'source_seals_SHA':{source+'/'+ep:sha256(p) for (source,ep),(p,s) in lookup.items()},
        'source_evaluator_SHA':sha256(Path(__file__)),'selection':'All registered source histories, no effect selection; three deterministic episode slots and no replacement for invalid clicks',
        'density':'Frozen current real candidate count <=4,5-8,>=9; frame target statistics per stratum. Whole-sequence TrackEval per stratum only if all original frames lie in it; mixed-density sequences kept separate, never truncate/hide hard frames.',
        'no_new_model_training_or_parameter_selection':True,'no_fresh_confirmation_VAL_TEST':True,'not_generalization_or_scientific_PASS':True}
    write_json('data/MULTITARGET_FULL_TRACKEVAL_PLAN_V1.json',plan)
    pinned=HISTORY/'third_party/MOTIP/TrackEval';commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip();assert commit=='12c8791b303e0a0b50f753af204249e622d0281a'
    root=ASSETS/'multitarget_trackeval_v1'
    if root.exists():raise FileExistsError('preserve multi-target evaluator attempt')
    storage(96<<20);root.mkdir(parents=True);metrics={};invocations=[]
    for slot in range(3):
        slot_events=[e for e in events if episode_slot[e['episode_uid']]==slot];sequences=sorted({e['sequence'] for e in slot_events})
        directory=root/('slot'+str(slot));directory.mkdir();seqmap=directory/'seqmap.txt'
        with seqmap.open('x') as f:f.write('name\n'+'\n'.join(sequences)+'\n')
        for source in sources:
            tracker=root/'trackers'/source/'data';tracker.mkdir(parents=True,exist_ok=True)
            # Separate slot tree avoids overwriting another click's output.
            tracker=directory/'trackers'/source/'data';tracker.mkdir(parents=True,exist_ok=True)
            for event in slot_events:
                _,s=lookup[source,event['episode_uid']];actual=Path(next(a['path'] for a in s['artifacts'] if a['kind']=='trajectory'))
                (tracker/(event['sequence']+'.txt')).symlink_to(actual)
        command=_trackeval_command(directory/'trackers',directory/'evaluation',sources,seqmap,gt_split='train',gt_folder=TRAIN)
        command[2]=str(pinned/'scripts/run_mot_challenge.py');log=directory/'trackeval.log';began=time.monotonic()
        with log.open('x') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
        receipt={'slot':slot,'sequences':sequences,'command':command,'returncode':result.returncode,'seconds':time.monotonic()-began,
            'log_path':str(log),'log_SHA':sha256(log),'pinned_TrackEval_commit':commit,'same_metric_dataset_settings_all4_sources':True}
        write_json('data/multitarget_trackeval_v1/slot'+str(slot)+'_INVOCATION.json',receipt);invocations.append(receipt)
        if result.returncode:raise RuntimeError('actual multi-target TrackEval failed, attempt retained')
        for source in sources:
            m=trackeval_summary(parse_trackeval(directory/'evaluation',source,sequences))
            assert all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN'))
            for event in slot_events:metrics[source+'/'+event['episode_uid']]=m['per_sequence'][event['sequence']]
        print(json.dumps({'actual_multi_target_TrackEval_slot':slot,'sequences':len(sequences),'source_trajectories':len(slot_events)*4}),flush=True)
    identities={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']}
    density=defaultdict(lambda:defaultdict(Counter));stats={};detections={};whole_density={}
    for sequence in sequence_order:
        frames,_=checked_frames(sequence);gt=dancetrack_annotations(TRAIN/sequence)
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        frame_density={int(p['frame']):'sparse' if len(rows)<=4 else 'medium' if len(rows)<=8 else 'crowded' for p,rows in frames}
        whole_density[sequence]=next(iter(set(frame_density.values()))) if len(set(frame_density.values()))==1 else 'MIXED_DENSITY_WHOLE_SEQUENCE'
        for event in [e for e in events if e['sequence']==sequence]:
            ep=event['episode_uid'];target=identities[ep];base_seal=lookup['JOINT_BASELINE_SHADOW_P0',ep][1]
            baseline=read_zstd_jsonl(Path(next(a['path'] for a in base_seal['artifacts'] if a['kind']=='trace')))
            base_public=baseline[event['frame']]['target_public_id'];base_text=Path(next(a['path'] for a in base_seal['artifacts'] if a['kind']=='trajectory')).read_text()
            for source in sources:
                seal=lookup[source,ep][1];trace=read_zstd_jsonl(Path(next(a['path'] for a in seal['artifacts'] if a['kind']=='trace')))
                assert [r['frame'] for r in trace]==list(range(len(frames)))
                public=trace[event['frame']]['target_public_id'];counter=Counter();observations=[]
                detections[source+'/'+ep]=audit_unchanged_detections(base_text,Path(next(a['path'] for a in seal['artifacts'] if a['kind']=='trajectory')).read_text())
                for a,b in zip(trace,baseline,strict=True):
                    f=a['frame']
                    if f<=event['frame']:continue
                    actual_uid=assignment_map(a).get(public);base_uid=assignment_map(b).get(base_public)
                    outcome=identity_outcome(actual_uid,matched[f],target);right=outcome=='TARGET';br=matched[f].get(base_uid)==target
                    own=a['authority'].get('own_KEEP_uid',a['base_assignments'].get(str(public)));effective=bool(a['authority'].get('effective_assignment_change'))
                    observations.append({'frame':f,'C0_correct':br,'actual_correct':right,'own_KEEP_correct':matched[f].get(own)==target,'effective_action':effective})
                    values={'frames':1,'candidate_available':int(target in matched[f].values()),'visible':int(any(r['identity']==target for r in gt.get(f,[]))),
                        'correct':int(right),'N01':int(right and not br),'N10':int(br and not right),'verified_other_takeover':int(outcome=='VERIFIED_OTHER'),
                        'UNKNOWN_selected':int(outcome=='UNKNOWN_UNMATCHED'),'effective_actions':int(effective)}
                    counter.update(values);density[frame_density[f]][source].update(values)
                stats[source+'/'+ep]={'counts':dict(counter),'target_events':analyze_target_events(observations),
                    'whole_sequence_density':whole_density[sequence],'historical_role':event['diagnostic_role']}
    aggregate={}
    for source in sources:
        # Average episodes inside their scene, then scenes; seeds/targets are
        # not additional independent sequence clusters.
        per_scene={s:{k:float(np.mean([metrics[source+'/'+e['episode_uid']][k] for e in events if e['sequence']==s])) for k in
            ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')} for s in sequence_order}
        aggregate[source]={'per_sequence_episode_mean':per_scene,'macro_sequence_mean':{k:float(np.mean([m[k] for m in per_scene.values()])) for k in next(iter(per_scene.values()))}}
    result={'stage':'N72R21R1','status':'COMPLETE_ACTUAL_88_MULTI_TARGET_FULL_TRACKEVAL_RECORDS',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','actual_previously_run_source_trajectories':88,'new_MOT_rollouts':0,
        'independent_exposed_TRAIN_scenes':8,'valid_single_click_episodes':22,'TrackEval_per_source_episode':metrics,'aggregate':aggregate,
        'target_statistics':stats,'density_frame_counts':{k:{s:dict(c) for s,c in v.items()} for k,v in density.items()},
        'whole_sequence_density':whole_density,'detection_multiset_audit':detections,'actual_invocations':invocations,
        'plan_SHA':sha256(OUT/'data/MULTITARGET_FULL_TRACKEVAL_PLAN_V1.json'),'old_model_FIT_INNER_scenes_not_independent_generalization':True,
        'no_fresh_confirmation_VAL_TEST_or_outer_effect_selection':True,'next_stage_authorized':False,'whole_goal_scientific_decision':'PENDING'}
    write_json('data/MULTITARGET_FULL_TRACKEVAL_RESULT_V1.json',result)
    update_status(actual_multitarget_source_TrackEval_records=88,multitarget_historical_density_metrics_complete=True)
    print(json.dumps({'actual_multitarget_metrics_records':88,'source_HOTA_macro':{s:m['macro_sequence_mean']['HOTA'] for s,m in aggregate.items()},
        'density_whole_sequence':whole_density,'fresh_confirmation_untouched':True}),flush=True)


if __name__=='__main__':run()
