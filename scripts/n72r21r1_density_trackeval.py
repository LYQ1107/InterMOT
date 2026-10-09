"""All frozen density strata on full causal trajectories, masked-frame diagnostic.

Original frame numbers/public IDs remain unchanged. Both GT and predictions
are masked after runtime, not regenerated, concatenated or called full-policy
MOT performance. The88 whole-sequence results stay the authoritative metrics.
"""
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,HISTORY,TRAIN,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_label_corpus import verify_registered
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r3r2r3_pipeline import _trackeval_command,parse_trackeval,trackeval_summary


def mask_original_frames(text,zero_based_frames):
    return ''.join(line+'\n' for line in text.splitlines() if line.strip() and int(line.split(',',1)[0])-1 in zero_based_frames)


def run():
    whole=read_json(OUT/'data/MULTITARGET_FULL_TRACKEVAL_RESULT_V1.json')
    inputs,seals,_=verify_registered();events=read_json(OUT/'protocol/COMMITTED_MEMORY_CURRICULUM_V1.json')['events']
    valid={e['episode_uid'] for e in events};sources=read_json(OUT/'protocol/JOINT_STATE_CORPUS.json')['state_sources']
    lookup={(s['source_distribution'],s['event']['episode_uid']):(p,s) for p,s in seals if s['event']['episode_uid'] in valid}
    assert len(lookup)==88
    boundaries=read_json(OUT/'data/DENSITY_GROUPS.json');frame_sets={};GT_SHA={};slots={}
    sequences=sorted({e['sequence'] for e in events})
    for sequence in sequences:
        frames,_=checked_frames(sequence)
        frame_sets[sequence]={name:{int(p['frame']) for p,rows in frames if low<=len(rows) and (high is None or len(rows)<=high)} for name,(low,high) in
            [(n,boundaries[n]) for n in ('sparse','medium','crowded')]}
        assert set.union(*frame_sets[sequence].values())==set(range(len(frames)))
        assert sum(map(len,frame_sets[sequence].values()))==len(frames)
        GT_SHA[sequence]=sha256(TRAIN/sequence/'gt/gt.txt')
        for slot,event in enumerate(sorted([e for e in inputs['inputs'] if e['sequence']==sequence],key=lambda e:e['episode_uid'])):slots[event['episode_uid']]=slot
    plan={'stage':'N72R21R1','formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','all3_strata_all4_sources_all22_valid_events':True,
        'whole_sequence_result_SHA':sha256(OUT/'data/MULTITARGET_FULL_TRACKEVAL_RESULT_V1.json'),'density_boundaries_SHA':sha256(OUT/'data/DENSITY_GROUPS.json'),
        'frame_axis_original_not_concatenated':True,'GT_source_SHA':GT_SHA,'source_evaluator_SHA':sha256(Path(__file__)),
        'claim':'POSTHOC_DENSITY_MASKED_TRACKEVAL_DIAGNOSTIC_OF_ACTUAL_FULL_CAUSAL_OUTPUTS_NOT_FULL_POLICY_SCORE',
        'empty_strata_explicit_no_metric_fabrication':True,'no_new_tracking_or_training_or_model_selection':True,'no_confirmation_VAL_TEST':True}
    write_json('data/DENSITY_MASKED_TRACKEVAL_PLAN_V1.json',plan)
    storage(96<<20);root=ASSETS/'density_masked_trackeval_v1'
    if root.exists():raise FileExistsError('preserve previous density diagnostic attempt')
    root.mkdir(parents=True);result={};invocations=[]
    pinned=HISTORY/'third_party/MOTIP/TrackEval';assert subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()=='12c8791b303e0a0b50f753af204249e622d0281a'
    for stratum in ('sparse','medium','crowded'):
        gt_root=root/stratum/'gt';GT_artifacts=[];per_episode={}
        for sequence in sequences:
            subset=frame_sets[sequence][stratum]
            gt_path=gt_root/sequence/'gt/gt.txt';gt_path.parent.mkdir(parents=True)
            with gt_path.open('x') as f:f.write(mask_original_frames((TRAIN/sequence/'gt/gt.txt').read_text(),subset))
            info=gt_root/sequence/'seqinfo.ini'
            with info.open('x') as f:f.write((TRAIN/sequence/'seqinfo.ini').read_text())
            GT_artifacts.append({'sequence':sequence,'masked_original_frames':len(subset),'path':str(gt_path),'sha256':sha256(gt_path)})
        for slot in range(3):
            slot_events=[e for e in events if slots[e['episode_uid']]==slot];scope=sorted({e['sequence'] for e in slot_events})
            directory=root/stratum/('slot'+str(slot));directory.mkdir();seqmap=directory/'seqmap.txt'
            with seqmap.open('x') as f:f.write('name\n'+'\n'.join(scope)+'\n')
            for source in sources:
                tracker=directory/'trackers'/source/'data';tracker.mkdir(parents=True)
                for event in slot_events:
                    _,s=lookup[source,event['episode_uid']];actual=Path(next(a['path'] for a in s['artifacts'] if a['kind']=='trajectory'))
                    with (tracker/(event['sequence']+'.txt')).open('x') as f:f.write(mask_original_frames(actual.read_text(),frame_sets[event['sequence']][stratum]))
            command=_trackeval_command(directory/'trackers',directory/'evaluation',sources,seqmap,gt_split='train',gt_folder=gt_root)
            command[2]=str(pinned/'scripts/run_mot_challenge.py');log=directory/'trackeval.log';began=time.monotonic()
            with log.open('x') as stream:actual=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            receipt={'stratum':stratum,'slot':slot,'command':command,'returncode':actual.returncode,'seconds':time.monotonic()-began,
                'log_path':str(log),'log_SHA':sha256(log),'same_settings_for_all_sources':True,
                'GT_is_deliberately_masked_diagnostic_not_original_full_GT':True,'original_timeline_seqinfo_unchanged':True}
            write_json('data/density_trackeval_v1/'+stratum+'_slot'+str(slot)+'_INVOCATION.json',receipt);invocations.append(receipt)
            if actual.returncode:raise RuntimeError('masked density TrackEval failed, attempt retained')
            for source in sources:
                metrics=trackeval_summary(parse_trackeval(directory/'evaluation',source,scope))
                assert all(metrics[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN'))
                for event in slot_events:
                    if frame_sets[event['sequence']][stratum]:per_episode[source+'/'+event['episode_uid']]=metrics['per_sequence'][event['sequence']]
            print(json.dumps({'density_actual_TrackEval':stratum,'slot':slot,'source_episodes':len(slot_events)*4}),flush=True)
        supported=[s for s in sequences if frame_sets[s][stratum]];aggregate={}
        for source in sources:
            per_sequence={s:{k:float(np.mean([per_episode[source+'/'+e['episode_uid']][k] for e in events if e['sequence']==s])) for k in
                ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')} for s in supported}
            aggregate[source]={'per_sequence_episode_mean':per_sequence,'macro_supported_sequence_mean':{k:float(np.mean([m[k] for m in per_sequence.values()]))
                for k in next(iter(per_sequence.values()))} if supported else None}
        result[stratum]={'supported_sequences':supported,'zero_frame_sequences':[s for s in sequences if s not in supported],
            'masked_frame_counts':{s:len(frame_sets[s][stratum]) for s in sequences},'GT_subset_artifacts':GT_artifacts,
            'actual_density_frame_target_statistics':whole['density_frame_counts'].get(stratum,{}),'TrackEval_per_episode':per_episode,'aggregate':aggregate,
            'NOT_full_MOT_metric_and_NOT_independent_confirmation':True}
    write_json('data/DENSITY_MASKED_TRACKEVAL_RESULT_V1.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_ALL_STRATA_MASKED_FRAME_DIAGNOSTIC',
        'formal_goal_file':'outputs/N72R21R1/FINAL_GOAL.json','plan_SHA':sha256(OUT/'data/DENSITY_MASKED_TRACKEVAL_PLAN_V1.json'),
        'strata':result,'actual_invocations':invocations,'no_new_tracking_training_or_model_selection':True,
        'original88_full_policy_metrics_unchanged':True,'no_confirmation_VAL_TEST_used':True,'next_stage_authorized':False})
    update_status(all_density_strata_conditional_TrackEval_complete=True)
    print(json.dumps({'all3_density_strata_scored':True,'not_full_policy_or_generalization_PASS':True}),flush=True)


if __name__=='__main__':run()
