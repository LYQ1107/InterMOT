"""Actual same-setting TRAIN counterfactual-window TrackEval, not proxy metrics."""
from pathlib import Path
import json
import subprocess
import time
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, read_json, write_json, sha256, storage
from scripts.n72r21r1_label_corpus import verify_registered
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.evaluation.mot_detection_audit import audit_unchanged_detections


def run():
    storage(64<<20); inputs,seals,failed=verify_registered()
    chosen=[]
    for sequence in sorted({e['sequence'] for e in inputs['inputs'] if e['diagnostic_role']=='FIT'}):
        episodes=[(p,s) for p,s in seals if s['event']['sequence']==sequence and s['source_distribution']=='JOINT_BASELINE_SHADOW_P0']
        # Prospective first episode/event by original frame then stable UID.
        # No choice by measured benefit, initialization label, or GT-best action.
        for seal_path,source in sorted(episodes,key=lambda r:(r[1]['event']['frame'],r[1]['event']['episode_uid'])):
            found=False
            for ref in source['counterfactual_seals']:
                branch=read_json(ref['path'])
                if all(a['frames']==101 for a in branch['artifacts']):
                    chosen.append((source,branch,ref)); found=True; break
            if found: break
    assert len(chosen)==6
    root=ASSETS/'corpus/window_trackeval_v1'
    if root.exists(): raise FileExistsError('retain any prior evaluator attempt; explicit versioned recovery required')
    root.mkdir(parents=True); evaluations={}
    pinned=HISTORY/'third_party/MOTIP/TrackEval'
    commit=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip()
    assert commit=='12c8791b303e0a0b50f753af204249e622d0281a'
    for source,branch,ref in chosen:
        sequence=source['event']['sequence']; start=branch['frame']; arms=branch['artifacts']
        gt_root=root/'GT'/sequence; (gt_root/'gt').mkdir(parents=True)
        with (gt_root/'gt/gt.txt').open('x') as destination:
            for line in (TRAIN/sequence/'gt/gt.txt').read_text().splitlines():
                fields=line.split(','); frame=int(fields[0])
                if start+1<=frame<=start+101:
                    fields[0]=str(frame-start); destination.write(','.join(fields)+'\n')
        event=source['event']
        with (gt_root/'seqinfo.ini').open('x') as destination:
            destination.write(f'[Sequence]\nname={sequence}\nimDir=img1\nframeRate={event["fps"]}\nseqLength=101\nimWidth={event["width"]}\nimHeight={event["height"]}\nimExt=.jpg\n')
        tracker_root=root/'trackers'/sequence; names=[]; mapping={}; detection={}; views={}
        for slot,artifact in enumerate(arms):
            arm=read_json(artifact['path']); name='ARM'+str(slot); names.append(name)
            text=trajectory_text([dict(r,frame=r['frame']-start) for r in arm['rows']])
            path=tracker_root/name/'data'/(sequence+'.txt'); path.parent.mkdir(parents=True)
            with path.open('x') as destination: destination.write(text)
            mapping[name]={'action':arm['action'],'actual_branch_path':artifact['path'],'actual_branch_SHA':artifact['sha256']}
            views[name]=text
        keep=next(name for name in names if mapping[name]['action']['family']=='KEEP')
        for name in names: detection[name]=audit_unchanged_detections(views[keep],views[name])
        seqmap=root/(sequence+'_seqmap.txt')
        with seqmap.open('x') as destination: destination.write('name\n'+sequence+'\n')
        evaluation=root/'evaluation'/sequence; evaluation.mkdir(parents=True)
        command=_trackeval_command(tracker_root,evaluation,names,seqmap,gt_split='train',gt_folder=root/'GT')
        command[2]=str(pinned/'scripts/run_mot_challenge.py'); started=time.monotonic()
        log_path=evaluation/'trackeval.log'
        with log_path.open('x') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        receipt={'command':command,'returncode':process.returncode,'seconds':time.monotonic()-started,
            'log_path':str(log_path),'log_SHA':sha256(log_path),'TrackEval_commit':commit}
        write_json('corpus/window_trackeval_receipts/'+sequence+'.json',receipt)
        if process.returncode: raise RuntimeError('actual window evaluator failed; original attempt retained')
        metrics={name:trackeval_summary(parse_trackeval(evaluation,name,[sequence])) for name in names}
        assert all(all(m[k] is not None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')) for m in metrics.values())
        evaluations[sequence]={'episode_uid':event['episode_uid'],'original_frame':start,'frames':101,
            'branch_seal_SHA':ref['sha256'],'actions':mapping,'metrics':metrics,'KEEP_reference':keep,
            'paired_delta_vs_KEEP':{name:{k:metrics[name][k]-metrics[keep][k] for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')} for name in names},
            'unchanged_detection_multiset':detection,'GT_SHA':sha256(TRAIN/sequence/'gt/gt.txt'),
            'evaluation_only_time_origin_remap_not_runtime_rewrite':True,'receipt':receipt}
        print(json.dumps({'actual_window_TrackEval_complete':sequence,'arms':len(names),'frames':101}),flush=True)
    write_json('diagnostics/INTERVENTION_WINDOW_TRACKEVAL.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_PINNED_WINDOW_METRICS',
        'source_evaluator_SHA':sha256(Path(__file__)),'episodes':evaluations,'six_historical_FIT_sequences':True,
        'not_full_policy_MOT_metrics':True,'not_independent_confirmation':True,'no_GT_runtime_action_selection':True,
        'all_arms_same_GT_and_evaluation_config':True,'not_next_stage_authorization':True})


if __name__=='__main__': run()
