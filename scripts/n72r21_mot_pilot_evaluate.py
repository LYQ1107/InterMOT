"""Actual complete MOT TrackEval and target/write truth only after all seals."""
import json
from pathlib import Path
import numpy as np
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21_validation import cases,case_id
from scripts.n72r21_mot_pilot import PROTOCOL,CODE,checked_frames,DATASET
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3r2r3_pipeline import run_trackeval_many,parse_trackeval,trackeval_summary
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.evaluation.one_click_protocol import evaluate_episode,FrameTruth
from sam3_intermot.evaluation.one_click_identity import strict_identity_claim_metrics
from sam3_intermot.evaluation.mot_detection_audit import audit_unchanged_detections


REPAIR=OUT/'mot_pilot/EVALUATOR_ASSERT_REPAIR_R1.json'


def target_rows(trace,public,*,direct_identity):
    result=[]
    for row in trace:
        outputs={o['public_id']:o for o in row['outputs']};selected=outputs.get(public);uid=None if selected is None else selected['candidate_uid']
        decision=row['identity_decision'] if direct_identity else None
        r={'frame':row['frame'],'selected_candidate_uid':uid,'predicted_box_xyxy':None if selected is None else selected['box_xyxy'],
            'rank1_candidate_uid':decision['rank1_candidate_uid'] if decision is not None else uid,
            'candidate_available_probability':decision['candidate_available_probability'] if decision is not None else float(uid is not None),
            'rank1_identity_joint_probability':decision['rank1_identity_joint_probability'] if decision is not None else float(uid is not None),
            'target_present_probability':decision['candidate_available_probability'] if decision is not None else float(uid is not None),
            'memory_write':row['joint_identity_memory_write'],'memory_write_candidate_uid':row['joint_memory_write_candidate_uid'],
            'runtime_future_gt_used':False,'runtime_gt_used':False}
        if r['memory_write'] and r['memory_write_candidate_uid']!=uid:raise ValueError('write lacks actual target ownership')
        result.append(r)
    return result


def run():
    repair=read_json(REPAIR)
    if sha256(repair['original_evaluator_exact_backup'])!=repair['original_evaluator_sha256']:
        raise ValueError('original failed evaluator source must remain preserved')
    for path,digest in repair['pinned_primary_sources'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('pinned primary metric source changed')
    protocol=read_json(PROTOCOL);registered=[case_id(n,s) for n,s,_ in cases(protocol)];sequences=protocol['sequences'];all_seals={};sources={}
    code={p:sha256(ROOT/p) for p in CODE}
    # Every registered case for BOTH scenes must be sealed before future truth.
    for case in registered:
        for sequence in sequences:
            path=OUT/'mot_pilot/runtime_seals'/case/f'{sequence}.json';seal=read_json(path)
            if seal['code_sha256']!=code or seal['protocol_sha256']!=sha256(PROTOCOL):raise ValueError('frozen runtime source changed before evaluation')
            if not seal['one_to_one_complete_candidate_ownership_checked_every_frame']:raise ValueError('not full multi-object MOT')
            for a in seal['artifacts']:
                if sha256(a['path'])!=a['sha256']:raise ValueError('actual MOT artifact changed')
            all_seals[case,sequence]=seal;sources[f'{case}/{sequence}']=sha256(path)
    # Verify actual exported geometry/confidence, not identity-sensitive metric
    # equality. This uses no GT and preserves detection multiplicity.
    detection_audit={case:{} for case in registered}
    for sequence in sequences:
        reference_path=next(a['path'] for a in all_seals['CLICK_C0',sequence]['artifacts'] if a['kind']=='trajectory')
        reference=Path(reference_path).read_text()
        for case in registered:
            candidate_path=next(a['path'] for a in all_seals[case,sequence]['artifacts'] if a['kind']=='trajectory')
            detection_audit[case][sequence]=audit_unchanged_detections(reference,Path(candidate_path).read_text())
    labels_path=OUT/'development/INITIALIZATION_TRUTH.json';labels={r['episode_uid']:r['target_gt_identity'] for r in read_json(labels_path)['labels']}
    target_statistics={case:{} for case in registered};metadata={}
    for sequence in sequences:
        initpath=OUT/'mot_pilot/initialization'/f'{sequence}.json';init=read_json(initpath);event=init['event'];identity=labels[event['episode_uid']]
        if any(s['initialization_manifest_sha256']!=sha256(initpath) for (c,q),s in all_seals.items() if q==sequence):raise ValueError('initialization changed')
        gt=dancetrack_annotations(DATASET/sequence);frames,_=checked_frames(sequence)
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        truth=[]
        for p,rows in frames:
            f=int(p['frame']);target=[a for a in gt.get(f,[]) if a['identity']==identity]
            truth.append(FrameTruth(f,bool(target),tuple(target[0]['box']) if target else None,
                frozenset(u for u,label in matched[f].items() if label==identity),tuple(tuple(a['box']) for a in gt.get(f,[]) if a['identity']!=identity)))
        traces={case:read_zstd_jsonl(Path(next(a for a in all_seals[case,sequence]['artifacts'] if a['kind']=='trace')['path'])) for case in registered}
        if any([r['frame'] for r in t]!=list(range(len(frames))) for t in traces.values()):raise ValueError('complete original frame axis required')
        baseline=traces['CLICK_C0'];origin={}
        for r in baseline:
            for o in r['outputs']:
                label=matched[r['frame']].get(o['candidate_uid'])
                if o['public_id'] not in origin and label is not None:origin[o['public_id']]=label
        baseline_target=baseline[event['frame']]['target_public_id'];baseline_correct={r['frame']:matched[r['frame']].get(r['target_uid'])==identity for r in baseline}
        for case,trace in traces.items():
            seal=all_seals[case,sequence];click_row=trace[event['frame']]
            public=click_row['target_public_id']
            if seal['condition'].get('click') is False:
                public=next((o['public_id'] for o in click_row['outputs'] if o['candidate_uid']==init['clicked_candidate_uid']),None)
            if public is None:raise ValueError('initialization public binding missing, never substitute a target')
            runtime=target_rows(trace,public,direct_identity=seal['condition']['family']!='joint_baseline')
            first=event['frame']+1;future=runtime[first:];future_truth=truth[first:]
            known=[]
            for r in future:
                uid=r['rank1_candidate_uid'];axis=matched[r['frame']]
                if uid is not None and uid not in axis:raise ValueError('rank UID outside candidate axis')
                label=axis.get(uid);known.append(False if uid is None else None if label is None else bool(label==identity))
            summary=evaluate_episode(future,future_truth,fps=event['fps'],recording_id=sequence)
            summary['secondary_strict_identity_claim']=strict_identity_claim_metrics(future,future_truth,verified_rank1_labels=known)
            if seal['condition']['family']!='T2':summary['secondary_strict_identity_claim']['score']='UNCALIBRATED_RULE_OR_BINARY_JOINT_ASSIGNMENT_NOT_LEARNED_POSTERIOR'
            N01=N10=damage=changed=unknown_writes=verified_other_writes=0;correct={}
            for r,base in zip(trace,baseline,strict=True):
                if {o['candidate_uid'] for o in r['outputs']}!={o['candidate_uid'] for o in base['outputs']}:raise ValueError('detection output set changed')
                if r['frame']<=event['frame']:continue
                uid=next((o['candidate_uid'] for o in r['outputs'] if o['public_id']==public),None);label=matched[r['frame']].get(uid);right=label==identity
                correct[r['frame']]=right;bc=baseline_correct[r['frame']];N01+=right and not bc;N10+=bc and not right
                current={o['public_id']:o['candidate_uid'] for o in r['outputs']};old={o['public_id']:o['candidate_uid'] for o in base['outputs']}
                changed+={o['candidate_uid']:o['public_id'] for o in r['outputs']}!={o['candidate_uid']:o['public_id'] for o in base['outputs']}
                damage+=sum(pid!=baseline_target and matched[r['frame']].get(old.get(pid))==origin_identity and matched[r['frame']].get(current.get(pid))!=origin_identity for pid,origin_identity in origin.items())
                if r['joint_identity_memory_write']:
                    write_label=matched[r['frame']].get(r['joint_memory_write_candidate_uid']);unknown_writes+=write_label is None;verified_other_writes+=write_label is not None and write_label!=identity
            persistent={}
            for h in (20,50,100):
                eligible=[f for f,right in correct.items() if right and not baseline_correct[f] and f+h<len(frames)]
                count=sum(all(correct.get(f+k,False) and not baseline_correct[f+k] for k in range(1,h+1)) for f in eligible)
                persistent[f'H{h}']={'eligible_current_corrections':len(eligible),'all_future_frames_strict_target_benefit':count,
                    'rate':count/len(eligible) if eligible else None,'overlapping_corrections_not_independent_events':True}
            summary.update(episode_uid=event['episode_uid'],clicked_public_id=public,initialization_failure=False)
            summary.update(initial_clicked_UID_verified_target=matched[event['frame']].get(init['clicked_candidate_uid'])==identity,
                global_ownership_changed_frames=changed,N01=N01,N10=N10,non_target_initial_origin_correct_frame_damage=damage,
                UNKNOWN_unmatched_memory_writes=unknown_writes,verified_other_identity_memory_writes=verified_other_writes,
                strict_future_identity_benefit=persistent)
            target_statistics[case][sequence]=summary
        metadata[sequence]={'GT_sha256':sha256(DATASET/sequence/'gt/gt.txt'),'all_current_candidate_outputs_preserved':True,
            'initial_clicked_UID_verified_target':matched[event['frame']].get(init['clicked_candidate_uid'])==identity}
    storage(64<<20);seqmap=ASSETS/'mot_pilot/seqmap.txt'
    if seqmap.exists():
        if seqmap.read_text()!='name\n'+'\n'.join(sequences)+'\n':raise ValueError('seqmap changed')
    else:seqmap.parent.mkdir(parents=True,exist_ok=True);seqmap.write_text('name\n'+'\n'.join(sequences)+'\n')
    evalroot=Path(repair['new_TrackEval_output_root'])
    if evalroot.exists():raise ValueError('do not overwrite the versioned TrackEval repair output')
    result=run_trackeval_many(ASSETS/'mot_pilot/trackers',evalroot,registered,seqmap,gt_split='train',gt_folder=DATASET)
    metrics={case:trackeval_summary(parse_trackeval(evalroot,case,sequences)) for case in registered}
    for case,m in metrics.items():
        if any(m[k] is None for k in ('HOTA','AssA','IDF1','IDSW','DetA','LocA','MOTA','FP','FN')):raise ValueError('missing actual TrackEval metric')
        if set(m['per_sequence'])!=set(sequences):raise ValueError('missing actual per-sequence TrackEval metrics')
    deltas={case:{k:m[k]-metrics['CLICK_C0'][k] for k in ('HOTA','AssA','IDF1','IDSW')} for case,m in metrics.items()}
    write_json('mot_pilot/RESULT.json',{'status':'COMPLETE_TRAIN_ENGINEERING_TRANSFER_DIAGNOSTIC','final_goal_file':'outputs/N72R21/FINAL_GOAL.json',
        'protocol_sha256':sha256(PROTOCOL),'runtime_seal_sha256':sources,'actual_TrackEval_metrics':metrics,'deltas_vs_clicked_C0':deltas,
        'target_statistics':target_statistics,'metadata':metadata,'initialization_truth_source_sha256':sha256(labels_path),
        'exported_detection_multiset_audit':detection_audit,
        'original_evaluator_failure_file':'outputs/N72R21/mot_pilot/ORIGINAL_EVALUATOR_FAILURE.json',
        'versioned_evaluator_repair_sha256':sha256(REPAIR),'versioned_evaluator_repair':repair,
        'detection_audit_source_sha256':sha256(ROOT/'sam3_intermot/evaluation/mot_detection_audit.py'),
        'runtime_and_weights_unchanged_after_original_evaluator_failure':True,
        'TrackEval_command':result['command'],'TrackEval_returncode':result['returncode'],
        'TrackEval_log':{'path':result['log'],'sha256':sha256(result['log'])},
        'evaluator_source_sha256':sha256(Path(__file__)),'TrackEval_commit':submodule_head(),
        'independent_full_multi_object_rollouts_not_merged_one_target_episodes':True,
        'all16_cases_and_all3_seeds_reported':True,'two_TRAIN_sequences_not_generalization_or_cross_recording_evidence':True,
        'MOT_state_distribution_shift_explicit':True,'VAL_used':False,'test_used':False,'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps({'actual_full_MOT_pilot_complete':True,'scene_cases':32,'cases':16,'sequences':2,'TrackEval_returncode':result['returncode']}),flush=True)


def submodule_head():
    import subprocess
    return subprocess.check_output(['git','-C',str(ROOT/'third_party/MOTIP/TrackEval'),'rev-parse','HEAD'],text=True).strip()


if __name__=='__main__':run()
