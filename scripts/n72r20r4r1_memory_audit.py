"""Repeat only already-frozen P6 for rollback/semantic-write audit, SHA bound."""
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_inner import reliability_objects
from scripts.n72r20r4r1_evaluate import run_runtime,project
from scripts.n72r20r4_train_authority import target_truth
from scripts.n72r20r3_common import gt_by_frame,iou
from scripts.n72r20r4r1_supervision import match_frame


def run_fold(sequence):
    path=OUT/'memory/rollback_audit'/f'{sequence}.json'
    if path.exists():return read_json(path)
    done=read_json(OUT/'authority/outer'/f'{sequence}.json')
    if done['status']!='COMPLETE':raise ValueError('audit before frozen formal replay')
    frozen=read_json(OUT/'authority/frozen_outer'/f'{sequence}.json');case=frozen['selected']['P6']
    adapter=strict_ensemble(sequence);frames=load_frames(sequence);event=events()[sequence];encoded=project(adapter,frames)
    from sam3_intermot.association.opportunity_tracker import InterventionPolicy
    from sam3_intermot.association.opportunity_memory import MemoryPolicy
    trace,profile=run_runtime(sequence,frames,event,adapter,InterventionPolicy(**case['policy']),MemoryPolicy(**case['memory']),encoded=encoded,calibration_config=frozen['calibration'])
    formal=read_json(OUT/'evaluations/outer'/f'{sequence}.json')['manifests']['MEMORY_P6'][sequence]
    digest=hashlib.sha256(trajectory_text(trace).encode()).hexdigest()
    if digest!=formal['trajectory_sha256'] or profile['proposal_stream_sha256']!=formal['proposal_stream_sha256']:raise RuntimeError('P6 repeat differs from sealed formal run')
    gt=gt_by_frame(DATASET/'train'/sequence/'gt/gt.txt');truth=target_truth(event,gt);rollback=[];write_classification=__import__('collections').Counter()
    for ((payload,rows),d) in zip(frames,trace):
        m=d['memory'];f=d['frame']
        if m.get('rollback'):rollback.append({k:m.get(k) for k in ('frame','rollback_snapshot_frame','rollback_observed_frame','backdated')})
        if not m.get('accepted'):continue
        matched=match_frame(rows,gt.get(f,[]));row=next(r for r in rows if str(r['candidate_uid'])==m['candidate_uid'])
        nearest=max(((iou(row['box_xyxy'],b),p) for p,b in gt.get(f,[])),default=(0.,None))
        semantic=nearest[1] if nearest[0]>=.5 else None
        write_classification['accepted']+=1;write_classification['strict_correct']+=int(matched.get(m['candidate_uid'])==truth)
        write_classification['max_iou_semantic_correct_posthoc_secondary']+=int(semantic==truth)
        write_classification['unmatched_in_strict_matching']+=int(matched.get(m['candidate_uid']) is None)
    result={'sequence':sequence,'sealed_formal_MOT_SHA_reproduced':True,'sealed_proposal_stream_SHA_reproduced':True,'rollbacks':rollback,'rollback_count':len(rollback),
        'accepted_write_semantic_secondary':dict(write_classification),'primary_labels_unchanged':'preregistered one-to-one IoU>=0.5; unmatched candidates are not strict target observations',
        'semantic_secondary_never_used_for_training_thresholds_or_selection':True,'already_frozen_P6_only':True,'seconds':profile['seconds']}
    write_json(path,result);return result


if __name__=='__main__':
    import argparse
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',default=list(SEQUENCES));args=p.parse_args()
    for s in args.sequences:run_fold(s)
