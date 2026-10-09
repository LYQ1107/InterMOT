"""After all22 seals: candidate truth taxonomy distinct from causal features."""
from collections import Counter
import json
from pathlib import Path
import subprocess
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,TRAIN,read_json,write_json,sha256,storage,update_status
from scripts.n72r21r1_open_set_collect import PROTOCOL
from scripts.n72r21r1_reproduce_v2 import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching
from sam3_intermot.one_click.acib_runtime import overlap


def verify():
    protocol=read_json(PROTOCOL);assert all(sha256(ROOT/p)==s for p,s in protocol['source_code_SHA'].items());seals=[]
    for event in protocol['events']:
        path=OUT/'availability/current_axis_v1'/event['episode_uid']/'seal.json';s=read_json(path)
        assert s['protocol_SHA']==sha256(PROTOCOL) and s['source_code_SHA']==protocol['source_code_SHA']
        assert s['input_SHA']==protocol['input_SHA'] and s['full_C0_states_outputs_exactly_reproduced']
        assert not s['runtime_GT_input'] and not s['runtime_future_GT_input'] and not s['GT_history_replacement']
        assert all(sha256(a['path'])==a['sha256'] for a in s['artifacts']);seals.append((path,s))
    assert len(seals)==22
    return protocol,seals


def label_uid(uid,matched,target,available):
    if uid is None:return 'CORRECT_NONE' if not available else 'INCORRECT_NONE'
    identity=matched.get(uid)
    return 'UNKNOWN_UNMATCHED' if identity is None else 'TARGET' if identity==target else 'VERIFIED_OTHER'


def run():
    storage(64<<20);protocol,seals=verify();truth={r['episode_uid']:r['target_gt_identity'] for r in read_json(OUT/'corpus/INITIALIZATION_TRUTH.json')['labels']}
    destination=ASSETS/'availability/current_axis_v1/supervision.jsonl.zst'
    if destination.exists():raise FileExistsError('preserve partial/completed all-axis supervision')
    counts=Counter();taxonomy=Counter();GT_SHA={};groups=0;records=0
    with destination.open('xb') as stream:
        proc=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=stream)
        try:
            for sequence in sorted({s['event']['sequence'] for _,s in seals}):
                frames,index_SHA=checked_frames(sequence);gt=dancetrack_annotations(TRAIN/sequence);GT_SHA[sequence]=sha256(TRAIN/sequence/'gt/gt.txt')
                for path,seal in [(p,s) for p,s in seals if s['event']['sequence']==sequence]:
                    event=seal['event'];target=truth[event['episode_uid']]
                    assert strict_candidate_matching(frames[event['frame']][1],gt.get(event['frame'],[])).get(event['clicked_candidate_uid'])==target
                    data=read_zstd_jsonl(Path(seal['artifacts'][0]['path']))
                    assert len(data)==seal['sampled_current_groups'] and index_SHA==seal['candidate_index_SHA']
                    for record in data:
                        f=record['frame'];rows=frames[f][1];matched=strict_candidate_matching(rows,gt.get(f,[]));available=target in matched.values()
                        target_box=next((r['box'] for r in gt.get(f,[]) if r['identity']==target),None)
                        # Annotation rows contain box from the public loader;
                        # this box is taxonomy only, never a feature or candidate.
                        visible=target_box is not None;by_uid={str(r['candidate_uid']):r for r in rows}
                        assert [a['candidate_uid'] for a in record['axis']]==list(by_uid)+[None]
                        axis=[]
                        for proposal in record['axis']:
                            uid=proposal['candidate_uid'];outcome=label_uid(uid,matched,target,available);features=proposal['features'];tags=[]
                            if outcome=='VERIFIED_OTHER':
                                if features['anchor_cosine']>=.7:tags.append('VISUAL_ANCHOR_LOOKALIKE_VERIFIED_OTHER')
                                if features['native_same']:tags.append('SAME_NATIVE_CONTINUITY_VERIFIED_WRONG_IDENTITY')
                                if target_box is not None and overlap(by_uid[uid]['box_xyxy'],target_box)>0.:tags.append('GT_TARGET_BOX_OVERLAP_BUT_WRONG_UID')
                            if features['competitor_owned']:tags.append('OTHER_PUBLIC_ID_OWNED_CANDIDATE')
                            if outcome=='UNKNOWN_UNMATCHED':tags.append('UNMATCHED_OR_FALSE_CROP_UNKNOWN_NOT_VERIFIED_NEGATIVE')
                            if not available:tags.append('TARGET_UNAVAILABLE')
                            if not visible:tags.append('TARGET_GT_ABSENT')
                            counts[event['diagnostic_role']+'/'+outcome]+=1
                            for tag in tags:taxonomy[event['diagnostic_role']+'/'+tag]+=1
                            axis.append({**proposal,'current_outcome':outcome,'class':0 if outcome in ('TARGET','CORRECT_NONE') else 2 if outcome=='UNKNOWN_UNMATCHED' else 1,
                                'verified_identity_negative':outcome=='VERIFIED_OTHER','taxonomy_labels_not_features':tags})
                        output={**record,'axis':axis,'target_candidate_available_label':available,'target_visible_label':visible,
                            'runtime_seal_SHA':sha256(path),'candidate_index_SHA':index_SHA,'current_GT_labels_not_features':True,
                            'UNKNOWN_not_verified_negative':True,'no_GT_created_box_or_feature':True}
                        proc.stdin.write((json.dumps(output,sort_keys=True,allow_nan=False)+'\n').encode());groups+=1;records+=len(axis)
                print(json.dumps({'all_axis_labels_scene':sequence,'groups':groups,'rows':records}),flush=True)
            proc.stdin.close()
            if proc.wait():raise RuntimeError('all-axis label compressor failure')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
    write_json('availability/current_axis_v1/SUPERVISION_MANIFEST.json',{'stage':'N72R21R1','status':'COMPLETE_ACTUAL_CURRENT_AXIS_OPEN_SET_SUPERVISION',
        'path':str(destination),'sha256':sha256(destination),'groups':groups,'candidate_NONE_rows':records,'counts':dict(counts),
        'taxonomy_counts':dict(taxonomy),'actual_full_joint_source_rollouts':22,'GT_sources_SHA':GT_SHA,'protocol_SHA':sha256(PROTOCOL),
        'source_evaluator_SHA':sha256(Path(__file__)),'label_scope':'Current candidate correctness/NONE; no future-trajectory guarantee',
        'all22_source_seals_verified_before_truth_opened':True,'class_names':['CURRENT_CORRECT_UID_OR_CORRECT_NONE','VERIFIED_OTHER_OR_INCORRECT_NONE','UNKNOWN_UNMATCHED'],
        'correctness_and_availability_labels_separate':True,'no_VAL_TEST_confirmation_used':True,'not_actual_new_verifier_fit_yet':True})
    update_status(current_axis_open_set_supervision_complete=True,actual_open_set_source_rollouts=22)


if __name__=='__main__':run()
