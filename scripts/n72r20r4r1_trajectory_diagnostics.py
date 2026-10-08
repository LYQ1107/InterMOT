"""Posthoc all-output ownership and correction episodes, including native-only."""
from collections import Counter
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_evaluate import baseline_trace
from scripts.n72r20r4r1_supervision import match_frame
from scripts.n72r20r4_train_authority import target_truth
from scripts.n72r20r3_common import gt_by_frame


def sequence_audit(sequence):
    done=read_json(OUT/'authority/outer'/f'{sequence}.json')
    if done['status']!='COMPLETE':raise ValueError('diagnostics before frozen formal evaluation')
    manifests=read_json(OUT/'evaluations/outer'/f'{sequence}.json')['manifests'];frames=load_frames(sequence)
    gt=gt_by_frame(DATASET/'train'/sequence/'gt/gt.txt');event=events()[sequence];truth=target_truth(event,gt);base,_=baseline_trace(sequence)
    matches=[match_frame(rows,gt.get(f,[])) for f,(_,rows) in enumerate(frames)];result={};cache={}
    for name,axis in manifests.items():
        manifest=axis[sequence];digest=manifest['trajectory_sha256']
        if digest in cache:result[name]={**cache[digest],'exact_trajectory_alias':True};continue
        delta=manifest['delta']
        if sha256(Path(delta['path']))!=delta['sha256']:raise RuntimeError('delta SHA')
        changes={r['frame']:r['changed_ownership'] for r in read_zstd_jsonl(Path(delta['path']))}
        anchor_uid=base[event['event_frame']]['target_uid']
        pub_at_anchor={r['candidate_uid']:r['public_id'] for r in base[event['event_frame']]['outputs']}
        pub_at_anchor.update(changes.get(event['event_frame'],{}));target=pub_at_anchor[anchor_uid]
        observations={};pre_click_changed=global_changed=0;conflicts=0
        for f,b in enumerate(base):
            ownership={r['candidate_uid']:r['public_id'] for r in b['outputs']};ownership.update(changes.get(f,{}))
            conflicts+=len(ownership)-len(set(ownership.values()))
            changed=bool(changes.get(f));global_changed+=int(changed);pre_click_changed+=int(changed and f<=event['event_frame'])
            if f<=event['event_frame']:continue
            uid=next((u for u,p in ownership.items() if p==target),None)
            observations[f]={'baseline_correct':matches[f].get(b['target_uid'])==truth,'correct':matches[f].get(uid)==truth,
                'candidate_covered':truth in matches[f].values(),'target_uid_changed':uid!=b['target_uid'],'global_changed':changed}
        N01=sum(r['correct'] and not r['baseline_correct'] for r in observations.values());N10=sum(r['baseline_correct'] and not r['correct'] for r in observations.values())
        if N01!=done['statistics'][name]['target']['N01'] or N10!=done['statistics'][name]['target']['N10']:raise RuntimeError('independent N01/N10 reconstruction differs')
        onset=[f for f,r in observations.items() if r['correct'] and not r['baseline_correct'] and not (observations.get(f-1,{}).get('correct',False) and not observations.get(f-1,{}).get('baseline_correct',False))]
        persistence={}
        for h in (5,10,30):
            eligible=[f for f in onset if f+h<len(frames)]
            continuous=sum(all(observations[f+k]['correct'] and not observations[f+k]['baseline_correct'] for k in range(1,h+1)) for f in eligible)
            persistence[f'H{h}']={'onsets_with_full_horizon':len(eligible),'all_future_N01_frames':continuous,'rate':continuous/len(eligible) if eligible else None,
                'future_correct_rate':float(np.mean([observations[f+k]['correct'] for f in eligible for k in range(1,h+1)])) if eligible else None}
        entry={'N01':N01,'N10':N10,'correction_episode_onsets':onset,'persistence_including_native_only':persistence,'globally_changed_output_frames':global_changed,
            'pre_click_or_click_output_changed_frames':pre_click_changed,'same_preclick_outputs_as_R4':pre_click_changed==0,'global_public_ownership_conflicts':conflicts,
            'candidate_missing_frames':sum(not r['candidate_covered'] for r in observations.values()),'eligible_future_frames':len(observations),
            'GT_used_only_after_completed_formal_evaluation':True,'posthoc_only_not_online_selection':True}
        cache[digest]=entry;result[name]=entry
    write_json(OUT/'target_tracking/ownership_audits'/f'{sequence}.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',default=list(SEQUENCES));args=p.parse_args()
    for s in args.sequences:sequence_audit(s)
