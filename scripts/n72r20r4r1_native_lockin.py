"""Posthoc exact native soft-evidence audit on the mined natural episodes."""
from copy import deepcopy
from collections import Counter
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_fit import decode_sequence
from scripts.n72r20r4r1_supervision import prepare_anchors
from scripts.n72r20r3_common import gt_by_frame
from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.opportunity_scores import decompose_scores
from sam3_intermot.association.global_assignment_adapter import solve_global,public_map


def run():
    result={}
    for sequence in SEQUENCES:
        done=OUT/'opportunity/native_lockin_posthoc'/f'{sequence}.json'
        if done.exists():result[sequence]=read_json(done);continue
        # Explicit training-side/posthoc soft-evidence diagnostic; not a
        # runtime GT-assisted method and not an outer policy selector.
        frames=load_frames(sequence);gt=gt_by_frame(DATASET/'train'/sequence/'gt/gt.txt');anchors,matches=prepare_anchors(sequence,frames,gt)
        events_by_key,actions,summary=decode_sequence(sequence);counts=Counter();samples=[];margins=[];regrets=[]
        for identity,prepared in anchors.items():
            tracker=CausalIdentityTracker(config=AuthorityConfig(memory='P0'),event=prepared['event'],bank=make_bank())
            for payload,rows in frames:
                f=int(payload['frame']);key=(identity,f);pre=None;pre_states={}
                if key in events_by_key and tracker.target_public is not None:
                    pre_states={p:deepcopy(s) for p,s in tracker.states.items()};pre=pre_states[tracker.target_public]
                d=tracker.step(rows,f)
                if key not in events_by_key:continue
                event=events_by_key[key];matrix=np.asarray(event['base_scores']).reshape(len(rows),len(event['public_axis']))
                if not np.array_equal(d['base_matrix'],matrix):raise RuntimeError('source mined score replay mismatch')
                counts['audited_sampled_events']+=1
                correct=event['oracle_correct_uids'];baseline_uid=d['target_uid']
                if not correct or baseline_uid in correct or baseline_uid is None or pre is None:continue
                counts['wrong_assigned_target_with_correct_candidate']+=1
                i=next(i for i,r in enumerate(rows) if str(r['candidate_uid'])==baseline_uid);j=event['public_axis'].index(event['target_public'])
                parts=decompose_scores([pre],rows,f);soft=parts['native_core'][:,0]+parts['native_bonus'][:,0]
                active=float(soft[i]);counts['wrong_assigned_native_bonus_active']+=int(active>0)
                relaxed=matrix.copy();relaxed[:,j]-=soft;relaxed[parts['hard_mask'][:,0],j]=-1e9
                solved=solve_global(rows,relaxed,[pre_states[p] for p in event['public_axis']],frame=f);after=public_map(solved).get(event['target_public'])
                counts['target_native_removal_current_assignment_changed']+=int(after!=baseline_uid)
                counts['target_native_removal_current_correct']+=int(after in correct)
                indices=[z for z,r in enumerate(rows) if str(r['candidate_uid']) in correct]
                local_margin=float(matrix[i,j]-max(matrix[z,j] for z in indices));margins.append(local_margin)
                relevant=[r for r in actions[key] if r['action']['candidate_uid'] in correct]
                regret=min((r['global_cost'] for r in relevant),default=None)
                if regret is not None:regrets.append(regret)
                if len(samples)<12:samples.append({'sequence':sequence,'identity':identity,'frame':f,'baseline_uid':baseline_uid,'oracle_correct_uids':correct,
                    'wrong_native_soft_contribution':active,'local_wrong_minus_best_correct_score':local_margin,'correct_forced_global_regret':regret,
                    'exact_full_global_target_native_removed_uid':after,'correct_after_native_removal':after in correct,'hard_negative_override_preserved':True})
        result[sequence]={'counts':dict(counts),'samples':samples,'median_wrong_minus_correct_local_score':float(np.median(margins)) if margins else None,
            'median_correct_forced_global_regret':float(np.median(regrets)) if regrets else None,'corpus_source_sha256':summary['artifact']['sha256'],
            'current_frame_diagnostic_not_future_policy':True,'GT_used_only_posthoc':True,'positive_actual_activation':0,'positive_learned_model':'NOT_RUN_NO_NATURAL_ACTIVATION'}
        write_json(done,result[sequence]);print(json.dumps({'native_soft_audit':sequence,'counts':dict(counts)}),flush=True)
    write_json(OUT/'opportunity/NATIVE_POSITIVE_LOCKIN_POSTHOC.json',{'per_sequence':result,'not_used_for_policy_selection':True})


if __name__=='__main__':torch.set_num_threads(1);run()
