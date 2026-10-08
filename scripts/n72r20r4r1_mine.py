"""Natural multi-identity counterfactual outcomes on real TRAIN candidates.

The cache is representation-independent. Every fit reader checks sequence
exclusions; corpus labels never select a heldout policy. Full branch state is
copied before t and every future frame is independently scored and committed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import time
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_solver import AssociationAction
from sam3_intermot.association.identity_authority import AuthorityConfig
from scripts.n72r20r3_common import gt_by_frame
from scripts.n72r20r4r1_supervision import prepare_anchors,measure_branch
from scripts.n72r20r4r1_common import *


def compact(d):
    return {k:d[k] for k in ("frame","outputs","assignments","target_uid","target_public_id","memory","births","deaths")}


def future(branch,frames,start,horizon):
    return [compact(branch.step(frames[f][1],f)) for f in range(start,start+horizon)]


def baseline_origins(origins,trace,matches):
    result=dict(origins)
    for d in trace:
        for p,uid in d["assignments"].items():
            if int(p) not in result and uid is not None:result[int(p)]=matches[d["frame"]].get(uid)
    return result


def taxonomy(d,rows,matched,gt_frame,identity):
    uid=d["target_uid"];correct=[str(r["candidate_uid"]) for r in rows if matched[str(r["candidate_uid"])]==identity]
    visible=any(p==identity for p,_ in gt_frame)
    if not visible:kind="T5_VALID_REJECTION"
    elif not correct:kind="T4_CANDIDATE_MISSING"
    elif matched.get(uid)==identity:kind="T0_ALREADY_CORRECT"
    elif uid is None:kind="T2_RECOVERABLE_NONE"
    else:kind="T1_RECOVERABLE_WRONG_MATCH"
    owned=any(int(p)!=d["target_public_id"] and u in correct for p,u in d["assignments"].items())
    return kind,owned,correct,visible


def mine(sequence):
    torch.set_num_threads(1)
    target_path=ASSETS/"counterfactual"/f"{sequence}.jsonl.zst"
    summary_path=OUT/"counterfactual/per_sequence"/f"{sequence}.json"
    if target_path.exists() and summary_path.exists():
        sealed=read_json(summary_path)
        if sha256(target_path)!=sealed["artifact"]["sha256"]:raise RuntimeError("counterfactual SHA mismatch")
        return sealed
    check_storage(reserve_mib=3)
    frames=load_frames(sequence);gt=gt_by_frame(DATASET/"train"/sequence/"gt/gt.txt")
    anchors,matches=prepare_anchors(sequence,frames,gt)
    summaries={};tax=Counter();actions=Counter();long_records=[];started=time.perf_counter()

    def rows_generator():
        for identity,prepared in sorted(anchors.items()):
            event=prepared["event"];anchor_frame=event["event_frame"]
            tracker=OpportunityTracker(config=AuthorityConfig(source="raw"),event=event,bank=make_bank(),
                intervention_policy=InterventionPolicy(source="raw"),audit_hashes=False)
            origins={};count=positive=negative=neutral=oracle=changed=0;selected={"positive":[],"harmful":[],"neutral":[]}
            for payload,candidates in frames:
                f=int(payload["frame"])
                sampled=f>anchor_frame and (f-anchor_frame<=5 or (f-anchor_frame)%20==0) and f+5<len(frames)
                before=tracker.clone() if sampled else None
                baseline=tracker.step(candidates,f,collect_proposals=sampled,all_candidates=True)
                origins=baseline_origins(origins,[baseline],matches)
                if tracker.target_public is not None:origins[tracker.target_public]=identity
                if not sampled:continue
                target=tracker.target_public
                kind,owned,correct,visible=taxonomy(baseline,candidates,matches[f],gt.get(f,[]),identity)
                tax[kind]+=1;tax["T3_COMPETITOR_OWNED"]+=int(owned)
                btrace=[compact(baseline)]+future(tracker.clone(),frames,f+1,5)
                local_origins=baseline_origins(origins,btrace,matches)
                current_matrix=np.asarray(baseline["base_matrix"])
                event_record={"record_type":"event","sequence":sequence,"identity_key":[sequence,identity],"frame":f,
                    "anchor_frame":anchor_frame,"anchor_candidate_uid":event["target_candidate_uid"],
                    "target_public":target,"candidate_uids":[str(r["candidate_uid"]) for r in candidates],
                    "public_axis":baseline["states_before_commit_axis"],"base_scores":current_matrix.tolist(),
                    "base_assignments":baseline["base_assignments"],"kind":kind,"competitor_owned":owned,
                    "oracle_correct_uids":correct,"target_visible":visible,"runtime_gt_read":False,"labels_posthoc_only":True}
                yield event_record
                # Explicit KEEP reference prevents all-negative collapse being
                # hidden by an incorrect absolute contamination penalty.
                keep=measure_branch(btrace,btrace,matches,local_origins,target,identity)
                assert keep["value"]==0
                yield {"record_type":"action","sequence":sequence,"identity_key":[sequence,identity],"frame":f,
                    "action":{"family":"KEEP","public_id":target,"candidate_uid":None},"feasible":True,
                    "H1":measure_branch(btrace[:2],btrace[:2],matches,local_origins,target,identity),"H5":keep,"is_reference":True}
                for proposal in baseline["proposals"]:
                    action=AssociationAction(**proposal["action"]);actions[action.family]+=1
                    if not proposal["feasible"]:
                        actions["infeasible"]+=1
                        yield {"record_type":"action","sequence":sequence,"identity_key":[sequence,identity],"frame":f,**proposal}
                        continue
                    branch=before.clone();current=branch.step(candidates,f,forced_action=action)
                    treatment=[compact(current)]+future(branch,frames,f+1,5)
                    h5=measure_branch(btrace,treatment,matches,local_origins,target,identity)
                    h1=measure_branch(btrace[:2],treatment[:2],matches,local_origins,target,identity)
                    oracle_flag=action.candidate_uid in correct or (not correct and action.family=="REJECT_TARGET")
                    row={"record_type":"action","sequence":sequence,"identity_key":[sequence,identity],"frame":f,
                        "action":action.to_dict(),"feasible":True,"features":proposal["features"],
                        "global_cost":proposal["global_cost"],"H1":h1,"H5":h5,
                        "oracle_correct_action":oracle_flag,"forced_assignments":current["assignments"],
                        "baseline_assignments":baseline["assignments"],"own_future_state":True,"runtime_gt_read":False,"labels_posthoc_only":True}
                    yield row
                    count+=1;positive+=int(h5["beneficial"]);negative+=int(h5["harmful"]);neutral+=int(not h5["beneficial"] and not h5["harmful"])
                    changed+=int(h5["assignment_changed"]);oracle+=int(h5["beneficial"] and oracle_flag)
                    tax["T7_GLOBAL_INTERVENTION_HARM"]+=int(h5["current_target_gain"]>0 and h5["current_other_damage"]>0 and not h5["beneficial"])
                    if f+30<len(frames):
                        stratum="positive" if h5["beneficial"] else "harmful" if h5["harmful"] else "neutral"
                        key=hashlib.sha256(f"{sequence}:{identity}:{f}:{action.family}:{action.candidate_uid}".encode()).hexdigest()
                        selection=selected[stratum]
                        if len(selection)<6 or key<max(s[0] for s in selection):
                            selection.append((key,before.clone(),deepcopy(origins),row))
                            selection.sort(key=lambda x:x[0]);del selection[6:]
            for stratum,selection in selected.items():
                for key,before,origins_at_t,row in selection:
                    f=row["frame"];action=AssociationAction(**row["action"])
                    b=before.clone();t=before.clone()
                    btrace=[compact(b.step(frames[f][1],f))]+future(b,frames,f+1,30)
                    ttrace=[compact(t.step(frames[f][1],f,forced_action=action))]+future(t,frames,f+1,30)
                    local=baseline_origins(origins_at_t,btrace,matches)
                    long={"record_type":"long_horizon","sequence":sequence,"identity_key":[sequence,identity],"frame":f,"action":row["action"],"stratum":stratum,
                        "selection_sha256":key,"H10":measure_branch(btrace[:11],ttrace[:11],matches,local,row["action"]["public_id"],identity),
                        "H30":measure_branch(btrace,ttrace,matches,local,row["action"]["public_id"],identity),"own_future_state":True,"labels_posthoc_only":True}
                    yield long
                    # Window inputs remain in RAM until the dedicated bounded
                    # evaluator consumes them; no persistent raw MOT copies.
                    long_records.append({"sequence":sequence,"identity_key":[sequence,identity],"frame":f,"stratum":stratum,"H10":long["H10"],"H30":long["H30"]})
            summaries[str(identity)]={"anchor_frame":anchor_frame,"actions":count,"beneficial":positive,"harmful":negative,"neutral":neutral,"changed":changed,"oracle_beneficial":oracle,"H30_strata":{k:len(v) for k,v in selected.items()}}
            print(json.dumps({"sequence":sequence,"identity":identity,**summaries[str(identity)]}),flush=True)
            write_json(OUT/"counterfactual/PROGRESS.json",{"sequence":sequence,"identities_completed":summaries,"elapsed_seconds":time.perf_counter()-started,"formal_outer_not_evaluated":True})
    artifact=stream_zstd(target_path,rows_generator())
    result={"sequence":sequence,"identities":summaries,"eligible_identities":len(anchors),"taxonomy":dict(tax),"action_counts":dict(actions),"artifact":artifact,
        "H30_events":len(long_records),"seconds":time.perf_counter()-started,"base_environment":"R4_CAUSAL_DYNAMIC_P0","natural_actions_only":True,"no_positive_duplication":True,"own_future_state":True,"training_reader_requires_fit_sequence":True,"representation_independent_outcomes":True,"outer_labels_cannot_select_models":True}
    write_json(summary_path,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--sequences",nargs="+",default=list(SEQUENCES));args=parser.parse_args()
    if not set(args.sequences)<=set(SEQUENCES):raise ValueError("unregistered train sequence")
    for s in args.sequences:mine(s)
