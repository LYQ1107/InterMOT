#!/usr/bin/env python3
"""Posthoc score-vs-global-margin transfer, candidate coverage and state audit.

These results cannot change frozen inner/outer/VAL policies. GT is opened
after the independent runtime traces have been sealed.
"""
from __future__ import annotations
from scripts.n72r20r4_common import *
from scripts.n72r20r4_run_loso import load_trace
from scripts.n72r20r4_train_authority import target_truth, matched_identity
from scripts.n72r20r3_common import gt_by_frame, iou
from sam3_intermot.association.identity_authority import AuthorityConfig, calibrated_residual


def residual_bound(decision: dict, rows: list[dict], config: dict) -> dict:
    """A sufficient score-bound certificate, not an oracle association solve.

    The runtime stores the exact global objective loss from excluding the
    baseline target edge. Only the target column receives the residual; an
    alternative target edge (or NONE) cannot gain more than this bound.
    """
    uid = decision["base_assignments"].get(str(decision["target_public_id"]))
    axis = [str(r["candidate_uid"]) for r in rows]
    if uid not in axis:
        return {"baseline_target_assigned": False, "certified_blocked": False}
    current = axis.index(uid)
    residual = (decision["authority"] * config["strength"]
                * calibrated_residual(np.asarray(decision["identity_scores"]), AuthorityConfig(**config)))
    advantage = max([0.0] + [float(v) for i, v in enumerate(residual) if i != current]) - float(residual[current])
    global_margin = float(decision["authority_features"][2])
    return {"baseline_target_assigned": True,
            "max_possible_target_residual_advantage": advantage,
            "exact_global_assignment_margin": global_margin,
            "certified_blocked": global_margin > max(0.0, advantage) + 1e-5}


def run() -> dict:
    results={}
    for sequence in SEQUENCES:
        event=events()[sequence]
        gt=gt_by_frame(DATASET/"train"/sequence/"gt/gt.txt")
        target=target_truth(event,gt)
        frames=load_frames(sequence)
        sequences={}
        for name in ("RAW_REID","LEGACY_BASE_SCORE","B2_ADAPTER_NO_WRITES","B3_CONSENSUS_MEMORY","MEMORY_P2_RELIABLE","HISTORICAL_R3R2_ADAPTER7","SELECTED_TREATMENT"):
            trace=load_trace("dev",name,sequence)
            if len(frames)!=len(trace) or any(d["frame"]!=i for i,d in enumerate(trace)):
                raise ValueError("posthoc frame axis mismatch")
            profile=read_json(ASSETS/"dev/manifests"/name/f"{sequence}.json")
            config=profile["configuration"]
            visible=covered=competitive=win=0
            global_margins=[];score_margins=[];advantages=[]
            assigned=blocked=0;gap_resets=0;identity_changes=0;previous_identity=None
            state_hashes=[];divergences=[]
            cascade=max_cascade=0;first_wrong=None
            for frame,((_,rows),decision) in enumerate(zip(frames,trace)):
                if frame<=int(event["event_frame"]):continue
                boxes=[box for identity,box in gt.get(frame,[]) if identity==target]
                visible+=int(bool(boxes))
                positives=[i for i,r in enumerate(rows) if max((iou(r["box_xyxy"],box) for box in boxes),default=0)>=0.5]
                covered+=int(bool(positives))
                scores=np.asarray(decision["identity_scores"])
                negatives=[i for i in range(len(rows)) if i not in positives]
                if len(scores)!=len(rows):raise ValueError("posthoc score/candidate axis mismatch")
                if positives and negatives:
                    competitive+=1
                    difference=max(scores[positives])-max(scores[negatives])
                    score_margins.append(float(difference));win+=int(difference>0)
                feature=decision["authority_features"]
                global_margins.append(float(feature[2]))
                certificate=residual_bound(decision,rows,config)
                assigned+=int(certificate["baseline_target_assigned"])
                blocked+=int(certificate["certified_blocked"])
                if certificate["baseline_target_assigned"]:
                    advantages.append(certificate["max_possible_target_residual_advantage"])
                output=next((r for r in decision["outputs"] if r["public_id"]==decision["target_public_id"]),None)
                current_identity=None if output is None else matched_identity(output["box_xyxy"],gt.get(frame,[]))
                if current_identity is not None and previous_identity is not None:
                    identity_changes+=int(current_identity!=previous_identity)
                gap_resets+=int(current_identity is None and previous_identity is not None)
                previous_identity=current_identity
                write=decision["memory"]
                hash_value=write.get("score_state_hash")
                if hash_value is not None:state_hashes.append(hash_value)
                if "state_divergence_from_anchor" in write:divergences.append(write["state_divergence_from_anchor"])
                uid=write.get("candidate_uid")
                r=next((r for r in rows if r["candidate_uid"]==uid),None)
                correct=r is not None and matched_identity(r["box_xyxy"],gt.get(frame,[]))==target
                if write.get("accepted"):
                    if not correct:
                        first_wrong=first_wrong if first_wrong is not None else frame
                        cascade+=1
                    else:cascade=0
                    max_cascade=max(max_cascade,cascade)
            max_residual_pair_advantage=6*config["strength"]*config["base_scale"]
            sequences[name]={"target_visible_frames":visible,"real_candidate_covered_frames":covered,"candidate_coverage_given_visible":covered/visible if visible else None,"competitive_identity_frames":competitive,"hard_negative_identity_wins":win,"hard_negative_identity_win_rate":win/competitive if competitive else None,"negative_definition":"every other real candidate, including unmatched false detections; duplicate target-positive candidates excluded","median_positive_minus_hard_negative_margin":float(np.median(score_margins)) if score_margins else None,"median_global_assignment_margin":float(np.median(global_margins)) if global_margins else None,"max_pairwise_identity_residual_advantage_from_clip":max_residual_pair_advantage,"baseline_target_assigned_frames":assigned,"global_margin_certifies_no_target_reallocation_frames":blocked,"certified_blocked_fraction_given_assigned":blocked/assigned if assigned else None,"median_actual_max_target_residual_advantage":float(np.median(advantages)) if advantages else None,"base_scale":config["base_scale"],"strength":config["strength"],"query_state_unique_hashes":len(set(state_hashes)),"state_divergence_observation_count":len(divergences),"mean_state_divergence_from_anchor":float(np.mean(divergences)) if divergences else None,"target_identity_continuity_changes_between_consecutive_matched_outputs":identity_changes,"target_matched_to_unmatched_transitions":gap_resets,"first_wrong_write":first_wrong,"longest_wrong_accepted_write_cascade_within_sequence":max_cascade,"runtime_profile_seconds":profile["seconds"],"runtime_profile_fps":profile["fps"],"posthoc_only":True,"used_to_tune_policy":False}
        results[sequence]=sequences
    aggregate={}
    for name in next(iter(results.values())):
        values=[r[name] for r in results.values()]
        counters={k:sum(r[k] for r in values) for k in ("target_visible_frames","real_candidate_covered_frames","competitive_identity_frames","hard_negative_identity_wins","baseline_target_assigned_frames","global_margin_certifies_no_target_reallocation_frames")}
        aggregate[name]={**counters,"candidate_coverage_given_visible":counters["real_candidate_covered_frames"]/counters["target_visible_frames"] if counters["target_visible_frames"] else None,"hard_negative_identity_win_rate":counters["hard_negative_identity_wins"]/counters["competitive_identity_frames"] if counters["competitive_identity_frames"] else None,"certified_blocked_fraction_given_assigned":counters["global_margin_certifies_no_target_reallocation_frames"]/counters["baseline_target_assigned_frames"] if counters["baseline_target_assigned_frames"] else None}
    summary={"stage":STAGE,"status":"POSTHOC_COMPLETE","runtime_GT_used":False,"used_to_tune_policy":False,"per_sequence":results,"aggregate":aggregate,"score_margin_vs_global_authority_distinguished":True,"identity_rank_is_not_HOTA":True,"certificate_is_sufficient_not_necessary":True,"offline_profile_is_not_live_SAM3_end_to_end_latency":True}
    write_json(OUT/"adapter/POSTHOC_TRANSFER_DIAGNOSIS.json",summary)
    return summary


if __name__=="__main__":
    torch.set_num_threads(1)
    print(json.dumps(run(),sort_keys=True))
