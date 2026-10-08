#!/usr/bin/env python3
"""M5/M6/M10: paired complete trajectories, GT opened only post rollout."""
from __future__ import annotations
from scripts.n72r20r4_common import *
from scripts.n72r20r4_train_authority import target_truth, matched_identity
from sam3_intermot.association.causal_state_commit import memory_metrics
from scripts.n72r20r3_common import gt_by_frame, iou


def audit_pair(sequence: str, baseline: list[dict], treatment: list[dict], *, split: str = "train") -> dict:
    if len(baseline) != len(treatment):
        raise ValueError("counterfactual frame count mismatch")
    event = events(split)[sequence]
    gt = gt_by_frame(DATASET / split / sequence / "gt/gt.txt")
    target_gt = target_truth(event,gt)
    def is_correct(trace,frame):
        row = trace[frame]
        target = row.get("target_public_id")
        candidate = next((r for r in row["outputs"] if r["public_id"] == target), None)
        return candidate is not None and matched_identity(candidate["box_xyxy"],gt.get(frame,[])) == target_gt
    def target_id(trace,frame):
        r = trace[frame]
        p = r.get("target_public_id")
        output = next((o for o in r["outputs"] if o["public_id"] == p),None)
        return None if output is None else matched_identity(output["box_xyxy"],gt.get(frame,[]))
    summary = {"sequence":sequence,"changed_frames":0,"N01":0,"N10":0,"global_conflicts":0,"id_reallocations":0,"non_target_assignment_changes":0,"persistent_corrections_1":0,"persistent_corrections_5":0,"persistent_corrections_10":0,"persistent_corrections_30":0,"target_visible_frames":0,"target_correct_frames_baseline":0,"target_correct_frames_treatment":0,"wrong_identity_takeover_frames":0,"candidate_covered_frames":0}
    interventions=[]; writes=[]; origins={}; target_runs=[]; gap=0; recoveries=0; previous_correct=False
    for frame,(b,t) in enumerate(zip(baseline,treatment)):
        truth = gt.get(frame,[])
        visible = any(identity==target_gt for identity,_ in truth)
        pre_event = frame < int(event["event_frame"])
        bc,tc=is_correct(baseline,frame),is_correct(treatment,frame)
        summary["target_visible_frames"] += int(visible and not pre_event)
        summary["target_correct_frames_baseline"] += int(bc and not pre_event)
        summary["target_correct_frames_treatment"] += int(tc and not pre_event)
        identity=target_id(treatment,frame)
        summary["wrong_identity_takeover_frames"] += int(identity is not None and identity != target_gt and not pre_event)
        if not pre_event:
            if not tc and visible:
                gap+=1
            elif gap:
                target_runs.append(gap); recoveries+=int(tc); gap=0
        changed=b["assignments"] != t["assignments"]
        n01=not bc and tc and not pre_event
        n10=bc and not tc and not pre_event
        summary["changed_frames"]+=int(changed)
        summary["N01"]+=int(n01); summary["N10"]+=int(n10)
        publics=set(b["assignments"]) | set(t["assignments"])
        target=str(t.get("target_public_id"))
        non_target_changed=[p for p in sorted(publics) if p!=target and b["assignments"].get(p)!=t["assignments"].get(p)]
        summary["non_target_assignment_changes"]+=len(non_target_changed)
        uids=[r["candidate_uid"] for r in t["outputs"]]
        collision=len(uids)-len(set(uids))
        summary["global_conflicts"]+=collision
        if collision:
            raise RuntimeError("posthoc detects global collision")
        for r in b["outputs"]:
            origins.setdefault(r["public_id"],matched_identity(r["box_xyxy"],truth))
        bcorrect={r["public_id"]:origins.get(r["public_id"]) is not None and matched_identity(r["box_xyxy"],truth)==origins.get(r["public_id"]) for r in b["outputs"]}
        tcorrect={r["public_id"]:origins.get(r["public_id"]) is not None and matched_identity(r["box_xyxy"],truth)==origins.get(r["public_id"]) for r in t["outputs"]}
        stolen=[p for p,v in bcorrect.items() if v and p!=t.get("target_public_id") and not tcorrect.get(p,False)]
        if changed:
            persistent={}
            for h in (1,5,10,30):
                future=list(range(frame+1,min(frame+h+1,len(treatment))))
                relevant=[f for f in future if any(i==target_gt for i,_ in gt.get(f,[]))]
                sustained=bool(n01 and len(future)==h and relevant and all(is_correct(treatment,f) for f in relevant))
                persistent[str(h)]={"frames_available":len(future),"target_visible_future_frames":len(relevant),"sustained_correction":sustained,"baseline_correct_future":sum(is_correct(baseline,f) for f in relevant),"treatment_correct_future":sum(is_correct(treatment,f) for f in relevant)}
                summary[f"persistent_corrections_{h}"]+=int(sustained)
            interventions.append({"sequence":sequence,"frame":frame,"baseline_target_assignment":b.get("target_uid"),"treatment_target_assignment":t.get("target_uid"),"baseline_all_public_assignments":b["assignments"],"treatment_all_public_assignments":t["assignments"],"candidate_collision":False,"non_target_changed_public_ids":non_target_changed,"other_previously_correct_tracks_damaged":stolen,"N01":n01,"N10":n10,"future_continuity":persistent,"public_ID_binding_preserved":b.get("target_public_id")==t.get("target_public_id"),"local_TrackEval_status":"PENDING_INTERVENTION_WINDOW_EVALUATION","posthoc_GT_only":True,"runtime_future_gt_used":False})
        write={**t["memory"],"sequence":sequence,"correct":tc,"posthoc_GT_only":True}
        writes.append(write)
    if gap: target_runs.append(gap)
    denominator=summary["target_visible_frames"]
    summary["clicked_target_recall_baseline"]=summary["target_correct_frames_baseline"]/denominator if denominator else None
    summary["clicked_target_recall_treatment"]=summary["target_correct_frames_treatment"]/denominator if denominator else None
    summary["successful_reacquisitions"]=recoveries
    summary["target_missing_intervals"]=target_runs
    summary["target_ID_continuity_changes"]=sum(target_id(treatment,f)!=target_id(treatment,f-1) for f in range(int(event["event_frame"])+1,len(treatment)) if target_id(treatment,f) is not None and target_id(treatment,f-1) is not None)
    summary["memory"]=memory_metrics(writes)
    summary["state_feedback_divergent_frames"]=sum(b["state_after"]!=t["state_after"] for b,t in zip(baseline,treatment))
    return {"summary":summary,"interventions":interventions,"memory_rows":writes}
