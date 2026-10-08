#!/usr/bin/env python3
"""Posthoc-only intervention-window TrackEval and single-target headroom."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from scripts.n72r20r4_common import *
from scripts.n72r20r4_trackeval import evaluate, METRICS
from scripts.n72r20r4_run_loso import load_trace
from scripts.n72r20r4_run_causal_tracker import export_run
from scripts.n72r20r4_train_authority import target_truth, matched_identity
from scripts.n72r20r3_common import gt_by_frame, iou
from scripts.n72r20r3r2r3_pipeline import run_trackeval_many, parse_trackeval, trackeval_summary


def headroom() -> dict:
    records={}
    event_map=events()
    for s in SEQUENCES:
        baseline=load_trace("dev","B1_CAUSAL_BASELINE",s)
        event=event_map[s]
        gt=gt_by_frame(DATASET/"train"/s/"gt/gt.txt")
        target=target_truth(event,gt)
        frames=load_frames(s)
        oracle=deepcopy(baseline)
        candidate_coverage=0;conflicts=0;visible=0
        for frame,(_,rows) in enumerate(frames):
            if frame<int(event["event_frame"]):continue
            boxes=[box for identity,box in gt.get(frame,[]) if identity==target]
            visible+=int(bool(boxes))
            candidates=[r for r in rows if max((iou(r["box_xyxy"],box) for box in boxes),default=0)>=0.5]
            candidate_coverage+=int(bool(candidates))
            p=oracle[frame]["target_public_id"]
            oracle[frame]["outputs"]=[r for r in oracle[frame]["outputs"] if r["public_id"]!=p]
            if candidates:
                r=max(candidates,key=lambda r:max(iou(r["box_xyxy"],box) for box in boxes))
                uid=str(r["candidate_uid"])
                owners=[o for o in oracle[frame]["outputs"] if o["candidate_uid"]==uid]
                conflicts+=len(owners)
                oracle[frame]["outputs"]=[o for o in oracle[frame]["outputs"] if o["candidate_uid"]!=uid]
                oracle[frame]["outputs"].append({"public_id":p,"candidate_uid":uid,"box_xyxy":r["box_xyxy"],"confidence":float(r.get("confidence",1.0) or 1.0)})
        record=export_run("POSTHOC_SINGLE_TARGET_ORACLE",s,oracle,{"oracle_only":True,"runtime_eligible":False,"future_GT_used_offline":True},group="headroom")
        # Baseline is exported only as a diagnostic reference; it is not a
        # newly claimed independent A/A tracker.
        export_run("BASELINE",s,baseline,{"source":"R4 sealed causal baseline","oracle_only":False},group="headroom")
        records[s]={"target_visible_frames":visible,"real_candidate_covered_frames":candidate_coverage,"candidate_coverage":candidate_coverage/visible if visible else None,"oracle_reallocated_other_owner_frames":conflicts,"oracle_trajectory_manifest":record}
    evaluation=evaluate("headroom",["BASELINE","POSTHOC_SINGLE_TARGET_ORACLE"],list(SEQUENCES))
    b=evaluation["metrics"]["BASELINE"];o=evaluation["metrics"]["POSTHOC_SINGLE_TARGET_ORACLE"]
    result={"status":"POSTHOC_DIAGNOSTIC_COMPLETE","oracle_runtime_eligible":False,"not_a_guaranteed_global_upper_bound":True,"reason":"best correct clicked-target candidate can remove a competing owner; interaction with global TrackEval remains coupled","metrics":evaluation["metrics"],"deltas":{k:o[k]-b[k] for k in METRICS},"per_sequence":records,"runtime_GT_inputs":False}
    write_json(OUT/"target_tracking/POSTHOC_HEADROOM.json",result)
    return result


def intervention_windows() -> dict:
    """Merge overlapping [event-1,event+30] intervals; evaluate each cluster.

    Every intervention receives the exact official local metrics of its
    containing interval. This is clustered attribution, not an assertion that
    overlapping interventions independently caused additive metric gains.
    """
    root=ASSETS/"local_interventions"
    gt_root=root/"gt"
    tracker_root=root/"trackers"
    merged_records=[];all_events=[];seq_names=[]
    for sequence in SEQUENCES:
        baseline=load_trace("dev","B1_CAUSAL_BASELINE",sequence)
        treatment=load_trace("dev","SELECTED_TREATMENT",sequence)
        events_rows=read_zstd_jsonl(ASSETS/"dev/audits/SELECTED_TREATMENT"/f"{sequence}__interventions.jsonl.zst")
        intervals=[]
        for row in events_rows:
            lo=max(0,row["frame"]-1);hi=min(len(baseline)-1,row["frame"]+30)
            if intervals and lo<=intervals[-1][1]+1:
                intervals[-1]=(intervals[-1][0],max(hi,intervals[-1][1]))
            else:intervals.append((lo,hi))
        gt=gt_by_frame(DATASET/"train"/sequence/"gt/gt.txt")
        for i,(lo,hi) in enumerate(intervals):
            name=f"{sequence}__cluster{i:04d}"
            seq_names.append(name)
            d=gt_root/name; (d/"gt").mkdir(parents=True,exist_ok=True)
            (d/"seqinfo.ini").write_text(f"[Sequence]\nname={name}\nimDir=img1\nframeRate=20\nseqLength={hi-lo+1}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")
            lines=[]
            for f in range(lo,hi+1):
                for identity,box in gt.get(f,[]):
                    x1,y1,x2,y2=box
                    lines.append(f"{f-lo+1},{identity},{x1},{y1},{x2-x1},{y2-y1},1,1,1\n")
            (d/"gt/gt.txt").write_text("".join(lines))
            for tracker,source in (("BASELINE",baseline),("TREATMENT",treatment)):
                rows=[{**r,"frame":r["frame"]-lo} for r in source[lo:hi+1]]
                export_run(tracker,name,rows,{"posthoc_window_only":True,"source_sequence":sequence,"source_frames":[lo,hi]},group="local_interventions")
            cluster={"sequence":sequence,"window_name":name,"absolute_frame_start":lo,"absolute_frame_end":hi,"interventions_in_cluster":sum(lo<=r["frame"]<=hi for r in events_rows)}
            merged_records.append(cluster)
        for r in events_rows:
            cluster=next(c for c in merged_records if c["sequence"]==sequence and c["absolute_frame_start"]<=r["frame"]<=c["absolute_frame_end"])
            r["local_TrackEval_cluster"]=cluster["window_name"]
            all_events.append(r)
    if seq_names:
        seqmap=tracker_root/"seqmap.txt";seqmap.write_text("name\n"+"\n".join(seq_names)+"\n")
        evaluated=run_trackeval_many(tracker_root,root/"eval",["BASELINE","TREATMENT"],seqmap,gt_split="train",gt_folder=gt_root)
        metrics={n:trackeval_summary(parse_trackeval(root/"eval",n,seq_names)) for n in ("BASELINE","TREATMENT")}
        for row in all_events:
            cluster=row["local_TrackEval_cluster"]
            b=metrics["BASELINE"]["per_sequence"][cluster];t=metrics["TREATMENT"]["per_sequence"][cluster]
            row["local_TrackEval_status"]="COMPLETE_CLUSTERED_WINDOW"
            row["local_TrackEval_delta"]={k:t[k]-b[k] for k in METRICS}
            row["overlapping_intervention_effects_not_additive"]=True
    else:
        metrics={};evaluated={"status":"NO_SELECTED_INTERVENTIONS"}
    write_zstd(OUT/"association/TRAJECTORY_INTERVENTION_AUDIT.jsonl.zst",all_events)
    result={"status":"COMPLETE" if seq_names else "NO_SELECTED_INTERVENTIONS","interventions":len(all_events),"clusters":merged_records,"metrics":metrics,"clustered_local_trackeval":True,"independent_event_causal_effect_claimed":False,"evaluation":evaluated}
    write_json(OUT/"association/LOCAL_TRAJECTORY_TRACK_EVAL.json",result)
    return result


if __name__=="__main__":
    torch.set_num_threads(1)
    print(json.dumps({"headroom":headroom(),"intervention_windows":intervention_windows()},sort_keys=True))
