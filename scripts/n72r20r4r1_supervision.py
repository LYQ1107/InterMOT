"""Offline training-only truth, stable public origins and isolated value labels."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.n72r20r3_common import iou


def match_frame(rows,truth):
    """One-to-one real-candidate matching; no synthetic candidates are made."""
    identities=sorted(truth,key=lambda pair:pair[0]);n=len(rows);m=len(identities)
    matrix=np.zeros((n,m+n),dtype=float)
    for i,r in enumerate(rows):
        for j,(_,b) in enumerate(identities):
            overlap=iou(r["box_xyxy"],b);matrix[i,j]=overlap if overlap>=.5 else -1.
    mapping={str(r["candidate_uid"]):None for r in rows}
    ri,ci=linear_sum_assignment(-matrix)
    for i,j in zip(ri,ci):
        if j<m and matrix[i,j]>=.5:mapping[str(rows[i]["candidate_uid"])]=int(identities[j][0])
    return mapping


def prepare_anchors(sequence,frames,gt):
    matches=[match_frame(rows,gt.get(int(payload["frame"]),[])) for payload,rows in frames]
    anchors={}
    for f,((_,rows),matched) in enumerate(zip(frames,matches)):
        if f+31>=len(frames):continue
        identities=sorted({v for v in matched.values() if v is not None})
        for identity in identities:
            if identity in anchors:continue
            candidates=[r for r in rows if matched[str(r["candidate_uid"])]==identity]
            boxes=[b for p,b in gt.get(f,[]) if p==identity]
            row=min(candidates,key=lambda r:(-max(iou(r["box_xyxy"],b) for b in boxes),str(r["candidate_uid"])))
            # Runtime payload carries only the single simulated confirmation.
            event={"event_frame":f,"human_anchor":np.asarray(row["feature"]).tolist(),"target_candidate_uid":str(row["candidate_uid"]),"target_box_xyxy":list(row["box_xyxy"]),"interaction_source":"simulated_from_gt"}
            anchors[identity]={"identity_key":[sequence,identity],"event":event}
    return anchors,matches


def correctness(decision,matched,origins,target,identity):
    assigned={int(p):u for p,u in decision["assignments"].items()}
    public={p:origin is not None and matched.get(assigned.get(p))==origin for p,origin in origins.items()}
    uid=assigned.get(target);available=identity in matched.values()
    # Safe target identity/availability decision, including an explicit NONE.
    public[target]=(matched.get(uid)==identity) if available else uid is None
    return public


def measure_branch(baseline,treatment,matches,origins,target,identity):
    values=[]
    for b,t in zip(baseline,treatment):
        f=b["frame"];bc=correctness(b,matches[f],origins,target,identity);tc=correctness(t,matches[f],origins,target,identity)
        values.append({"target_gain":int(tc.get(target,False))-int(bc.get(target,False)),
            "other_damage":sum(v and not tc.get(p,False) for p,v in bc.items() if p!=target),
            "wrong_override":bool(bc.get(target) and not tc.get(target)),
            "wrong_write":bool(t["memory"].get("accepted") and matches[f].get(t["memory"].get("candidate_uid"))!=identity),
            "target_correct":bool(tc.get(target)),"baseline_target_correct":bool(bc.get(target)),
            "strict_identity_correct":matches[f].get(t["target_uid"])==identity,
            "target_available":identity in matches[f].values()})
    current=values[0];future=values[1:]
    uid=treatment[0]["target_uid"]
    risk=bool(uid is not None and matches[baseline[0]["frame"]].get(uid)!=identity)
    ft=float(np.mean([v["target_gain"] for v in future])) if future else 0.
    fd=float(np.mean([v["other_damage"] for v in future])) if future else 0.
    value=current["target_gain"]+ft-2*(current["other_damage"]+fd)-int(risk)
    changed=baseline[0]["assignments"]!=treatment[0]["assignments"]
    if not changed and all(b["assignments"]==t["assignments"] for b,t in zip(baseline,treatment)):
        value=0.  # KEEP is the zero-utility reference even if its match is wrong.
    return {"horizon":len(future),"value":float(value),"beneficial":bool(changed and value>0 and not current["wrong_override"]),
        "harmful":bool(value<0 or current["wrong_override"]),"assignment_changed":changed,
        "current_target_gain":current["target_gain"],"future_target_gain":ft,
        "current_other_damage":current["other_damage"],"future_other_damage":fd,
        "wrong_override":current["wrong_override"],"potential_wrong_write":risk,
        "actual_wrong_writes":sum(v["wrong_write"] for v in values),
        "target_takeover_frames":sum(v["target_gain"]>0 for v in future),"raw_frames":values}


def supervision_gate(examples):
    positive=[e for e in examples if e["H5"]["beneficial"]]
    negative=[e for e in examples if e["H5"]["harmful"]]
    identities={tuple(e["identity_key"]) for e in positive};sequences={e["sequence"] for e in positive}
    passed=len(positive)>=30 and len(negative)>=30 and len(identities)>=3 and len(sequences)>=2
    return {"pass":passed,"naturally_changed_beneficial":len(positive),"harmful":len(negative),"beneficial_identities":len(identities),"beneficial_sequences":len(sequences),"decision":"PASS_IDENTIFIABLE_SUPERVISION" if passed else "FAIL_INTERVENTION_SUPERVISION_IDENTIFIABILITY"}
