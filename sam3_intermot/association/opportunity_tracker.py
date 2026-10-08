"""Full causal R4 lifecycle with current-frame constrained-edge interventions.

No historical module is patched and no baseline future state is replayed.
The KEEP path uses the original scorer, solver and machine updates exactly.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import numpy as np
from .causal_identity_tracker import CausalIdentityTracker,state_digest,margin
from .identity_state import IdentityState
from .online_associator import score_matrix_pairwise,predicted_iou
from .global_assignment_adapter import solve_global,public_map,validate_global
from .opportunity_scores import observations,decompose_scores
from .opportunity_solver import AssociationAction,action_for_candidate,solve_counterfactual_global_assignment
from .opportunity_memory import MemoryPolicy,commit_causal_memory

FEATURE_NAMES=("identity_score","identity_margin","raw_anchor_score","anchor_query_agreement",
    "base_target_score","base_column_margin","global_regret","owned_by_competitor",
    "displaced_publics","explicit_none_count","motion_consistency","quality",
    "native_same","track_gap","trusted_gap","candidate_count","reject_action",
    "recover_action","identity_rank","anchor_rank","target_lost","native_streak",
    "state_drift","memory_updates")


@dataclass(frozen=True)
class InterventionPolicy:
    family:str="C0"
    strength:float=1.
    top_k:int=5
    source:str="adapter"
    beneficial_min:float=.7
    harmful_max:float=.3
    value_min:float=0.
    native_discount:float=1.
    positive_discount:float=1.
    discount_scope:str="target"

    def __post_init__(self):
        if self.family not in tuple(f"C{i}" for i in range(7)):raise ValueError("unknown controller")
        if self.top_k not in (1,3,5):raise ValueError("unregistered proposal K")
        if self.source not in ("adapter","raw","shuffled","legacy"):raise ValueError("unknown identity source")
        if not 0<=self.native_discount<=1 or not 0<=self.positive_discount<=1:raise ValueError("reliability must be bounded")


class OpportunityTracker(CausalIdentityTracker):
    def __init__(self,*,intervention_policy=None,memory_policy=None,memory_predictor=None,native_predictor=None,audit_hashes=True,**kwargs):
        super().__init__(**kwargs)
        self.intervention_policy=intervention_policy or InterventionPolicy()
        self.memory_policy=memory_policy or MemoryPolicy(self.config.memory)
        self.memory_predictor=memory_predictor;self.native_predictor=native_predictor;self.audit_hashes=audit_hashes
        self.pending=[];self.native_streak=0;self.last_native_key=None;self.last_observation_frame=-1
        self.observed_native_streak=0
        self.contradiction_streak=0;self.trust_snapshot=None

    def _scores(self,ordered,rows,frame):
        matrix=np.asarray(score_matrix_pairwise(ordered,observations(rows),frame,None,
            reid_weights={"sim":1.5,"iou":1.,"native":.5,"gap":.1},native_bonus=3.,positive_bonus=5.),dtype=np.float64)
        p=self.intervention_policy
        if p.native_discount!=1 or p.positive_discount!=1 or self.native_predictor is not None:
            parts=decompose_scores(ordered,rows,frame)
            columns=range(len(ordered)) if p.discount_scope=="global" else [j for j,s in enumerate(ordered) if s.public_id==self.target_public]
            for j in columns:
                matrix[:,j]-=(1-p.native_discount)*(parts["native_core"][:,j]+parts["native_bonus"][:,j])+(1-p.positive_discount)*parts["positive_bonus"][:,j]
                if self.native_predictor is not None and ordered[j].public_id==self.target_public and frame>int(self.event["event_frame"]):
                    record=self.bank.records[self.target_public];query=record.current_state
                    raw=np.asarray([float(np.dot(query,r["feature"])) for r in rows])
                    for i,r in enumerate(rows):
                        wf=self.native_features(ordered[j],r,rows,i,raw,query,frame)
                        reliability=self.native_predictor.predict(wf)["beneficial"]
                        matrix[i,j]-=p.native_discount*(1-reliability)*(parts["native_core"][i,j]+parts["native_bonus"][i,j])
            matrix[parts["hard_mask"]]=-1e9
        return matrix

    def native_features(self,state,row,rows,index,scores,query,frame):
        return [float(scores[index]),margin(scores,index),float(np.dot(self.event["human_anchor"],row["feature"])),
            float(np.dot(query,self.event["human_anchor"])),predicted_iou(state,np.asarray(row["box_xyxy"]),frame),float(row.get("conf",0.)),
            min(1.,self.observed_native_streak/5.),min(1.,(frame-self.last_trusted_frame)/100.),
            float(state.last_native_tid==int(row["native_tid"]) and state.last_native_scope==row.get("native_scope")),min(1.,len(rows)/50.)]

    def _identity(self,rows,encoded):
        record=None if self.bank is None else self.bank.records.get(self.target_public)
        query=np.asarray(self.event["human_anchor"],dtype=np.float32) if record is None else record.current_state
        x=np.stack([r["feature"] for r in rows]) if rows else np.empty((0,512),dtype=np.float32)
        if self.intervention_policy.source in ("adapter","shuffled"):
            if self.adapter is None:return np.zeros(len(rows)),query
            self.adapter_calls+=1
            return self.adapter.scores(np.roll(query,137) if self.intervention_policy.source=="shuffled" else query,x,encoded),query
        return x@query,query

    def proposals(self,rows,ordered,base,solver,identity,query,frame,*,all_candidates=False):
        if self.target_public is None:return []
        j=next(j for j,s in enumerate(ordered) if s.public_id==self.target_public);s=ordered[j]
        uids=[str(r["candidate_uid"]) for r in rows];base_uid=public_map(solver).get(self.target_public)
        raw=np.asarray([float(np.dot(np.asarray(self.event["human_anchor"]),r["feature"])) for r in rows])
        def top(values):return sorted(range(len(rows)),key=lambda i:(-float(values[i]),uids[i]))[:self.intervention_policy.top_k]
        indices=sorted(set(range(len(rows))) if all_candidates else set(top(identity)+top(raw)+top(base[:,j])))
        actions=[action_for_candidate(self.target_public,uids[i],solver) for i in indices if uids[i]!=base_uid]
        actions.append(AssociationAction("REJECT_TARGET",self.target_public))
        result=[]
        for action in actions:
            forced=solve_counterfactual_global_assignment(rows,base,[s.pid for s in ordered],[s.public_id for s in ordered],action,frame=frame)
            if not forced["feasible"]:
                result.append({"action":action.to_dict(),"feasible":False,"reason":forced["reason"],"hard_constraint_violation":forced["hard_constraint_violation"]});continue
            i=uids.index(action.candidate_uid) if action.candidate_uid is not None else None
            current=uids.index(base_uid) if base_uid in uids else None
            score=-1. if i is None else float(identity[i]);baseline_score=-1. if current is None else float(identity[current])
            feature=np.asarray([score,margin(identity,i),-1. if i is None else raw[i],float(np.dot(query,self.event["human_anchor"])),
                0. if i is None else base[i,j]/5.,0. if i is None else margin(base[:,j],i)/5.,forced["global_cost"]/5.,
                float(action.family=="GLOBAL_SWAP"),len(forced["displaced_public_ids"])/10.,len(forced["unassigned_candidate_uids"])/10.,
                0. if i is None else predicted_iou(s,np.asarray(rows[i]["box_xyxy"]),frame),
                0. if i is None else min(1.,max(0.,float(rows[i].get("conf",rows[i].get("confidence",0.))))),
                0. if i is None else float(s.last_native_tid==int(rows[i]["native_tid"]) and s.last_native_scope==rows[i].get("native_scope")),
                min(1.,(frame-s.last_seen_frame)/100.),min(1.,(frame-self.last_trusted_frame)/100.),len(rows)/50.,
                float(action.family=="REJECT_TARGET"),float(base_uid is None),
                1. if i is None else sum(identity>identity[i])/max(1,len(rows)),1. if i is None else sum(raw>raw[i])/max(1,len(rows)),
                float(s.state==IdentityState.LOST),min(1.,self.native_streak/5.),
                1.-float(np.dot(query,self.event["human_anchor"])),0. if self.bank is None else self.bank.records[self.target_public].machine_update_count/1000.],dtype=np.float32)
            result.append({"action":action.to_dict(),"feasible":True,"features":feature.tolist(),"global_cost":forced["global_cost"],
                "identity_advantage":score-baseline_score,"forced":forced,"assignment_changed":forced["assignment_changed"]})
        return result

    def step(self,rows,frame,*,encoded_candidates=None,forced_action=None,collect_proposals=False,all_candidates=False,baseline_solver=None):
        if frame!=self.frame+1:raise ValueError("frames must be consumed exactly once")
        rows=list(rows)
        for r in rows:
            if any(k in r for k in ("gt","gt_id","target_gt_id","oracle","label","label_index")) or r.get("runtime_gt_read") or r.get("runtime_future_gt_used"):raise ValueError("GT is not a runtime feature")
            x=np.asarray(r["feature"]);b=np.asarray(r["box_xyxy"])
            if x.shape!=(512,) or not np.isfinite(x).all() or np.linalg.norm(x)<=1e-6 or b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:raise ValueError("invalid real observation")
        before=state_digest(self.states) if self.audit_hashes else None
        self._initialize(rows,frame)
        ordered=[s for _,s in sorted(self.states.items()) if self.config.lifecycle=="legacy_fixed" or s.state!=IdentityState.TERMINATED]
        if frame>0 and any(s.last_seen_frame>=frame for s in ordered):raise RuntimeError("current/future state read")
        base=self._scores(ordered,rows,frame);bs=solve_global(rows,base,ordered,frame=frame)
        self._click(rows,bs,frame)
        if self.target_public is not None and self.target_public not in [s.public_id for s in ordered]:
            ordered.append(self.states[self.target_public]);base=self._scores(ordered,rows,frame);bs=solve_global(rows,base,ordered,frame=frame)
        identity=np.zeros(len(rows));query=None;score_hash=None;proposals=[];solver=bs;selected=None
        postclick=self.target_public is not None and frame>int(self.event["event_frame"])
        if postclick:
            identity,query=self._identity(rows,encoded_candidates)
            if self.intervention_policy.source=="legacy":
                j=next(j for j,s in enumerate(ordered) if s.public_id==self.target_public)
                identity=1/(1+np.exp(-(base[:,j]-2)/1.5))
            record=None if self.bank is None else self.bank.records.get(self.target_public)
            score_hash=None if record is None else record.state_hash()
            if collect_proposals or self.intervention_policy.family!="C0":proposals=self.proposals(rows,ordered,base,bs,identity,query,frame,all_candidates=all_candidates)
            if forced_action is not None:
                selected=solve_counterfactual_global_assignment(rows,base,[s.pid for s in ordered],[s.public_id for s in ordered],forced_action,frame=frame)
                if not selected["feasible"]:raise ValueError(selected["reason"])
                solver=selected["solver"]
            elif self.intervention_policy.family!="C0":
                approved=[]
                for p in proposals:
                    if not p["feasible"]:continue
                    if self.intervention_policy.family=="C1":
                        value=self.intervention_policy.strength*p["identity_advantage"]-p["global_cost"]
                        prediction={"beneficial":float(value>0),"harmful":0.,"value":value}
                    else:
                        if self.controller is None:raise ValueError("learned controller checkpoint required")
                        prediction=self.controller.predict(p["features"])
                    p["prediction"]=prediction
                    if prediction["beneficial"]>=self.intervention_policy.beneficial_min and prediction["harmful"]<=self.intervention_policy.harmful_max and prediction["value"]>self.intervention_policy.value_min:approved.append((prediction["value"],p))
                if approved:
                    selected=max(approved,key=lambda pair:pair[0])[1]["forced"];solver=selected["solver"]
        elif forced_action is not None:raise ValueError("intervention before human initialization")
        validate_global(solver,rows)
        uid=public_map(solver).get(self.target_public);idx=next((i for i,r in enumerate(rows) if str(r["candidate_uid"])==uid),None)
        state=self.states.get(self.target_public)
        wf={"motion":0. if idx is None else predicted_iou(state,np.asarray(rows[idx]["box_xyxy"]),frame),"quality":0. if idx is None else float(rows[idx].get("conf",0.)),"vector":None}
        if idx is not None and postclick:
            j=ordered.index(state)
            native=(rows[idx].get("native_scope"),int(rows[idx]["native_tid"]))
            self.observed_native_streak=self.observed_native_streak+1 if native==self.last_native_key and self.last_observation_frame==frame-1 else 1
            self.last_native_key=native;self.last_observation_frame=frame
            # Action-model confirmation streak is inactive under the frozen
            # P0/P1/P2 source policies, matching the sealed mining features.
            if self.memory_policy.family in ("P3","P4","P5","P6"):self.native_streak=self.observed_native_streak
            wf["vector"]=[float(identity[idx]),margin(identity,idx),float(np.dot(self.event["human_anchor"],rows[idx]["feature"])),float(np.dot(query,self.event["human_anchor"])),wf["motion"],wf["quality"],min(1.,self.observed_native_streak/5.),min(1.,(frame-self.last_trusted_frame)/100.),float(state.last_native_tid==int(rows[idx]["native_tid"])),min(1.,len(rows)/50.)]
        elif postclick:
            self.native_streak=0;self.observed_native_streak=0;self.last_native_key=None;self.pending=[]
        commit=commit_causal_memory(self,rows=rows,frame=frame,baseline=baseline_solver or bs,solver=solver,scores=identity,score_hash=score_hash,write_features=wf)
        if commit["accepted"]:self.last_trusted_frame=frame
        by_uid={str(r["candidate_uid"]):r for r in rows};assignments=public_map(solver);final=dict(assignments);births=[];deaths=[]
        for s in ordered:
            uid=assignments.get(s.public_id)
            if uid is not None:
                r=by_uid[uid];s.update_machine(np.asarray(r["feature"]),np.asarray(r["box_xyxy"]),frame,int(r["native_tid"]),.9,update_prototype=not(frame==int(self.event["event_frame"]) and s.public_id==self.target_public),native_scope=r.get("native_scope"))
            else:
                if s.state==IdentityState.ACTIVE:s.mark_lost(frame)
                else:s.advance_lost()
                if self.config.lifecycle=="dynamic" and frame-s.last_seen_frame>self.max_lost_gap and s.public_id!=self.target_public:s.terminate();deaths.append(s.public_id)
        if self.config.lifecycle=="dynamic":
            for d in solver["assignment_rows"]:
                if d["public_id"] is None:
                    r=by_uid[d["candidate_uid"]];s=self._birth(r,frame);final[s.public_id]=str(r["candidate_uid"]);births.append(s.public_id)
        assigned=[u for u in final.values() if u is not None]
        if len(assigned)!=len(set(assigned)):raise RuntimeError("duplicate candidate ownership")
        output=[{"public_id":p,"candidate_uid":u,"box_xyxy":list(by_uid[u]["box_xyxy"]),"confidence":float(by_uid[u].get("confidence",1.) or 1.)} for p,u in sorted(final.items()) if u is not None]
        self.frame=frame
        return {"frame":frame,"outputs":output,"assignments":{str(p):u for p,u in final.items()},"base_assignments":{str(p):u for p,u in public_map(bs).items()},"target_public_id":self.target_public,"target_uid":final.get(self.target_public),"identity_scores":identity.tolist(),"memory":commit,"write_features":wf["vector"],"births":births,"deaths":deaths,"state_before":before,"state_after":state_digest(self.states) if self.audit_hashes else None,"proposals":proposals,"selected_action":None if selected is None else selected["action"],"global_assignment_changed":False if selected is None else selected["assignment_changed"],"base_matrix":base,"solver":solver,"states_before_commit_axis":[s.public_id for s in ordered],"runtime_gt_read":False,"runtime_future_gt_used":False}
