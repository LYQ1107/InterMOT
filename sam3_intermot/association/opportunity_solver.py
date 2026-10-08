"""Exact constrained global counterfactuals; no GT, score boosts or target merges."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib
import json
from typing import Any,Mapping,Sequence

import numpy as np

from .public_assignment import solve_exact_public_assignment,validate_exact_public_assignment
from .global_assignment_adapter import public_map

ACTION_FAMILIES=("KEEP","SWITCH_TO_CANDIDATE","REJECT_TARGET","RECOVER_FROM_NONE","GLOBAL_SWAP")


@dataclass(frozen=True)
class AssociationAction:
    family: str
    public_id: int
    candidate_uid: str | None=None

    def __post_init__(self):
        if self.family not in ACTION_FAMILIES:raise ValueError("unknown action family")
        if self.public_id<=0:raise ValueError("invalid public authority")
        if self.family not in {"KEEP","REJECT_TARGET"} and not self.candidate_uid:raise ValueError("action requires a real candidate UID")
        if self.family in {"KEEP","REJECT_TARGET"} and self.candidate_uid is not None:raise ValueError("KEEP/REJECT cannot carry a candidate")

    def to_dict(self):return asdict(self)


def objective(result: Mapping[str,Any]) -> float:
    return float(sum(row["score"] for row in result["assignment_rows"]))


def solve_counterfactual_global_assignment(candidate_rows: Sequence[Mapping[str,Any]], base_scores: np.ndarray,
        association_state_axis: Sequence[int], public_id_axis: Sequence[int], proposed_action: AssociationAction,
        *,none_score: float=0.0,hard_negative_mask: np.ndarray | None=None,frame: int=0) -> dict[str,Any]:
    rows=[dict(r) for r in candidate_rows];matrix=np.asarray(base_scores,dtype=np.float64)
    publics=[int(p) for p in public_id_axis];states=[int(s) for s in association_state_axis]
    original=solve_exact_public_assignment(rows,matrix,states,publics,none_score=none_score,source_run_id=f"N72R20R4:{frame}",runtime_future_gt_used=False)
    hard=matrix<=-1e8
    if hard_negative_mask is not None:
        supplied=np.asarray(hard_negative_mask,dtype=bool)
        if supplied.shape!=matrix.shape:raise ValueError("hard-negative axis mismatch")
        hard=hard|supplied
    # An explicit mask supplied independently of score sentinels is also
    # enforced for every non-forced edge. All baseline comparisons use it.
    constrained=matrix.copy();constrained[hard]=-1e9
    if not np.array_equal(constrained,matrix):
        original=solve_exact_public_assignment(rows,constrained,states,publics,none_score=none_score,source_run_id=f"N72R20R4:{frame}",runtime_future_gt_used=False)
        matrix=constrained
    base_map=public_map(original)
    result={"action":proposed_action.to_dict(),"feasible":False,"status":"FEASIBILITY_FAIL","hard_constraint_violation":False,"solver":None,"baseline_solver":original,"baseline_objective":objective(original),"runtime_gt_used":False}
    def failure(reason,hard_violation=False):
        return {**result,"reason":reason,"hard_constraint_violation":hard_violation}
    target=proposed_action.public_id
    if target not in publics:return failure("TARGET_PUBLIC_NOT_ON_AXIS")
    j=publics.index(target);uid_axis=[str(r["candidate_uid"]) for r in rows]
    if proposed_action.family=="KEEP":solver=deepcopy(original)
    else:
        if proposed_action.family=="REJECT_TARGET":
            excluded_row=None
        else:
            if proposed_action.candidate_uid not in uid_axis:return failure("CANDIDATE_NOT_ON_CURRENT_AXIS")
            excluded_row=uid_axis.index(proposed_action.candidate_uid)
            if hard[excluded_row,j]:return failure("IMMUTABLE_HARD_NEGATIVE",True)
            if proposed_action.family=="RECOVER_FROM_NONE" and base_map[target] is not None:return failure("RECOVERY_REQUIRES_BASELINE_NONE")
            if proposed_action.family=="GLOBAL_SWAP" and not any(p!=target and u==proposed_action.candidate_uid for p,u in base_map.items()):return failure("GLOBAL_SWAP_REQUIRES_COMPETITOR_OWNER")
        keep_rows=[i for i in range(len(rows)) if i!=excluded_row]
        keep_cols=[k for k in range(len(publics)) if k!=j]
        reduced=solve_exact_public_assignment([rows[i] for i in keep_rows],matrix[np.ix_(keep_rows,keep_cols)],
            [states[k] for k in keep_cols],[publics[k] for k in keep_cols],none_score=none_score,
            source_run_id=f"N72R20R4:{frame}",runtime_future_gt_used=False)
        solver=deepcopy(original);merged={}
        for local,item in enumerate(reduced["assignment_rows"]):
            full_i=keep_rows[local];entry=dict(item)
            entry["candidate_index"]=int(rows[full_i].get("candidate_index",full_i))
            if entry["public_id"] is not None:
                entry["public_column_index"]=publics.index(entry["public_id"])
            else:
                entry["none_column_index"]=keep_rows[entry["none_column_index"]]
            merged[full_i]=entry
        if excluded_row is not None:
            merged[excluded_row]={"candidate_index":int(rows[excluded_row].get("candidate_index",excluded_row)),"candidate_uid":uid_axis[excluded_row],"association_state_id":states[j],"public_id":target,"public_column_index":j,"none_column_index":None,"score":float(matrix[excluded_row,j]),"status":"ASSIGNED_TO_PUBLIC_ID","source_run_id":f"N72R20R4:{frame}","session_id":None}
        solver["assignment_rows"]=[merged[i] for i in range(len(rows))]
        owned={r["public_id"]:r["candidate_uid"] for r in solver["assignment_rows"] if r["public_id"] is not None}
        for item in solver["public_assignments"]:
            item["candidate_uid"]=owned.get(item["public_id"])
            item["status"]="ASSIGNED_CANDIDATE" if item["candidate_uid"] is not None else "NO_CANDIDATE_ASSIGNED"
        solver["assigned_public_count"]=len(owned)
        solver["explicit_none_count"]=sum(r["public_id"] is None for r in solver["assignment_rows"])
        solver["solver"]="exact_forced_edge_reduced_global_Hungarian_plus_NONE"
    errors=validate_exact_public_assignment(solver)
    for i,row in enumerate(solver["assignment_rows"]):
        if row["candidate_uid"]!=uid_axis[i]:errors.append("candidate_axis_changed")
        if row["public_id"] is not None and hard[i,publics.index(row["public_id"])]:errors.append("hard_negative_used")
    if errors:raise RuntimeError(f"counterfactual invariant: {errors}")
    changed=public_map(solver)!=base_map
    return {**result,"feasible":True,"status":"FEASIBLE","solver":solver,"forced_assignment":public_map(solver),"assignment_changed":changed,
            "assignment_objective":objective(solver),"global_cost":objective(original)-objective(solver),
            "displaced_public_ids":[p for p,u in base_map.items() if p!=target and public_map(solver).get(p)!=u],
            "unassigned_candidate_uids":[r["candidate_uid"] for r in solver["assignment_rows"] if r["public_id"] is None],"hard_constraint_violation":False}


def action_for_candidate(public: int,uid: str,baseline_solver: Mapping[str,Any]) -> AssociationAction:
    ownership=public_map(baseline_solver)
    if any(p!=public and u==uid for p,u in ownership.items()):family="GLOBAL_SWAP"
    elif ownership.get(public) is None:family="RECOVER_FROM_NONE"
    else:family="SWITCH_TO_CANDIDATE"
    return AssociationAction(family,int(public),str(uid))
