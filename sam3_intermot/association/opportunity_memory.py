"""Causal write confirmation and non-retroactive rollback diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .global_assignment_adapter import public_map
from .causal_state_commit import commit_memory


@dataclass(frozen=True)
class MemoryPolicy:
    family: str = "P0"
    confirmation_k: int = 2
    anchor_min: float = .3
    probability_min: float = .9

    def __post_init__(self):
        if self.family not in tuple(f"P{i}" for i in range(7)): raise ValueError("unknown memory family")
        if self.confirmation_k not in (2,3): raise ValueError("confirmation K must be 2 or 3")


def commit_causal_memory(tracker, *, rows, frame, baseline, solver, scores, score_hash, write_features):
    policy=tracker.memory_policy
    if policy.family in ("P0","P1","P2"):
        return commit_memory(tracker.bank, tracker.config, frame=frame, rows=rows,
            base_solver=baseline, solver=solver, public_id=tracker.target_public,
            scores=scores, score_state_hash=score_hash)
    result={"frame":frame,"policy":policy.family,"eligible":False,"accepted":False,
        "candidate_uid":None,"score_state_hash":score_hash,"score_before_update":True,
        "runtime_future_gt_used":False,"rollback":False,"confirmation_frame":None}
    record=None if tracker.bank is None else tracker.bank.records.get(tracker.target_public)
    if record is None or frame<=record.initialized_frame:return result
    if record.last_update_frame>=frame or record.state_hash()!=score_hash:raise RuntimeError("noncausal memory write")
    anchor_hash=record.anchor_hash()
    uid=public_map(solver).get(tracker.target_public);result["candidate_uid"]=uid
    if uid is None:
        tracker.pending=[];tracker.native_streak=0;tracker.last_native_key=None
        return result
    i=next(i for i,r in enumerate(rows) if str(r["candidate_uid"])==uid);row=rows[i]
    feature=np.asarray(row["feature"],dtype=np.float32)
    result["eligible"]=True
    native=(row.get("native_scope"),int(row["native_tid"]))
    agree=bool(len(scores) and i==int(np.argmax(scores)))
    anchor_agreement=float(np.dot(record.human_anchor,feature))
    safe_causal=agree and anchor_agreement>=policy.anchor_min and write_features["motion"]>=.1 and write_features["quality"]>=.5
    tracker.pending.append({"frame":frame,"candidate_uid":uid,"native":native,"causal_agreement":safe_causal})
    tracker.pending=tracker.pending[-policy.confirmation_k:]
    confirmed=tracker.native_streak>=policy.confirmation_k and len(tracker.pending)>=policy.confirmation_k and all(p["causal_agreement"] for p in tracker.pending)
    probability=None
    if policy.family=="P3":accept=tracker.native_streak>=policy.confirmation_k and safe_causal
    elif policy.family in ("P4","P6"):accept=confirmed and safe_causal
    else:
        if tracker.memory_predictor is None:raise ValueError("P5 requires a real fit-only write-safety classifier")
        probability=float(tracker.memory_predictor.predict(write_features["vector"])["beneficial"])
        accept=probability>=policy.probability_min and anchor_agreement>=policy.anchor_min
    result.update(native_streak=tracker.native_streak,anchor_agreement=anchor_agreement,
        pending_frames=[p["frame"] for p in tracker.pending],safety_probability=probability)
    if policy.family=="P6":
        contradicted=not agree or anchor_agreement<policy.anchor_min
        tracker.contradiction_streak=tracker.contradiction_streak+1 if contradicted else 0
        if tracker.contradiction_streak>=3 and tracker.trust_snapshot is not None:
            record._current_state=tracker.trust_snapshot["state"].copy()
            record.last_update_frame=frame
            result.update(rollback=True,rollback_snapshot_frame=tracker.trust_snapshot["frame"],
                rollback_observed_frame=frame,rollback_after_state_hash=record.state_hash(),backdated=False)
            tracker.contradiction_streak=0;accept=False
    if accept:
        target_only={"public_assignments":[{"public_id":tracker.target_public,"candidate_uid":uid}]}
        update=tracker.bank.update_from_consensus(frame=frame,candidate_rows=rows,
            base_solver=target_only,treatment_solver=target_only,source=f"R4R1_{policy.family}_causal_confirmation")
        result["accepted"]=tracker.target_public in update["updated_public_ids"]
        result["confirmation_frame"]=frame;result["update"]=update["updates"]
        if result["accepted"]:
            tracker.trust_snapshot={"frame":frame,"state":record.current_state}
    if anchor_hash!=record.anchor_hash():raise RuntimeError("immutable human anchor changed")
    result["state_divergence_from_anchor"]=1.-float(np.dot(record.human_anchor,record.current_state))
    return result
