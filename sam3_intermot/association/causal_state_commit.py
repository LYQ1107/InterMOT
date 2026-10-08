"""Score-before-write commits and explicit safety denominators."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

from .global_assignment_adapter import public_map
from .identity_authority import AuthorityConfig
from .learned_identity_memory import LearnedIdentityMemoryBank


def commit_memory(bank: LearnedIdentityMemoryBank | None, config: AuthorityConfig, *, frame: int, rows: Sequence[Mapping[str, Any]], base_solver: Mapping[str, Any], solver: Mapping[str, Any], public_id: int | None, scores: np.ndarray, score_state_hash: str | None) -> dict[str, Any]:
    record = None if bank is None or public_id is None else bank.records.get(public_id)
    result = {"frame": frame, "eligible": False, "accepted": False, "candidate_uid": None, "policy": config.memory, "score_state_hash": score_state_hash, "score_before_update": True, "runtime_future_gt_used": False}
    if record is None or frame <= record.initialized_frame:
        return result
    if record.last_update_frame >= frame or record.state_hash() != score_state_hash:
        raise RuntimeError("score-before-update or causal state invariant failed")
    uid = public_map(solver).get(public_id)
    result["candidate_uid"] = uid
    by_uid = {str(r["candidate_uid"]): (i, r) for i, r in enumerate(rows)}
    if uid is None:
        return result
    i, row = by_uid[uid]
    feature = np.asarray(row["feature"])
    result["eligible"] = bool(feature.shape == (512,) and np.isfinite(feature).all() and np.linalg.norm(feature) > 1e-6)
    if not result["eligible"] or config.memory == "P0":
        return result
    if config.memory == "P1":
        accept = public_map(base_solver).get(public_id) == uid
    elif config.memory == "P2":
        second = max([float(s) for j, s in enumerate(scores) if j != i], default=-1.0)
        accept = bool(scores[i] >= config.write_score and scores[i] - second >= config.write_margin and float(row.get("conf", 0)) >= config.write_quality and float(np.dot(record.human_anchor, feature)) >= 0.3)
    else:
        raise ValueError("unknown memory policy")
    if accept:
        # P2 uses the same frozen updater through its trusted-input consensus
        # API; acceptance is determined above and is separately logged.
        target_only = {"public_assignments": [{"public_id": public_id, "candidate_uid": uid}]}
        update = bank.update_from_consensus(frame=frame, candidate_rows=rows, base_solver=target_only, treatment_solver=target_only, source=f"R4_{config.memory}_causal_commit")
        result["accepted"] = public_id in update["updated_public_ids"]
        result["update"] = update["updates"]
    result["state_divergence_from_anchor"] = 1.0 - float(np.dot(record.human_anchor, record.current_state))
    return result


def memory_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = sum(bool(r.get("eligible")) for r in rows)
    accepted = sum(bool(r.get("accepted")) for r in rows)
    correct_opportunities = sum(bool(r.get("eligible")) and bool(r.get("correct")) for r in rows)
    correct = sum(bool(r.get("accepted")) and bool(r.get("correct")) for r in rows)
    wrong = accepted - correct
    wrong_rate = wrong / accepted if accepted else None
    retention = correct / correct_opportunities if correct_opportunities else None
    cascade = longest = 0
    first_wrong = None
    for row in rows:
        if row.get("accepted") and not row.get("correct"):
            first_wrong = first_wrong or {"sequence": row.get("sequence"), "frame": row["frame"]}
            cascade += 1
        elif row.get("accepted"):
            cascade = 0
        longest = max(longest, cascade)
    return {"eligible_write_opportunities": eligible, "accepted_writes": accepted, "correct_writes": correct, "wrong_writes": wrong, "correct_opportunities": correct_opportunities, "wrong_write_rate": wrong_rate, "correct_write_retention": retention, "wrong_write_denominator": "accepted_writes", "retention_denominator": "eligible_correct_observation_opportunities", "first_wrong_write": first_wrong, "contamination_cascade_length_accepted_writes": longest, "safety_pass": bool(accepted and wrong_rate <= 0.02 and retention is not None and retention >= 0.6)}
