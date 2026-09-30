#!/usr/bin/env python3
"""Replay the N72R20 learned-memory seam on a frozen runtime/base-score tape.

The source tape is intentionally required.  This runner never fabricates a
base association matrix from GT, never invokes SAM3, and never changes the
exact assignment solver.  Its compact source schema is documented in
``outputs/N72R20/protocol.json`` and is suitable for adapting a recovered
N72R15 runtime tape without changing the candidate axis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sam3_intermot.association.effect_assignment import solve_effect_assignment
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from sam3_intermot.association.learned_identity_state_edge import (
    build_learned_identity_state_edge_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"


@dataclass(frozen=True)
class ExplicitAssociationState:
    association_state_id: int
    public_id: int


def _map_assignments(solver: Mapping[str, Any]) -> dict[int, str | None]:
    result: dict[int, str | None] = {}
    for item in solver.get("public_assignments", []):
        public = item.get("public_id")
        if public is None:
            continue
        public = int(public)
        if public in result:
            raise ValueError(f"duplicate public assignment {public}")
        uid = item.get("candidate_uid")
        result[public] = None if uid in (None, "", "None") else str(uid)
    return result


def _matrix_hash(value: Any) -> str:
    payload = json.dumps(np.asarray(value, dtype=np.float64).tolist(), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def replay_frame(
    *,
    bank: LearnedIdentityMemoryBank,
    frame: int,
    candidate_rows: Sequence[Mapping[str, Any]],
    states: Sequence[ExplicitAssociationState],
    base_candidate_state_scores: Sequence[Sequence[float]],
    source_run_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Run one causal E2 frame and return auditable baseline/treatment output."""

    public_axis = [int(state.public_id) for state in states]
    state_axis = [int(state.association_state_id) for state in states]
    base_matrix = np.asarray(base_candidate_state_scores, dtype=np.float64)
    if base_matrix.shape != (len(candidate_rows), len(states)):
        raise ValueError(
            f"base candidate×state shape {base_matrix.shape} != {(len(candidate_rows), len(states))}"
        )
    base_solver = solve_effect_assignment(
        candidate_rows=candidate_rows,
        persistent_states=states,
        fused_state_candidate_scores=base_matrix.T,
        source_run_id=f"{source_run_id}:base",
        session_id=session_id,
        none_score=0.0,
    )
    edge = build_learned_identity_state_edge_matrix(
        bank=bank,
        candidate_rows=candidate_rows,
        public_id_axis=public_axis,
        frame=int(frame),
        base_candidate_public_scores=base_matrix,
        state_edge_scale=1.0,
    )
    fused_matrix = np.asarray(edge["fused_candidate_public_scores"], dtype=np.float64)
    treatment_solver = solve_effect_assignment(
        candidate_rows=candidate_rows,
        persistent_states=states,
        fused_state_candidate_scores=fused_matrix.T,
        source_run_id=f"{source_run_id}:treatment",
        session_id=session_id,
        none_score=0.0,
    )
    digest_before = bank.digest()
    state_update = bank.update_from_consensus(
        frame=int(frame),
        candidate_rows=candidate_rows,
        base_solver=base_solver,
        treatment_solver=treatment_solver,
    )
    digest_after = bank.digest()
    base_map = _map_assignments(base_solver)
    treatment_map = _map_assignments(treatment_solver)
    changed_public_ids = sorted(
        public for public in set(base_map) | set(treatment_map) if base_map.get(public) != treatment_map.get(public)
    )
    candidate_none_changes = [
        public
        for public in changed_public_ids
        if (base_map.get(public) is None) != (treatment_map.get(public) is None)
    ]
    return {
        "frame": int(frame),
        "candidate_uids": [str(row["candidate_uid"]) for row in candidate_rows],
        "association_state_axis": state_axis,
        "public_id_axis": public_axis,
        "base_solver": base_solver,
        "treatment_solver": treatment_solver,
        "edge": edge,
        "base_matrix_sha256": _matrix_hash(base_matrix),
        "fused_matrix_sha256": _matrix_hash(fused_matrix),
        "row_max_preserved": bool(edge["row_max_preserved"]),
        "changed_public_ids": changed_public_ids,
        "assignment_changed": bool(changed_public_ids),
        "candidate_none_changes": candidate_none_changes,
        "state_digest_before": digest_before,
        "state_digest_after": digest_after,
        "state_changed": bool(digest_before != digest_after),
        "state_update": state_update,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _validate_source(source: Mapping[str, Any]) -> None:
    if source.get("runtime_future_gt_used") is True or source.get("runtime_gt_read") is True:
        raise ValueError("source tape contains runtime GT usage")
    if not isinstance(source.get("frames"), list):
        raise ValueError("source tape requires a frames list")
    if not source.get("event_id"):
        raise ValueError("source tape requires event_id")


def run_source(source_path: Path, output_dir: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text())
    _validate_source(source)
    states = [
        ExplicitAssociationState(int(state), int(public))
        for state, public in zip(source["association_state_axis"], source["public_id_axis"])
    ]
    if len(states) != len(set(state.association_state_id for state in states)):
        raise ValueError("source association state axis contains duplicates")
    if len(states) != len(set(state.public_id for state in states)):
        raise ValueError("source public authority axis contains duplicates")
    bank = LearnedIdentityMemoryBank.from_checkpoint(
        CHECKPOINT,
        encoder_sha256=ENCODER_SHA,
        expected_encoder_sha256=ENCODER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
    )
    bank.initialize_public_identity(
        public_id=int(source["target_public_id"]),
        association_state_id=int(source["target_association_state_id"]),
        frame=int(source["event_frame"]),
        human_anchor=np.asarray(source["human_anchor"], dtype=np.float32),
        interaction_source=str(source.get("interaction_source", "simulated_from_gt")),
    )
    rows: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    for frame in source["frames"]:
        result = replay_frame(
            bank=bank,
            frame=int(frame["frame"]),
            candidate_rows=list(frame["candidate_rows"]),
            states=states,
            base_candidate_state_scores=frame["base_candidate_state_scores"],
            source_run_id=f"n72r20:{source['event_id']}:{frame['frame']}",
            session_id=f"n72r20:{source['event_id']}",
        )
        rows.append(result)
        writes.extend(result["state_update"].get("updates", []))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assignment_change_audit.json").write_text(
        json.dumps(
            {
                "stage": "N72R20",
                "variant": "N72R20_LEARNED_IDENTITY_TRUSTED",
                "source": str(source_path.resolve()),
                "frames": len(rows),
                "assignment_changed_frames": sum(row["assignment_changed"] for row in rows),
                "changed_public_id_count": sum(len(row["changed_public_ids"]) for row in rows),
                "candidate_none_change_count": sum(len(row["candidate_none_changes"]) for row in rows),
                "row_max_preservation_failures": sum(not row["row_max_preserved"] for row in rows),
                "runtime_future_gt_used": False,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with (output_dir / "memory_write_audit.jsonl").open("w") as handle:
        for item in writes:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    return {
        "status": "PASS_N72R20_ASSIGNMENT_SHADOW_REPLAY",
        "frames": len(rows),
        "assignment_changed_frames": sum(row["assignment_changed"] for row in rows),
        "memory_updates": len(writes),
        "runtime_future_gt_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/N72R20")
    args = parser.parse_args()
    result = run_source(args.source.resolve(), args.output_dir.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
