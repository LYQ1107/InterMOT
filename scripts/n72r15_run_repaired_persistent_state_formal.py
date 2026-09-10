#!/usr/bin/env python3
"""Run the N72R15 repaired persistent identity association replay.

This runner consumes only the frozen N72R9 candidate streams and the frozen
N72R9 baseline rows.  It does not open dataset GT, instantiate SAM3, or infer
the public identity of a candidate.  The exact public solver remains the
single assignment implementation; all GT access is deferred to posthoc
scripts after the runtime artifacts are sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.association.relative_persistent_state_edge import (  # noqa: E402
    STATE_EDGE_SCALE,
    build_relative_persistent_state_edge_matrix,
    fuse_row_max_preserving,
)
from sam3_intermot.association.trusted_persistent_public_state import (  # noqa: E402
    TrustedPersistentPublicAssociationBank,
    collect_explicit_competitor_embeddings,
)
from sam3_intermot.reacquisition.target_candidate_pool import build_candidate_pool  # noqa: E402
from scripts import n72r11_on_demand_replay as replay  # noqa: E402


HORIZON = 100
VARIANTS = (
    "E0_BASELINE_B0",
    "E1I_HUMAN_RELATIVE_STATE",
    "E1J_TRUSTED_GLOBAL_RELATIVE_STATE",
)
H1 = VARIANTS[1]
G2 = VARIANTS[2]
DEFAULT_OUTPUT = ROOT / "outputs/N72R15"
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
CORRECTED_E0_ROOT = ROOT / "outputs/N72R14/attempt_02/formal"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            np.asarray(value, dtype=np.float64).tolist(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _events() -> list[dict[str, Any]]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    events = payload.get("source_event_selection", {}).get("events", [])
    if not isinstance(events, list) or len(events) != 32:
        raise RuntimeError(f"frozen protocol must contain 32 events, got {len(events) if isinstance(events, list) else None}")
    result = [dict(item) for item in events]
    if len({str(item.get("event_id")) for item in result}) != 32:
        raise RuntimeError("frozen event IDs are not unique")
    return result


def _semantic_assignment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(item.get("candidate_uid")): item.get("solver_public_id", item.get("public_id"))
        for item in row.get("candidate_rows", [])
    }


def _public_assignment_map(solver: Mapping[str, Any]) -> dict[int, str | None]:
    result: dict[int, str | None] = {}
    for item in solver.get("public_assignments", []):
        if item.get("public_id") is None:
            continue
        public = int(item["public_id"])
        if public in result:
            raise RuntimeError(f"duplicate public assignment {public}")
        uid = item.get("candidate_uid")
        result[public] = None if uid in (None, "", "None") else str(uid)
    return result


def _find_target_uid(solver: Mapping[str, Any], public_id: int) -> str | None:
    values = [
        item
        for item in solver.get("assignment_rows", [])
        if item.get("public_id") is not None and int(item["public_id"]) == int(public_id)
    ]
    if len(values) > 1:
        raise RuntimeError(f"duplicate public assignment for {public_id}")
    return None if not values else str(values[0]["candidate_uid"])


def _exact_base(
    inputs: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    *,
    event_id: str,
    variant: str,
) -> dict[str, Any]:
    pairs, vectors = replay.legacy._base_vectors(inputs, int(frame), pool)
    if pool:
        matrix = np.stack(
            [np.asarray(vectors[str(item["candidate_uid"])], dtype=np.float64) for item in pool],
            axis=0,
        )
    else:
        matrix = np.zeros((0, len(pairs)), dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError(f"{event_id}/{variant}:{frame}: base matrix invalid")
    states = replay._state_objects(pairs)
    solver = solve_effect_assignment(
        candidate_rows=pool,
        persistent_states=states,
        fused_state_candidate_scores=matrix.T,
        source_run_id=f"n72r15:{variant}:base:{event_id}:{frame}",
        session_id=f"n72r15:{variant}:{event_id}",
        none_score=0.0,
    )
    return {
        "pairs": pairs,
        "state_objects": states,
        "base_matrix": matrix,
        "base_solver": solver,
    }


def _corrected_e0_rows(event_id: str) -> list[dict[str, Any]]:
    """Read the frozen positive-geometry corrected E0 control only."""

    path = CORRECTED_E0_ROOT / _slug(event_id) / "E0_BASELINE_B0" / "runtime_frames.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"corrected E0 control is missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != HORIZON + 1:
        raise RuntimeError(f"corrected E0 control has {len(rows)} rows, expected {HORIZON + 1}: {event_id}")
    return rows


def _zero_edge(pool: Sequence[Mapping[str, Any]], public_axis: Sequence[int], frame: int) -> dict[str, Any]:
    shape = (len(pool), len(public_axis))
    zeros = np.zeros(shape, dtype=np.float64).tolist()
    return {
        "schema_version": "N72R15_RELATIVE_PERSISTENT_STATE_EDGE_V1",
        "frame": int(frame),
        "candidate_uids": [str(item["candidate_uid"]) for item in pool],
        "public_id_axis": [int(value) for value in public_axis],
        "shape": [shape[0], shape[1]],
        "state_scope": "OFF",
        "target_public_id": None,
        "state_edge_scale": float(STATE_EDGE_SCALE),
        "feature_available_by_candidate": [bool(item.get("feature") is not None) for item in pool],
        "state_available": [[False for _ in public_axis] for _ in pool],
        "evidence_available": [[False for _ in public_axis] for _ in pool],
        "appearance": zeros,
        "appearance_prototype": zeros,
        "appearance_positive": zeros,
        "appearance_negative": zeros,
        "motion": zeros,
        "native_continuity": zeros,
        "gap": zeros,
        "raw": zeros,
        "row_centered": zeros,
        "column_centered": zeros,
        "relative": zeros,
        "relative_delta": zeros,
        "runtime_future_gt_used": False,
    }


def _state_digest(bank: TrustedPersistentPublicAssociationBank | None) -> str:
    return "NONE" if bank is None else bank.digest()


def _initialize_bank(
    inputs: Mapping[str, Any],
) -> tuple[TrustedPersistentPublicAssociationBank, dict[str, Any], dict[str, Any]]:
    event_frame = int(inputs["event_frame"])
    event_id = str(inputs["event_id"])
    c0 = inputs["rows"]["c0_source"][event_frame]
    c0_pool, c0_audit = build_candidate_pool(
        list(c0.get("candidate_rows", [])),
        (),
        sequence=str(inputs["sequence"]),
        frame=event_frame,
        include_target_session=False,
        require_positive_geometry=True,
    )
    initial = _exact_base(inputs, event_frame, c0_pool, event_id=event_id, variant="INITIAL_EVENT_FRAME_B0")
    target_rows = list(inputs["rows"]["target_stream_source"][event_frame].get("candidate_rows", []))
    target_candidate = replay._strip_target_authority(target_rows[0]) if target_rows else None
    competitors, competitor_audit = collect_explicit_competitor_embeddings(
        event=inputs.get("protocol_event", {}),
        candidate_rows=c0_pool,
        solver=initial["base_solver"],
        target_public_id=int(inputs["target_public_id"]),
    )
    bank = TrustedPersistentPublicAssociationBank()
    initialization = bank.initialize_from_event_frame(
        event_frame=event_frame,
        candidate_rows=c0_pool,
        solver=initial["base_solver"],
        association_state_axis=[pair[0] for pair in initial["pairs"]],
        public_id_axis=[pair[1] for pair in initial["pairs"]],
        target_public_id=int(inputs["target_public_id"]),
        target_anchor=inputs["anchor"],
        target_box=inputs["anchor_box"],
        target_candidate=target_candidate,
        competing_embeddings=competitors,
        event_id=event_id,
    )
    initialization["event_frame_candidate_axis"] = [str(item["candidate_uid"]) for item in c0_pool]
    initialization["event_frame_base_solver_public_axis"] = [int(pair[1]) for pair in initial["pairs"]]
    initialization["event_frame_pool_audit"] = c0_audit
    initialization["competitor_audit"] = competitor_audit
    initialization["target_session_candidate_uid"] = (
        None if target_candidate is None else str(target_candidate.get("candidate_uid"))
    )
    return bank, initial, initialization


def _event_row(
    inputs: Mapping[str, Any],
    variant: str,
    *,
    initialization: Mapping[str, Any] | None,
    bank: TrustedPersistentPublicAssociationBank | None,
) -> dict[str, Any]:
    event_frame = int(inputs["event_frame"])
    state_enabled = variant in {H1, G2}
    return {
        "schema_version": "N72R15_REPAIRED_PERSISTENT_STATE_RUNTIME_FRAME_V1",
        "record_kind": "event_frame_correction",
        "variant": variant,
        "event_id": str(inputs["event_id"]),
        "sequence": str(inputs["sequence"]),
        "event_frame": event_frame,
        "frame": event_frame,
        "frame_horizon": 0,
        "target_public_id": int(inputs["target_public_id"]),
        "candidate_rows": [],
        "candidate_count": 0,
        "candidate_pool": None,
        "assignment": None,
        "base_assignment": None,
        "score_audit": None,
        "memory_read": False,
        "memory_write": bool(state_enabled and initialization and initialization.get("human_memory_written")),
        "event_frame_memory_read": False,
        "first_memory_visible_frame": event_frame + 1,
        "human_correction_provenance": (
            None
            if initialization is None
            else {
                "target_public_id": int(inputs["target_public_id"]),
                "target_session_candidate_uid": initialization.get("target_session_candidate_uid"),
                "human_write_count": int(initialization.get("human_write_count", 0)),
                "explicit_competitor_count": int(initialization.get("explicit_competitor_count", 0)),
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
            }
        ),
        "persistent_state_association": {
            "candidate_pool_policy": "MAIN_B0_ONLY",
            "target_session_candidate_in_solver": False,
            "state_scope": "OFF" if not state_enabled else ("HUMAN_TARGET_ONLY" if variant == H1 else "TRUSTED_GLOBAL"),
            "trusted_update_policy": "OFF" if variant == H1 else ("BASE_TREATMENT_CONSENSUS" if variant == G2 else "OFF"),
            "event_frame_initialization": None if initialization is None else dict(initialization),
            "event_frame_memory_read": False,
            "first_memory_visible_frame": event_frame + 1,
            "state_snapshot_after_event": [] if bank is None else bank.compact_snapshot(event_frame),
            "runtime_future_gt_used": False,
        },
        "state_update": None,
        "public_id_immutable": True,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _runtime_row(
    inputs: Mapping[str, Any],
    *,
    variant: str,
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    pool_audit: Mapping[str, Any],
    scored: Mapping[str, Any],
    bank: TrustedPersistentPublicAssociationBank | None,
    edge: Mapping[str, Any],
    fusion: Mapping[str, Any],
    solver: Mapping[str, Any],
    state_update: Mapping[str, Any],
    digest_before: str,
    digest_after: str,
    state_snapshot_before: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_candidates = replay._solver_rows(pool, solver)
    public_axis = [int(value) for value in scored["public_axis"]]
    base_matrix = np.asarray(scored["base_matrix"], dtype=np.float64)
    fused_matrix = np.asarray(fusion["fused"], dtype=np.float64)
    target_public = int(inputs["target_public_id"])
    target_uid = _find_target_uid(solver, target_public)
    base_target_uid = _find_target_uid(scored["base_solver"], target_public)
    pool_record = dict(pool_audit)
    pool_record.update(
        {
            "candidate_pool_policy": "MAIN_B0_ONLY",
            "target_session_candidate_in_solver": False,
            "target_session_candidate_count": 0,
            "candidate_rows": [replay.serializable_candidate(item, include_feature=False) for item in pool],
        }
    )
    base_map = _public_assignment_map(scored["base_solver"])
    treatment_map = _public_assignment_map(solver)
    changed_public_ids = sorted(
        public for public in set(base_map) | set(treatment_map) if base_map.get(public) != treatment_map.get(public)
    )
    state_info = {
        "candidate_pool_policy": "MAIN_B0_ONLY",
        "target_session_candidate_in_solver": False,
        "state_scope": "OFF" if variant == "E0_BASELINE_B0" else ("HUMAN_TARGET_ONLY" if variant == H1 else "TRUSTED_GLOBAL"),
        "candidate_uids": [str(value) for value in edge.get("candidate_uids", [])],
        "public_id_axis": [int(value) for value in edge.get("public_id_axis", [])],
        "shape": list(edge.get("shape", [0, 0])),
        "state_edge_scale": float(edge.get("state_edge_scale", STATE_EDGE_SCALE)),
        "appearance": edge.get("appearance", []),
        "appearance_prototype": edge.get("appearance_prototype", []),
        "appearance_positive": edge.get("appearance_positive", []),
        "appearance_negative": edge.get("appearance_negative", []),
        "motion": edge.get("motion", []),
        "native_continuity": edge.get("native_continuity", []),
        "gap": edge.get("gap", []),
        "raw_evidence_matrix": edge.get("raw", []),
        "row_centered_matrix": edge.get("row_centered", []),
        "column_centered_matrix": edge.get("column_centered", []),
        "relative_evidence_matrix": edge.get("relative", []),
        "relative_delta_matrix": edge.get("relative_delta", []),
        "proposal_matrix": np.asarray(fusion["proposal"], dtype=np.float64).tolist(),
        "row_max_before": np.asarray(fusion["row_max_before"], dtype=np.float64).tolist(),
        "row_max_after": np.asarray(fusion["row_max_after"], dtype=np.float64).tolist(),
        "row_shift": np.asarray(fusion["row_shift"], dtype=np.float64).tolist(),
        "row_max_preserved": bool(fusion["row_max_preserved"]),
        "trusted_update_policy": "OFF" if variant == H1 else ("BASE_TREATMENT_CONSENSUS" if variant == G2 else "OFF"),
        "consensus_public_count": int(state_update.get("consensus_public_count", 0)),
        "disagreement_public_count": int(state_update.get("disagreement_public_count", 0)),
        "machine_memory_write_count": len(state_update.get("machine_memory_write_public_ids", [])),
        "motion_state_write_count": len(state_update.get("motion_state_write_public_ids", [])),
        "target_consensus": state_update.get("target_consensus"),
        "target_disagreement": state_update.get("target_disagreement"),
        "state_snapshot_before_update": list(state_snapshot_before),
        "state_snapshot_after_update": [] if bank is None else bank.compact_snapshot(frame),
        "state_digest_before": digest_before,
        "state_digest_after": digest_after,
        "state_changed": bool(digest_before != digest_after),
        "treatment_changed_public_ids": changed_public_ids,
        "runtime_future_gt_used": False,
    }
    return {
        "schema_version": "N72R15_REPAIRED_PERSISTENT_STATE_RUNTIME_FRAME_V1",
        "record_kind": "future_association_frame",
        "variant": variant,
        "event_id": str(inputs["event_id"]),
        "sequence": str(inputs["sequence"]),
        "event_frame": int(inputs["event_frame"]),
        "frame": int(frame),
        "frame_horizon": int(frame) - int(inputs["event_frame"]),
        "target_public_id": target_public,
        "candidate_rows": output_candidates,
        "candidate_count": len(output_candidates),
        "candidate_pool": pool_record,
        "assignment": solver,
        "base_assignment": scored["base_solver"],
        "score_audit": {
            "association_state_axis": [int(value) for value in scored["state_axis"]],
            "public_id_axis": public_axis,
            "base_score_matrix": base_matrix.tolist(),
            "fused_score_matrix": fused_matrix.tolist(),
            "base_score_matrix_shape": list(base_matrix.shape),
            "fused_score_matrix_shape": list(fused_matrix.shape),
            "base_score_matrix_sha256": matrix_sha256(base_matrix),
            "fused_score_matrix_sha256": matrix_sha256(fused_matrix),
            "base_target_uid": base_target_uid,
            "target_uid": target_uid,
            "assignment_changed_from_base": bool(base_target_uid != target_uid),
            "global_assignment_changed": bool(changed_public_ids),
            "changed_public_ids": changed_public_ids,
            "base_assigned_public_count": int(scored["base_solver"].get("assigned_public_count", 0)),
            "treatment_assigned_public_count": int(solver.get("assigned_public_count", 0)),
            "assignment_cardinality_changed": bool(
                scored["base_solver"].get("assigned_public_count", 0) != solver.get("assigned_public_count", 0)
            ),
            "none_score": 0.0,
            "runtime_future_gt_used": False,
        },
        "persistent_state_association": state_info,
        "memory_read": bool(variant in {H1, G2}),
        "memory_write": bool(state_update.get("machine_memory_write_public_ids")),
        "event_frame_memory_read": False,
        "first_memory_visible_frame": int(inputs["event_frame"]) + 1,
        "state_update": dict(state_update),
        "raw_binding_switch": None,
        "public_id_immutable": True,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }


def _run_variant(
    inputs: Mapping[str, Any],
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown N72R15 variant {variant}")
    event_frame = int(inputs["event_frame"])
    bank: TrustedPersistentPublicAssociationBank | None = None
    initialization: dict[str, Any] | None = None
    if variant in {H1, G2}:
        bank, initial, initialization = _initialize_bank(inputs)
        del initial
    rows: list[dict[str, Any]] = [
        _event_row(inputs, variant, initialization=initialization, bank=bank)
    ]
    control_matches: list[bool] = []
    previous_target_uid: str | None = None
    row_max_checks = 0
    state_edge_nonzero_checks = 0
    consensus_updates = 0
    disagreement_rejections = 0
    future_machine_writes = 0
    for frame in range(event_frame + 1, event_frame + HORIZON + 1):
        c0 = inputs["rows"]["c0_source"][frame]
        main = list(c0.get("candidate_rows", []))
        # The target-session stream is deliberately not passed to the pool.
        # It remains correction provenance at the event frame only and is not
        # a second detection axis in any formal future solver.
        pool, pool_audit = build_candidate_pool(
            main,
            (),
            sequence=str(inputs["sequence"]),
            frame=frame,
            include_target_session=False,
            require_positive_geometry=True,
        )
        scored = _exact_base(
            inputs,
            frame,
            pool,
            event_id=str(inputs["event_id"]),
            variant=variant,
        )
        scored["state_axis"] = [pair[0] for pair in scored["pairs"]]
        scored["public_axis"] = [pair[1] for pair in scored["pairs"]]
        if variant == "E0_BASELINE_B0":
            edge = _zero_edge(pool, scored["public_axis"], frame)
            fusion = fuse_row_max_preserving(scored["base_matrix"], np.zeros_like(scored["base_matrix"]))
            solver = scored["base_solver"]
            before = after = "NONE"
            snapshot_before: list[dict[str, Any]] = []
            state_update: dict[str, Any] = {
                "frame": int(frame),
                "consensus_public_ids": [],
                "consensus_public_count": 0,
                "disagreement_public_ids": [],
                "disagreement_public_count": 0,
                "unassigned_public_ids": [],
                "machine_memory_write_public_ids": [],
                "motion_state_write_public_ids": [],
                "runtime_future_gt_used": False,
            }
        else:
            assert bank is not None
            bank.states_for_pairs(scored["pairs"])
            snapshot_before = bank.compact_snapshot(frame - 1)
            before = bank.digest()
            scope = "HUMAN_TARGET_ONLY" if variant == H1 else "TRUSTED_GLOBAL"
            edge = build_relative_persistent_state_edge_matrix(
                bank=bank,
                candidate_rows=pool,
                public_id_axis=scored["public_axis"],
                frame=frame,
                target_public_id=int(inputs["target_public_id"]),
                state_scope=scope,
            )
            delta = np.asarray(edge["relative_delta"], dtype=np.float64)
            fusion = fuse_row_max_preserving(
                scored["base_matrix"],
                delta,
                state_edge_scale=float(STATE_EDGE_SCALE),
            )
            row_max_checks += 1
            state_edge_nonzero_checks += int(np.any(np.abs(delta) > 1.0e-12))
            solver = solve_effect_assignment(
                candidate_rows=pool,
                persistent_states=scored["state_objects"],
                fused_state_candidate_scores=np.asarray(fusion["fused"], dtype=np.float64).T,
                source_run_id=f"n72r15:{variant}:{inputs['event_id']}:{frame}",
                session_id=f"n72r15:{variant}:{inputs['event_id']}",
                none_score=0.0,
            )
            if variant == H1:
                # H1 keeps the human-confirmed target state as the only
                # enabled identity evidence and ages it without accepting any
                # future machine observation.
                state_update = bank.age_without_machine(frame)
            else:
                state_update = bank.update_from_consensus(
                    frame=frame,
                    candidate_rows=pool,
                    base_solver=scored["base_solver"],
                    treatment_solver=solver,
                    target_public_id=int(inputs["target_public_id"]),
                    none_score=0.0,
                )
            after = bank.digest()
            consensus_updates += int(state_update.get("consensus_public_count", 0))
            disagreement_rejections += int(state_update.get("disagreement_public_count", 0))
            future_machine_writes += len(state_update.get("machine_memory_write_public_ids", []))
        row = _runtime_row(
            inputs,
            variant=variant,
            frame=frame,
            pool=pool,
            pool_audit=pool_audit,
            scored=scored,
            bank=bank,
            edge=edge,
            fusion=fusion,
            solver=solver,
            state_update=state_update,
            digest_before=before,
            digest_after=after,
            state_snapshot_before=snapshot_before,
        )
        current_target_uid = _find_target_uid(solver, int(inputs["target_public_id"]))
        if previous_target_uid is not None and current_target_uid is not None and current_target_uid != previous_target_uid:
            row["raw_binding_switch"] = {"from": previous_target_uid, "to": current_target_uid}
        previous_target_uid = current_target_uid or previous_target_uid
        frozen = inputs["corrected_e0_rows"][frame - event_frame]
        if variant == "E0_BASELINE_B0":
            frozen_audit = frozen.get("score_audit", {})
            control_matches.append(
                [str(item["candidate_uid"]) for item in row["candidate_rows"]]
                == [str(item["candidate_uid"]) for item in frozen.get("candidate_rows", [])]
                and scored["public_axis"] == [int(value) for value in frozen_audit.get("public_id_axis", [])]
                and matrix_sha256(scored["base_matrix"]) == matrix_sha256(frozen_audit.get("fused_score_matrix", []))
                and _semantic_assignment(row) == _semantic_assignment(frozen)
            )
        rows.append(row)
    expected_frames = list(range(event_frame, event_frame + HORIZON + 1))
    if len(rows) != HORIZON + 1 or [int(row["frame"]) for row in rows] != expected_frames:
        raise RuntimeError(f"{inputs['event_id']}/{variant}: frame axis is not 101 rows")
    for row in rows:
        for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
            if row.get(key) is not False:
                raise RuntimeError(f"{inputs['event_id']}/{variant}: runtime flag {key} is not false")
    if variant == "E0_BASELINE_B0" and not all(control_matches):
        raise RuntimeError(f"{inputs['event_id']}: B0 is not equivalent to frozen corrected E0")
    return rows, {
        "variant": variant,
        "frame_count": len(rows),
        "control_matches_frozen_e0": all(control_matches) if variant == "E0_BASELINE_B0" else None,
        "control_checked_future_frames": len(control_matches),
        "row_max_preservation_checks": row_max_checks,
        "row_max_preservation_pass_count": row_max_checks,
        "state_edge_nonzero_frame_count": state_edge_nonzero_checks,
        "consensus_update_count": consensus_updates,
        "disagreement_rejection_count": disagreement_rejections,
        "future_machine_write_count": future_machine_writes,
        "runtime_future_gt_used": False,
    }


def _validate_cross_variant_axes(variant_rows: Mapping[str, Sequence[Mapping[str, Any]]], event_id: str) -> dict[str, Any]:
    baseline = variant_rows["E0_BASELINE_B0"]
    checks = 0
    failures: list[str] = []
    for index in range(1, HORIZON + 1):
        reference_uids = [str(item.get("candidate_uid")) for item in baseline[index].get("candidate_rows", [])]
        reference_public = [int(value) for value in baseline[index].get("score_audit", {}).get("public_id_axis", [])]
        for variant in VARIANTS[1:]:
            rows = variant_rows[variant]
            uids = [str(item.get("candidate_uid")) for item in rows[index].get("candidate_rows", [])]
            publics = [int(value) for value in rows[index].get("score_audit", {}).get("public_id_axis", [])]
            checks += 1
            if uids != reference_uids or publics != reference_public:
                failures.append(f"{variant}:{index}")
    if failures:
        raise RuntimeError(f"{event_id}: BLOCKED_CANDIDATE_AXIS_MISMATCH {failures[:3]}")
    return {
        "checks": checks,
        "failures": failures,
        "candidate_axis_identical": True,
        "public_axis_identical": True,
    }


def _write_event(output_root: Path, event: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_dir = output_root / _slug(event_id)
    event_dir.mkdir(parents=True, exist_ok=True)
    variant_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        try:
            rows, stats = _run_variant(inputs, variant)
            variant_rows[variant] = rows
            frames_path = event_dir / variant / "runtime_frames.jsonl"
            atomic_jsonl(frames_path, rows)
            done = {
                "schema_version": "N72R15_POLICY_DONE_V1",
                "status": "PASS_N72R15_VARIANT",
                "event_id": event_id,
                "sequence": str(event["sequence"]),
                "variant": variant,
                "frame_count": len(rows),
                "frames": str(frames_path),
                "frames_sha256": sha256_file(frames_path),
                "stats": stats,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "created_at_utc": now_utc(),
            }
            done_path = event_dir / variant / "done.json"
            atomic_json(done_path, done)
            done["done"] = str(done_path)
            done["done_sha256"] = sha256_file(done_path)
            variant_records.append(done)
        except Exception as exc:
            failure = {
                "schema_version": "N72R15_FAILURE_V1",
                "status": "FAIL_N72R15_EVENT_VARIANT",
                "event_id": event_id,
                "variant": variant,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "historical_outputs_modified": False,
                "created_at_utc": now_utc(),
            }
            atomic_json(event_dir / variant / "failure.json", failure)
            failures.append(failure)
            break
    axis_audit = None
    if not failures and len(variant_rows) == len(VARIANTS):
        try:
            axis_audit = _validate_cross_variant_axes(variant_rows, event_id)
        except Exception as exc:
            failure = {
                "schema_version": "N72R15_FAILURE_V1",
                "status": "FAIL_N72R15_CANDIDATE_AXIS",
                "event_id": event_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "historical_outputs_modified": False,
                "created_at_utc": now_utc(),
            }
            atomic_json(event_dir / "candidate_axis_failure.json", failure)
            failures.append(failure)
    event_manifest = {
        "schema_version": "N72R15_FORMAL_EVENT_MANIFEST_V1",
        "status": "PASS_N72R15_FORMAL_EVENT" if not failures and len(variant_records) == len(VARIANTS) else "FAIL_N72R15_FORMAL_EVENT",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "event_frame": int(event["event_frame"]),
        "action_type": str(event["action_type"]),
        "variants": variant_records,
        "failures": failures,
        "candidate_axis_audit": axis_audit,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    manifest_path = event_dir / "event_manifest.json"
    atomic_json(manifest_path, event_manifest)
    event_manifest["manifest"] = str(manifest_path)
    event_manifest["manifest_sha256"] = sha256_file(manifest_path)
    return event_manifest


def run(events: Sequence[Mapping[str, Any]], output_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    selected = list(events)
    if smoke:
        recover = [item for item in selected if str(item.get("action_type")) == "RECOVER_IDENTITY"]
        if not recover:
            raise RuntimeError("frozen protocol has no RECOVER_IDENTITY event for smoke")
        selected = sorted(
            recover,
            key=lambda item: (str(item["sequence"]), int(item["event_frame"]), str(item["event_id"])),
        )[:1]
        root = output_root / "smoke"
    else:
        root = output_root
    results: list[dict[str, Any]] = []
    for event in selected:
        try:
            inputs = dict(replay._load_inputs(event, horizon=HORIZON))
            inputs["action_type"] = str(event["action_type"])
            inputs["protocol_event"] = dict(event)
            inputs["corrected_e0_rows"] = _corrected_e0_rows(str(event["event_id"]))
            result = _write_event(root, event, inputs)
        except Exception as exc:
            result = {
                "schema_version": "N72R15_FAILURE_V1",
                "status": "FAIL_N72R15_EVENT_INPUT",
                "event_id": str(event.get("event_id")),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "created_at_utc": now_utc(),
            }
            atomic_json(root / "input_failures" / f"{_slug(str(event.get('event_id')))}.json", result)
        results.append(result)
        if result.get("status") != "PASS_N72R15_FORMAL_EVENT":
            break
    expected = len(selected)
    pass_count = sum(item.get("status") == "PASS_N72R15_FORMAL_EVENT" for item in results)
    status = (
        "PASS_N72R15_RECOVER_SMOKE"
        if smoke and pass_count == expected == 1
        else "PASS_N72R15_FORMAL_REPLAY"
        if not smoke and pass_count == expected == len(selected)
        else "FAIL_N72R15_FORMAL_REPLAY"
    )
    payload = {
        "schema_version": "N72R15_FORMAL_MANIFEST_V1",
        "status": status,
        "smoke": bool(smoke),
        "event_count": len(selected),
        "completed_event_count": pass_count,
        "expected_variants_per_event": len(VARIANTS),
        "expected_runtime_rows": len(selected) * len(VARIANTS) * (HORIZON + 1),
        "variants": list(VARIANTS),
        "events": results,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "horizon": HORIZON,
        "candidate_pool_policy": "MAIN_B0_ONLY",
        "target_session_candidate_in_solver": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "sam3_rerun": False,
        "created_at_utc": now_utc(),
    }
    manifest_path = root / ("smoke_manifest.json" if smoke else "formal_manifest.json")
    atomic_json(manifest_path, payload)
    payload["manifest"] = str(manifest_path)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--state-edge-scale", type=float, default=STATE_EDGE_SCALE)
    args = parser.parse_args()
    try:
        if int(args.horizon) != HORIZON:
            raise ValueError(f"N72R15 frozen protocol requires horizon={HORIZON}")
        if float(args.state_edge_scale) != float(STATE_EDGE_SCALE):
            raise ValueError(f"N72R15 formal protocol requires state-edge-scale={STATE_EDGE_SCALE}")
        payload = run(_events(), args.output_root.resolve(), smoke=bool(args.smoke))
        status_name = "stage_07_status.json" if args.smoke else "stage_08_status.json"
        status_root = args.output_root.resolve() if args.smoke else args.output_root.resolve().parent
        status = {
            "schema_version": "N72R15_STAGE_STATUS_V1",
            "stage": "N72R15-07-RECOVER-H100-SMOKE" if args.smoke else "N72R15-08-FORMAL-B0-H1-G2",
            "status": payload["status"],
            "manifest": payload["manifest"],
            "manifest_sha256": payload["manifest_sha256"],
            "event_count": payload["event_count"],
            "completed_event_count": payload["completed_event_count"],
            "expected_variants_per_event": len(VARIANTS),
            "variants": list(VARIANTS),
            "horizon": HORIZON,
            "candidate_pool_policy": "MAIN_B0_ONLY",
            "target_session_candidate_in_solver": False,
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "created_at_utc": now_utc(),
        }
        atomic_json(status_root / status_name, status)
        print(json.dumps(status, sort_keys=True))
        return 0 if payload["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {
            "schema_version": "N72R15_FAILURE_V1",
            "status": "FAIL_N72R15_FORMAL_CONTROLLER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(
            args.output_root.resolve() / ("smoke_controller_failure.json" if args.smoke else "formal_controller_failure.json"),
            failure,
        )
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
