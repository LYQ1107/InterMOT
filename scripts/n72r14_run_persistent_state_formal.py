#!/usr/bin/env python3
"""Run the N72R14 B0/P0/T1/G1 causal public-state replay.

The runner only consumes the frozen N72R9 candidate streams.  It does not
instantiate SAM3, open GT, or use a second assignment implementation.  GT is
opened later by ``n72r14_aggregate_metrics.py`` after all runtime JSONL files
have been atomically sealed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
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
from sam3_intermot.association.persistent_public_state import (  # noqa: E402
    PersistentPublicAssociationBank,
)
from sam3_intermot.association.persistent_state_edge import (  # noqa: E402
    STATE_EDGE_SCALE,
    build_persistent_state_edge_matrix,
)
from sam3_intermot.reacquisition.target_candidate_pool import build_candidate_pool  # noqa: E402
from scripts import n72r11_on_demand_replay as replay  # noqa: E402


HORIZON = 100
VARIANTS = (
    "E0_BASELINE_B0",
    "E1F_TARGET_POOL_ONLY",
    "E1G_TARGET_STATE_ONLY",
    "E1H_PERSISTENT_GLOBAL_STATE",
)
TARGET_VARIANT = "E1G_TARGET_STATE_ONLY"
GLOBAL_VARIANT = "E1H_PERSISTENT_GLOBAL_STATE"
DEFAULT_OUTPUT = ROOT / "outputs/N72R14"
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"


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
        raise RuntimeError(f"frozen protocol must have 32 events, got {len(events) if isinstance(events, list) else None}")
    result = [dict(event) for event in events]
    if len({str(event.get("event_id")) for event in result}) != 32:
        raise RuntimeError("frozen event IDs are not unique")
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


def _semantic_assignment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(item.get("candidate_uid")): item.get("solver_public_id", item.get("public_id"))
        for item in row.get("candidate_rows", [])
    }


def _exact_base(
    inputs: Mapping[str, Any],
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    *,
    event_id: str,
    variant: str,
) -> dict[str, Any]:
    pairs, vectors = replay.legacy._base_vectors(inputs, int(frame), pool)
    matrix = np.stack(
        [np.asarray(vectors[str(candidate["candidate_uid"])], dtype=np.float64) for candidate in pool],
        axis=0,
    )
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError(f"{event_id}/{variant}:{frame}: base matrix invalid")
    states = replay._state_objects(pairs)
    solver = solve_effect_assignment(
        candidate_rows=pool,
        persistent_states=states,
        fused_state_candidate_scores=matrix.T,
        source_run_id=f"n72r14:{variant}:base:{event_id}:{frame}",
        session_id=f"n72r14:{variant}:{event_id}",
        none_score=0.0,
    )
    return {
        "pairs": pairs,
        "state_objects": states,
        "base_matrix": matrix,
        "base_solver": solver,
    }


def _event_row(inputs: Mapping[str, Any], variant: str) -> dict[str, Any]:
    row = replay._event_frame_row(inputs, variant)
    row.update(
        {
            "schema_version": "N72R14_PERSISTENT_STATE_RUNTIME_FRAME_V1",
            "action_type": str(inputs.get("action_type", "UNKNOWN")),
            "persistent_state_association": {
                "enabled": variant in {TARGET_VARIANT, GLOBAL_VARIANT},
                "scope": "OFF" if variant == "E0_BASELINE_B0" else "NO_EDGE_EVENT_FRAME",
                "state_edge_scale": float(STATE_EDGE_SCALE),
                "edge_shape": [0, 0],
                "event_frame_memory_read": False,
                "first_memory_visible_frame": int(inputs["event_frame"]) + 1,
                "runtime_future_gt_used": False,
            },
            "base_score_matrix_sha256": None,
            "fused_score_matrix_sha256": None,
            "state_update": None,
            "state_digest_before": None,
            "state_digest_after": None,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
    )
    return row


def _zero_edge(pool: Sequence[Mapping[str, Any]], public_axis: Sequence[int], frame: int, *, enabled: bool, scope: str) -> dict[str, Any]:
    shape = [len(pool), len(public_axis)]
    zeros = np.zeros((len(pool), len(public_axis)), dtype=np.float64).tolist()
    return {
        "schema_version": "N72R14_PERSISTENT_STATE_EDGE_V1",
        "frame": int(frame),
        "candidate_uids": [str(item["candidate_uid"]) for item in pool],
        "public_id_axis": [int(value) for value in public_axis],
        "shape": shape,
        "state_edge_scale": float(STATE_EDGE_SCALE),
        "feature_available_by_candidate": [bool(item.get("feature") is not None) for item in pool],
        "state_available": [[False for _ in public_axis] for _ in pool],
        "appearance": zeros,
        "motion": zeros,
        "native_continuity": zeros,
        "gap": zeros,
        "raw": zeros,
        "delta": zeros,
        "enabled": bool(enabled),
        "scope": scope,
        "runtime_future_gt_used": False,
    }


def _state_digest(bank: PersistentPublicAssociationBank) -> str:
    # The full snapshot is used only for an in-memory digest.  Runtime rows
    # store compact public-state snapshots and never serialize the vectors.
    payload = json.dumps(bank.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_row(
    inputs: Mapping[str, Any],
    *,
    variant: str,
    frame: int,
    pool: Sequence[Mapping[str, Any]],
    pool_audit: Mapping[str, Any],
    scored: Mapping[str, Any],
    bank: PersistentPublicAssociationBank,
    edge: Mapping[str, Any],
    fused: np.ndarray,
    solver: Mapping[str, Any],
    state_update: Mapping[str, Any],
    digest_before: str,
    digest_after: str,
    state_snapshot_before: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_candidates = replay._solver_rows(pool, solver)
    public_axis = [int(value) for value in scored["public_axis"]]
    base_matrix = np.asarray(scored["base_matrix"], dtype=np.float64)
    fused_matrix = np.asarray(fused, dtype=np.float64)
    target_public = int(inputs["target_public_id"])
    target_uid = _find_target_uid(solver, target_public)
    base_target_uid = _find_target_uid(scored["base_solver"], target_public)
    pool_record = dict(pool_audit)
    pool_record["candidate_rows"] = [replay.serializable_candidate(item, include_feature=False) for item in pool]
    # ``state_update`` has already been computed, but the bank now contains the
    # after-state.  The before snapshot is passed in by the caller through the
    # digest and is intentionally compact rather than a full memory dump.
    state_snapshot_after = bank.compact_snapshot(frame)
    edge_record = dict(edge)
    edge_record["base_score_matrix_sha256"] = matrix_sha256(base_matrix)
    edge_record["fused_score_matrix_sha256"] = matrix_sha256(fused_matrix)
    edge_record["enabled"] = variant in {TARGET_VARIANT, GLOBAL_VARIANT}
    edge_record["scope"] = (
        "TARGET_PUBLIC_ID_ONLY" if variant == TARGET_VARIANT else
        "GLOBAL_PUBLIC_ID" if variant == GLOBAL_VARIANT else "OFF"
    )
    row = {
        "schema_version": "N72R14_PERSISTENT_STATE_RUNTIME_FRAME_V1",
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
            "none_score": 0.0,
            "runtime_future_gt_used": False,
        },
        "persistent_state_association": {
            "enabled": variant in {TARGET_VARIANT, GLOBAL_VARIANT},
            "scope": edge_record["scope"],
            "state_edge_scale": float(STATE_EDGE_SCALE),
            "edge": edge_record,
            "state_snapshot_before_update": list(state_snapshot_before),
            "state_snapshot_after_update": state_snapshot_after,
            "state_digest_before": digest_before,
            "state_digest_after": digest_after,
            "state_changed": bool(digest_before != digest_after),
            "runtime_future_gt_used": False,
        },
        "memory_read": bool(variant in {TARGET_VARIANT, GLOBAL_VARIANT}),
        "memory_write": bool(state_update.get("machine_memory_write_public_ids")),
        "event_frame_memory_read": False,
        "first_memory_visible_frame": int(inputs["event_frame"]) + 1,
        "state_update": dict(state_update),
        "base_score_matrix_sha256": matrix_sha256(base_matrix),
        "fused_score_matrix_sha256": matrix_sha256(fused_matrix),
        "raw_binding_switch": None,
        "public_id_immutable": True,
        "public_id_inference": False,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }
    return row


def _initialize_bank(inputs: Mapping[str, Any]) -> PersistentPublicAssociationBank:
    bank = PersistentPublicAssociationBank()
    event_frame = int(inputs["event_frame"])
    c0 = inputs["rows"]["c0_source"][event_frame]
    c0_pool, _ = build_candidate_pool(
        list(c0.get("candidate_rows", [])),
        (),
        sequence=str(inputs["sequence"]),
        frame=event_frame,
        include_target_session=False,
        # Match the corrected N72R11R5R1 E0 contract: reject degenerate
        # candidates before the official solver and retain the rejection in
        # the pool audit.  Post-hoc filtering would invalidate the control.
        require_positive_geometry=True,
    )
    initial = _exact_base(
        inputs,
        event_frame,
        c0_pool,
        event_id=str(inputs["event_id"]),
        variant="INITIAL_EVENT_FRAME",
    )
    target_rows = list(inputs["rows"]["target_stream_source"][event_frame].get("candidate_rows", []))
    target_candidate = target_rows[0] if target_rows else None
    bank.initialize_from_event_frame(
        event_frame=event_frame,
        candidate_rows=c0_pool,
        solver=initial["base_solver"],
        association_state_axis=[pair[0] for pair in initial["pairs"]],
        public_id_axis=[pair[1] for pair in initial["pairs"]],
        target_public_id=int(inputs["target_public_id"]),
        target_anchor=inputs["anchor"],
        target_box=inputs["anchor_box"],
        target_candidate=target_candidate,
        event_id=str(inputs["event_id"]),
    )
    return bank


def _run_variant(inputs: Mapping[str, Any], variant: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_frame = int(inputs["event_frame"])
    rows: list[dict[str, Any]] = [_event_row(inputs, variant)]
    if variant == "E0_BASELINE_B0":
        bank = PersistentPublicAssociationBank()
    else:
        bank = _initialize_bank(inputs)
    control_matches: list[bool] = []
    previous_target_uid: str | None = None
    for frame in range(event_frame + 1, event_frame + HORIZON + 1):
        c0 = inputs["rows"]["c0_source"][frame]
        target_row = inputs["rows"]["target_stream_source"][frame]
        main = list(c0.get("candidate_rows", []))
        target = list(target_row.get("candidate_rows", [])) if variant != "E0_BASELINE_B0" else []
        pool, pool_audit = build_candidate_pool(
            main,
            target,
            sequence=str(inputs["sequence"]),
            frame=frame,
            include_target_session=variant != "E0_BASELINE_B0",
            require_positive_geometry=True,
        )
        scored = _exact_base(
            inputs,
            frame,
            pool,
            event_id=str(inputs["event_id"]),
            variant=variant,
        )
        public_axis = [int(pair[1]) for pair in scored["pairs"]]
        if variant == "E0_BASELINE_B0":
            edge = _zero_edge(pool, public_axis, frame, enabled=False, scope="OFF")
            fused = np.asarray(scored["base_matrix"], dtype=np.float64)
            solver = scored["base_solver"]
            before = _state_digest(bank)
            state_update = {
                "frame": int(frame),
                "assigned_public_ids": [],
                "machine_memory_write_public_ids": [],
                "unassigned_public_ids": [],
                "runtime_future_gt_used": False,
            }
            after = before
            snapshot_before = []
        elif variant == "E1F_TARGET_POOL_ONLY":
            bank.states_for_pairs(scored["pairs"])
            edge = _zero_edge(pool, public_axis, frame, enabled=False, scope="OFF")
            fused = np.asarray(scored["base_matrix"], dtype=np.float64)
            solver = scored["base_solver"]
            snapshot_before = bank.compact_snapshot(frame - 1)
            before = _state_digest(bank)
            state_update = bank.update_from_solver(
                frame=frame,
                candidate_rows=pool,
                solver=solver,
                none_score=0.0,
            )
            after = _state_digest(bank)
        else:
            bank.states_for_pairs(scored["pairs"])
            snapshot_before = bank.compact_snapshot(frame - 1)
            before = _state_digest(bank)
            edge = build_persistent_state_edge_matrix(
                bank=bank,
                candidate_rows=pool,
                public_id_axis=public_axis,
                frame=frame,
            )
            deltas = np.asarray(edge["delta"], dtype=np.float64)
            if variant == TARGET_VARIANT:
                target_col = public_axis.index(int(inputs["target_public_id"]))
                applied_edge = np.zeros_like(deltas)
                applied_edge[:, target_col] = deltas[:, target_col]
                edge = dict(edge)
                edge["delta_before_scope"] = deltas.tolist()
                edge["delta"] = applied_edge.tolist()
                edge["scope"] = "TARGET_PUBLIC_ID_ONLY"
                edge["enabled"] = True
                deltas = applied_edge
            elif variant == GLOBAL_VARIANT:
                edge = dict(edge)
                edge["scope"] = "GLOBAL_PUBLIC_ID"
                edge["enabled"] = True
            else:
                raise ValueError(f"unknown N72R14 variant {variant}")
            fused = np.asarray(scored["base_matrix"], dtype=np.float64) + float(STATE_EDGE_SCALE) * deltas
            if fused.shape != scored["base_matrix"].shape or not np.isfinite(fused).all():
                raise RuntimeError(f"{inputs['event_id']}:{frame}: fused matrix invalid")
            solver = solve_effect_assignment(
                candidate_rows=pool,
                persistent_states=scored["state_objects"],
                fused_state_candidate_scores=fused.T,
                source_run_id=f"n72r14:{variant}:{inputs['event_id']}:{frame}",
                session_id=f"n72r14:{variant}:{inputs['event_id']}",
                none_score=0.0,
            )
            state_update = bank.update_from_solver(
                frame=frame,
                candidate_rows=pool,
                solver=solver,
                none_score=0.0,
            )
            after = _state_digest(bank)
        row = _runtime_row(
            inputs,
            variant=variant,
            frame=frame,
            pool=pool,
            pool_audit=pool_audit,
            scored={**scored, "state_axis": [pair[0] for pair in scored["pairs"]], "public_axis": public_axis},
            bank=bank,
            edge=edge,
            fused=fused,
            solver=solver,
            state_update=state_update,
            digest_before=before,
            digest_after=after,
            state_snapshot_before=snapshot_before,
        )
        row["raw_binding_switch"] = None
        current_target_uid = _find_target_uid(solver, int(inputs["target_public_id"]))
        if previous_target_uid is not None and current_target_uid is not None and current_target_uid != previous_target_uid:
            row["raw_binding_switch"] = {"from": previous_target_uid, "to": current_target_uid}
        previous_target_uid = current_target_uid or previous_target_uid
        frozen = inputs["baseline_rows"][frame - event_frame]
        if variant == "E0_BASELINE_B0":
            frozen_audit = frozen.get("score_audit", {})
            control_matches.append(
                [str(item["candidate_uid"]) for item in row["candidate_rows"]]
                == [str(item["candidate_uid"]) for item in frozen.get("candidate_rows", [])]
                and public_axis == [int(value) for value in frozen_audit.get("public_id_axis", [])]
                and matrix_sha256(scored["base_matrix"]) == matrix_sha256(frozen_audit.get("fused_score_matrix", []))
                and _semantic_assignment(row) == _semantic_assignment(frozen)
            )
        rows.append(row)
    if len(rows) != HORIZON + 1:
        raise RuntimeError(f"{inputs['event_id']}/{variant}: expected 101 rows")
    expected_frames = list(range(event_frame, event_frame + HORIZON + 1))
    if [int(row["frame"]) for row in rows] != expected_frames:
        raise RuntimeError(f"{inputs['event_id']}/{variant}: frame axis mismatch")
    for row in rows:
        for key in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
            if row.get(key) is not False:
                raise RuntimeError(f"{inputs['event_id']}/{variant}: runtime flag {key} is not false")
    return rows, {
        "variant": variant,
        "frame_count": len(rows),
        "control_matches_frozen_e0": all(control_matches) if variant == "E0_BASELINE_B0" else None,
        "control_checked_future_frames": len(control_matches),
        "runtime_future_gt_used": False,
    }


def _write_event(output_root: Path, event: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    event_id = str(event["event_id"])
    event_dir = output_root / "formal" / _slug(event_id)
    event_dir.mkdir(parents=True, exist_ok=True)
    variant_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for variant in VARIANTS:
        try:
            rows, stats = _run_variant(inputs, variant)
            frames_path = event_dir / variant / "runtime_frames.jsonl"
            atomic_jsonl(frames_path, rows)
            done = {
                "schema_version": "N72R14_POLICY_DONE_V1",
                "status": "PASS_N72R14_VARIANT",
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
                "schema_version": "N72R14_FAILURE_V1",
                "status": "FAIL_N72R14_EVENT_VARIANT",
                "event_id": event_id,
                "variant": variant,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "historical_outputs_modified": False,
                "created_at_utc": now_utc(),
            }
            failure_path = event_dir / variant / "failure.json"
            atomic_json(failure_path, failure)
            failures.append(failure)
            break
    event_manifest = {
        "schema_version": "N72R14_FORMAL_EVENT_MANIFEST_V1",
        "status": "PASS_N72R14_FORMAL_EVENT" if not failures and len(variant_records) == len(VARIANTS) else "FAIL_N72R14_FORMAL_EVENT",
        "event_id": event_id,
        "sequence": str(event["sequence"]),
        "event_frame": int(event["event_frame"]),
        "action_type": str(event["action_type"]),
        "variants": variant_records,
        "failures": failures,
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
        recover = [event for event in selected if str(event.get("action_type")) == "RECOVER_IDENTITY"]
        if not recover:
            raise RuntimeError("frozen protocol has no RECOVER_IDENTITY event for smoke")
        selected = sorted(recover, key=lambda event: (str(event["sequence"]), int(event["event_frame"]), str(event["event_id"])))[:1]
        root = output_root / "smoke"
    else:
        root = output_root
    results: list[dict[str, Any]] = []
    for event in selected:
        try:
            inputs = replay._load_inputs(event, horizon=HORIZON)
            inputs = dict(inputs)
            inputs["action_type"] = str(event["action_type"])
            result = _write_event(root, event, inputs)
        except Exception as exc:
            result = {
                "schema_version": "N72R14_FAILURE_V1",
                "status": "FAIL_N72R14_EVENT_INPUT",
                "event_id": str(event.get("event_id")),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "created_at_utc": now_utc(),
            }
            atomic_json(root / "input_failures" / f"{_slug(str(event.get('event_id')))}.json", result)
        results.append(result)
        if result.get("status") != "PASS_N72R14_FORMAL_EVENT":
            break
    expected = len(selected)
    pass_count = sum(item.get("status") == "PASS_N72R14_FORMAL_EVENT" for item in results)
    status = "PASS_N72R14_RECOVER_SMOKE" if smoke and pass_count == expected == 1 else (
        "PASS_N72R14_FORMAL_REPLAY" if not smoke and pass_count == expected == len(selected) else
        "FAIL_N72R14_FORMAL_REPLAY"
    )
    payload = {
        "schema_version": "N72R14_FORMAL_MANIFEST_V1",
        "status": status,
        "smoke": bool(smoke),
        "event_count": len(selected),
        "completed_event_count": pass_count,
        "expected_variants_per_event": len(VARIANTS),
        "variants": list(VARIANTS),
        "events": results,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "horizon": HORIZON,
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
    args = parser.parse_args()
    try:
        payload = run(_events(), args.output_root.resolve(), smoke=bool(args.smoke))
        status_name = "stage_02_status.json" if args.smoke else "stage_03_status.json"
        status = {
            "schema_version": "N72R14_STAGE_STATUS_V1",
            "stage": "N72R14-02-RECOVER-H100-SMOKE" if args.smoke else "N72R14-03-FORMAL-B0-P0-T1-G1",
            "status": payload["status"],
            "manifest": payload["manifest"],
            "manifest_sha256": payload["manifest_sha256"],
            "event_count": payload["event_count"],
            "completed_event_count": payload["completed_event_count"],
            "variants": list(VARIANTS),
            "horizon": HORIZON,
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "created_at_utc": now_utc(),
        }
        atomic_json(args.output_root.resolve() / status_name, status)
        print(json.dumps(status, sort_keys=True))
        return 0 if payload["status"].startswith("PASS") else 2
    except Exception as exc:
        failure = {
            "schema_version": "N72R14_FAILURE_V1",
            "status": "FAIL_N72R14_FORMAL_CONTROLLER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_future_gt_used": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(args.output_root.resolve() / ("smoke_controller_failure.json" if args.smoke else "formal_controller_failure.json"), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
