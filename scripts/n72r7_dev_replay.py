#!/usr/bin/env python3
"""Run the cheap causal D1/D2 candidate-pool development replay.

D0 is the sealed N72R6 B0 stream.  D1 replays the complete B0 pool and lets a
target-conditioned selector add evidence to the target public row.  D2 adds
the current target-session row to that same pool.  The exact public solver is
called once per frame; runtime artifacts contain no GT values.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.effect_assignment import solve_effect_assignment  # noqa: E402
from sam3_intermot.reacquisition.target_candidate_pool import (  # noqa: E402
    MAIN_B0_CANDIDATE,
    TARGET_SESSION_CURRENT_RAW,
    build_candidate_pool,
    serializable_candidate,
)
from sam3_intermot.reacquisition.target_candidate_selector import (  # noqa: E402
    SelectorConfig,
    TargetCandidateSelector,
    TargetSelectionContext,
    box_iou,
)


N72R5 = ROOT / "outputs/N72R5"
N72R5R1 = ROOT / "outputs/N72R5R1"
N72R6 = ROOT / "outputs/N72R6"
EVENT_POLICY = N72R5 / "mechanism_rounds/round_06_event_policy/real_event_manifest.json"
STAGE08 = N72R5R1 / "controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
TARGET_MANIFEST = N72R6 / "recovery_target_stream_manifest_attempt3.json"
REPLAY_ROOT = N72R6 / "public_replay/human_anchor_fallback_attempt1"
DEV_PROTOCOL = ROOT / "outputs/N72R7/dev_protocol.json"
HORIZON = 100
TARGET_EVIDENCE_INJECTION_SCALE = 1.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    batch = read_json(REPLAY_ROOT / "replay_batch_status.json")
    if batch.get("status") != "PASS_N72R6_C0_C1_REPLAY" or int(batch.get("completed_event_count", -1)) != 32:
        raise RuntimeError(f"N72R6 frozen replay is not complete: {batch.get('status')}")
    policy = read_json(EVENT_POLICY)
    events = {str(item["event_id"]): item for item in policy.get("events", [])}
    stage08 = read_json(STAGE08)
    eligible = set()
    for item in stage08.get("events", []):
        branches = {str(row.get("branch")): row for row in item.get("branches", [])}
        if branches.get("B1_SPATIAL_CORRECTION_ONLY", {}).get("action_precondition_status") == "APPLIED":
            eligible.add(str(item["event_id"]))
    target = read_json(TARGET_MANIFEST)
    selected = {str(item["event_id"]): item for item in target.get("selected", [])}
    if len(eligible) != 32 or set(selected) != eligible:
        raise RuntimeError("N72R6 eligible and target-stream event sets do not match")
    manifests: dict[str, dict[str, Any]] = {}
    for event_id in sorted(eligible):
        manifest = read_json(REPLAY_ROOT / event_id / "event_manifest.json")
        if manifest.get("status") != "PASS_N72R6_C0_C1_EVENT_REPLAY":
            raise RuntimeError(f"N72R6 event is not PASS: {event_id}")
        for key in ("c0", "c1"):
            path = resolve(str(manifest[key]["path"]))
            if not path.is_file() or sha256_file(path) != str(manifest[key]["sha256"]):
                raise RuntimeError(f"N72R6 {key} hash mismatch: {event_id}")
            rows = read_jsonl(path)
            expected = list(range(int(manifest["event_frame"]), int(manifest["event_frame"]) + HORIZON + 1))
            if len(rows) != HORIZON + 1 or [int(row["frame"]) for row in rows] != expected:
                raise RuntimeError(f"N72R6 {key} frame axis mismatch: {event_id}")
            if any(row.get("runtime_future_gt_used") is not False for row in rows):
                raise RuntimeError(f"N72R6 {key} runtime GT flag violation: {event_id}")
        target_path = resolve(str(manifest["target_stream_frames"]))
        target_rows = read_jsonl(target_path)
        if len(target_rows) != HORIZON + 1:
            raise RuntimeError(f"target stream frame count mismatch: {event_id}")
        for row in target_rows:
            for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                if row.get(flag) is not False:
                    raise RuntimeError(f"target stream flag violation: {event_id}:{row.get('frame')}:{flag}")
        manifests[event_id] = manifest
    return batch, policy, manifests


def write_development_protocol() -> dict[str, Any]:
    config = SelectorConfig()
    body = {
        "schema_version": "N72R7_DEVELOPMENT_REPLAY_PROTOCOL_V1",
        "created_at_utc": now_utc(),
        "source_protocol": str(ROOT / "outputs/N72R7/protocol.json"),
        "source_protocol_sha256": sha256_file(ROOT / "outputs/N72R7/protocol.json"),
        "variants": {
            "D0": "frozen N72R6 B0 candidate rows and solver outputs; no new selection",
            "D1": "complete B0 pool plus causal target selector",
            "D2": "complete B0 pool plus target-session current raw rows plus same selector",
        },
        "candidate_sources": [MAIN_B0_CANDIDATE, TARGET_SESSION_CURRENT_RAW],
        "selector_config": {
            key: float(value) for key, value in vars(config).items()
        },
        "target_evidence_injection_scale": TARGET_EVIDENCE_INJECTION_SCALE,
        "target_evidence_injection_rule": "selected target candidate only; max(selector_score-none_score,0) added to target public column",
        "memory_admission_rule": "selected candidate assigned to target public, score>=admission_score, margin>=admission_margin, and finite feature",
        "distractor_admission_rule": "ambiguous or rejected selected feature is retained as distractor evidence; never target memory",
        "raw_binding_rule": "raw ID and native scope are weak continuity features; a raw switch cannot change public ID",
        "solver": "sam3_intermot.association.effect_assignment.solve_effect_assignment",
        "metrics": ["H20", "H50", "H100", "identity_error", "missing", "wrong_reassociation", "protected_regression", "sequence_cluster_bootstrap"],
        "bootstrap": {"seed": 7202, "repetitions": 2000, "cluster_unit": "independent_sequence"},
        "post_treatment_fields_forbidden_in_runtime": ["GT", "future_identity_error", "future_IoU", "H20", "H50", "H100", "IDSW"],
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    body["protocol_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    atomic_json(DEV_PROTOCOL, body)
    return body


def _explicit_authority_pairs(row: Mapping[str, Any], *, label: str) -> list[tuple[int, int]]:
    """Extract the state→public authority pairs without using row indices."""

    state_axis = [int(value) for value in row.get("association_state_axis", [])]
    if len(state_axis) != len(set(state_axis)):
        raise RuntimeError(f"{label} association state axis contains duplicates")
    by_state: dict[int, int] = {}
    for item in row.get("identity_rows", []):
        state_id = item.get("association_state_id")
        public_id = item.get("public_id")
        if state_id is None or public_id is None:
            continue
        state_id = int(state_id)
        public_id = int(public_id)
        if state_id in by_state and by_state[state_id] != public_id:
            raise RuntimeError(f"{label} has conflicting public authority for state {state_id}")
        by_state[state_id] = public_id
    # Some frozen rows carry the same explicit authority on candidate rows;
    # use it only as a provenance fallback, never as a row-index inference.
    for item in row.get("candidate_rows", []):
        state_id = item.get("solver_association_state_id")
        public_id = item.get("solver_public_id")
        if state_id is None or public_id is None:
            continue
        state_id = int(state_id)
        public_id = int(public_id)
        if state_id in by_state and by_state[state_id] != public_id:
            raise RuntimeError(f"{label} candidate/identity authority conflict for state {state_id}")
        by_state[state_id] = public_id
    missing = [state_id for state_id in state_axis if state_id not in by_state]
    if missing:
        raise RuntimeError(f"{label} lacks explicit state→public authority for {missing}")
    pairs = [(state_id, by_state[state_id]) for state_id in state_axis]
    public_ids = [public_id for _, public_id in pairs]
    if len(public_ids) != len(set(public_ids)):
        raise RuntimeError(f"{label} explicit public authority axis contains duplicates")
    return pairs


def _map_matrix_row_to_public_axis(
    row_values: Sequence[float],
    source_pairs: Sequence[tuple[int, int]],
    target_pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Map a frozen score row through explicit public authority bindings."""

    source_public_to_column = {public_id: index for index, (_, public_id) in enumerate(source_pairs)}
    result = np.zeros(len(target_pairs), dtype=np.float64)
    values = np.asarray(row_values, dtype=np.float64).reshape(-1)
    if values.size != len(source_pairs):
        raise RuntimeError(
            f"score row/authority mismatch: values={values.size} source_pairs={len(source_pairs)}"
        )
    for index, (_, public_id) in enumerate(target_pairs):
        source_index = source_public_to_column.get(public_id)
        if source_index is not None:
            result[index] = float(values[source_index])
    return result


def _target_base_vector(
    target_uid: str,
    c1_row: Mapping[str, Any],
    replay_pairs: Sequence[tuple[int, int]],
    c1_pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, str]:
    c1_candidates = list(c1_row.get("candidate_rows", []))
    matches = [index for index, item in enumerate(c1_candidates) if str(item.get("candidate_uid")) == target_uid]
    c1_matrix = np.asarray(c1_row.get("base_score_matrix", []), dtype=np.float64)
    if len(matches) == 1 and c1_matrix.ndim == 2 and c1_matrix.shape[0] == len(c1_candidates) and c1_matrix.shape[1] == len(c1_pairs):
        return (
            _map_matrix_row_to_public_axis(c1_matrix[matches[0]], c1_pairs, replay_pairs),
            "N72R6_C1_TARGET_ROW_BASE_SCORE",
        )
    return np.zeros(len(replay_pairs), dtype=np.float64), "NEUTRAL_ZERO_NO_ACCEPTED_C1_TARGET_ROW"


def _solver_rows(
    candidates: Sequence[Mapping[str, Any]],
    solver: Mapping[str, Any],
) -> list[dict[str, Any]]:
    decisions = {str(row["candidate_uid"]): row for row in solver["assignment_rows"]}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        uid = str(candidate["candidate_uid"])
        decision = decisions[uid]
        item = serializable_candidate(candidate, include_feature=False)
        public_id = decision.get("public_id")
        item.update({
            "candidate_public_id_before_solver": None,
            "solver_public_id": None if public_id is None else int(public_id),
            "solver_association_state_id": decision.get("association_state_id"),
            "solver_status": str(decision["status"]),
            "solver_score": float(decision["score"]),
            "public_id": None if public_id is None else int(public_id),
            "public_id_authority": "exact_global_solver_output" if public_id is not None else None,
            "assignment_status": "EXPLICIT_NONE" if public_id is None else "ASSIGNED_TO_PUBLIC_ID",
            "assigned_public_id": None if public_id is None else int(public_id),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": False,
        })
        rows.append(item)
    return rows


def _find_assigned_uid(solver: Mapping[str, Any], public_id: int) -> str | None:
    for row in solver.get("assignment_rows", []):
        if row.get("public_id") is not None and int(row["public_id"]) == int(public_id):
            return str(row["candidate_uid"])
    return None


def run_event(
    event: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any],
    *,
    variant: str,
    output_root: Path,
    selector: TargetCandidateSelector,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    event_frame = int(frozen_manifest["event_frame"])
    target_public = int(frozen_manifest["target_public_id"])
    c0_rows = read_jsonl(resolve(str(frozen_manifest["c0"]["path"])))
    c1_rows = read_jsonl(resolve(str(frozen_manifest["c1"]["path"])))
    target_stream_rows = read_jsonl(resolve(str(frozen_manifest["target_stream_frames"])))
    c0_by_frame = {int(row["frame"]): row for row in c0_rows}
    c1_by_frame = {int(row["frame"]): row for row in c1_rows}
    target_by_frame = {int(row["frame"]): row for row in target_stream_rows}
    done = read_json(resolve(str(frozen_manifest["target_stream_done"])))
    anchor = read_json(resolve(str(done["human_anchor"])))
    anchor_feature = np.asarray(anchor["feature"], dtype=np.float32).reshape(-1)
    anchor_feature /= float(np.linalg.norm(anchor_feature))
    anchor_box = [float(value) for value in anchor["box_xyxy"]]
    event_target_rows = list(target_by_frame[event_frame].get("candidate_rows", []))
    old_raw = None if not event_target_rows else event_target_rows[0].get("official_raw_sam_id")
    old_scope = None if not event_target_rows else event_target_rows[0].get("native_tid_scope")
    context = TargetSelectionContext(
        human_anchor=anchor_feature,
        trusted_features=[],
        distractor_features=[],
        predicted_box=anchor_box,
        previous_raw_sam_id=None if old_raw is None else int(old_raw),
        previous_native_scope=None if old_scope is None else str(old_scope),
        frame=event_frame + 1,
        event_frame=event_frame,
        memory_read=True,
    )
    trusted_hashes: list[str] = []
    distractor_hashes: list[str] = []
    velocity = np.zeros(2, dtype=np.float64)
    output_rows: list[dict[str, Any]] = [{
        "schema_version": "N72R7_CLOSED_LOOP_FRAME_V1",
        "record_kind": "event_frame_correction",
        "variant": variant,
        "event_id": event_id,
        "sequence": sequence,
        "event_frame": event_frame,
        "frame": event_frame,
        "frame_horizon": 0,
        "target_public_id": target_public,
        "candidate_rows": [],
        "candidate_count": 0,
        "candidate_pool": None,
        "selection_audit": None,
        "assignment": None,
        "memory_read": False,
        "memory_write": True,
        "event_frame_memory_read": False,
        "first_memory_visible_frame": event_frame + 1,
        "raw_binding_switch": None,
        "trusted_memory_update": "HUMAN_ANCHOR_INITIALIZED",
        "distractor_memory_update_count": 0,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
        "public_id_immutable": True,
    }]
    raw_switches: list[dict[str, Any]] = []
    target_assignments = 0
    target_selections = 0
    none_selections = 0
    trusted_updates = 0
    distractor_updates = 0
    candidate_source_counts: Counter[str] = Counter()
    for frame in range(event_frame + 1, event_frame + HORIZON + 1):
        c0 = c0_by_frame[frame]
        c1 = c1_by_frame[frame]
        raw_target_rows = list(target_by_frame[frame].get("candidate_rows", []))
        include_target = variant == "D2"
        pool, pool_audit = build_candidate_pool(
            c0.get("candidate_rows", []),
            raw_target_rows if include_target else (),
            sequence=sequence,
            frame=frame,
            include_target_session=include_target,
        )
        for candidate in pool:
            candidate_source_counts[str(candidate["candidate_source"])] += 1
        c0_candidates = list(c0.get("candidate_rows", []))
        c0_matrix = np.asarray(c0.get("base_score_matrix", []), dtype=np.float64)
        c0_pairs = _explicit_authority_pairs(c0, label=f"B0:{event_id}:{frame}")
        c1_pairs = _explicit_authority_pairs(c1, label=f"C1:{event_id}:{frame}")
        if c0_matrix.shape != (len(c0_candidates), len(c0_pairs)):
            raise RuntimeError(f"B0 matrix/axis mismatch: {event_id}:{frame}")
        replay_pairs = list(c0_pairs)
        if target_public not in {public_id for _, public_id in replay_pairs}:
            target_pair = next((pair for pair in c1_pairs if pair[1] == target_public), None)
            if target_pair is None:
                raise RuntimeError(f"target public authority missing from C1: {event_id}:{frame}:{target_public}")
            if target_pair[0] in {state_id for state_id, _ in replay_pairs}:
                raise RuntimeError(f"target state authority collides with B0 state: {event_id}:{frame}:{target_pair}")
            replay_pairs.append(target_pair)
        state_axis = [state_id for state_id, _ in replay_pairs]
        public_axis = [public_id for _, public_id in replay_pairs]
        base_vectors: dict[str, np.ndarray] = {}
        for index, candidate in enumerate(c0_candidates):
            base_vectors[str(candidate["candidate_uid"])] = _map_matrix_row_to_public_axis(
                c0_matrix[index], c0_pairs, replay_pairs
            )
        for candidate in pool[len(c0_candidates):]:
            vector, source = _target_base_vector(
                str(candidate["candidate_uid"]), c1, replay_pairs, c1_pairs
            )
            base_vectors[str(candidate["candidate_uid"])] = vector
            candidate["base_score_source"] = source
        base_target_scores = {
            str(candidate["candidate_uid"]): (
                None
                if target_public not in public_axis
                else float(base_vectors[str(candidate["candidate_uid"])] [public_axis.index(target_public)])
            )
            for candidate in pool
        }
        context.frame = frame
        context.predicted_box = None if context.predicted_box is None else [
            float(context.predicted_box[0] + velocity[0]),
            float(context.predicted_box[1] + velocity[1]),
            float(context.predicted_box[2] + velocity[0]),
            float(context.predicted_box[3] + velocity[1]),
        ]
        selection = selector.select(pool, context=context, base_target_scores=base_target_scores)
        selected_uid = selection["selected_candidate_uid"]
        target_selections += int(selected_uid is not None)
        none_selections += int(selected_uid is None)
        fused = (
            np.stack([base_vectors[str(candidate["candidate_uid"])] for candidate in pool], axis=0)
            if pool
            else np.zeros((0, len(state_axis)))
        )
        target_col = None if target_public not in public_axis else public_axis.index(target_public)
        injected_delta = 0.0
        if selected_uid is not None and target_col is not None:
            selected_index = next(index for index, item in enumerate(pool) if str(item["candidate_uid"]) == str(selected_uid))
            selected_score = float(selection["selected_score"] or 0.0)
            injected_delta = TARGET_EVIDENCE_INJECTION_SCALE * max(selected_score - selector.config.none_score, 0.0)
            fused[selected_index, target_col] += injected_delta
        state_objects = [
            type("ReplayState", (), {"association_state_id": state_id, "public_id": public_id})()
            for state_id, public_id in zip(state_axis, public_axis)
        ]
        solver = solve_effect_assignment(
            candidate_rows=pool,
            persistent_states=state_objects,
            fused_state_candidate_scores=fused.T,
            source_run_id=f"n72r7:{variant}:{event_id}",
            session_id=f"n72r7:{variant}:{event_id}",
            none_score=0.0,
        )
        assigned_target_uid = _find_assigned_uid(solver, target_public)
        target_assignments += int(assigned_target_uid is not None)
        assigned_candidate = None if assigned_target_uid is None else next(
            item for item in pool if str(item["candidate_uid"]) == str(assigned_target_uid)
        )
        binding_switch = None
        if assigned_candidate is not None:
            new_raw = assigned_candidate.get("official_raw_sam_id")
            new_scope = assigned_candidate.get("native_scope")
            changed = (old_raw != new_raw) or (str(old_scope) != str(new_scope))
            if changed:
                binding_switch = {
                    "public_id": target_public,
                    "frame": frame,
                    "old_raw_sam_id": None if old_raw is None else int(old_raw),
                    "new_raw_sam_id": None if new_raw is None else int(new_raw),
                    "old_source": "previous_target_binding",
                    "new_source": str(assigned_candidate["candidate_source"]),
                    "selector_score": selection.get("selected_score"),
                    "selector_margin": selection.get("best_minus_second_margin"),
                    "selected_candidate_uid": selected_uid,
                    "assigned_candidate_uid": assigned_target_uid,
                    "reason": "target_selector_assignment_rebinding",
                    "public_id_changed": False,
                    "runtime_future_gt_used": False,
                }
                raw_switches.append(binding_switch)
            old_raw = None if new_raw is None else int(new_raw)
            old_scope = None if new_scope is None else str(new_scope)
            new_box = [float(value) for value in assigned_candidate["box_xyxy"]]
            if context.predicted_box is not None:
                old_center = np.asarray([(context.predicted_box[0] + context.predicted_box[2]) / 2, (context.predicted_box[1] + context.predicted_box[3]) / 2])
                new_center = np.asarray([(new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2])
                velocity = 0.5 * velocity + 0.5 * (new_center - old_center)
            context.predicted_box = new_box
            if str(assigned_target_uid) == str(selected_uid) and bool(selection["reliable_for_memory_admission"]):
                feature = assigned_candidate.get("feature")
                if feature is not None:
                    digest = str(assigned_candidate.get("feature_sha256"))
                    context.trusted_features.append(np.asarray(feature, dtype=np.float32))
                    trusted_hashes.append(digest)
                    trusted_updates += 1
            elif selected_uid is not None:
                selected_candidate = next(item for item in pool if str(item["candidate_uid"]) == str(selected_uid))
                if selected_candidate.get("feature") is not None:
                    context.distractor_features.append(np.asarray(selected_candidate["feature"], dtype=np.float32))
                    distractor_hashes.append(str(selected_candidate.get("feature_sha256")))
                    distractor_updates += 1
        elif selected_uid is not None:
            selected_candidate = next(item for item in pool if str(item["candidate_uid"]) == str(selected_uid))
            if selected_candidate.get("feature") is not None:
                context.distractor_features.append(np.asarray(selected_candidate["feature"], dtype=np.float32))
                distractor_hashes.append(str(selected_candidate.get("feature_sha256")))
                distractor_updates += 1
        # Keep the causal memory bounded, as in the frozen protocol.
        context.trusted_features = context.trusted_features[-3:]
        context.distractor_features = context.distractor_features[-8:]
        for item in pool:
            item["raw_continuity"] = float(
                item.get("official_raw_sam_id") is not None
                and old_raw is not None
                and int(item["official_raw_sam_id"]) == int(old_raw)
                and item.get("native_scope") is not None
                and old_scope is not None
                and str(item.get("native_scope")) == str(old_scope)
            )
        output_rows.append({
            "schema_version": "N72R7_CLOSED_LOOP_FRAME_V1",
            "record_kind": "future_association_frame",
            "variant": variant,
            "event_id": event_id,
            "sequence": sequence,
            "event_frame": event_frame,
            "frame": frame,
            "frame_horizon": int(frame - event_frame),
            "target_public_id": target_public,
            "candidate_rows": _solver_rows(pool, solver),
            "candidate_count": len(pool),
            "candidate_pool": {
                **pool_audit,
                "candidate_rows": [serializable_candidate(item, include_feature=False) for item in pool],
            },
            "selection_audit": selection,
            "assignment": {
                "target_public_id": target_public,
                "target_selected_candidate_uid": selected_uid,
                "target_assigned_candidate_uid": assigned_target_uid,
                "target_selector_and_solver_agree": bool(selected_uid is not None and selected_uid == assigned_target_uid),
                "target_public_in_state_axis": target_col is not None,
                "solver_public_id_immutable": True,
                "solver": solver,
            },
            "score_audit": {
                "base_score_matrix": np.asarray([base_vectors[str(item["candidate_uid"])] for item in pool], dtype=np.float64).tolist(),
                "fused_score_matrix": fused.astype(float).tolist(),
                "score_matrix_orientation": "candidate_x_association_state",
                "association_state_axis": state_axis,
                "public_id_axis": public_axis,
                "explicit_authority_pairs": [
                    {"association_state_id": state_id, "public_id": public_id}
                    for state_id, public_id in replay_pairs
                ],
                "target_public_column_index": target_col,
                "target_public_evidence_injected_candidate_uid": selected_uid if injected_delta > 0.0 else None,
                "target_public_evidence_delta": float(injected_delta),
                "target_evidence_injection_scale": TARGET_EVIDENCE_INJECTION_SCALE,
            },
            "memory_read": True,
            "memory_write": bool(assigned_target_uid is not None),
            "event_frame_memory_read": False,
            "first_memory_visible_frame": event_frame + 1,
            "raw_binding_switch": binding_switch,
            "trusted_memory_update": bool(
                assigned_target_uid is not None
                and selected_uid == assigned_target_uid
                and selection["reliable_for_memory_admission"]
                and next((item for item in pool if str(item["candidate_uid"]) == str(assigned_target_uid)), {}).get("feature") is not None
            ),
            "distractor_memory_update_count": int(distractor_updates),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "public_id_inference": False,
            "public_id_immutable": True,
        })
    if len(output_rows) != HORIZON + 1:
        raise RuntimeError(f"development replay row count mismatch: {event_id}")
    event_out = output_root / event_id
    event_out.mkdir(parents=True, exist_ok=True)
    frames_path = event_out / f"{variant}.jsonl"
    manifest_path = event_out / "event_manifest.json"
    atomic_jsonl(frames_path, output_rows)
    manifest = {
        "schema_version": "N72R7_CLOSED_LOOP_EVENT_MANIFEST_V1",
        "status": "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY",
        "variant": variant,
        "event_id": event_id,
        "sequence": sequence,
        "event_frame": event_frame,
        "future_window": [event_frame + 1, event_frame + HORIZON],
        "frame_count": len(output_rows),
        "target_public_id": target_public,
        "frames": str(frames_path),
        "frames_sha256": sha256_file(frames_path),
        "n72r6_c0_source": str(resolve(str(frozen_manifest["c0"]["path"]))),
        "n72r6_c0_source_sha256": str(frozen_manifest["c0"]["sha256"]),
        "n72r6_c1_source": str(resolve(str(frozen_manifest["c1"]["path"]))),
        "n72r6_target_stream_source": str(resolve(str(frozen_manifest["target_stream_frames"]))),
        "protocol_sha256": protocol["protocol_sha256"],
        "target_selection_count": target_selections,
        "explicit_none_selection_count": none_selections,
        "target_assignment_count": target_assignments,
        "raw_binding_switch_count": len(raw_switches),
        "raw_binding_switches": raw_switches,
        "trusted_memory_update_count": trusted_updates,
        "distractor_memory_update_count": distractor_updates,
        "trusted_feature_hashes": trusted_hashes,
        "distractor_feature_hashes": distractor_hashes,
        "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
        "candidate_pool_complete": True,
        "candidate_pool_source_order": [MAIN_B0_CANDIDATE] if variant == "D1" else [MAIN_B0_CANDIDATE, TARGET_SESSION_CURRENT_RAW],
        "raw_switch_preserves_public_id": all(not item["public_id_changed"] for item in raw_switches),
        "event_frame_memory_read": False,
        "first_memory_visible_frame": event_frame + 1,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "public_id_inference": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def write_failure(output_root: Path, event_id: str, variant: str, exc: BaseException) -> Path:
    path = output_root / "attempts" / f"{event_id}.{variant}.failure.json"
    atomic_json(path, {
        "schema_version": "N72R7_CLOSED_LOOP_FAILURE_V1",
        "status": "FAIL_N72R7_CLOSED_LOOP_EVENT_REPLAY",
        "event_id": event_id,
        "variant": variant,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "created_at_utc": now_utc(),
    })
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("D1", "D2"), required=False)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--output-root", default="outputs/N72R7/dev_replay/full_attempt1")
    parser.add_argument("--write-protocol", action="store_true")
    args = parser.parse_args()
    if args.write_protocol:
        payload = write_development_protocol()
        print(json.dumps({"status": "PASS_DEVELOPMENT_PROTOCOL_FROZEN", "protocol_sha256": payload["protocol_sha256"]}))
        return 0
    if not args.variant:
        parser.error("--variant is required unless --write-protocol is used")
    _, policy, frozen = validate_inputs()
    protocol = read_json(DEV_PROTOCOL)
    if protocol.get("schema_version") != "N72R7_DEVELOPMENT_REPLAY_PROTOCOL_V1":
        raise RuntimeError("development protocol is missing or invalid")
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    events = {str(item["event_id"]): item for item in policy["events"]}
    requested = sorted(args.event_id) if args.event_id else sorted(frozen)
    unknown = sorted(set(requested) - set(frozen))
    if unknown:
        error = ValueError(
            "requested event IDs are not in the frozen N72R6 replay/target-stream set: "
            f"{unknown}; available_count={len(frozen)}"
        )
        failure = write_failure(output_root, "__input_validation__", args.variant, error)
        print(json.dumps({"status": "FAIL_INPUT_VALIDATION", "artifact": str(failure), "error": str(error)}))
        return 1
    selector = TargetCandidateSelector(SelectorConfig())
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event_id in requested:
        try:
            result = run_event(events[event_id], frozen[event_id], variant=args.variant, output_root=output_root, selector=selector, protocol=protocol)
            results.append(result)
            print(json.dumps({"event_id": event_id, "variant": args.variant, "status": result["status"]}, ensure_ascii=False))
        except Exception as exc:  # preserve the first failure and continue independent events
            failure = write_failure(output_root, event_id, args.variant, exc)
            failures.append({"event_id": event_id, "artifact": str(failure), "error": str(exc)})
            print(json.dumps({"event_id": event_id, "variant": args.variant, "status": "FAIL", "artifact": str(failure)}, ensure_ascii=False))
    batch_status = {
        "schema_version": "N72R7_CLOSED_LOOP_BATCH_V1",
        "status": "PASS_N72R7_CLOSED_LOOP_BATCH" if len(results) == len(requested) and not failures else "PARTIAL_N72R7_CLOSED_LOOP_BATCH",
        "variant": args.variant,
        "requested_event_count": len(requested),
        "completed_event_count": len(results),
        "failed_event_count": len(failures),
        "results": [{"event_id": str(item["event_id"]), "manifest": str(output_root / str(item["event_id"]) / "event_manifest.json")} for item in results],
        "failures": failures,
        "protocol": str(DEV_PROTOCOL),
        "protocol_sha256": protocol["protocol_sha256"],
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "batch_status.json", batch_status)
    print(json.dumps({"status": batch_status["status"], "completed": len(results), "failed": len(failures)}, ensure_ascii=False))
    return 0 if batch_status["status"] == "PASS_N72R7_CLOSED_LOOP_BATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
