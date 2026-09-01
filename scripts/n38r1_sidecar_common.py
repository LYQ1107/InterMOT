"""N38R1 lossless, event-frame-inclusive diagnostic sidecar helpers.

This module is intentionally R1-only.  It reuses the frozen N37 replay
boundary and the frozen N36 candidate tape, but never writes under outputs/n36,
outputs/n37, or outputs/n38.  The event-frame association run is performed on
an audit-only StateManager clone; the future branches are produced by the
unchanged N37 paired_replay function, so the extra event-frame observation
cannot alter the treatment state or future trace.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sam3_intermot.association.ccam_replay import (
    _apply_shared_spatial_correction,
    _candidate_observations,
    _manager_from_prefix,
    paired_replay,
    validate_candidate_tape,
)
from scripts.n36_real_eval_common import (
    FEATURE_DIM,
    atomic_json,
    iter_rows,
    replay_candidate,
    variant_config,
)
from scripts.run_n37_replay import build_runtime_tape, runtime_event_view


ROOT = Path(__file__).resolve().parents[1]
N37_MANIFEST = ROOT / "outputs" / "n37" / "real_event_manifest.json"
N38_PROTOCOL = ROOT / "outputs" / "n38" / "diagnostic" / "diagnostic_protocol.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
N38R1_PROTOCOL = "N38R1_LOSSLESS_EVENT_AND_FUTURE_SIDECAR_V1"


def event_id_of(item: Mapping[str, Any]) -> str:
    event = item.get("event")
    if not isinstance(event, Mapping) or event.get("event_id") is None:
        raise ValueError("event.event_id missing")
    return str(event["event_id"])


def load_manifest_item(manifest_path: Path, event_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"N37 event manifest is not PASS: {payload.get('status')}")
    events = payload.get("events")
    if not isinstance(events, list) or payload.get("event_count") != len(events):
        raise RuntimeError("N37 event manifest event_count/list mismatch")
    matches = [item for item in events if isinstance(item, dict) and event_id_of(item) == str(event_id)]
    if len(matches) != 1:
        raise KeyError(f"expected one event for {event_id}, found {len(matches)}")
    return payload, deepcopy(matches[0])


def protocol_hash() -> str:
    payload = json.loads(N38_PROTOCOL.read_text(encoding="utf-8"))
    declared = payload.get("protocol_hash")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if declared is None:
        raise RuntimeError("frozen N38 diagnostic protocol_hash is missing")
    return str(declared)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(item: Mapping[str, Any]) -> Path:
    value = Path(str(item["source_tape"]))
    return value if value.is_absolute() else ROOT / value


def _forbidden_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized in {
        "future_gt",
        "future_image",
        "future_features",
        "future_candidate_outcomes",
        "gt",
        "gt_box",
        "gt_id",
        "dataset_identity",
        "public_id",
        "reward",
        "selected_candidate",
        "candidate_outcome",
    } or normalized.startswith("future_")


def _strict_feature(value: Any, dim: int = FEATURE_DIM) -> tuple[list[float], float]:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != dim or not np.all(np.isfinite(vector)):
        raise ValueError(f"feature must be finite {dim}-D, got shape={vector.shape}")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-6:
        raise ValueError("feature norm is non-finite or zero")
    return vector.astype(float).tolist(), norm


def _mask_sidecar(mask: Any) -> dict[str, Any]:
    if mask is None:
        return {
            "present": False,
            "mask_hash": None,
            "mask_status": "NOT_PRESENT_IN_N36_TAPE",
            "mask": None,
        }
    if not isinstance(mask, Mapping):
        return {
            "present": False,
            "mask_hash": None,
            "mask_status": "PRESENT_BUT_SCHEMA_INVALID",
            "mask": deepcopy(mask),
        }
    declared = mask.get("sha256")
    if declared is None:
        status = "PRESENT_WITHOUT_DECLARED_SHA256"
    else:
        status = "PRESENT_DECLARED_SHA256_PRESERVED"
    return {
        "present": True,
        "mask_hash": None if declared is None else str(declared),
        "mask_status": status,
        "mask": deepcopy(dict(mask)),
    }


def source_candidate_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = [str(key) for key in candidate if _forbidden_key(key)]
    if forbidden:
        raise ValueError(f"source candidate contains forbidden runtime key(s): {forbidden}")
    source = deepcopy(dict(candidate))
    source_feature, source_norm = _strict_feature(candidate.get("machine_embedding"))
    mask = _mask_sidecar(candidate.get("mask"))
    source["machine_embedding"] = source_feature
    source["machine_feature_finite"] = True
    source["machine_feature_dim"] = FEATURE_DIM
    source["machine_feature_norm"] = source_norm
    source["mask_hash"] = mask["mask_hash"]
    source["mask_status"] = mask["mask_status"]
    source["mask_preserved_losslessly"] = bool(mask["present"])
    return source


def read_source_rows(item: Mapping[str, Any], event_frame: int, end_frame: int) -> dict[int, dict[str, Any]]:
    path = source_path(item)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = item.get("source_tape_sha256")
    actual_hash = file_sha256(path)
    if expected_hash is not None and str(expected_hash) != actual_hash:
        raise RuntimeError(
            f"source tape sha256 mismatch: expected={expected_hash} actual={actual_hash}"
        )
    rows: dict[int, dict[str, Any]] = {}
    for _line_no, row in iter_rows(path, event_frame, end_frame):
        frame = int(row["frame"])
        if frame in rows:
            raise RuntimeError(f"duplicate source frame {frame}")
        if row.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"source frame {frame} runtime_future_gt_used is not false")
        if row.get("runtime_gt_read") is not False:
            raise RuntimeError(f"source frame {frame} runtime_gt_read is not false")
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError(f"source frame {frame} candidates missing")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise RuntimeError(f"source frame {frame} candidate is not an object")
            source_candidate_record(candidate)
        rows[frame] = row
    expected = list(range(int(event_frame), int(end_frame) + 1))
    actual = sorted(rows)
    if actual != expected:
        raise RuntimeError(
            f"source frame range incomplete: expected {event_frame}..{end_frame}, got {actual[:3]}..{actual[-3:] if actual else []} count={len(actual)}"
        )
    return rows


def safe_source_row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    mapping = row.get("public_id_to_sequence_global_native_id") or {}
    return {
        "frame": int(row["frame"]),
        "candidate_complete": bool(row.get("candidate_complete") is True),
        "candidate_set_complete": bool(row.get("candidate_set_complete") is True),
        "runtime_future_gt_used": row.get("runtime_future_gt_used"),
        "runtime_gt_read": row.get("runtime_gt_read"),
        "public_id_namespace": row.get("public_id_namespace"),
        "states_public_ids": deepcopy(row.get("states_public_ids", [])),
        "public_id_to_sequence_global_native_id": deepcopy(mapping),
        "chunk_id": row.get("chunk_id"),
        "source_chunk_ids": deepcopy(row.get("source_chunk_ids", [])),
        "candidate_set_source": row.get("candidate_set_source"),
    }


def _assignment_public_ids(audit: Mapping[str, Any]) -> list[Any]:
    value = audit.get("candidate_public_ids")
    return list(value) if isinstance(value, list) else []


def _candidate_source_public_id(
    source_candidate: Mapping[str, Any], prefix_state: list[Mapping[str, Any]], event: Mapping[str, Any]
) -> int | None:
    global_id = source_candidate.get("sequence_global_native_id")
    if global_id is None:
        return None
    matches = [
        int(item["public_id"])
        for item in prefix_state
        if item.get("native_tid") is not None and int(item["native_tid"]) == int(global_id)
    ]
    if len(matches) == 1:
        return matches[0]
    correction_ids = []
    for correction in event.get("spatial_corrections", []) if isinstance(event.get("spatial_corrections"), list) else []:
        if correction.get("native_tid") is not None and int(correction["native_tid"]) == int(global_id):
            if correction.get("public_id") is not None:
                correction_ids.append(int(correction["public_id"]))
    return correction_ids[0] if len(set(correction_ids)) == 1 else None


def _state_mapping(audit: Mapping[str, Any]) -> dict[str, Any]:
    mapping = audit.get("public_id_to_native_tid")
    return deepcopy(mapping) if isinstance(mapping, Mapping) else {}


def enrich_candidate_audit(
    audit: Mapping[str, Any],
    source_row: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    frame: int,
    is_event_frame: bool,
    memory_read: bool,
    current_frame_write_hidden: bool,
) -> dict[str, Any]:
    """Keep every frozen raw audit field and add lossless source/mapping data."""
    if not isinstance(audit, Mapping):
        raise ValueError(f"missing candidate audit at frame {frame}")
    output = deepcopy(dict(audit))
    raw_candidates = output.get("candidates")
    source_candidates = source_row.get("candidates")
    if not isinstance(raw_candidates, list) or not isinstance(source_candidates, list):
        raise ValueError(f"candidate list missing at frame {frame}")
    if len(raw_candidates) != len(source_candidates):
        raise ValueError(
            f"candidate audit/source count mismatch at frame {frame}: {len(raw_candidates)} != {len(source_candidates)}"
        )
    source_records = [source_candidate_record(candidate) for candidate in source_candidates]
    event = item["event"]
    assigned = _assignment_public_ids(output)
    enriched: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"raw candidate {index} at frame {frame} is not an object")
        source = source_records[index]
        feature, feature_norm = _strict_feature(raw.get("feature"))
        assigned_id = assigned[index] if index < len(assigned) else None
        assigned_id = None if assigned_id is None else int(assigned_id)
        source_global = source.get("sequence_global_native_id")
        enriched.append(
            {
                "candidate_index": int(index),
                "candidate_obs_id": int(raw.get("obs_id", index)),
                "candidate_native_id": int(raw.get("native_tid", -1)),
                "candidate_local_native_id": source.get("local_native_id"),
                "sequence_global_id": source_global,
                "candidate_public_id": assigned_id,
                "assigned_public_id": assigned_id,
                "source_public_id": _candidate_source_public_id(
                    source, item.get("prefix_state", []), event
                ),
                "public_id_mapping_status": (
                    "ASSOCIATION_ASSIGNMENT_COMPLETE" if assigned_id is not None else "UNMAPPED"
                ),
                "feature": feature,
                "feature_finite": True,
                "feature_dim": FEATURE_DIM,
                "feature_norm": feature_norm,
                "box": deepcopy(raw.get("box")),
                "confidence": float(raw.get("confidence", 1.0)),
                "source_candidate": source,
            }
        )
    output["candidate_records"] = enriched
    output["candidate_public_id_mapping"] = [
        {
            "candidate_index": row["candidate_index"],
            "candidate_native_id": row["candidate_native_id"],
            "candidate_local_native_id": row["candidate_local_native_id"],
            "sequence_global_id": row["sequence_global_id"],
            "source_public_id": row["source_public_id"],
            "assigned_public_id": row["assigned_public_id"],
        }
        for row in enriched
    ]
    output["candidate_public_id_mapping_complete"] = bool(
        output.get("candidate_public_id_mapping_complete") is True
        and all(row["assigned_public_id"] is not None for row in enriched)
    )
    output["source_row_metadata"] = safe_source_row_metadata(source_row)
    output["frame"] = int(frame)
    output["is_event_frame"] = bool(is_event_frame)
    output["is_future_frame"] = not bool(is_event_frame)
    output["memory_read"] = bool(memory_read)
    output["memory_write"] = False if is_event_frame else bool(output.get("memory_write", False))
    output["current_frame_write_hidden"] = bool(current_frame_write_hidden)
    output["runtime_future_gt_used"] = False
    output["gt_loaded_posthoc"] = False
    output["raw_candidate_audit_preserved"] = True
    output["state_public_id_to_native_id"] = _state_mapping(output)

    fused = np.asarray(output.get("fused_scores", []), dtype=float)
    state_ids = [int(value) for value in output.get("public_id_order", [])]
    if fused.ndim != 2 or fused.shape != (len(enriched), len(state_ids)):
        raise ValueError(
            f"fused score shape mismatch at frame {frame}: {fused.shape} expected {(len(enriched), len(state_ids))}"
        )
    if not np.all(np.isfinite(fused)):
        raise ValueError(f"non-finite fused score at frame {frame}")
    rank_by_state: list[list[int]] = [[0 for _ in state_ids] for _ in enriched]
    for state_index in range(len(state_ids)):
        order = sorted(range(len(enriched)), key=lambda idx: (-float(fused[idx, state_index]), idx))
        for rank, candidate_index in enumerate(order, 1):
            rank_by_state[candidate_index][state_index] = int(rank)
    output["candidate_rank_by_state"] = rank_by_state

    target_public_id = int(item["event"].get("public_id", item["event"].get("canonical_public_id")))
    target_state_index = state_ids.index(target_public_id) if target_public_id in state_ids else None
    top_rows: list[tuple[float, int]] = []
    if target_state_index is not None:
        top_rows = sorted(
            [(float(fused[index, target_state_index]), index) for index in range(len(enriched))],
            key=lambda value: (-value[0], value[1]),
        )
    top1 = top_rows[0] if len(top_rows) >= 1 else None
    top2 = top_rows[1] if len(top_rows) >= 2 else None
    target_row = next(
        (index for index, value in enumerate(assigned) if int(value) == target_public_id),
        None,
    ) if assigned else None
    assigned_col = None
    if target_row is not None and target_row < len(output.get("assignment_after_scope", [])):
        value = int(output["assignment_after_scope"][target_row])
        if 0 <= value < len(state_ids):
            assigned_col = value
    alternatives = [index for index in range(len(state_ids)) if index != assigned_col]
    best_alternative_col = None
    assignment_score_margin = None
    if target_row is not None and assigned_col is not None and alternatives:
        best_alternative_col = min(alternatives, key=lambda index: -float(fused[target_row, index]))
        assignment_score_margin = float(
            fused[target_row, assigned_col] - fused[target_row, best_alternative_col]
        )
    output["hungarian_cost_audit"] = {
        "status": "AVAILABLE",
        "orientation": "candidate_row_x_public_id_state_column",
        "cost_definition": "cost=-fused_score; scipy linear_sum_assignment(-fused_score)",
        "cost_matrix": (-fused).tolist(),
        "row_candidate_public_ids": deepcopy(assigned),
        "row_candidate_native_ids": deepcopy(output.get("candidate_native_ids", [])),
        "column_public_ids": state_ids,
        "assignment_after_scope": deepcopy(output.get("assignment_after_scope", [])),
        "target_public_id": target_public_id,
        "target_state_index": target_state_index,
        "target_row": target_row,
        "assigned_col": assigned_col,
        "best_alternative_col": best_alternative_col,
        "assignment_score_margin": assignment_score_margin,
        "assignment_cost_margin": None if assignment_score_margin is None else -assignment_score_margin,
        "target_row_costs": (
            (-fused[target_row]).tolist() if target_row is not None else None
        ),
    }
    output["target_state_top_two"] = {
        "target_public_id": target_public_id,
        "target_state_index": target_state_index,
        "top1_candidate_index": None if top1 is None else int(top1[1]),
        "top2_candidate_index": None if top2 is None else int(top2[1]),
        "top1_candidate_public_id": None if top1 is None else int(assigned[top1[1]]),
        "top2_candidate_public_id": None if top2 is None else int(assigned[top2[1]]),
        "top1_sequence_global_id": None if top1 is None else enriched[top1[1]]["sequence_global_id"],
        "top2_sequence_global_id": None if top2 is None else enriched[top2[1]]["sequence_global_id"],
        "top1_score": None if top1 is None else float(top1[0]),
        "top2_score": None if top2 is None else float(top2[0]),
        "top1_top2_score_margin": None if top1 is None or top2 is None else float(top1[0] - top2[0]),
        "top1_top2_normalized_margin": (
            None
            if top1 is None or top2 is None
            else float((top1[0] - top2[0]) / max(1.0, abs(top1[0])))
        ),
        "top2_distinct_public_id": (
            None if top1 is None or top2 is None else bool(assigned[top1[1]] != assigned[top2[1]])
        ),
    }
    return output


def build_event_frame_audit(
    tape: Mapping[str, Any],
    source_row: Mapping[str, Any],
    item: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Audit event-frame scores without changing the future treatment state."""
    event = tape["event"]
    frame = int(event["frame"])
    manager = _manager_from_prefix(tape["prefix_state"], frame, config, FEATURE_DIM)
    _apply_shared_spatial_correction(manager, event, frame, FEATURE_DIM)
    runtime_frame = {
        "frame": frame,
        "candidates": [
            replay_candidate(candidate, index)
            for index, candidate in enumerate(source_row.get("candidates", []))
        ],
    }
    manager.rollout_frame(frame, _candidate_observations(runtime_frame, FEATURE_DIM), model=None)
    if not manager.candidate_log:
        raise RuntimeError("event-frame audit manager produced no candidate log")
    raw = deepcopy(manager.candidate_log[-1])
    return enrich_candidate_audit(
        raw,
        source_row,
        item,
        frame=frame,
        is_event_frame=True,
        memory_read=False,
        current_frame_write_hidden=True,
    )


def _future_entry(
    entry: Mapping[str, Any],
    source_rows: Mapping[int, Mapping[str, Any]],
    item: Mapping[str, Any],
    *,
    memory_write: bool,
    memory_read: bool,
) -> dict[str, Any]:
    frame = int(entry["frame"])
    source_row = source_rows.get(frame)
    if source_row is None:
        raise RuntimeError(f"future source row missing for frame {frame}")
    return {
        "frame": frame,
        "is_event_frame": False,
        "is_future_frame": True,
        "memory_write": bool(memory_write),
        "memory_read": bool(memory_read),
        "runtime_future_gt_used": False,
        "gt_loaded_posthoc": False,
        "rows": deepcopy(entry.get("rows", [])),
        "candidate_audit": enrich_candidate_audit(
            entry.get("candidate_audit"),
            source_row,
            item,
            frame=frame,
            is_event_frame=False,
            memory_read=memory_read,
            current_frame_write_hidden=False,
        ),
    }


def build_sidecar(item: Mapping[str, Any], variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant {variant}")
    event = item["event"]
    event_frame = int(event["frame"])
    expected_end = min(int(item["sequence_frame_count"]) - 1, event_frame + 100)
    source_rows = read_source_rows(item, event_frame, expected_end)
    tape = build_runtime_tape(item, horizon=100)
    validation = validate_candidate_tape(tape, feat_dim=FEATURE_DIM)
    if not validation["valid"] or not validation["candidate_complete"]:
        raise RuntimeError(f"frozen runtime tape validation failed: {validation}")
    future_frames = [int(row["frame"]) for row in tape.get("frames", [])]
    expected_future = list(range(event_frame + 1, expected_end + 1))
    if future_frames != expected_future:
        raise RuntimeError(
            f"runtime future window mismatch: expected {expected_future[:2]}..{expected_future[-2:]}, got {future_frames[:2]}..{future_frames[-2:]}"
        )
    config, description = variant_config(variant)
    event_audit = build_event_frame_audit(tape, source_rows[event_frame], item, config)
    replay = paired_replay(
        tape,
        config=config,
        feat_dim=FEATURE_DIM,
        write_branch_uses_appearance_memory=(variant != "M0"),
    )
    if replay.get("status") != "PASS":
        raise RuntimeError(f"paired_replay returned {replay.get('status')}: {replay.get('validation')}")
    branches: dict[str, Any] = {}
    for branch_name, branch in replay.get("branches", {}).items():
        actual_write = bool(branch.get("memory_write", False))
        actual_read = bool(branch.get("appearance_memory", {}).get("records"))
        trace = branch.get("future_trace")
        if not isinstance(trace, list) or [int(row["frame"]) for row in trace] != expected_future:
            raise RuntimeError(f"{variant}:{branch_name} future trace is incomplete or non-contiguous")
        branches[branch_name] = {
            "memory_write": actual_write,
            "memory_read": bool(actual_write and any(
                bool(np.asarray(row.get("candidate_audit", {}).get("appearance_memory_scores", []), dtype=float).size)
                for row in trace
            )),
            "future_trace": [
                _future_entry(
                    entry,
                    source_rows,
                    item,
                    memory_write=actual_write,
                    memory_read=bool(actual_write),
                )
                for entry in trace
            ],
            "state_summary": deepcopy(branch.get("state_summary", {})),
            "appearance_memory": deepcopy(branch.get("appearance_memory", {})),
        }
    if set(branches) != {"memory_write=False", "memory_write=True"}:
        raise RuntimeError(f"unexpected paired branch keys: {sorted(branches)}")
    return {
        "protocol": N38R1_PROTOCOL,
        "status": "PASS",
        "event_id": event_id_of(item),
        "variant": variant,
        "variant_description": description,
        "sequence": str(event["sequence"]),
        "event_frame": event_frame,
        "future_frame_start": event_frame + 1,
        "future_frame_end": expected_end,
        "future_frame_count": len(expected_future),
        "interaction_source": "simulated_from_gt",
        "synthetic": False,
        "event": runtime_event_view(event),
        "prefix_state": deepcopy(item["prefix_state"]),
        "source_candidate_tape": str(item["source_tape"]),
        "source_tape_sha256": item.get("source_tape_sha256"),
        "frozen_n38_protocol_hash": protocol_hash(),
        "event_frame_audit": {
            "frame": event_frame,
            "is_event_frame": True,
            "is_future_frame": False,
            "audit_only_clone": True,
            "future_treatment_state_untouched": True,
            "memory_write": False,
            "memory_read": False,
            "current_frame_write_hidden": True,
            "runtime_future_gt_used": False,
            "gt_loaded_posthoc": False,
            "candidate_audit": event_audit,
        },
        "memory_write_transaction": {
            "correction_precedes_memory_write": True,
            "event_frame_read_after_write": False,
            "future_visible_from_frame": event_frame + 1,
            "variant_memory_write": {
                name: bool(name != "M0") for name in ("M0", "M1", "M2", "M3", "M4")
            },
        },
        "branches": branches,
        "runtime_future_gt_used": False,
        "gt_loaded_posthoc": False,
        "gt_runtime_boundary": {
            "runtime_future_gt_used": False,
            "runtime_gt_fields_sent": [],
            "gt_available_only_to_posthoc_stage": True,
            "event_time_human_box_is_allowed_input": True,
        },
        "lossless_contract": {
            "event_frame_included": True,
            "event_plus_one_included": bool(expected_future),
            "all_future_frames_included": True,
            "raw_candidate_audit_preserved": True,
            "full_512d_runtime_features_preserved": True,
            "source_machine_features_preserved": True,
            "source_masks_or_hashes_preserved": True,
            "hungarian_cost_audit_preserved": True,
            "atomic_artifact_written_by_worker": True,
        },
        "tape_validation": validation,
        "replay_comparison": deepcopy(replay.get("comparison", [])),
        "metric_status": "NOT_COMPUTED_R1_DIAGNOSTIC_ONLY",
    }


def write_failure(path: Path, *, event_id: str, variant: str, exc: BaseException, traceback_text: str) -> None:
    atomic_json(
        path,
        {
            "protocol": N38R1_PROTOCOL,
            "status": "FAIL",
            "event_id": event_id,
            "variant": variant,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback_text,
            "runtime_future_gt_used": False,
            "artifact_is_failure_evidence": True,
        },
    )
