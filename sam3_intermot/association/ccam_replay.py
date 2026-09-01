"""Future-only, candidate-complete replay utilities for N33 CCAM.

The replay boundary is deliberately small and explicit.  By default a tape
contains the state immediately before a human event, an explicit shared
spatial correction transaction, and observations from frames strictly after
the event.  Both branches apply that same spatial transaction; their only
intended difference is whether CCAM is written.  An explicit
``prefix_state_is_post_correction`` compatibility flag is available for a
precomputed state snapshot, but is not the default protocol.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from sam3_intermot.association.state_manager import StateManager, StateManagerConfig


FORBIDDEN_CANDIDATE_KEYS = {
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
}


def _feature(value: Any, dim: int = 512) -> Optional[np.ndarray]:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != int(dim) or not np.all(np.isfinite(vector)):
        return None
    if float(np.linalg.norm(vector)) <= 1e-6:
        return None
    return vector


def _has_forbidden_candidate_key(mapping: Mapping[str, Any]) -> Optional[str]:
    for key in mapping:
        normalized = str(key).strip().lower()
        if normalized in FORBIDDEN_CANDIDATE_KEYS or normalized.startswith("future_"):
            return str(key)
    return None


def validate_candidate_tape(tape: Mapping[str, Any], feat_dim: int = 512) -> dict:
    """Validate structure and causal completeness without reading labels.

    A structurally valid but incomplete tape is returned with
    ``candidate_complete=False`` so the caller can preserve a useful audit
    artifact while refusing to claim an identity treatment effect.
    """
    issues: List[str] = []
    if not isinstance(tape, Mapping):
        return {"valid": False, "candidate_complete": False, "issues": ["tape_not_mapping"]}
    frames = tape.get("frames")
    if not isinstance(frames, list) or not frames:
        issues.append("frames_missing_or_empty")
        frames = []
    frame_ids: List[int] = []
    candidate_count = 0
    feature_missing_count = 0
    for frame_item in frames:
        if not isinstance(frame_item, Mapping):
            issues.append("frame_not_mapping")
            continue
        try:
            frame_id = int(frame_item["frame"])
        except (KeyError, TypeError, ValueError):
            issues.append("frame_id_invalid")
            continue
        frame_ids.append(frame_id)
        candidates = frame_item.get("candidates")
        if not isinstance(candidates, list):
            issues.append(f"frame_{frame_id}_candidates_missing")
            continue
        for candidate in candidates:
            candidate_count += 1
            if not isinstance(candidate, Mapping):
                issues.append(f"frame_{frame_id}_candidate_not_mapping")
                feature_missing_count += 1
                continue
            forbidden = _has_forbidden_candidate_key(candidate)
            if forbidden is not None:
                issues.append(f"frame_{frame_id}_candidate_forbidden_key:{forbidden}")
            try:
                box = np.asarray(candidate["box"], dtype=float).reshape(-1)
                if box.size != 4 or not np.all(np.isfinite(box)):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                issues.append(f"frame_{frame_id}_candidate_box_invalid")
            embedding = candidate.get("embedding", candidate.get("feature"))
            if _feature(embedding, feat_dim) is None:
                feature_missing_count += 1
                issues.append(f"frame_{frame_id}_candidate_feature_unavailable")
            try:
                confidence = float(candidate.get("confidence", candidate.get("conf", 1.0)))
                if not np.isfinite(confidence):
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(f"frame_{frame_id}_candidate_confidence_invalid")
    if frame_ids != sorted(set(frame_ids)):
        issues.append("frame_ids_not_strictly_increasing_and_unique")

    prefix = tape.get("prefix_state")
    prefix_is_empty = not isinstance(prefix, list) or not prefix
    if prefix_is_empty:
        issues.append("prefix_state_missing_or_empty")
        prefix = []
    prefix_ids = set()
    for item in prefix:
        if not isinstance(item, Mapping):
            issues.append("prefix_state_item_not_mapping")
            continue
        try:
            pid = int(item["public_id"])
        except (KeyError, TypeError, ValueError):
            issues.append("prefix_public_id_invalid")
            continue
        if pid in prefix_ids:
            issues.append(f"prefix_public_id_duplicate:{pid}")
        prefix_ids.add(pid)
        if _feature(item.get("embedding", item.get("feature")), feat_dim) is None:
            issues.append(f"prefix_feature_unavailable:{pid}")
        try:
            box = np.asarray(item["box"], dtype=float).reshape(-1)
            if box.size != 4 or not np.all(np.isfinite(box)):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append(f"prefix_box_invalid:{pid}")

    authorized_empty_prefix_add = False
    event = tape.get("event")
    event_valid = isinstance(event, Mapping)
    if not event_valid:
        issues.append("human_event_missing")
    else:
        try:
            event_frame = int(event["frame"])
            public_id = int(event.get("public_id", event.get("canonical_public_id")))
            if public_id < 1:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            issues.append("human_event_frame_or_public_id_invalid")
            event_frame = None
            public_id = None
        if _feature(event.get("human_embedding", event.get("embedding")), feat_dim) is None:
            issues.append("human_embedding_unavailable")
        if event_frame is not None and any(frame <= event_frame for frame in frame_ids):
            issues.append("future_tape_contains_event_or_prefix_frame")
        if public_id is not None and prefix_ids and public_id not in prefix_ids:
            # ADD_NEW_IDENTITY is the one legal transaction whose
            # authoritative public ID is created by the current-frame spatial
            # correction rather than inherited from the pre-event prefix.
            # Keep this exception narrow: the event must explicitly name the
            # action and carry a correction entry for the same new ID.
            add_correction_ids = {
                int(item.get("public_id"))
                for item in _correction_entries(event)
                if isinstance(item, Mapping) and item.get("public_id") is not None
            }
            is_authorized_add = (
                str(event.get("action_type", "")) == "ADD_NEW_IDENTITY"
                and public_id in add_correction_ids
            )
            if not is_authorized_add:
                issues.append("human_public_id_absent_from_prefix_state")
        if prefix_is_empty:
            # ADD_NEW_IDENTITY at frame zero has no pre-event identity state.
            # It is valid only when the event explicitly supplies the new
            # public ID through the current-frame spatial correction.  This is
            # the same narrow exception used above for an absent new ID; all
            # other actions still require a non-empty prefix.
            add_correction_ids = {
                int(item.get("public_id"))
                for item in _correction_entries(event)
                if isinstance(item, Mapping) and item.get("public_id") is not None
            }
            authorized_empty_prefix_add = bool(
                str(event.get("action_type", "")) == "ADD_NEW_IDENTITY"
                and public_id is not None
                and public_id in add_correction_ids
            )
            if authorized_empty_prefix_add:
                issues = [
                    issue
                    for issue in issues
                    if issue != "prefix_state_missing_or_empty"
                ]
        if not bool(tape.get("prefix_state_is_post_correction", False)):
            has_spatial_correction = any(
                event.get(key) is not None
                for key in (
                    "spatial_correction",
                    "spatial_corrections",
                    "correction_observation",
                    "correction_box",
                    "gt_box",
                )
            )
            if not has_spatial_correction:
                issues.append("spatial_correction_missing_for_pre_correction_prefix")

    # A caller must explicitly attest that every candidate available at each
    # future frame is present.  A tape containing only the selected candidate
    # is not complete merely because its feature is finite.
    explicitly_complete = tape.get(
        "candidate_set_complete", tape.get("candidate_complete", False)
    )
    candidate_complete = bool(
        explicitly_complete
        and bool(frames)
        and candidate_count >= 0
        and feature_missing_count == 0
        and not any(issue.startswith("prefix_feature_unavailable") for issue in issues)
        and not any("forbidden_key" in issue for issue in issues)
    )
    structural_issues = [
        issue
        for issue in issues
        if not issue.endswith("_feature_unavailable")
        and "candidate_feature_unavailable" not in issue
    ]
    valid = not structural_issues and event_valid and (
        bool(prefix) or authorized_empty_prefix_add
    )
    return {
        "valid": bool(valid),
        "candidate_complete": bool(candidate_complete and valid),
        "issues": issues,
        "frame_count": len(frames),
        "candidate_count": candidate_count,
        "feature_missing_count": feature_missing_count,
        "authorized_empty_prefix_add": bool(authorized_empty_prefix_add),
    }


def _candidate_observations(frame_item: Mapping[str, Any], feat_dim: int = 512) -> List[dict]:
    observations = []
    for index, candidate in enumerate(frame_item.get("candidates", [])):
        embedding = _feature(candidate.get("embedding", candidate.get("feature")), feat_dim)
        if embedding is None:
            raise ValueError(f"candidate feature unavailable at frame {frame_item.get('frame')} index {index}")
        observations.append(
            {
                "obs_id": int(candidate.get("obs_id", index)),
                "box": np.asarray(candidate["box"], dtype=float).copy(),
                "native_tid": int(candidate.get("native_tid", -1)),
                "native_age": float(candidate.get("native_age", 0.0)),
                "conf": float(candidate.get("confidence", candidate.get("conf", 1.0))),
                "feat": embedding.copy(),
                "has_feat": 1.0,
            }
        )
    return observations


def _manager_from_prefix(
    prefix_state: Sequence[Mapping[str, Any]],
    event_frame: int,
    config: StateManagerConfig,
    feat_dim: int = 512,
) -> StateManager:
    manager = StateManager(config)
    for item in prefix_state:
        pid = int(item["public_id"])
        embedding = _feature(item.get("embedding", item.get("feature")), feat_dim)
        if embedding is None:
            raise ValueError(f"prefix feature unavailable for public_id={pid}")
        observation = {
            "feat": embedding,
            "box": np.asarray(item["box"], dtype=float),
            "native_tid": int(item.get("native_tid", -1)),
        }
        state = manager.get_or_create(pid, observation, event_frame)
        state.last_native_tid = int(item.get("native_tid", -1))
        manager.next_pid = max(manager.next_pid, pid + 1)
    return manager


def _correction_entries(event: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    corrections = event.get("spatial_corrections")
    if isinstance(corrections, list):
        return [item for item in corrections if isinstance(item, Mapping)]
    correction = event.get("spatial_correction") or event.get("correction_observation")
    if isinstance(correction, Mapping):
        return [correction]
    return [
        {
            "public_id": event.get("public_id", event.get("canonical_public_id")),
            "box": event.get("correction_box", event.get("gt_box")),
            "native_tid": event.get("native_tid", -1),
            "embedding": event.get("correction_embedding"),
        }
    ]


def _apply_shared_spatial_correction(
    manager: StateManager,
    event: Mapping[str, Any],
    frame: int,
    feat_dim: int,
) -> None:
    """Apply the same current-frame spatial transaction to one branch.

    This is a compact replay adapter, not a second identity selector.  The
    authoritative public IDs and current correction observations are supplied
    by the event; no future label is consulted.  The full N10 transaction
    remains the production spatial path.
    """
    if bool(event.get("prefix_state_is_post_correction", False)):
        return
    for correction in _correction_entries(event):
        public_value = correction.get(
            "public_id", event.get("public_id", event.get("canonical_public_id"))
        )
        if public_value is None:
            raise ValueError("spatial correction public_id unavailable")
        public_id = int(public_value)
        box_value = correction.get("box", correction.get("correction_box"))
        if box_value is None:
            raise ValueError(f"spatial correction box unavailable for public_id={public_id}")
        box = np.asarray(box_value, dtype=float).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)):
            raise ValueError(f"spatial correction box invalid for public_id={public_id}")
        native_tid = int(correction.get("native_tid", -1))
        embedding = _feature(
            correction.get("embedding", correction.get("feature")), feat_dim
        )
        state = manager.states.get(public_id)
        if state is None:
            if embedding is None:
                embedding = np.zeros(feat_dim, dtype=np.float32)
            state = manager.get_or_create(
                public_id,
                {"feat": embedding, "box": box, "native_tid": native_tid},
                frame,
            )
            manager.next_pid = max(manager.next_pid, public_id + 1)
        else:
            if embedding is not None:
                state.update_machine(
                    embedding,
                    box,
                    frame,
                    native_tid,
                    manager.cfg.ema,
                    update_prototype=True,
                )
            else:
                state.last_box = box.copy()
                state.last_seen_frame = int(frame)
                state.last_native_tid = native_tid
                state.state = state.ACTIVE
                state.lost_age = 0
        if native_tid >= 0:
            state.add_positive(native_tid, manager.native_expiry(frame))


def _safe_trace(manager: StateManager) -> List[dict]:
    return deepcopy(manager.candidate_log)


def _run_branch(
    tape: Mapping[str, Any],
    event: Mapping[str, Any],
    write_memory: bool,
    config: StateManagerConfig,
    feat_dim: int,
) -> dict:
    event_frame = int(event["frame"])
    branch_config = replace(config, use_appearance_memory=bool(write_memory))
    manager = _manager_from_prefix(tape["prefix_state"], event_frame, branch_config, feat_dim)
    _apply_shared_spatial_correction(manager, event, event_frame, feat_dim)
    public_id = int(event.get("public_id", event.get("canonical_public_id")))
    embedding = _feature(event.get("human_embedding", event.get("embedding")), feat_dim)
    if write_memory:
        if embedding is None:
            raise ValueError("human_embedding_unavailable")
        competitors = [
            value
            for value in (
                _feature(item, feat_dim)
                for item in event.get("competing_embeddings", [])
            )
            if value is not None
        ]
        manager.update_human_appearance(
            public_id,
            event_frame,
            embedding,
            quality=float(event.get("quality", 1.0)),
            competing_embeddings=competitors,
            write_event_id=(None if event.get("event_id") is None else str(event["event_id"])),
        )
    frame_trace = []
    for frame_item in tape["frames"]:
        frame = int(frame_item["frame"])
        observations = _candidate_observations(frame_item, feat_dim)
        rows = manager.rollout_frame(frame, observations, model=None)
        frame_trace.append(
            {
                "frame": frame,
                "rows": [[int(pid), np.asarray(box, dtype=float).tolist()] for pid, box in rows],
                "candidate_audit": deepcopy(manager.candidate_log[-1]),
            }
        )
    return {
        "memory_write": bool(write_memory),
        "status": "PASS",
        "future_trace": frame_trace,
        "state_summary": manager.state_summary(),
        "appearance_memory": manager.appearance_memory.serialize(),
        "candidate_log": _safe_trace(manager),
    }


def paired_replay(
    tape: Mapping[str, Any],
    *,
    config: Optional[StateManagerConfig] = None,
    feat_dim: int = 512,
    write_branch_uses_appearance_memory: Optional[bool] = None,
) -> dict:
    """Run identical future candidates with and without the CCAM write.

    Historically the default write branch enabled appearance memory even
    when the caller supplied the legacy-disabled config; N33's focused smoke
    relies on that behavior.  N34 passes an explicit value so M0 can be a
    genuine disabled baseline while M1-M4 compare the same enabled variant
    with only the human write toggled.
    """
    validation = validate_candidate_tape(tape, feat_dim=feat_dim)
    if not validation["valid"] or not validation["candidate_complete"]:
        return {
            "status": "NOT_AVAILABLE",
            "candidate_complete": bool(validation["candidate_complete"]),
            "validation": validation,
            "identity_effect": "NOT_COMPUTABLE",
        }
    event = tape["event"]
    base_config = config or StateManagerConfig(variant="reid")
    write_branch_enabled = (
        True
        if write_branch_uses_appearance_memory is None
        else bool(write_branch_uses_appearance_memory)
    )
    branches = {
        "memory_write=False": _run_branch(tape, event, False, base_config, feat_dim),
        "memory_write=True": _run_branch(
            tape, event, write_branch_enabled, base_config, feat_dim
        ),
    }
    no_write = branches["memory_write=False"]["future_trace"]
    with_write = branches["memory_write=True"]["future_trace"]
    comparison = []
    for left, right in zip(no_write, with_write):
        left_audit = left["candidate_audit"]
        right_audit = right["candidate_audit"]
        left_scores = np.asarray(left_audit.get("fused_scores", []), dtype=float)
        right_scores = np.asarray(right_audit.get("fused_scores", []), dtype=float)
        left_assign = left_audit.get("assignment_after_scope", left_audit.get("assignment", []))
        right_assign = right_audit.get("assignment_after_scope", right_audit.get("assignment", []))
        same_shape = left_scores.shape == right_scores.shape
        score_delta = right_scores - left_scores if same_shape else None
        comparison.append(
            {
                "frame": int(left["frame"]),
                "memory_score_delta": None if score_delta is None else score_delta.tolist(),
                "max_abs_score_delta": (
                    None
                    if score_delta is None
                    else (float(np.max(np.abs(score_delta))) if score_delta.size else 0.0)
                ),
                "score_shape_equal": bool(same_shape),
                "assignment_changed": bool(left_assign != right_assign),
            }
        )
    metrics = {
        "future_h20_iou": None,
        "future_h50_iou": None,
        "future_missing_rate": None,
        "id_switch_count": None,
        "re_correction_count": None,
        "protected_identity_regression": None,
    }
    return {
        "status": "PASS",
        "candidate_complete": True,
        "validation": validation,
        "event_frame": int(event["frame"]),
        "public_id": int(event.get("public_id", event.get("canonical_public_id"))),
        "branches": branches,
        "comparison": comparison,
        "metrics": metrics,
        "metric_status": "NOT_COMPUTABLE_NO_POSTHOC_IDENTITY_LABELS",
        "write_branch_uses_appearance_memory": bool(write_branch_enabled),
        "identity_effect": "NOT_COMPUTABLE",
    }
