"""Causal correction-epoch state for target-scoped SAM sessions.

An epoch is the auditable boundary between an old native SAM binding and a
new target-only correction stream.  Public/lineage IDs do not change.  The
epoch clears only session-local native constraints and re-anchors motion and
human appearance evidence for the named public identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping, Optional

import numpy as np


def feature_sha256(feature: Any) -> str:
    value = np.asarray(feature, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("human anchor feature must be non-empty and finite")
    return hashlib.sha256(value.tobytes()).hexdigest()


@dataclass(frozen=True)
class CorrectionEpoch:
    epoch_id: str
    public_id: int
    start_frame: int
    human_anchor_sha256: str
    target_session_scope: str
    previous_native_tid: Optional[int]
    previous_native_scope: Optional[str]
    native_constraints_cleared: bool
    motion_reanchored: bool
    machine_prototype_frozen: bool

    def __post_init__(self) -> None:
        if not self.epoch_id or not self.target_session_scope:
            raise ValueError("epoch_id and target_session_scope are required")
        if int(self.public_id) <= 0:
            raise ValueError("public_id must be positive")
        if int(self.start_frame) < 0:
            raise ValueError("start_frame must be non-negative")
        if len(str(self.human_anchor_sha256)) != 64:
            raise ValueError("human_anchor_sha256 must be a SHA256 hex digest")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_correction_epoch(
    *,
    epoch_id: str,
    public_id: int,
    start_frame: int,
    human_anchor: Any,
    target_session_scope: str,
    previous_native_tid: Optional[int],
    previous_native_scope: Optional[str],
) -> CorrectionEpoch:
    return CorrectionEpoch(
        epoch_id=str(epoch_id),
        public_id=int(public_id),
        start_frame=int(start_frame),
        human_anchor_sha256=feature_sha256(human_anchor),
        target_session_scope=str(target_session_scope),
        previous_native_tid=(None if previous_native_tid is None else int(previous_native_tid)),
        previous_native_scope=(None if previous_native_scope in (None, "") else str(previous_native_scope)),
        native_constraints_cleared=True,
        motion_reanchored=True,
        machine_prototype_frozen=True,
    )


def apply_epoch_to_identity_state(
    state: Any,
    epoch: CorrectionEpoch,
    human_anchor: Any,
    authoritative_box: Any,
    *,
    target_native_tid: Optional[int] = None,
) -> None:
    """Apply epoch semantics to an association ``IdentityState`` in place."""

    clear = getattr(state, "clear_native_constraints", None)
    if callable(clear):
        clear()
    anchor = np.asarray(human_anchor, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(anchor))
    if anchor.size == 0 or not np.all(np.isfinite(anchor)) or norm <= 1.0e-6:
        raise ValueError("human anchor must be finite and non-zero")
    state.anchors = [anchor / norm]
    state.authority = 1.0
    state.anchor_frame = int(epoch.start_frame)
    state.last_box = np.asarray(authoritative_box, dtype=float).reshape(-1)[:4].copy()
    if state.last_box.size != 4 or not np.all(np.isfinite(state.last_box)):
        raise ValueError("authoritative_box must contain four finite values")
    state.velocity = np.zeros(2, dtype=float)
    if target_native_tid is not None:
        state.last_native_tid = int(target_native_tid)
    state.last_native_scope = str(epoch.target_session_scope)
    state.correction_epoch_id = str(epoch.epoch_id)
    state.target_session_scope = str(epoch.target_session_scope)
    state.human_anchor_sha256 = str(epoch.human_anchor_sha256)
    state.native_constraints_cleared = bool(epoch.native_constraints_cleared)
    state.motion_reanchored = bool(epoch.motion_reanchored)
    state.machine_prototype_frozen = bool(epoch.machine_prototype_frozen)


def apply_epoch_to_persistent_record(
    record: Any,
    epoch: CorrectionEpoch,
    human_anchor: Any,
    authoritative_box: Any,
) -> None:
    """Persist epoch provenance on a public identity record in place."""

    anchor = np.asarray(human_anchor, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(anchor))
    if anchor.size == 0 or not np.all(np.isfinite(anchor)) or norm <= 1.0e-6:
        raise ValueError("human anchor must be finite and non-zero")
    box = np.asarray(authoritative_box, dtype=float).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)):
        raise ValueError("authoritative_box must contain four finite values")
    clear = getattr(record, "appearance_state", None)
    if not isinstance(clear, dict):
        record.appearance_state = {}
    record.appearance_state["human_anchor_feature"] = (anchor / norm).astype(float).tolist()
    record.appearance_state["human_anchor_sha256"] = str(epoch.human_anchor_sha256)
    record.appearance_state["human_anchor_frame"] = int(epoch.start_frame)
    record.appearance_state["machine_prototype_frozen"] = bool(epoch.machine_prototype_frozen)
    record.last_box = box.astype(float).tolist()
    record.motion_state_ref = {
        "last_box": box.astype(float).tolist(),
        "last_frame": int(epoch.start_frame),
        "velocity_reset": True,
    }
    record.correction_epoch_id = str(epoch.epoch_id)
    record.human_anchor_feature = (anchor / norm).astype(float).tolist()
    record.human_anchor_sha256 = str(epoch.human_anchor_sha256)
    record.target_session_scope = str(epoch.target_session_scope)
    record.previous_native_tid = epoch.previous_native_tid
    record.previous_native_scope = epoch.previous_native_scope
    record.last_native_scope = str(epoch.target_session_scope)
    record.native_constraints_cleared = bool(epoch.native_constraints_cleared)
    record.motion_reanchored = bool(epoch.motion_reanchored)
    record.machine_prototype_frozen = bool(epoch.machine_prototype_frozen)
    record.correction_epoch_active = True
    record.event_history.append(
        {
            "frame_idx": int(epoch.start_frame),
            "operation": "BEGIN_CORRECTION_EPOCH",
            "candidate_uid": None,
            "public_id": int(record.public_id),
            "association_state_id": int(record.association_state_id),
            "status": str(record.status),
            "correction_epoch": epoch.as_dict(),
            "runtime_future_gt_used": False,
        }
    )


__all__ = [
    "CorrectionEpoch",
    "apply_epoch_to_identity_state",
    "apply_epoch_to_persistent_record",
    "feature_sha256",
    "make_correction_epoch",
]
