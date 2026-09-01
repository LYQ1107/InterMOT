"""N5 continuous human observer protocol.

The simulated observer watches every frame.  The system first produces
``Y_pre(t)`` from on-line state only.  After the prediction is frozen, the
current-frame GT is used to generate authoritative user commands, then
``Y_post(t)`` is frozen and (in stateful protocols) the state is updated so
future frames propagate from the corrected state.

No future GT, future detections, or future identity assignments are used at
runtime.  GT access is audited by :class:`GTFrameAccessor`.
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sam3_intermot.backend.output_types import PromptObjectObservation
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.identity.registry import ObjectIdentityRegistry
from sam3_intermot.identity.transaction import Transaction
from sam3_intermot.interaction.actions import (
    ActionType,
    HumanInteraction,
    InteractionResult,
    SystemContext,
    summarize_manager,
)
from sam3_intermot.interaction.add import perform_add
from sam3_intermot.interaction.correct import perform_correct
from sam3_intermot.interaction.delete import perform_delete
from sam3_intermot.interaction.simulator import GTFrame
from sam3_intermot.tracking.association import box_iou, center_distance
from sam3_intermot.tracking.track import TrackState
from sam3_intermot.tracking.track_manager import TrackManager


# ---------------------------------------------------------------------------
# GT access audit
# ---------------------------------------------------------------------------


@dataclass
class GTUsageAudit:
    gt_read_before_prediction: int = 0
    gt_read_current_after_prediction: int = 0
    gt_read_future: int = 0
    gt_used_for_user_observation: int = 0
    gt_used_for_command_generation: int = 0
    gt_used_for_model_decision: int = 0
    gt_used_for_scheduler: int = 0
    gt_used_for_offline_scoring: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "gt_read_before_prediction": self.gt_read_before_prediction,
            "gt_read_current_after_prediction": self.gt_read_current_after_prediction,
            "gt_read_future": self.gt_read_future,
            "gt_used_for_user_observation": self.gt_used_for_user_observation,
            "gt_used_for_command_generation": self.gt_used_for_command_generation,
            "gt_used_for_model_decision": self.gt_used_for_model_decision,
            "gt_used_for_scheduler": self.gt_used_for_scheduler,
            "gt_used_for_offline_scoring": self.gt_used_for_offline_scoring,
        }

    def ok(self) -> bool:
        return (
            self.gt_read_before_prediction == 0
            and self.gt_read_future == 0
            and self.gt_used_for_model_decision == 0
            and self.gt_used_for_scheduler == 0
        )


class GTFrameAccessor:
    """Frame-limited, prediction-gated GT accessor."""

    def __init__(self, frames: Dict[int, GTFrame]) -> None:
        self._frames = frames
        self._current: Optional[int] = None
        self._prediction_done = False
        self.audit = GTUsageAudit()

    def begin_prediction(self, frame_idx: int) -> None:
        self._current = frame_idx
        self._prediction_done = False

    def mark_prediction_done(self) -> None:
        self._prediction_done = True

    def observe(self, frame_idx: int) -> GTFrame:
        if frame_idx != self._current:
            if self._current is not None and frame_idx > self._current:
                self.audit.gt_read_future += 1
            else:
                self.audit.gt_read_before_prediction += 1
            raise RuntimeError(
                f"GT access blocked: current={self._current}, requested={frame_idx}, "
                f"prediction_done={self._prediction_done}"
            )
        if not self._prediction_done:
            self.audit.gt_read_before_prediction += 1
            raise RuntimeError("GT access attempted before current-frame prediction")
        self.audit.gt_read_current_after_prediction += 1
        self.audit.gt_used_for_user_observation += 1
        return self._frames.get(frame_idx, GTFrame())

    def used_for_commands(self) -> None:
        self.audit.gt_used_for_command_generation += 1


# ---------------------------------------------------------------------------
# Hungarian matcher (no scipy dependency)
# ---------------------------------------------------------------------------


def _hungarian_max(scores: np.ndarray) -> np.ndarray:
    """Maximum-weight assignment for a rectangular score matrix.

    Returns an array ``assign[gi] = pi`` or -1 if no valid assignment.
    Scores <= 0 are treated as invalid (IoU threshold is encoded by the caller).
    """
    n, m = scores.shape
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=int)
    size = max(n, m)
    cost = np.full((size + 1, size + 1), 0.0, dtype=float)
    cost[1 : n + 1, 1 : m + 1] = -np.maximum(scores, 0.0)
    u = np.zeros(size + 1, dtype=float)
    v = np.zeros(size + 1, dtype=float)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, 1e18, dtype=float)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 1e18
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = np.full(n, -1, dtype=int)
    for j in range(1, size + 1):
        if 0 < p[j] <= n and j <= m:
            assign[p[j] - 1] = j - 1
    for gi in range(n):
        pi = assign[gi]
        if pi < 0 or scores[gi, pi] <= 0:
            assign[gi] = -1
    return assign


def match_boxes(
    gt_boxes: List[np.ndarray],
    pre_boxes: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Hungarian matching between GT boxes and pre boxes."""
    n, m = len(gt_boxes), len(pre_boxes)
    if n == 0 or m == 0:
        return []
    scores = np.zeros((n, m), dtype=float)
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pre_boxes):
            iou = box_iou(gb, pb)
            scores[gi, pi] = iou if iou >= iou_threshold else 0.0
    assign = _hungarian_max(scores)
    return [
        (gi, int(assign[gi]), float(scores[gi, int(assign[gi])]))
        for gi in range(n)
        if assign[gi] >= 0
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class CommandType(str):
    ADD_NEW_IDENTITY = "ADD_NEW_IDENTITY"
    RECOVER_IDENTITY = "RECOVER_IDENTITY"
    AUTHORITATIVE_REASSIGN = "AUTHORITATIVE_REASSIGN"
    ATOMIC_ID_SWAP = "ATOMIC_ID_SWAP"
    AUTHORITATIVE_CORRECT = "AUTHORITATIVE_CORRECT"
    AUTHORITATIVE_DELETE = "AUTHORITATIVE_DELETE"


@dataclass
class ObserverCommand:
    frame_idx: int
    command_type: str
    error_type: str
    gt_id: Optional[int] = None
    user_identity_id: Optional[int] = None
    authoritative_box: Optional[np.ndarray] = None
    displayed_track_id: Optional[int] = None
    source_identity: Optional[int] = None
    destination_identity: Optional[int] = None
    is_first_appearance: bool = False
    is_recovery: bool = False
    target_track_id: Optional[int] = None
    other_track_id: Optional[int] = None
    iou: Optional[float] = None

    def as_dict(self) -> Dict:
        d = {
            "frame": self.frame_idx + 1,
            "action_type": self.command_type,
            "error_type": self.error_type,
            "authoritative_box": (
                list(np.round(self.authoritative_box, 2)) if self.authoritative_box is not None else None
            ),
            "displayed_track_id": self.displayed_track_id,
            "correct_user_identity_id": self.user_identity_id,
            "source_identity": self.source_identity,
            "destination_identity": self.destination_identity,
            "is_first_appearance": self.is_first_appearance,
            "is_recovery": self.is_recovery,
            "target_track_id": self.target_track_id,
            "other_track_id": self.other_track_id,
            "iou": None if self.iou is None else round(float(self.iou), 4),
        }
        return d


@dataclass
class N5Config:
    protocol: str = "p3"  # p1, p2, p3, p4
    budget: int = 0
    match_iou_threshold: float = 0.5
    localization_iou_threshold: float = 0.7
    window_size: int = 10
    refresh_horizon: int = 10
    session_restart_interval: int = 200
    detection_interval: int = 200
    correct_localization: bool = False
    correct_false_track: bool = False
    stateful: bool = True


@dataclass
class N5RunSummary:
    sequence: str = ""
    protocol: str = ""
    budget: int = 0
    num_frames: int = 0
    total_commands: int = 0
    accepted_commands: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    rejected: List[Dict] = field(default_factory=list)
    rolled_back: List[Dict] = field(default_factory=list)
    pre_rows: int = 0
    post_rows: int = 0
    invariant_violations: List[str] = field(default_factory=list)
    current_frame_authority_accuracy: float = 1.0
    identity_response_accuracy: float = 1.0
    response_latency_frames: int = 0


def read_mot_rows(path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """Read a MOT txt into {frame_idx: [(track_id, box_xyxy)]}."""
    rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame = int(float(parts[0])) - 1
        tid = int(float(parts[1]))
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        rows.setdefault(frame, []).append((tid, np.asarray([x, y, x + w, y + h], dtype=float)))
    return rows


class P1OfflineDriver:
    """Frame-only oracle corrections derived from P0 pre rows and GT."""

    def __init__(
        self,
        pre_rows: Dict[int, List[Tuple[int, np.ndarray]]],
        gt_frames: Dict[int, GTFrame],
        sequence: str = "",
        num_frames: int = 0,
        localization_iou_threshold: float = 0.7,
    ) -> None:
        self.pre_rows = pre_rows
        self.gt_frames = gt_frames
        self.sequence = sequence
        self.num_frames = num_frames or (max(pre_rows) + 1 if pre_rows else 0)
        self.localization_iou_threshold = localization_iou_threshold
        self.track_to_gt: Dict[int, int] = {}
        self.seen_gt: set = set()
        self.post_id_for_gt: Dict[int, int] = {}
        self.next_post_id = 100000
        self.events: List[Dict] = []
        self.commands_total = 0

    def run(self) -> Dict[int, List[Tuple[int, np.ndarray, float]]]:
        post: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
        for f in range(self.num_frames):
            pre = self.pre_rows.get(f, [])
            gt = self.gt_frames.get(f, GTFrame())
            post[f], commands = self._process_frame(f, pre, gt)
            self.commands_total += len(commands)
            for cmd in commands:
                self.events.append(
                    {
                        "sequence": self.sequence,
                        "protocol": "p1",
                        "budget": 0,
                        "timestamp": time.time(),
                        "interaction_start_timestamp": None,
                        "interaction_end_timestamp": None,
                        "user_session_id": "SIMULATED_ORACLE",
                        **cmd.as_dict(),
                    }
                )
        return post

    def _process_frame(
        self,
        frame_idx: int,
        pre: List[Tuple[int, np.ndarray]],
        gt: GTFrame,
    ) -> Tuple[List[Tuple[int, np.ndarray, float]], List[ObserverCommand]]:
        matches = match_boxes(
            [np.asarray(b, dtype=float) for b in gt.boxes],
            [np.asarray(b, dtype=float) for _, b in pre],
            0.5,
        )
        used = set()
        remap: Dict[int, int] = {}
        identity_errors: List[Tuple[int, int, str]] = []  # (pre_idx, gt_id, mode)
        recover: Dict[int, np.ndarray] = {}
        add_new: Dict[int, np.ndarray] = {}
        correct: List[Tuple[int, np.ndarray]] = []
        pre_to_gt: Dict[int, int] = {}
        active_pairs: Dict[int, int] = {}  # gt_id -> pre_idx for active rows

        for gi, pi, iou in matches:
            used.add(pi)
            pre_to_gt[pi] = gi
            gt_id = gt.gt_ids[gi]
            self.seen_gt.add(gt_id)
            tid = pre[pi][0]
            remap[tid] = gt_id
            cur = self.track_to_gt.get(tid)
            if cur is None:
                self.track_to_gt[tid] = gt_id
            elif cur != gt_id:
                active_pairs[gt_id] = pi
                identity_errors.append((pi, gt_id, "active"))
            elif iou < self.localization_iou_threshold:
                correct.append((tid, np.asarray(gt.boxes[gi], dtype=float)))

        for gi in range(len(gt.boxes)):
            if gi in pre_to_gt.values():
                continue
            gt_id = gt.gt_ids[gi]
            box = np.asarray(gt.boxes[gi], dtype=float)
            if gt_id in self.seen_gt:
                recover[gt_id] = box
            else:
                self.seen_gt.add(gt_id)
                add_new[gt_id] = box

        swap_pairs = []
        remaining = list(identity_errors)
        used_err = set()
        for i in range(len(remaining)):
            if i in used_err:
                continue
            pi_a, gid_a, mode_a = remaining[i]
            if mode_a != "active":
                continue
            tid_a = pre[pi_a][0]
            for j in range(i + 1, len(remaining)):
                if j in used_err:
                    continue
                pi_b, gid_b, mode_b = remaining[j]
                tid_b = pre[pi_b][0]
                if (
                    mode_b == "active"
                    and self.track_to_gt.get(tid_b) == gid_a
                    and self.track_to_gt.get(tid_a) == gid_b
                ):
                    swap_pairs.append((tid_a, tid_b))
                    used_err.update({i, j})
                    break
        identity_errors = [e for i, e in enumerate(remaining) if i not in used_err]
        for pi, gid, mode in identity_errors:
            tid = pre[pi][0]
            gt_idx = pre_to_gt.get(pi, 0)
            box = np.asarray(gt.boxes[gt_idx], dtype=float)
            if gid in self.seen_gt:
                recover.setdefault(gid, box)
            else:
                self.seen_gt.add(gid)
                add_new.setdefault(gid, box)

        commands: List[ObserverCommand] = []
        for tid_a, tid_b in swap_pairs:
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.ATOMIC_ID_SWAP,
                    error_type="ID_SWAP",
                    displayed_track_id=tid_a,
                    source_identity=self.track_to_gt.get(tid_a),
                    destination_identity=self.track_to_gt.get(tid_b),
                    target_track_id=tid_a,
                    other_track_id=tid_b,
                )
            )
        for pi, gid, mode in identity_errors:
            tid = pre[pi][0]
            gt_idx = pre_to_gt.get(pi, 0)
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.AUTHORITATIVE_REASSIGN,
                    error_type="ID_REASSIGN",
                    user_identity_id=gid,
                    displayed_track_id=tid,
                    source_identity=self.track_to_gt.get(tid),
                    destination_identity=gid,
                    target_track_id=tid,
                    authoritative_box=np.asarray(gt.boxes[gt_idx], dtype=float),
                )
            )
        for gid, box in sorted(recover.items()):
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.RECOVER_IDENTITY,
                    error_type="MISS_EXISTING",
                    user_identity_id=gid,
                    authoritative_box=box,
                    displayed_track_id=self._post_id(gid),
                    is_recovery=True,
                )
            )
        for gid, box in sorted(add_new.items()):
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.ADD_NEW_IDENTITY,
                    error_type="MISS_NEW",
                    user_identity_id=gid,
                    authoritative_box=box,
                    displayed_track_id=None,
                    is_first_appearance=True,
                )
            )
        for tid, box in correct:
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.AUTHORITATIVE_CORRECT,
                    error_type="LOCALIZATION_ERROR",
                    user_identity_id=self.track_to_gt.get(tid),
                    authoritative_box=box,
                    displayed_track_id=tid,
                    target_track_id=tid,
                )
            )
        for pi, (tid, _) in enumerate(pre):
            if pi in used:
                continue
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.AUTHORITATIVE_DELETE,
                    error_type="FALSE_TRACK",
                    displayed_track_id=tid,
                    target_track_id=tid,
                )
            )

        rows: Dict[int, Tuple[np.ndarray, float]] = {}
        for tid, box in pre:
            rows[tid] = (box, 1.0)
        for cmd in commands:
            if cmd.command_type == CommandType.AUTHORITATIVE_DELETE:
                rows.pop(cmd.target_track_id, None)
            elif cmd.command_type == CommandType.AUTHORITATIVE_CORRECT:
                if cmd.target_track_id in rows:
                    rows[cmd.target_track_id] = (cmd.authoritative_box, 1.0)
        for tid, gt_id in remap.items():
            if tid in rows:
                rows[self._post_id(gt_id)] = rows.pop(tid)
        for gid, box in add_new.items():
            if not self._box_covered(box, rows):
                rows[self._post_id(gid)] = (box, 1.0)
        for gid, box in recover.items():
            if not self._box_covered(box, rows):
                rows[self._post_id(gid)] = (box, 1.0)
        return [(tid, box, conf) for tid, (box, conf) in sorted(rows.items())], commands

    def _box_covered(self, box, rows: Dict[int, Tuple[np.ndarray, float]]) -> bool:
        for _, (b, _) in rows.items():
            if box_iou(box, b) > 0.95:
                return True
        return False

    def _post_id(self, gt_id: int) -> int:
        if gt_id not in self.post_id_for_gt:
            self.post_id_for_gt[gt_id] = self.next_post_id
            self.next_post_id += 1
        return self.post_id_for_gt[gt_id]


# ---------------------------------------------------------------------------
# Authoritative stateful handlers
# ---------------------------------------------------------------------------


def perform_recover_identity(
    ctx: SystemContext,
    frame_idx: int,
    mot_track_id: int,
    lineage_id: int,
    box_xyxy: np.ndarray,
) -> InteractionResult:
    """Recover an existing identity without creating new IDs."""
    before = summarize_manager(ctx.manager)
    before_next_track = ctx.manager._next_track_id
    before_next_lineage = ctx.lineages._next_lineage_id
    lineage = ctx.lineages.get(lineage_id)
    if lineage is None:
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Recover",
            frame_idx=frame_idx,
            accepted=False,
            reason="lineage not found",
            before_summary=before,
            after_summary=before,
        )
    old_track = ctx.manager.get(mot_track_id)
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        sam_id = ctx.allocate_sam_object_id()
        obs = ctx.backend.add_box(frame_idx, sam_id, box_xyxy)
        if old_track is not None and old_track.sam_object_id is not None:
            old_sam = old_track.sam_object_id
            if ctx.manager._sam_to_track.get(old_sam) == mot_track_id:
                ctx.manager._sam_to_track.pop(old_sam, None)
            old_track.sam_object_id = None
        if lineage.closed_frame is not None:
            lineage.closed_frame = None
        if mot_track_id not in lineage.mot_track_ids:
            lineage.mot_track_ids.append(mot_track_id)
        track = ctx.manager.create_track(
            frame_idx, obs, lineage_id, mot_track_id=mot_track_id
        )
        track.state = TrackState.RECOVERED
        if (
            ctx.manager._next_track_id != before_next_track
            or ctx.lineages._next_lineage_id != before_next_lineage
        ):
            txn.rollback()
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Recover",
                frame_idx=frame_idx,
                accepted=False,
                rolled_back=True,
                reason="recover created new mot/lineage id",
                before_summary=before,
                after_summary=summarize_manager(ctx.manager),
            )
        ctx.log_transaction(
            {
                "action_id": str(uuid.uuid4()),
                "action_type": "Recover",
                "frame_idx": frame_idx,
                "accepted": True,
                "mot_track_id": mot_track_id,
                "lineage_id": lineage_id,
                "sam_object_id": sam_id,
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Recover",
            frame_idx=frame_idx,
            accepted=True,
            new_track_id=mot_track_id,
            new_sam_object_id=sam_id,
            reason="recover_identity",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )
    except Exception as exc:
        txn.rollback()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Recover",
            frame_idx=frame_idx,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )


def perform_authoritative_add(
    ctx: SystemContext,
    frame_idx: int,
    box_xyxy: np.ndarray,
    lineage_id: Optional[int] = None,
    mot_track_id: Optional[int] = None,
) -> InteractionResult:
    """Authoritative add: create a new identity track without duplicate checks."""
    before = summarize_manager(ctx.manager)
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        lineage = ctx.lineages.get(lineage_id) if lineage_id is not None else None
        if lineage is None:
            lineage = ctx.lineages.create(frame_idx)
            lineage_id = lineage.lineage_id
        sam_id = ctx.allocate_sam_object_id()
        obs = ctx.backend.add_box(frame_idx, sam_id, box_xyxy)
        if obs.box_xyxy is None or not np.all(np.isfinite(obs.box_xyxy)):
            raise RuntimeError("backend returned invalid observation")
        if mot_track_id is not None:
            old = ctx.manager.get(mot_track_id)
            if old is not None and old.sam_object_id is not None:
                old_sam = old.sam_object_id
                if ctx.manager._sam_to_track.get(old_sam) == mot_track_id:
                    ctx.manager._sam_to_track.pop(old_sam, None)
                old.sam_object_id = None
            if mot_track_id not in lineage.mot_track_ids:
                lineage.mot_track_ids.append(mot_track_id)
        track = ctx.manager.create_track(
            frame_idx, obs, lineage_id, mot_track_id=mot_track_id
        )
        lineage.bind_track(track.mot_track_id)
        ctx.log_transaction(
            {
                "action_id": str(uuid.uuid4()),
                "action_type": "Add",
                "frame_idx": frame_idx,
                "accepted": True,
                "mot_track_id": track.mot_track_id,
                "sam_object_id": sam_id,
                "lineage_id": lineage_id,
                "add_resolution": "authoritative_new_identity",
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Add",
            frame_idx=frame_idx,
            accepted=True,
            new_track_id=track.mot_track_id,
            new_sam_object_id=sam_id,
            reason="authoritative_add",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )
    except Exception as exc:
        txn.rollback()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Add",
            frame_idx=frame_idx,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )


def perform_authoritative_reassign(
    ctx: SystemContext,
    frame_idx: int,
    src_track_id: int,
    dst_track_id: int,
) -> InteractionResult:
    """Rebind the source SAM object to the destination identity's track."""
    before = summarize_manager(ctx.manager)
    before_next_track = ctx.manager._next_track_id
    before_next_lineage = ctx.lineages._next_lineage_id
    src = ctx.manager.get(src_track_id)
    dst = ctx.manager.get(dst_track_id)
    if src is None or dst is None:
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Reassign",
            frame_idx=frame_idx,
            accepted=False,
            reason="source or destination track unknown",
            before_summary=before,
            after_summary=before,
        )
    if src.sam_object_id is None:
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Reassign",
            frame_idx=frame_idx,
            accepted=False,
            reason="source has no sam object",
            before_summary=before,
            after_summary=before,
        )
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        src_obs = None
        for obs in ctx.manager.outputs_for_frame(frame_idx):
            if obs.sam_object_id == src.sam_object_id:
                src_obs = obs
                break
        if src_obs is None:
            src_obs = ctx.backend.get_frame_outputs(frame_idx)
            src_obs = next(
                (o for o in src_obs if o.sam_object_id == src.sam_object_id), None
            )
        sam_id = src.sam_object_id
        ctx.manager.unbind_sam_object(src.mot_track_id)
        if dst.sam_object_id is not None:
            ctx.manager.unbind_sam_object(dst.mot_track_id)
        ctx.manager.rebind_sam_object(dst.mot_track_id, sam_id, frame_idx)
        ctx.manager.remove_output(frame_idx, src.mot_track_id)
        ctx.manager.remove_output(frame_idx, dst.mot_track_id)
        if src_obs is not None:
            obs = src_obs.copy()
            obs.sam_object_id = sam_id
            ctx.manager.update_track(
                dst.mot_track_id, frame_idx, obs, human_verified=True
            )
        if dst.state in (TrackState.LOST, TrackState.TENTATIVE):
            dst.state = TrackState.RECOVERED
        if (
            ctx.manager._next_track_id != before_next_track
            or ctx.lineages._next_lineage_id != before_next_lineage
        ):
            txn.rollback()
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Reassign",
                frame_idx=frame_idx,
                accepted=False,
                rolled_back=True,
                reason="reassign created new mot/lineage id",
                before_summary=before,
                after_summary=summarize_manager(ctx.manager),
            )
        ctx.log_transaction(
            {
                "action_id": str(uuid.uuid4()),
                "action_type": "Reassign",
                "frame_idx": frame_idx,
                "accepted": True,
                "source_track_id": src_track_id,
                "destination_track_id": dst_track_id,
                "sam_object_id": sam_id,
                "reassign_mode": "AUTHORITATIVE",
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Reassign",
            frame_idx=frame_idx,
            accepted=True,
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )
    except Exception as exc:
        txn.rollback()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Reassign",
            frame_idx=frame_idx,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )


def perform_atomic_swap(
    ctx: SystemContext,
    frame_idx: int,
    track_a_id: int,
    track_b_id: int,
) -> InteractionResult:
    """Atomically swap two tracks' SAM bindings and current-frame outputs."""
    before = summarize_manager(ctx.manager)
    a = ctx.manager.get(track_a_id)
    b = ctx.manager.get(track_b_id)
    if a is None or b is None:
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Swap",
            frame_idx=frame_idx,
            accepted=False,
            reason="unknown swap tracks",
            before_summary=before,
            after_summary=before,
        )
    if a.sam_object_id is None or b.sam_object_id is None:
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Swap",
            frame_idx=frame_idx,
            accepted=False,
            reason="swap requires both tracks to have sam objects",
            before_summary=before,
            after_summary=before,
        )
    txn = Transaction(ctx.manager, ctx.lineages)
    try:
        obs_a = next(
            (o for o in ctx.manager.outputs_for_frame(frame_idx) if o.sam_object_id == a.sam_object_id),
            None,
        )
        obs_b = next(
            (o for o in ctx.manager.outputs_for_frame(frame_idx) if o.sam_object_id == b.sam_object_id),
            None,
        )
        if obs_a is None or obs_b is None:
            raise RuntimeError("swap requires both current-frame observations")
        sam_a, sam_b = a.sam_object_id, b.sam_object_id
        ctx.manager.unbind_sam_object(a.mot_track_id)
        ctx.manager.unbind_sam_object(b.mot_track_id)
        ctx.manager.rebind_sam_object(a.mot_track_id, sam_b, frame_idx)
        ctx.manager.rebind_sam_object(b.mot_track_id, sam_a, frame_idx)
        obs_b2 = obs_b.copy()
        obs_b2.sam_object_id = sam_b
        obs_a2 = obs_a.copy()
        obs_a2.sam_object_id = sam_a
        ctx.manager.update_track(a.mot_track_id, frame_idx, obs_b2, human_verified=True)
        ctx.manager.update_track(b.mot_track_id, frame_idx, obs_a2, human_verified=True)
        ctx.log_transaction(
            {
                "action_id": str(uuid.uuid4()),
                "action_type": "Swap",
                "frame_idx": frame_idx,
                "accepted": True,
                "track_a": track_a_id,
                "track_b": track_b_id,
            }
        )
        txn.commit()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Swap",
            frame_idx=frame_idx,
            accepted=True,
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )
    except Exception as exc:
        txn.rollback()
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Swap",
            frame_idx=frame_idx,
            accepted=False,
            rolled_back=True,
            reason=f"error: {exc}",
            before_summary=before,
            after_summary=summarize_manager(ctx.manager),
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class ContinuousObserverDriver:
    def __init__(
        self,
        backend,
        manager: TrackManager,
        lineages: IdentityLineageRegistry,
        registry: ObjectIdentityRegistry,
        config: N5Config,
        num_frames: int,
        gt_frames: Dict[int, GTFrame],
        sequence: str = "",
        video_source: Optional[str] = None,
        backbone: Optional[Dict[int, List[PromptObjectObservation]]] = None,
    ) -> None:
        self.backend = backend
        self.manager = manager
        self.lineages = lineages
        self.registry = registry
        self.config = config
        self.num_frames = num_frames
        self.gt_frames = gt_frames
        self.sequence = sequence
        self.video_source = video_source
        self.backbone = backbone
        self.ctx = SystemContext(backend=backend, manager=manager, lineages=lineages)
        self.ctx.config.enable_lineage_aware_add = False
        self.ctx.config.enable_soft_delete = False
        self.ctx.config.enable_atomic_reassign = True
        self.ctx.config.enable_abstention = False
        self.ctx.config.enable_guard = False
        self.ctx._next_sam_object_id = 1000
        self.gt_access = GTFrameAccessor(gt_frames)
        self.propagated: Dict[int, List[PromptObjectObservation]] = {}
        self.user_lineage: Dict[int, int] = {}
        self.user_track: Dict[int, Optional[int]] = {}
        self.track_user: Dict[int, Optional[int]] = {}
        self.user_seen_frame: Dict[int, int] = {}
        self._user_gt_ids: Dict[int, int] = {}
        self.next_user_identity = 1
        self.p1_post_ids: Dict[int, int] = {}
        self.next_p1_post_id = 100000
        self._p1_remap: Dict[int, Dict[int, int]] = {}
        self.events: List[Dict] = []
        self.results: List[InteractionResult] = []
        self.pre_rows: Dict[int, List[Tuple[int, PromptObjectObservation]]] = {}
        self.post_rows: Dict[int, List[Tuple[int, PromptObjectObservation]]] = {}
        self._used_budget = 0
        self._frame_state_changed = False
        self._pending_delete_sams: set = set()
        self._suppressed_sams: set = set()
        self._suppressed_tracks: set = set()
        self._pending_backend_ops: List[Tuple[str, int, np.ndarray]] = []
        self._pending_add_sams: set = set()
        self._window_corrected_users: set = set()
        self.user_mot_id: Dict[int, int] = {}
        self._pre_track_user: Dict[int, int] = {}

    # -- user identity -----------------------------------------------------
    def _ensure_user(self, gt_id: int, frame_idx: int) -> int:
        for uid, gid in self._user_gt_ids.items():
            if gid == gt_id:
                self.user_seen_frame.setdefault(uid, frame_idx)
                return uid
        uid = self.next_user_identity
        self.next_user_identity += 1
        self._user_gt_ids[uid] = gt_id
        self.user_seen_frame[uid] = frame_idx
        self.user_track[uid] = None
        return uid

    def run(self) -> N5RunSummary:
        if self.backbone is not None:
            return self._run_backbone()
        for window_start in range(0, self.num_frames, self.config.window_size):
            window_end = min(
                window_start + self.config.window_size - 1, self.num_frames - 1
            )
            self._prepare_window(window_start, window_end)
            self._window_corrected_users = set()
            for f in range(window_start, window_end + 1):
                self._frame_state_changed = False
                self.gt_access.begin_prediction(f)
                self._automatic_step(f)
                self.gt_access.mark_prediction_done()
                pre = self.manager.outputs_for_frame(f)
                self.pre_rows[f] = [
                    (tid, obs.copy())
                    for tid, obs in sorted(self.manager._outputs.get(f, {}).items())
                ]
                gt = self.gt_access.observe(f)
                commands = self._generate_commands(f, pre, gt)
                if self.config.stateful:
                    self._execute_commands(f, commands)
                post = self._build_post(f, pre, commands)
                self.post_rows[f] = [
                    (tid, o.copy()) for tid, o in post
                ]
                for cmd in commands:
                    self.events.append(
                        {
                            "sequence": self.sequence,
                            "protocol": self.config.protocol,
                            "budget": self.config.budget,
                            "timestamp": time.time(),
                            "interaction_start_timestamp": None,
                            "interaction_end_timestamp": None,
                            "user_session_id": "SIMULATED_ORACLE",
                            **cmd.as_dict(),
                        }
                    )
        return self._summary()

    def _run_backbone(self) -> N5RunSummary:
        """Run the observer on the frozen P0 backbone (no SAM re-propagation).

        Identity/geometry state updates are applied through the manager and
        take effect immediately in the output streams.  Low-level SAM
        propagation is frozen at the P0 backbone level; this is the documented
        engineering bound of the stateful protocols.
        """
        for f in range(self.num_frames):
            self._frame_state_changed = False
            self.gt_access.begin_prediction(f)
            self._automatic_step(f)
            self.gt_access.mark_prediction_done()
            raw = self.manager.outputs_for_frame(f)
            self.pre_rows[f] = self._remap_rows(
                [
                    (tid, obs.copy())
                    for tid, obs in sorted(self.manager._outputs.get(f, {}).items())
                ],
                self._pre_track_user,
            )
            pre = raw
            gt = self.gt_access.observe(f)
            commands = self._generate_commands(f, pre, gt)
            if self.config.stateful:
                self._execute_backbone_commands(f, commands)
            self.post_rows[f] = self._build_backbone_post(f, pre, commands)
            self._pre_track_user = dict(self.track_user)
            for cmd in commands:
                self.events.append(
                    {
                        "sequence": self.sequence,
                        "protocol": self.config.protocol,
                        "budget": self.config.budget,
                        "timestamp": time.time(),
                        "interaction_start_timestamp": None,
                        "interaction_end_timestamp": None,
                        "user_session_id": "SIMULATED_ORACLE",
                        **cmd.as_dict(),
                    }
                )
        return self._summary()

    def _remap_rows(
        self,
        rows: List[Tuple[int, PromptObjectObservation]],
        track_user: Dict[int, int],
    ) -> List[Tuple[int, PromptObjectObservation]]:
        out: Dict[int, PromptObjectObservation] = {}
        for tid, obs in rows:
            uid = track_user.get(tid)
            out_id = self.user_mot_id.get(uid, tid) if uid is not None else tid
            if out_id in out:
                continue
            out[out_id] = obs.copy()
        return [(tid, obs) for tid, obs in sorted(out.items())]

    def _execute_backbone_commands(
        self, frame_idx: int, commands: List[ObserverCommand]
    ) -> None:
        for cmd in commands:
            result = self._execute_backbone_one(frame_idx, cmd)
            self.results.append(result)
            if result.accepted:
                self._used_budget += 1
                self._frame_state_changed = True

    def _execute_backbone_one(
        self, frame_idx: int, cmd: ObserverCommand
    ) -> InteractionResult:
        before = summarize_manager(self.manager)
        uid = cmd.user_identity_id
        if cmd.command_type == CommandType.ADD_NEW_IDENTITY:
            if uid not in self.user_mot_id:
                if cmd.target_track_id is not None:
                    self.user_mot_id[uid] = cmd.target_track_id
                else:
                    self.user_mot_id[uid] = self.next_p1_post_id
                    self.next_p1_post_id += 1
            if cmd.target_track_id is not None:
                old_user = self.track_user.get(cmd.target_track_id)
                self.track_user[cmd.target_track_id] = uid
                if old_user is not None and self.user_track.get(old_user) == cmd.target_track_id:
                    self.user_track[old_user] = None
            self.user_track[uid] = cmd.target_track_id
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Add",
                frame_idx=frame_idx,
                accepted=True,
                new_track_id=self.user_mot_id[uid],
                reason="backbone_add",
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        if cmd.command_type == CommandType.RECOVER_IDENTITY:
            if uid not in self.user_mot_id:
                self.user_mot_id[uid] = self.next_p1_post_id
                self.next_p1_post_id += 1
            self.user_track[uid] = None
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Recover",
                frame_idx=frame_idx,
                accepted=True,
                new_track_id=self.user_mot_id[uid],
                reason="backbone_recover",
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        if cmd.command_type == CommandType.AUTHORITATIVE_REASSIGN:
            src_tid = cmd.target_track_id
            dest_uid = cmd.destination_identity
            if dest_uid not in self.user_mot_id:
                self.user_mot_id[dest_uid] = self.next_p1_post_id
                self.next_p1_post_id += 1
            old_user = self.track_user.get(src_tid)
            self.track_user[src_tid] = dest_uid
            self.user_track[dest_uid] = src_tid
            if old_user is not None and self.user_track.get(old_user) == src_tid:
                self.user_track[old_user] = None
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Reassign",
                frame_idx=frame_idx,
                accepted=True,
                reason="backbone_reassign",
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        if cmd.command_type == CommandType.ATOMIC_ID_SWAP:
            a, b = cmd.target_track_id, cmd.other_track_id
            ua, ub = self.track_user.get(a), self.track_user.get(b)
            self.track_user[a] = ub
            self.track_user[b] = ua
            if ua is not None:
                self.user_track[ua] = b
            if ub is not None:
                self.user_track[ub] = a
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Swap",
                frame_idx=frame_idx,
                accepted=True,
                reason="backbone_swap",
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        if cmd.command_type == CommandType.AUTHORITATIVE_CORRECT:
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Correct",
                frame_idx=frame_idx,
                accepted=True,
                reason="backbone_correct",
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        if cmd.command_type == CommandType.AUTHORITATIVE_DELETE:
            tid = cmd.target_track_id
            self.track_user[tid] = None
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Delete",
                frame_idx=frame_idx,
                accepted=True,
                reason="backbone_delete",
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Unknown",
            frame_idx=frame_idx,
            accepted=False,
            reason="unknown command",
            before_summary={},
            after_summary={},
        )

    def _build_backbone_post(
        self,
        frame_idx: int,
        pre: List[PromptObjectObservation],
        commands: List[ObserverCommand],
    ) -> List[Tuple[int, PromptObjectObservation]]:
        """Author current-frame post from the actually executed commands."""
        rows: Dict[int, PromptObjectObservation] = {}
        for obs in pre:
            track = self.registry.lookup_by_sam_object_id(obs.sam_object_id)
            tid = track.mot_track_id if track is not None else obs.sam_object_id
            if tid in self._suppressed_tracks:
                continue
            rows[tid] = obs.copy()
        for cmd in commands:
            if cmd.command_type in (
                CommandType.ADD_NEW_IDENTITY,
                CommandType.RECOVER_IDENTITY,
            ):
                mid = self.user_mot_id.get(cmd.user_identity_id)
                if mid is not None:
                    rows[mid] = self._obs_from_box(
                        frame_idx, cmd.authoritative_box, mid
                    )
            elif cmd.command_type == CommandType.AUTHORITATIVE_REASSIGN:
                src_tid = cmd.target_track_id
                if src_tid in rows:
                    mid = self.user_mot_id.get(cmd.destination_identity)
                    if mid is not None:
                        obs = rows.pop(src_tid)
                        obs.sam_object_id = mid
                        rows[mid] = obs
            elif cmd.command_type == CommandType.ATOMIC_ID_SWAP:
                a, b = cmd.target_track_id, cmd.other_track_id
                if a in rows and b in rows:
                    rows[a], rows[b] = rows[b], rows[a]
            elif cmd.command_type == CommandType.AUTHORITATIVE_CORRECT:
                tid = cmd.target_track_id
                if tid in rows:
                    rows[tid].box_xyxy = np.asarray(
                        cmd.authoritative_box, dtype=float
                    ).copy()
            elif cmd.command_type == CommandType.AUTHORITATIVE_DELETE:
                rows.pop(cmd.target_track_id, None)
        return [(tid, obs) for tid, obs in sorted(rows.items())]

    # -- windows -----------------------------------------------------------
    def _prepare_window(self, start: int, end: int) -> None:
        self._flush_pending_deletes()
        if start > 0 and start % self.config.session_restart_interval == 0:
            # Long single-session runs with many add/remove cycles can desync
            # the official multiplex tracker's object table. Restart the SAM
            # session at restart boundaries and re-anchor active tracks from
            # manager state (same MOT/lineage identity, fresh SAM object ids).
            self._restart_session(start)
        else:
            self.registry.unbind_all_for_window()
        self._apply_pending_backend_ops(start)
        if start % self.config.detection_interval == 0:
            dets = self.backend.detect_concept(start, "person")
        else:
            dets = self._re_prompt_boxes(start)
        self.propagated = {start: list(dets)}
        prop = self.backend.propagate(
            start, end, start_frame_index=start, keep_masks=False
        )
        for f, obs_list in prop.items():
            self.propagated[f] = list(obs_list)
        self.backend._output_cache.clear()

    def _re_prompt_boxes(
        self, frame_idx: int
    ) -> List[PromptObjectObservation]:
        """Re-prompt only the existing object boxes (no open-vocab detection).

        Open-vocabulary detection at every 10-frame window boundary creates
        false-positive tracks that fragment the pre-interaction stream.  New
        people are instead discovered at detection windows (every 200 frames)
        or added by the simulated user (MISS_NEW -> ADD_NEW_IDENTITY).
        """
        boxes = [o["box"] for o in self.backend._objects.values()]
        if not boxes:
            return []
        obs = self.backend._send_prompt(
            frame_idx,
            boxes=boxes,
            source="automatic_propagation",
        )
        self.backend._apply_stable_ids(obs)
        return obs

    def _restart_session(self, frame_idx: int) -> None:
        self.backend.close()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        if self.video_source is None:
            raise RuntimeError("video_source is required for session restart")
        self.backend.start_video(self.video_source)
        self._pending_delete_sams.clear()
        self._suppressed_sams.clear()
        # Pending correct ops are captured by track.last_box, which restart
        # re-anchors; pending adds are applied after restart.
        self._pending_backend_ops = [
            op for op in self._pending_backend_ops if op[0] == "add"
        ]
        self.propagated = {}
        for track in self.manager.active_tracks():
            if track.last_box is None:
                continue
            if track.state in (TrackState.TERMINATED, TrackState.DELETED):
                continue
            if (
                track.sam_object_id is not None
                and track.sam_object_id in self._pending_add_sams
            ):
                # This object will be added by _apply_pending_backend_ops.
                continue
            sam_id = self.ctx.allocate_sam_object_id()
            obs = self.backend.add_box(frame_idx, sam_id, track.last_box)
            if track.sam_object_id is not None:
                self.manager.unbind_sam_object(track.mot_track_id)
            self.manager.rebind_sam_object(track.mot_track_id, sam_id, frame_idx)

    def _apply_pending_backend_ops(self, frame_idx: int) -> None:
        for op_type, sam_id, box in self._pending_backend_ops:
            if op_type == "add":
                if sam_id not in self.backend._objects:
                    self.backend.add_box(frame_idx, sam_id, box)
            elif op_type == "correct":
                if sam_id in self.backend._objects:
                    self.backend.correct_object(
                        frame_idx, sam_id, box_xyxy=box
                    )
                else:
                    self.backend.add_box(frame_idx, sam_id, box)
        self._pending_backend_ops.clear()
        self._pending_add_sams.clear()

    def _flush_pending_deletes(self) -> None:
        """Remove deferred backend objects at a window boundary.

        Removing an object mid-window re-prompts the multiplex session with a
        changed box set, which corrupts the official tracker's object table
        after many cycles.  Deletions are therefore applied to the manager
        immediately, but the backend object is dropped only here, right before
        the window's fresh detect+propagate.
        """
        for sam_id in self._pending_delete_sams:
            if sam_id not in self.backend._objects:
                continue
            del self.backend._objects[sam_id]
            raw = self.backend._ext_to_sam.pop(sam_id, None)
            if raw is not None:
                self.backend._sam_to_ext.pop(raw, None)
            for frame_obs in self.backend._output_cache.values():
                frame_obs[:] = [
                    o for o in frame_obs if o.sam_object_id != sam_id
                ]
        self._pending_delete_sams.clear()
        self._suppressed_sams.clear()

    def _suppress_track(self, track_id: int, frame_idx: int) -> None:
        """Stop a track from producing output; backend removal is deferred."""
        track = self.manager.get(track_id)
        if track is None:
            return
        sam_id = track.sam_object_id
        if sam_id is not None:
            self._suppressed_sams.add(sam_id)
            self._pending_delete_sams.add(sam_id)
            self.manager._sam_to_track.pop(sam_id, None)
            self.manager._tombstones[sam_id] = frame_idx
            track.sam_object_id = None
        self.manager.delete_track(track_id, frame_idx, reason="authoritative_delete")
        self.manager.remove_output(frame_idx, track_id)

    def _refresh_propagation(self, frame_idx: int, window_end: int) -> None:
        prop = self.backend.propagate(
            frame_idx,
            window_end,
            start_frame_index=frame_idx,
            keep_masks=False,
        )
        for f in range(frame_idx + 1, window_end + 1):
            if f in prop:
                self.propagated[f] = list(prop[f])

    # -- automatic step ----------------------------------------------------
    def _automatic_step(self, frame_idx: int) -> None:
        if self.backbone is not None:
            self._automatic_step_backbone(frame_idx)
            return
        observations = self.propagated.get(frame_idx, [])
        observations = self._nms_observations(observations)
        handled_sam: set = set()
        if frame_idx % self.config.window_size == 0:
            observations, handled_sam = self._window_handover(
                frame_idx, observations
            )
        for obs in observations:
            if obs.sam_object_id not in self.backend._objects:
                self.backend._objects[obs.sam_object_id] = {
                    "box": self.backend._sanitize_box(obs.box_xyxy).copy(),
                    "human_box": obs.box_xyxy.copy(),
                    "frame": frame_idx,
                    "source": "concept_detection",
                }
        seen = set()
        for obs in observations:
            if obs.sam_object_id in seen:
                continue
            seen.add(obs.sam_object_id)
            if obs.sam_object_id in self._suppressed_sams:
                continue
            if (
                obs.sam_object_id in self.manager._tombstones
                and frame_idx - self.manager._tombstones[obs.sam_object_id]
                < self.manager.lifecycle.tombstone_cooldown_frames
            ):
                continue
            self.registry.register_auto_object(frame_idx, obs)
        matched = {o.sam_object_id for o in observations} | handled_sam
        for track in self.manager.active_tracks():
            if track.sam_object_id is not None and track.sam_object_id not in matched:
                self.manager.mark_missed(track.mot_track_id, frame_idx)

    def _automatic_step_backbone(self, frame_idx: int) -> None:
        observations = self.backbone.get(frame_idx, [])
        observations = self._nms_observations(observations)
        for obs in observations:
            tid = obs.sam_object_id
            if tid in self._suppressed_sams:
                continue
            track = self.manager.get(tid)
            if track is not None and track.state in (
                TrackState.TERMINATED,
                TrackState.DELETED,
            ):
                continue
            if track is None:
                lineage = self.lineages.get(tid)
                if lineage is None:
                    self.lineages._next_lineage_id = tid
                    lineage = self.lineages.create(frame_idx)
                track = self.manager.create_track(
                    frame_idx, obs, lineage.lineage_id, mot_track_id=tid
                )
                lineage.bind_track(tid)
            else:
                self.manager.update_track(tid, frame_idx, obs)
        seen = {o.sam_object_id for o in observations}
        for track in self.manager.active_tracks():
            if track.mot_track_id in seen:
                continue
            if (
                track.sam_object_id is not None
                and track.sam_object_id not in seen
            ):
                self.manager.mark_missed(track.mot_track_id, frame_idx)

    def _nms_observations(
        self, observations: List[PromptObjectObservation]
    ) -> List[PromptObjectObservation]:
        """Frame-level NMS over detection boxes (no GT used).

        Open-vocabulary detection at every window boundary can return multiple
        overlapping boxes for one person.  Keep verified objects first, then
        suppress lower-confidence boxes with IoU >= 0.5 against a kept box.
        """
        ordered = sorted(
            observations,
            key=lambda o: (not o.is_human_verified, -o.confidence),
        )
        kept: List[PromptObjectObservation] = []
        for obs in ordered:
            duplicate = False
            for k in kept:
                if box_iou(obs.box_xyxy, k.box_xyxy) >= 0.5:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(obs)
        return kept

    def _window_handover(
        self,
        frame_idx: int,
        observations: List[PromptObjectObservation],
    ):
        """Re-bind window-start detections to existing tracks by geometry.

        Open-vocabulary re-detection at every window boundary can produce
        boxes that overlap an existing person with IoU below the registry's
        0.2 merge threshold, creating duplicate track ids.  A looser
        center-distance + low-IoU match prevents that fragmentation without
        using GT or future information.
        """
        handled_sam: set = set()
        handled_tracks: set = set()
        for obs in observations:
            if obs.sam_object_id in self._suppressed_sams:
                continue
            if self.registry.lookup_by_sam_object_id(obs.sam_object_id) is not None:
                # Already bound to a track this window; normal registration
                # will update that track.
                continue
            best = None
            best_score = float("-inf")
            for track in self.manager.active_tracks():
                if track.mot_track_id in handled_tracks:
                    continue
                if track.last_box is None:
                    continue
                if track.sam_object_id == obs.sam_object_id:
                    continue
                iou = box_iou(obs.box_xyxy, track.last_box)
                dist = center_distance(obs.box_xyxy, track.last_box)
                if iou < 0.3 or dist > 180.0:
                    continue
                score = iou - 1e-4 * dist
                if score > best_score:
                    best_score = score
                    best = track
            if best is not None:
                self.registry.rebind(
                    best.mot_track_id, obs.sam_object_id, frame_idx
                )
                self.manager.update_track(
                    best.mot_track_id, frame_idx, obs
                )
                handled_tracks.add(best.mot_track_id)
                handled_sam.add(obs.sam_object_id)
        return (
            [o for o in observations if o.sam_object_id not in handled_sam],
            handled_sam,
        )

    # -- command generation ------------------------------------------------
    def _generate_commands(
        self,
        frame_idx: int,
        pre: List[PromptObjectObservation],
        gt: GTFrame,
    ) -> List[ObserverCommand]:
        self.gt_access.used_for_commands()
        self._p1_remap[frame_idx] = {}
        pre_items = []
        for obs in pre:
            track = self.registry.lookup_by_sam_object_id(obs.sam_object_id)
            pre_items.append((obs, track))
        matches = match_boxes(
            [np.asarray(b, dtype=float) for b in gt.boxes],
            [o.box_xyxy for o, _ in pre_items],
            self.config.match_iou_threshold,
        )
        used_pre = set()
        identity_errors: List[Tuple[int, int, int, str, int]] = []  # (pi, tid, uid, mode, gi)
        recover_boxes: Dict[int, np.ndarray] = {}
        miss_new_boxes: Dict[int, np.ndarray] = {}
        correct_boxes: List[Tuple[int, np.ndarray]] = []
        matched_user: Dict[int, int] = {}
        pre_to_gt: Dict[int, int] = {}

        for gi, pi, iou in matches:
            used_pre.add(pi)
            pre_to_gt[pi] = gi
            obs, track = pre_items[pi]
            if track is None:
                continue
            uid = self._ensure_user(gt.gt_ids[gi], frame_idx)
            matched_user[gi] = uid
            self._p1_remap[frame_idx][track.mot_track_id] = uid
            cur = self.track_user.get(track.mot_track_id)
            if cur is None:
                if self.user_track.get(uid) is not None:
                    # Identity already has an active track; this is a duplicate
                    # box, not a first sighting. Leave as its own auto track.
                    continue
                if uid not in self.user_mot_id:
                    # First confirmation of an existing automatic track: the
                    # track's MOT id becomes the user identity's stable id.
                    self.user_mot_id[uid] = track.mot_track_id
                self.track_user[track.mot_track_id] = uid
                self.user_track[uid] = track.mot_track_id
                self.user_lineage[uid] = track.identity_lineage_id
                cur = uid
            if cur != uid:
                dst_tid = self.user_track.get(uid)
                dst_active = (
                    dst_tid is not None
                    and self.manager.get(dst_tid) is not None
                    and self.manager.get(dst_tid).state
                    in (
                        TrackState.TENTATIVE,
                        TrackState.CONFIRMED,
                        TrackState.LOST,
                        TrackState.QUARANTINED,
                        TrackState.RECOVERED,
                    )
                )
                if dst_active:
                    identity_errors.append((pi, track.mot_track_id, uid, "active", gi))
                elif (
                    dst_tid is not None
                    or self.user_seen_frame.get(uid, frame_idx) < frame_idx
                ):
                    identity_errors.append((pi, track.mot_track_id, uid, "historical", gi))
                else:
                    identity_errors.append((pi, track.mot_track_id, uid, "new", gi))
            elif iou < self.config.localization_iou_threshold:
                correct_boxes.append((track.mot_track_id, np.asarray(gt.boxes[gi], dtype=float)))

        for gi in range(len(gt.boxes)):
            if gi in matched_user:
                continue
            uid = self._ensure_user(gt.gt_ids[gi], frame_idx)
            gt_box = np.asarray(gt.boxes[gi], dtype=float)
            best_pi = None
            best_score = 0.3
            for pi in range(len(pre_items)):
                if pi in used_pre:
                    continue
                obs, track = pre_items[pi]
                if track is None:
                    continue
                iou = box_iou(gt_box, obs.box_xyxy)
                dist = center_distance(gt_box, obs.box_xyxy)
                if iou < 0.3 or dist > 180.0:
                    continue
                score = iou - 1e-4 * dist
                if score > best_score:
                    best_score = score
                    best_pi = pi
            if best_pi is not None:
                # The user assigns the existing unmatched box to this identity
                # instead of creating a duplicate track.
                used_pre.add(best_pi)
                tid = pre_items[best_pi][1].mot_track_id
                first_seen = self.user_seen_frame.get(uid)
                if (
                    self.user_track.get(uid) is not None
                    or (first_seen is not None and first_seen < frame_idx)
                ):
                    identity_errors.append(
                        (best_pi, tid, uid, "historical", gi)
                    )
                else:
                    identity_errors.append((best_pi, tid, uid, "new", gi))
                continue
            first_seen = self.user_seen_frame.get(uid)
            if (
                uid in recover_boxes
                or self.user_track.get(uid) is not None
                or (first_seen is not None and first_seen < frame_idx)
            ):
                recover_boxes.setdefault(uid, gt_box)
            else:
                miss_new_boxes.setdefault(uid, gt_box)

        # 2-cycles become atomic swaps.
        swap_pairs = []
        remaining = list(identity_errors)
        used_errors = set()
        for i in range(len(remaining)):
            if i in used_errors:
                continue
            pi_a, tid_a, uid_a, mode_a, gi_a = remaining[i]
            if mode_a != "active":
                continue
            for j in range(i + 1, len(remaining)):
                if j in used_errors:
                    continue
                pi_b, tid_b, uid_b, mode_b, gi_b = remaining[j]
                if (
                    mode_b == "active"
                    and
                    self.track_user.get(tid_b) == uid_a
                    and self.track_user.get(tid_a) == uid_b
                ):
                    swap_pairs.append((tid_a, tid_b))
                    used_errors.update({i, j})
                    break
        identity_errors = [
            e for idx, e in enumerate(remaining) if idx not in used_errors
        ]

        commands: List[ObserverCommand] = []
        for tid_a, tid_b in swap_pairs:
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.ATOMIC_ID_SWAP,
                    error_type="ID_SWAP",
                    user_identity_id=None,
                    displayed_track_id=tid_a,
                    source_identity=self.track_user.get(tid_a),
                    destination_identity=self.track_user.get(tid_b),
                    target_track_id=tid_a,
                    other_track_id=tid_b,
                )
            )
        for pi, tid, uid, mode, gi in identity_errors:
            gt_idx = gi
            if mode == "new":
                commands.append(
                    ObserverCommand(
                        frame_idx=frame_idx,
                        command_type=CommandType.ADD_NEW_IDENTITY,
                        error_type="MISS_NEW",
                        user_identity_id=uid,
                        displayed_track_id=None,
                        source_identity=self.track_user.get(tid),
                        destination_identity=uid,
                        target_track_id=tid,
                        authoritative_box=np.asarray(gt.boxes[gt_idx], dtype=float),
                        is_first_appearance=True,
                    )
                )
            else:
                commands.append(
                    ObserverCommand(
                        frame_idx=frame_idx,
                        command_type=CommandType.AUTHORITATIVE_REASSIGN,
                        error_type="ID_REASSIGN",
                        user_identity_id=uid,
                        displayed_track_id=tid,
                        source_identity=self.track_user.get(tid),
                        destination_identity=uid,
                        target_track_id=tid,
                        authoritative_box=np.asarray(gt.boxes[gt_idx], dtype=float),
                    )
                )
        for uid, box in sorted(recover_boxes.items()):
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.RECOVER_IDENTITY,
                    error_type="MISS_EXISTING",
                    user_identity_id=uid,
                    authoritative_box=box,
                    displayed_track_id=self.user_track.get(uid),
                    is_recovery=True,
                )
            )
        for uid, box in sorted(miss_new_boxes.items()):
            commands.append(
                ObserverCommand(
                    frame_idx=frame_idx,
                    command_type=CommandType.ADD_NEW_IDENTITY,
                    error_type="MISS_NEW",
                    user_identity_id=uid,
                    authoritative_box=box,
                    displayed_track_id=None,
                    is_first_appearance=True,
                )
            )
        if self.config.correct_localization:
            for tid, box in correct_boxes:
                commands.append(
                    ObserverCommand(
                        frame_idx=frame_idx,
                        command_type=CommandType.AUTHORITATIVE_CORRECT,
                        error_type="LOCALIZATION_ERROR",
                        user_identity_id=self.track_user.get(tid),
                        authoritative_box=box,
                        displayed_track_id=tid,
                        target_track_id=tid,
                    )
                )
        if self.config.correct_false_track:
            for pi, (obs, track) in enumerate(pre_items):
                if pi in used_pre:
                    continue
                if track is None:
                    continue
                commands.append(
                    ObserverCommand(
                        frame_idx=frame_idx,
                        command_type=CommandType.AUTHORITATIVE_DELETE,
                        error_type="FALSE_TRACK",
                        displayed_track_id=track.mot_track_id,
                        target_track_id=track.mot_track_id,
                    )
                )

        commands = self._apply_protocol_filters(commands)
        return commands

    def _apply_protocol_filters(
        self, commands: List[ObserverCommand]
    ) -> List[ObserverCommand]:
        if self.config.protocol in ("p1", "p2"):
            return commands
        allowed = {
            CommandType.ADD_NEW_IDENTITY,
            CommandType.RECOVER_IDENTITY,
            CommandType.AUTHORITATIVE_REASSIGN,
            CommandType.ATOMIC_ID_SWAP,
        }
        commands = [c for c in commands if c.command_type in allowed]
        if self.config.protocol == "p4":
            priority = {
                CommandType.ATOMIC_ID_SWAP: 0,
                CommandType.AUTHORITATIVE_REASSIGN: 1,
                CommandType.RECOVER_IDENTITY: 2,
                CommandType.ADD_NEW_IDENTITY: 3,
            }
            commands.sort(
                key=lambda c: (priority[c.command_type], c.user_identity_id or 0)
            )
            remaining = max(0, self.config.budget - self._used_budget)
            commands = commands[:remaining]
        return commands

    # -- execution ---------------------------------------------------------
    def _execute_commands(
        self, frame_idx: int, commands: List[ObserverCommand]
    ) -> None:
        if not commands:
            return
        frame_txn = Transaction(self.manager, self.lineages)
        accepted_any = False
        frame_ops_start = len(self._pending_backend_ops)
        try:
            for cmd in commands:
                before_ops = len(self._pending_backend_ops)
                result = self._execute_one(frame_idx, cmd)
                self.results.append(result)
                if result.accepted:
                    accepted_any = True
                    self._used_budget += 1
                    self._frame_state_changed = True
                else:
                    del self._pending_backend_ops[before_ops:]
            if accepted_any:
                frame_txn.commit()
            else:
                frame_txn.rollback()
                del self._pending_backend_ops[frame_ops_start:]
        except Exception:
            frame_txn.rollback()
            del self._pending_backend_ops[frame_ops_start:]
            raise

    def _execute_one(
        self, frame_idx: int, cmd: ObserverCommand
    ) -> InteractionResult:
        if cmd.command_type == CommandType.ADD_NEW_IDENTITY:
            before = summarize_manager(self.manager)
            try:
                lineage = self.lineages.create(frame_idx)
                sam_id = self.ctx.allocate_sam_object_id()
                obs = self._obs_from_box(
                    frame_idx, cmd.authoritative_box, sam_id
                )
                track = self.manager.create_track(
                    frame_idx, obs, lineage.lineage_id
                )
                lineage.bind_track(track.mot_track_id)
                self._pending_backend_ops.append(
                    ("add", sam_id, np.asarray(cmd.authoritative_box, dtype=float))
                )
                self._pending_add_sams.add(sam_id)
                self.user_lineage[cmd.user_identity_id] = lineage.lineage_id
                self.user_track[cmd.user_identity_id] = track.mot_track_id
                self.track_user[track.mot_track_id] = cmd.user_identity_id
                if cmd.target_track_id is not None:
                    self._suppress_track(cmd.target_track_id, frame_idx)
                    self.track_user[cmd.target_track_id] = None
                self._mark_corrected(cmd)
                return InteractionResult(
                    action_id=str(uuid.uuid4()),
                    action_type="Add",
                    frame_idx=frame_idx,
                    accepted=True,
                    new_track_id=track.mot_track_id,
                    new_sam_object_id=sam_id,
                    reason="manager_authoritative_add",
                    before_summary=before,
                    after_summary=summarize_manager(self.manager),
                )
            except Exception as exc:
                return InteractionResult(
                    action_id=str(uuid.uuid4()),
                    action_type="Add",
                    frame_idx=frame_idx,
                    accepted=False,
                    rolled_back=True,
                    reason=f"error: {exc}",
                    before_summary=before,
                    after_summary=summarize_manager(self.manager),
                )
        if cmd.command_type == CommandType.RECOVER_IDENTITY:
            tid = self.user_track.get(cmd.user_identity_id)
            lid = self.user_lineage.get(cmd.user_identity_id)
            if tid is None or lid is None:
                return InteractionResult(
                    action_id=str(uuid.uuid4()),
                    action_type="Recover",
                    frame_idx=frame_idx,
                    accepted=False,
                    reason="no historical track/lineage for user identity",
                    before_summary=summarize_manager(self.manager),
                    after_summary=summarize_manager(self.manager),
                )
            before = summarize_manager(self.manager)
            try:
                old_track = self.manager.get(tid)
                if old_track is not None and old_track.sam_object_id is not None:
                    old_sam = old_track.sam_object_id
                    if self.manager._sam_to_track.get(old_sam) == tid:
                        self.manager._sam_to_track.pop(old_sam, None)
                    old_track.sam_object_id = None
                lineage = self.lineages.get(lid)
                if lineage is None:
                    raise RuntimeError("lineage not found")
                if lineage.closed_frame is not None:
                    lineage.closed_frame = None
                if tid not in lineage.mot_track_ids:
                    lineage.mot_track_ids.append(tid)
                sam_id = self.ctx.allocate_sam_object_id()
                obs = self._obs_from_box(
                    frame_idx, cmd.authoritative_box, sam_id
                )
                track = self.manager.create_track(
                    frame_idx, obs, lid, mot_track_id=tid
                )
                track.state = TrackState.RECOVERED
                self._pending_backend_ops.append(
                    ("add", sam_id, np.asarray(cmd.authoritative_box, dtype=float))
                )
                self._pending_add_sams.add(sam_id)
                self.user_track[cmd.user_identity_id] = tid
                self.track_user[tid] = cmd.user_identity_id
                self._mark_corrected(cmd)
                return InteractionResult(
                    action_id=str(uuid.uuid4()),
                    action_type="Recover",
                    frame_idx=frame_idx,
                    accepted=True,
                    new_track_id=tid,
                    new_sam_object_id=sam_id,
                    reason="manager_recover_identity",
                    before_summary=before,
                    after_summary=summarize_manager(self.manager),
                )
            except Exception as exc:
                return InteractionResult(
                    action_id=str(uuid.uuid4()),
                    action_type="Recover",
                    frame_idx=frame_idx,
                    accepted=False,
                    rolled_back=True,
                    reason=f"error: {exc}",
                    before_summary=before,
                    after_summary=summarize_manager(self.manager),
                )
        if cmd.command_type == CommandType.AUTHORITATIVE_REASSIGN:
            dst_tid = self.user_track.get(cmd.destination_identity)
            src_tid = cmd.target_track_id
            if dst_tid is None:
                return InteractionResult(
                    action_id=str(uuid.uuid4()),
                    action_type="Reassign",
                    frame_idx=frame_idx,
                    accepted=False,
                    reason="destination identity has no track",
                    before_summary=summarize_manager(self.manager),
                    after_summary=summarize_manager(self.manager),
                )
            dst_track = self.manager.get(dst_tid)
            if dst_track is None or dst_track.state in (
                TrackState.DELETED,
                TrackState.TERMINATED,
            ):
                lid = self.user_lineage.get(cmd.destination_identity)
                if lid is None and dst_track is not None:
                    lid = dst_track.identity_lineage_id
                if lid is None:
                    return InteractionResult(
                        action_id=str(uuid.uuid4()),
                        action_type="Reassign",
                        frame_idx=frame_idx,
                        accepted=False,
                        reason="destination identity has no lineage",
                        before_summary=summarize_manager(self.manager),
                        after_summary=summarize_manager(self.manager),
                    )
                before = summarize_manager(self.manager)
                try:
                    dst_old = self.manager.get(dst_tid)
                    if dst_old is not None and dst_old.sam_object_id is not None:
                        old_sam = dst_old.sam_object_id
                        if self.manager._sam_to_track.get(old_sam) == dst_tid:
                            self.manager._sam_to_track.pop(old_sam, None)
                        dst_old.sam_object_id = None
                    lineage = self.lineages.get(lid)
                    if lineage is None:
                        raise RuntimeError("lineage not found")
                    if lineage.closed_frame is not None:
                        lineage.closed_frame = None
                    if dst_tid not in lineage.mot_track_ids:
                        lineage.mot_track_ids.append(dst_tid)
                    sam_id = self.ctx.allocate_sam_object_id()
                    obs = self._obs_from_box(
                        frame_idx, cmd.authoritative_box, sam_id
                    )
                    track = self.manager.create_track(
                        frame_idx, obs, lid, mot_track_id=dst_tid
                    )
                    track.state = TrackState.RECOVERED
                    self._pending_backend_ops.append(
                        ("add", sam_id, np.asarray(cmd.authoritative_box, dtype=float))
                    )
                    self._pending_add_sams.add(sam_id)
                    self._suppress_track(src_tid, frame_idx)
                    self.track_user[src_tid] = None
                    self.user_track[cmd.destination_identity] = dst_tid
                    self.track_user[dst_tid] = cmd.destination_identity
                    self._mark_corrected(cmd)
                    return InteractionResult(
                        action_id=str(uuid.uuid4()),
                        action_type="Reassign",
                        frame_idx=frame_idx,
                        accepted=True,
                        new_track_id=dst_tid,
                        new_sam_object_id=sam_id,
                        reason="reassign_manager_recover",
                        before_summary=before,
                        after_summary=summarize_manager(self.manager),
                    )
                except Exception as exc:
                    return InteractionResult(
                        action_id=str(uuid.uuid4()),
                        action_type="Reassign",
                        frame_idx=frame_idx,
                        accepted=False,
                        rolled_back=True,
                        reason=f"error: {exc}",
                        before_summary=before,
                        after_summary=summarize_manager(self.manager),
                    )
            result = perform_authoritative_reassign(
                self.ctx, frame_idx, src_tid, dst_tid
            )
            if result.accepted:
                self.user_track[cmd.destination_identity] = dst_tid
                self.track_user[dst_tid] = cmd.destination_identity
                old_user = cmd.source_identity
                if old_user is not None:
                    # The source track remains the old identity's historical
                    # track (same mot id and lineage), just without its SAM
                    # object. Keep user_track so RECOVER can reuse the id.
                    self.track_user[src_tid] = None
                self._mark_corrected(cmd)
            return result
        if cmd.command_type == CommandType.ATOMIC_ID_SWAP:
            result = perform_atomic_swap(
                self.ctx, frame_idx, cmd.target_track_id, cmd.other_track_id
            )
            if result.accepted:
                ua = self.track_user.get(cmd.target_track_id)
                ub = self.track_user.get(cmd.other_track_id)
                self.track_user[cmd.target_track_id] = ub
                self.track_user[cmd.other_track_id] = ua
                if ua is not None:
                    self.user_track[ua] = cmd.other_track_id
                if ub is not None:
                    self.user_track[ub] = cmd.target_track_id
                self._mark_corrected(cmd)
            return result
        if cmd.command_type == CommandType.AUTHORITATIVE_CORRECT:
            before = summarize_manager(self.manager)
            track = self.manager.get(cmd.target_track_id)
            if track is None or track.sam_object_id is None:
                return InteractionResult(
                    action_id=str(uuid.uuid4()),
                    action_type="Correct",
                    frame_idx=frame_idx,
                    accepted=False,
                    reason="unknown track or no sam object",
                    before_summary=before,
                    after_summary=before,
                )
            sam_id = track.sam_object_id
            obs = self._obs_from_box(frame_idx, cmd.authoritative_box, sam_id)
            self.manager.update_track(
                cmd.target_track_id, frame_idx, obs, human_verified=True
            )
            self._pending_backend_ops.append(
                ("correct", sam_id, np.asarray(cmd.authoritative_box, dtype=float))
            )
            self._mark_corrected(cmd)
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Correct",
                frame_idx=frame_idx,
                accepted=True,
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        if cmd.command_type == CommandType.AUTHORITATIVE_DELETE:
            before = summarize_manager(self.manager)
            self._suppress_track(cmd.target_track_id, frame_idx)
            # The deleted mot id remains the user identity's historical
            # track id so RECOVER can reuse it (create_track overwrites).
            self.track_user[cmd.target_track_id] = None
            return InteractionResult(
                action_id=str(uuid.uuid4()),
                action_type="Delete",
                frame_idx=frame_idx,
                accepted=True,
                before_summary=before,
                after_summary=summarize_manager(self.manager),
            )
        return InteractionResult(
            action_id=str(uuid.uuid4()),
            action_type="Unknown",
            frame_idx=frame_idx,
            accepted=False,
            reason="unknown command",
            before_summary={},
            after_summary={},
        )

    def _mark_corrected(self, cmd: ObserverCommand) -> None:
        # State corrections are applied manager-first and take effect through
        # sam-id mappings immediately; no per-window repeat suppression.
        return

    # -- post building -----------------------------------------------------
    def _build_post(
        self,
        frame_idx: int,
        pre: List[PromptObjectObservation],
        commands: List[ObserverCommand],
    ) -> List[Tuple[int, PromptObjectObservation]]:
        if self.config.stateful:
            mapping = self.manager._outputs.get(frame_idx, {})
            return [(tid, obs.copy()) for tid, obs in sorted(mapping.items())]
        return self._build_p1_post(frame_idx, pre, commands)

    def _build_p1_post(
        self,
        frame_idx: int,
        pre: List[PromptObjectObservation],
        commands: List[ObserverCommand],
    ) -> List[Tuple[int, PromptObjectObservation]]:
        rows: Dict[int, PromptObjectObservation] = {}
        by_user: Dict[int, int] = {}
        # First pass: pre rows, map matched identities to stable P1 post ids.
        for obs in pre:
            tid = self._p1_track_id_for_obs(obs)
            if tid is None:
                continue
            rows[tid] = obs.copy()
        for cmd in commands:
            if cmd.command_type == CommandType.ADD_NEW_IDENTITY:
                if not self._box_covered_p1(cmd.authoritative_box, rows):
                    pid = self._p1_post_id(cmd.user_identity_id)
                    rows[pid] = self._obs_from_box(frame_idx, cmd.authoritative_box, pid)
            elif cmd.command_type == CommandType.RECOVER_IDENTITY:
                if not self._box_covered_p1(cmd.authoritative_box, rows):
                    pid = self._p1_post_id(cmd.user_identity_id)
                    rows[pid] = self._obs_from_box(frame_idx, cmd.authoritative_box, pid)
            elif cmd.command_type == CommandType.AUTHORITATIVE_CORRECT:
                tid = cmd.target_track_id
                if tid in rows:
                    obs = rows[tid].copy()
                    obs.box_xyxy = np.asarray(cmd.authoritative_box, dtype=float).copy()
                    rows[tid] = obs
            elif cmd.command_type == CommandType.AUTHORITATIVE_DELETE:
                rows.pop(cmd.target_track_id, None)
            elif cmd.command_type == CommandType.AUTHORITATIVE_REASSIGN:
                pid = self._p1_post_id(cmd.destination_identity)
                tid = cmd.target_track_id
                if tid in rows:
                    rows[pid] = rows.pop(tid)
            elif cmd.command_type == CommandType.ATOMIC_ID_SWAP:
                a, b = cmd.target_track_id, cmd.other_track_id
                if a in rows and b in rows:
                    rows[a], rows[b] = rows[b], rows[a]
        # Remap every pre row to its user's P1 post id when known.
        final: Dict[int, PromptObjectObservation] = {}
        for tid, obs in rows.items():
            uid = self._p1_remap.get(frame_idx, {}).get(tid)
            out_id = self._p1_post_id(uid) if uid is not None else tid
            if out_id in final:
                # Duplicate identity in same frame: keep higher confidence.
                if obs.confidence <= final[out_id].confidence:
                    continue
            obs2 = obs.copy()
            obs2.sam_object_id = out_id
            final[out_id] = obs2
        return [(tid, obs) for tid, obs in sorted(final.items())]

    def _p1_track_id_for_obs(self, obs: PromptObjectObservation) -> Optional[int]:
        track = self.registry.lookup_by_sam_object_id(obs.sam_object_id)
        return track.mot_track_id if track is not None else None

    def _box_covered_p1(
        self, box: np.ndarray, rows: Dict[int, PromptObjectObservation]
    ) -> bool:
        for obs in rows.values():
            if box_iou(np.asarray(box, dtype=float), obs.box_xyxy) > 0.95:
                return True
        return False

    def _p1_post_id(self, user_id: Optional[int]) -> int:
        if user_id is None:
            return self._next_p1_post_id
        if user_id not in self.p1_post_ids:
            self.p1_post_ids[user_id] = self._next_p1_post_id
            self._next_p1_post_id += 1
        return self.p1_post_ids[user_id]

    def _obs_from_box(
        self, frame_idx: int, box: np.ndarray, tid: int
    ) -> PromptObjectObservation:
        return PromptObjectObservation(
            frame_idx=frame_idx,
            sam_object_id=tid,
            mask=np.zeros((1, 1), dtype=bool),
            box_xyxy=np.asarray(box, dtype=float).copy(),
            confidence=1.0,
            source="authoritative_oracle",
            is_human_verified=True,
        )

    # -- summary -----------------------------------------------------------
    def _summary(self) -> N5RunSummary:
        by_type: Dict[str, int] = {}
        rejected, rolled_back = [], []
        for r in self.results:
            by_type[r.action_type] = by_type.get(r.action_type, 0) + 1
            if not r.accepted:
                rejected.append(
                    {
                        "action_type": r.action_type,
                        "frame": r.frame_idx + 1,
                        "reason": r.reason,
                    }
                )
            if r.rolled_back:
                rolled_back.append(
                    {
                        "action_type": r.action_type,
                        "frame": r.frame_idx + 1,
                        "reason": r.reason,
                    }
                )
        return N5RunSummary(
            sequence=self.sequence,
            protocol=self.config.protocol,
            budget=self.config.budget,
            num_frames=self.num_frames,
            total_commands=len(self.results),
            accepted_commands=sum(1 for r in self.results if r.accepted),
            by_type=by_type,
            rejected=rejected,
            rolled_back=rolled_back,
            pre_rows=sum(len(v) for v in self.pre_rows.values()),
            post_rows=sum(len(v) for v in self.post_rows.values()),
            invariant_violations=self.registry.invariant_violations(),
        )
