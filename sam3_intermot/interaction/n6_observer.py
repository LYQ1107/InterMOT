"""N6 continuous-observer driver on the frozen P0 backbone.

Four-layer identity namespace (never mixed):

dataset_gt_id -> user_identity_id -> identity_lineage_id -> public_mot_id

Command generation is read-only; all namespace mutations happen inside
``_apply_commands`` for accepted authoritative actions only.  The post frame
is assembled once from the committed state by FrameOutputAssembler.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3_intermot.evaluation.frame_output import (
    FrameOutputAssembler,
    FrameOutputRow,
)
from sam3_intermot.identity.namespace import IdentityNamespace
from sam3_intermot.interaction.continuous_observer import match_boxes
from sam3_intermot.interaction.simulator import GTFrame
from sam3_intermot.tracking.association import box_iou


@dataclass
class N6Config:
    protocol: str = "p3"
    budget: int = 0
    match_iou_threshold: float = 0.5
    localization_iou_threshold: float = 0.7
    correct_localization: bool = False
    correct_false_track: bool = False
    stateful: bool = True


@dataclass
class N6Event:
    frame: int
    action_type: str
    gt_id: Optional[int] = None
    user_identity_id: Optional[int] = None
    public_mot_id: Optional[int] = None
    authoritative_box: Optional[np.ndarray] = None
    target_auto_tid: Optional[int] = None
    other_auto_tid: Optional[int] = None
    accepted: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "frame": self.frame + 1,
            "action_type": self.action_type,
            "user_identity_id": self.user_identity_id,
            "public_mot_id": self.public_mot_id,
            "authoritative_box": (
                list(np.round(self.authoritative_box, 2))
                if self.authoritative_box is not None
                else None
            ),
            "accepted": self.accepted,
            "reason": self.reason,
        }


class N6BackboneObserver:
    def __init__(
        self,
        backbone: Dict[int, List[Tuple[int, np.ndarray]]],
        gt_frames: Dict[int, GTFrame],
        num_frames: int,
        config: N6Config,
        sequence: str = "",
    ) -> None:
        self.backbone = backbone
        self.gt_frames = gt_frames
        self.num_frames = num_frames
        self.config = config
        self.sequence = sequence
        self.ns = IdentityNamespace()
        self.gt_to_user: Dict[int, int] = {}
        self.raw_fn = None
        self.assembler = FrameOutputAssembler()
        self.pre_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.post_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.events: List[dict] = []
        self.accepted_count = 0
        self.invariant_violations: List[str] = []
        self.state_hashes: List[dict] = []

    # ------------------------------------------------------------------
    def run(self) -> None:
        for f in range(self.num_frames):
            self._process_frame(f)
        self.invariant_violations += self.ns.violations()

    def _raw(self, frame_idx: int) -> List[Tuple[int, np.ndarray]]:
        if self.raw_fn is not None:
            return self.raw_fn(frame_idx)
        return self.backbone.get(frame_idx, [])

    def _process_frame(self, frame_idx: int) -> None:
        raw = self._raw(frame_idx)
        gt = self.gt_frames.get(frame_idx, GTFrame())
        self.pre_rows[frame_idx] = self._assemble_pre(raw)
        commands = self._generate_commands(frame_idx, raw, gt)
        accepted = self._apply_commands(frame_idx, commands)
        self.post_rows[frame_idx] = self._assemble_post(frame_idx, raw, gt, accepted)
        for e in accepted:
            self.events.append(
                {
                    "sequence": self.sequence,
                    "protocol": self.config.protocol,
                    "budget": self.config.budget,
                    "timestamp": time.time(),
                    "interaction_start_timestamp": None,
                    "interaction_end_timestamp": None,
                    "user_session_id": "SIMULATED_ORACLE",
                    **e.as_dict(),
                }
            )

    # -- helpers --------------------------------------------------------
    def _pid_for_auto(self, tid: int) -> int:
        uid = self.ns.user_for_auto(tid)
        pid = self.ns.public_id_for(uid) if uid is not None else None
        return pid if pid is not None else tid

    def _assemble_pre(
        self, raw: List[Tuple[int, np.ndarray]]
    ) -> List[Tuple[int, np.ndarray]]:
        out: Dict[int, np.ndarray] = {}
        for tid, box in raw:
            box = np.asarray(box, dtype=float)
            if not self._valid_box(box):
                continue
            pid = self._pid_for_auto(tid)
            out[pid] = box.copy()
        return [(pid, box) for pid, box in sorted(out.items())]

    def _valid_box(self, box: np.ndarray) -> bool:
        if box.size != 4:
            return False
        x1, y1, x2, y2 = box
        return bool(
            np.all(np.isfinite(box))
            and x2 > x1
            and y2 > y1
        )

    # -- generation (read-only) -----------------------------------------
    def _generate_commands(
        self,
        frame_idx: int,
        raw: List[Tuple[int, np.ndarray]],
        gt: GTFrame,
    ) -> List[N6Event]:
        commands: List[N6Event] = []
        gt_boxes = [np.asarray(b, float) for b in gt.boxes]
        matches = match_boxes(
            gt_boxes,
            [np.asarray(b, float) for _, b in raw],
            self.config.match_iou_threshold,
        )
        used_pi = set()
        matched_g = set()
        identity_errors: List[Tuple[int, int, int]] = []  # (pi, tid, uid)
        pi_to_gi = {}
        for gi, pi, iou in matches:
            used_pi.add(pi)
            matched_g.add(gi)
            pi_to_gi[pi] = gi
            tid, box = raw[pi]
            gid = gt.gt_ids[gi]
            gt_uid = self.gt_to_user.get(gid)
            auto_uid = self.ns.user_for_auto(tid)
            if gt_uid is None:
                # first identification of this dataset identity
                commands.append(
                    N6Event(
                        frame_idx,
                        "ADD_NEW_IDENTITY",
                        gt_id=gid,
                        target_auto_tid=tid,
                        authoritative_box=np.asarray(box, dtype=float).copy(),
                    )
                )
                continue
            if auto_uid is None:
                commands.append(
                    N6Event(
                        frame_idx,
                        "AUTO_BIND",
                        gt_id=gid,
                        user_identity_id=gt_uid,
                        target_auto_tid=tid,
                    )
                )
                continue
            if auto_uid != gt_uid:
                identity_errors.append((pi, tid, gt_uid))
                continue
            if (
                iou < self.config.localization_iou_threshold
                and self.config.correct_localization
            ):
                box = np.asarray(gt.boxes[gi], dtype=float)
                commands.append(
                    N6Event(
                        frame_idx,
                        "AUTHORITATIVE_CORRECT",
                        gt_id=gid,
                        user_identity_id=gt_uid,
                        public_mot_id=self.ns.public_id_for(gt_uid),
                        authoritative_box=box.copy(),
                        target_auto_tid=tid,
                    )
                )
        # identity errors: detect 2-cycles -> swap, else reassign
        used_err = set()
        swaps = []
        for i in range(len(identity_errors)):
            if i in used_err:
                continue
            pi_a, tid_a, uid_a = identity_errors[i]
            for j in range(i + 1, len(identity_errors)):
                if j in used_err:
                    continue
                pi_b, tid_b, uid_b = identity_errors[j]
                if self.ns.user_for_auto(tid_b) == uid_a and self.ns.user_for_auto(tid_a) == uid_b:
                    swaps.append((tid_a, tid_b, uid_a, uid_b))
                    used_err.update({i, j})
                    break
        for tid_a, tid_b, uid_a, uid_b in swaps:
            commands.append(
                N6Event(
                    frame_idx,
                    "ATOMIC_ID_SWAP",
                    user_identity_id=uid_a,
                    target_auto_tid=tid_a,
                    other_auto_tid=tid_b,
                )
            )
        for idx, (pi, tid, uid) in enumerate(identity_errors):
            if idx in used_err:
                continue
            gid = gt.gt_ids[pi_to_gi[pi]] if pi in pi_to_gi else None
            box = raw[pi][1]
            commands.append(
                N6Event(
                    frame_idx,
                    "AUTHORITATIVE_REASSIGN",
                    gt_id=gid,
                    user_identity_id=uid,
                    public_mot_id=self.ns.public_id_for(uid),
                    authoritative_box=np.asarray(box, dtype=float).copy(),
                    target_auto_tid=tid,
                )
            )
        # unmatched GT -> recover or add-new
        for gi in range(len(gt.boxes)):
            if gi in matched_g:
                continue
            gid = gt.gt_ids[gi]
            box = np.asarray(gt.boxes[gi], dtype=float)
            uid = self.gt_to_user.get(gid)
            if uid is not None:
                commands.append(
                    N6Event(
                        frame_idx,
                        "RECOVER_IDENTITY",
                        gt_id=gid,
                        user_identity_id=uid,
                        public_mot_id=self.ns.public_id_for(uid),
                        authoritative_box=box.copy(),
                    )
                )
            else:
                commands.append(
                    N6Event(
                        frame_idx,
                        "ADD_NEW_IDENTITY",
                        gt_id=gid,
                        authoritative_box=box.copy(),
                    )
                )
        # unmatched pre rows -> delete for P2
        if self.config.correct_false_track:
            for pi, (tid, _box) in enumerate(raw):
                if pi in used_pi:
                    continue
                commands.append(
                    N6Event(
                        frame_idx,
                        "AUTHORITATIVE_DELETE",
                        target_auto_tid=tid,
                        public_mot_id=tid,
                    )
                )
        return self._budget_filter(commands)

    def _budget_filter(self, commands: List[N6Event]) -> List[N6Event]:
        if self.config.protocol != "p4":
            return commands
        remaining = max(0, self.config.budget - self.accepted_count)
        free = [c for c in commands if c.action_type == "AUTO_BIND"]
        costed = [c for c in commands if c.action_type != "AUTO_BIND"]
        return free + costed[:remaining]

    # -- execution (only accepted mutations) ----------------------------
    def _apply_commands(self, frame_idx: int, commands: List[N6Event]) -> List[N6Event]:
        accepted: List[N6Event] = []
        before = self.ns.mutable_state_hash()
        any_auto_bind = False
        for cmd in commands:
            if self.config.protocol == "p4" and self.accepted_count >= self.config.budget:
                cmd.accepted = False
                cmd.reason = "BUDGET_EXHAUSTED"
                self.state_hashes.append(
                    {"frame": frame_idx, "kind": "BUDGET_EXHAUSTED", "changed": False}
                )
                continue
            try:
                self._execute(cmd)
            except Exception as exc:
                cmd.accepted = False
                cmd.reason = f"FAILED_PRECONDITION: {exc}"
                self.state_hashes.append(
                    {"frame": frame_idx, "kind": "REJECTED", "changed": False}
                )
                continue
            if cmd.action_type == "AUTO_BIND":
                # internal binding: not a user correction, no budget cost
                any_auto_bind = True
                cmd.accepted = True
                continue
            cmd.accepted = True
            accepted.append(cmd)
            self.accepted_count += 1
        after = self.ns.mutable_state_hash()
        if before != after and not accepted and not any_auto_bind:
            self.invariant_violations.append(
                f"frame {frame_idx}: state changed with zero accepted actions"
            )
        return accepted

    def _execute(self, cmd: N6Event) -> None:
        if cmd.action_type == "AUTO_BIND":
            if cmd.target_auto_tid is None or cmd.user_identity_id is None:
                raise ValueError("AUTO_BIND requires target track and user")
            self.ns.bind_auto(cmd.target_auto_tid, cmd.user_identity_id)
        elif cmd.action_type == "ADD_NEW_IDENTITY":
            uid = self.gt_to_user.get(cmd.gt_id)
            if uid is None:
                uid = self.ns.create_user(cmd.frame)[0]
                self.gt_to_user[cmd.gt_id] = uid
            if cmd.target_auto_tid is not None:
                self.ns.bind_auto(cmd.target_auto_tid, uid)
            cmd.user_identity_id = uid
            cmd.public_mot_id = self.ns.public_id_for(uid)
        elif cmd.action_type == "RECOVER_IDENTITY":
            uid = cmd.user_identity_id
            if uid is None:
                uid = self.gt_to_user.get(cmd.gt_id)
            if uid is None:
                raise ValueError("recover without known user identity")
            self.ns.recover(uid)
            cmd.user_identity_id = uid
            cmd.public_mot_id = self.ns.public_id_for(uid)
        elif cmd.action_type == "AUTHORITATIVE_REASSIGN":
            if cmd.target_auto_tid is None or cmd.user_identity_id is None:
                raise ValueError("reassign requires target track and user")
            self.ns.reassign(cmd.target_auto_tid, cmd.user_identity_id)
            cmd.public_mot_id = self.ns.public_id_for(cmd.user_identity_id)
        elif cmd.action_type == "ATOMIC_ID_SWAP":
            self.ns.swap(cmd.target_auto_tid, cmd.other_auto_tid)
        elif cmd.action_type == "AUTHORITATIVE_CORRECT":
            if cmd.user_identity_id is None:
                raise ValueError("correct requires user identity")
            cmd.public_mot_id = self.ns.public_id_for(cmd.user_identity_id)
        elif cmd.action_type == "AUTHORITATIVE_DELETE":
            pass
        else:
            raise ValueError(f"unknown action {cmd.action_type}")

    # -- post -----------------------------------------------------------
    def _assemble_post(
        self,
        frame_idx: int,
        raw: List[Tuple[int, np.ndarray]],
        gt: GTFrame,
        accepted: List[N6Event],
    ) -> List[Tuple[int, np.ndarray]]:
        rows: Dict[int, np.ndarray] = {}
        cmd_by_gt: Dict[int, N6Event] = {}
        for e in accepted:
            if e.gt_id is not None:
                cmd_by_gt[e.gt_id] = e
        for gi, gid in enumerate(gt.gt_ids):
            cmd = cmd_by_gt.get(gid)
            if cmd is not None and cmd.authoritative_box is not None:
                rows[cmd.public_mot_id] = np.asarray(cmd.authoritative_box, dtype=float).copy()
                continue
            # matched pre row for this GT
            best = None
            best_iou = self.config.match_iou_threshold
            for tid, box in raw:
                iou = box_iou(np.asarray(gt.boxes[gi], dtype=float), np.asarray(box, dtype=float))
                if iou > best_iou:
                    best_iou = iou
                    best = (tid, box)
            if best is not None:
                pid = self._pid_for_auto(best[0])
                rows[pid] = np.asarray(best[1], dtype=float).copy()
        # unmatched pre rows
        gt_boxes = [np.asarray(b, float) for b in gt.boxes]
        matches = match_boxes(
            gt_boxes, [np.asarray(b, float) for _, b in raw], self.config.match_iou_threshold
        )
        matched_pi = {pi for _, pi, _ in matches}
        if not self.config.correct_false_track:
            for pi, (tid, box) in enumerate(raw):
                if pi in matched_pi:
                    continue
                box = np.asarray(box, dtype=float)
                if not self._valid_box(box):
                    continue
                pid = self._pid_for_auto(tid)
                if pid in rows:
                    continue
                rows[pid] = box.copy()
        fr = [
            FrameOutputRow(int(pid), np.asarray(box, dtype=float))
            for pid, box in rows.items()
        ]
        try:
            fr = self.assembler.assemble(frame_idx, fr)
        except Exception as exc:
            self.invariant_violations.append(f"frame {frame_idx}: {exc}")
            fr = []
        return [(r.public_mot_id, r.box_xyxy.copy()) for r in fr]
