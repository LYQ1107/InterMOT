"""N8 temporal-error observer on the frozen P0 backbone.

Semantics (frozen for N8):
- FIRST_APPEARANCE_MATCHED is never an interaction and never consumes budget,
  even when the public MOT id differs numerically from the dataset GT id.
- Only four verified temporal errors consume budget:
    TRUE_MISS_NEW, RECOVERABLE_MISS, TEMPORAL_ID_BREAK, TEMPORAL_ID_SWAP.
- HumanObserverMemory may change from current-frame GT observation after the
  prediction is frozen; SystemState (namespace + canonical_map + post rows)
  may change only through accepted interactions.
- Fixed per-frame application priority:
    TEMPORAL_ID_SWAP > TEMPORAL_ID_BREAK > RECOVERABLE_MISS > TRUE_MISS_NEW,
  with ties broken by stable memory user_identity_id.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3_intermot.evaluation.frame_output import FrameOutputAssembler, FrameOutputRow
from sam3_intermot.identity.namespace import IdentityNamespace
from sam3_intermot.interaction.continuous_observer import match_boxes
from sam3_intermot.interaction.simulator import GTFrame
from sam3_intermot.tracking.association import box_iou


class EventType:
    FIRST_APPEARANCE_MATCHED = "FIRST_APPEARANCE_MATCHED"
    TRUE_MISS_NEW = "TRUE_MISS_NEW"
    RECOVERABLE_MISS = "RECOVERABLE_MISS"
    TEMPORAL_ID_BREAK = "TEMPORAL_ID_BREAK"
    TEMPORAL_ID_SWAP = "TEMPORAL_ID_SWAP"
    LOCALIZATION_ONLY_ERROR = "LOCALIZATION_ONLY_ERROR"
    FALSE_POSITIVE = "FALSE_POSITIVE"


ERROR_PRIORITY = {
    EventType.TEMPORAL_ID_SWAP: 0,
    EventType.TEMPORAL_ID_BREAK: 1,
    EventType.RECOVERABLE_MISS: 2,
    EventType.TRUE_MISS_NEW: 3,
}


@dataclass
class VerifiedInteractionEvent:
    sequence: str
    frame: int
    event_type: str
    user_identity_id: Optional[int]
    dataset_gt_id: Optional[int]
    current_public_id: Optional[int]
    canonical_public_id: Optional[int]
    pre_box: Optional[np.ndarray]
    gt_box: Optional[np.ndarray]
    seen_before: bool
    last_seen_frame: Optional[int]
    gap_length: Optional[int]
    matched_prediction: bool
    interaction_required: bool
    budget_available: bool
    accepted: bool
    action_type: str
    system_state_hash_before: str
    system_state_hash_after: str
    observer_memory_hash_before: str
    observer_memory_hash_after: str
    target_auto_tid: Optional[int] = None
    other_auto_tid: Optional[int] = None
    other_dataset_gt_id: Optional[int] = None
    other_canonical_public_id: Optional[int] = None
    iou: Optional[float] = None
    reason: str = ""

    def as_dict(self) -> dict:
        def box(b):
            return None if b is None else [round(float(x), 2) for x in b]

        return {
            "sequence": self.sequence,
            "frame": self.frame + 1,
            "event_type": self.event_type,
            "user_identity_id": self.user_identity_id,
            "dataset_gt_id": self.dataset_gt_id,
            "current_public_id": self.current_public_id,
            "canonical_public_id": self.canonical_public_id,
            "pre_box": box(self.pre_box),
            "gt_box": box(self.gt_box),
            "seen_before": self.seen_before,
            "last_seen_frame": None if self.last_seen_frame is None else self.last_seen_frame + 1,
            "gap_length": self.gap_length,
            "matched_prediction": self.matched_prediction,
            "interaction_required": self.interaction_required,
            "budget_available": self.budget_available,
            "accepted": self.accepted,
            "action_type": self.action_type,
            "target_auto_tid": self.target_auto_tid,
            "other_auto_tid": self.other_auto_tid,
            "other_dataset_gt_id": self.other_dataset_gt_id,
            "other_canonical_public_id": self.other_canonical_public_id,
            "iou": None if self.iou is None else round(float(self.iou), 4),
            "reason": self.reason,
        }


class HumanObserverMemory:
    """Simulated human observer history.  GT is used only here, after Y_pre."""

    def __init__(self) -> None:
        self.records: Dict[int, dict] = {}
        self.next_uid = 1

    def first_seen(self, gid: int, public_id: int, frame: int, box) -> int:
        uid = self.next_uid
        self.next_uid += 1
        self.records[gid] = {
            "user_identity_id": uid,
            "first_seen_frame": frame,
            "last_seen_frame": frame,
            "last_matched_frame": frame if public_id >= 0 else None,
            "canonical_public_id": int(public_id),
            "last_observed_public_id": int(public_id),
            "last_correct_public_id": int(public_id),
            "currently_visible": public_id >= 0,
            "ever_seen": True,
            "last_matched_box": [round(float(x), 2) for x in box],
            "accepted_interactions": 0,
            "history": [],
        }
        return uid

    def observe(self, gid: int, frame: int, public_id: Optional[int], box=None) -> None:
        rec = self.records.get(gid)
        if rec is None:
            return
        rec["last_seen_frame"] = frame
        rec["currently_visible"] = public_id is not None
        if public_id is not None:
            rec["last_observed_public_id"] = int(public_id)
            rec["last_matched_frame"] = frame
            rec["last_matched_box"] = [round(float(x), 2) for x in box]
            # canonical is never overwritten by an error observation
        rec["history"].append(
            {
                "frame": frame,
                "public_id": public_id,
                "canonical": rec["canonical_public_id"],
            }
        )

    def hash(self) -> str:
        payload = {
            str(gid): {
                k: v for k, v in rec.items() if k != "history"
            }
            | {"history_len": len(rec["history"])}
            for gid, rec in self.records.items()
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


@dataclass
class N8Config:
    budget: int = 1  # -1 = unlimited
    match_iou_threshold: float = 0.5
    localization_iou_threshold: float = 0.7
    sequence: str = ""


class N8TemporalObserver:
    def __init__(
        self,
        backbone: Dict[int, List[Tuple[int, np.ndarray]]],
        gt_frames: Dict[int, GTFrame],
        num_frames: int,
        config: N8Config,
        sequence: str = "",
    ) -> None:
        self.backbone = backbone
        self.gt_frames = gt_frames
        self.num_frames = num_frames
        self.config = config
        self.sequence = sequence
        self.ns = IdentityNamespace()
        self.memory = HumanObserverMemory()
        self.canonical_map: Dict[int, int] = {}
        self.assembler = FrameOutputAssembler()
        self.pre_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.post_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.verified_errors: List[dict] = []
        self.interaction_events: List[dict] = []
        self.observer_audit: List[dict] = []
        self.state_hashes: List[dict] = []
        self.accepted_count = 0
        self.invariant_violations: List[str] = []
        self.gt_audit = {
            "frames": 0,
            "gt_read_current_after_prediction": 0,
            "gt_read_before_prediction": 0,
            "gt_read_future": 0,
            "system_mutation_without_accepted_action": 0,
        }

    # ------------------------------------------------------------------
    def run(self) -> None:
        for f in range(self.num_frames):
            self._process_frame(f)
        self.invariant_violations += self.ns.violations()
        self.gt_audit["frames"] = self.num_frames

    def _process_frame(self, frame_idx: int) -> None:
        raw = self.backbone.get(frame_idx, [])
        # Y_pre(t) is frozen first; GT is read only after this point.
        self.pre_rows[frame_idx] = self._assemble_pre(raw)
        gt = self.gt_frames.get(frame_idx, GTFrame())
        self.gt_audit["gt_read_current_after_prediction"] += 1
        events = self._detect_errors(frame_idx, raw, gt)
        self._apply_events(frame_idx, events)
        self.post_rows[frame_idx] = self._assemble_post(frame_idx, raw, gt)

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def _make_event(
        self,
        frame_idx: int,
        event_type: str,
        *,
        user_identity_id: Optional[int] = None,
        dataset_gt_id: Optional[int] = None,
        current_public_id: Optional[int] = None,
        canonical_public_id: Optional[int] = None,
        pre_box: Optional[np.ndarray] = None,
        gt_box: Optional[np.ndarray] = None,
        seen_before: bool = False,
        last_seen_frame: Optional[int] = None,
        gap_length: Optional[int] = None,
        matched_prediction: bool = False,
        interaction_required: bool = False,
        action_type: str = "NONE",
        target_auto_tid: Optional[int] = None,
        other_auto_tid: Optional[int] = None,
        other_dataset_gt_id: Optional[int] = None,
        other_canonical_public_id: Optional[int] = None,
        iou: Optional[float] = None,
        sys_before: Optional[str] = None,
        ob_before: Optional[str] = None,
        reason: str = "",
    ) -> dict:
        sys = sys_before or self.system_state_hash()
        ob = ob_before or self.memory.hash()
        return VerifiedInteractionEvent(
            sequence=self.sequence,
            frame=frame_idx,
            event_type=event_type,
            user_identity_id=user_identity_id,
            dataset_gt_id=dataset_gt_id,
            current_public_id=current_public_id,
            canonical_public_id=canonical_public_id,
            pre_box=pre_box,
            gt_box=gt_box,
            seen_before=seen_before,
            last_seen_frame=last_seen_frame,
            gap_length=gap_length,
            matched_prediction=matched_prediction,
            interaction_required=interaction_required,
            budget_available=self._budget_left() > 0,
            accepted=False,
            action_type=action_type,
            system_state_hash_before=sys,
            system_state_hash_after=sys,
            observer_memory_hash_before=ob,
            observer_memory_hash_after=ob,
            target_auto_tid=target_auto_tid,
            other_auto_tid=other_auto_tid,
            other_dataset_gt_id=other_dataset_gt_id,
            other_canonical_public_id=other_canonical_public_id,
            iou=iou,
            reason=reason,
        ).as_dict()

    def _detect_errors(self, frame_idx: int, raw, gt: GTFrame) -> List[dict]:
        ob_before = self.memory.hash()
        sys_before = self.system_state_hash()
        events: List[dict] = []
        gt_boxes = [np.asarray(b, float) for b in gt.boxes]
        raw_boxes = [np.asarray(b, float) for _, b in raw]
        matches = match_boxes(
            gt_boxes, raw_boxes, self.config.match_iou_threshold
        )
        matched_gi = set()
        matched_pi = set()
        breaks: List[dict] = []
        for gi, pi, iou in matches:
            matched_gi.add(gi)
            matched_pi.add(pi)
            gid = gt.gt_ids[gi]
            tid = int(raw[pi][0])
            box = np.asarray(raw[pi][1], dtype=float)
            gbox = gt_boxes[gi]
            rec = self.memory.records.get(gid)
            if rec is None:
                self.memory.first_seen(gid, tid, frame_idx, box)
                events.append(
                    self._make_event(
                        frame_idx,
                        EventType.FIRST_APPEARANCE_MATCHED,
                        user_identity_id=self.memory.records[gid]["user_identity_id"],
                        dataset_gt_id=gid,
                        current_public_id=tid,
                        canonical_public_id=tid,
                        pre_box=box,
                        gt_box=gbox,
                        seen_before=False,
                        matched_prediction=True,
                        iou=iou,
                        sys_before=sys_before,
                        ob_before=ob_before,
                    )
                )
            else:
                self.memory.observe(gid, frame_idx, tid, box)
                canonical = rec["canonical_public_id"]
                if tid != canonical:
                    breaks.append(
                        {
                            "gid": gid,
                            "tid": tid,
                            "canonical": canonical,
                            "box": box,
                            "gt_box": gbox,
                            "iou": iou,
                        }
                    )
                elif iou < self.config.localization_iou_threshold:
                    events.append(
                        self._make_event(
                            frame_idx,
                            EventType.LOCALIZATION_ONLY_ERROR,
                            user_identity_id=rec["user_identity_id"],
                            dataset_gt_id=gid,
                            current_public_id=tid,
                            canonical_public_id=canonical,
                            pre_box=box,
                            gt_box=gbox,
                            seen_before=True,
                            matched_prediction=True,
                            iou=iou,
                            sys_before=sys_before,
                            ob_before=ob_before,
                        )
                    )
        # unmatched GT -> true new miss or recoverable miss
        for gi in range(len(gt.boxes)):
            if gi in matched_gi:
                continue
            gid = gt.gt_ids[gi]
            gb = np.asarray(gt.boxes[gi], dtype=float)
            rec = self.memory.records.get(gid)
            if rec is None:
                uid = self.memory.first_seen(gid, -1, frame_idx, gb)
                self.memory.observe(gid, frame_idx, None)
                events.append(
                    self._make_event(
                        frame_idx,
                        EventType.TRUE_MISS_NEW,
                        user_identity_id=uid,
                        dataset_gt_id=gid,
                        current_public_id=None,
                        canonical_public_id=None,
                        gt_box=gb,
                        seen_before=False,
                        matched_prediction=False,
                        interaction_required=True,
                        action_type="ADD_NEW_IDENTITY",
                        sys_before=sys_before,
                        ob_before=ob_before,
                    )
                )
            else:
                self.memory.observe(gid, frame_idx, None)
                events.append(
                    self._make_event(
                        frame_idx,
                        EventType.RECOVERABLE_MISS,
                        user_identity_id=rec["user_identity_id"],
                        dataset_gt_id=gid,
                        current_public_id=None,
                        canonical_public_id=rec["canonical_public_id"],
                        gt_box=gb,
                        seen_before=True,
                        last_seen_frame=rec["last_seen_frame"],
                        gap_length=frame_idx - rec["last_seen_frame"],
                        matched_prediction=False,
                        interaction_required=True,
                        action_type="RECOVER_IDENTITY",
                        sys_before=sys_before,
                        ob_before=ob_before,
                    )
                )
        # swap detection among breaks
        by_tid = {br["tid"]: i for i, br in enumerate(breaks)}
        used_breaks = set()
        for i, ba in enumerate(breaks):
            if i in used_breaks:
                continue
            j = by_tid.get(ba["canonical"])
            if j is not None and j != i and j not in used_breaks:
                bb = breaks[j]
                if bb["tid"] == ba["canonical"] and ba["tid"] == bb["canonical"]:
                    used_breaks.update({i, j})
                    events.append(
                        self._make_event(
                            frame_idx,
                            EventType.TEMPORAL_ID_SWAP,
                            user_identity_id=self.memory.records[ba["gid"]]["user_identity_id"],
                            dataset_gt_id=ba["gid"],
                            current_public_id=ba["tid"],
                            canonical_public_id=ba["canonical"],
                            pre_box=ba["box"],
                            gt_box=ba["gt_box"],
                            seen_before=True,
                            matched_prediction=True,
                            interaction_required=True,
                            action_type="ATOMIC_ID_SWAP",
                            target_auto_tid=ba["tid"],
                            other_auto_tid=bb["tid"],
                            other_dataset_gt_id=bb["gid"],
                            other_canonical_public_id=bb["canonical"],
                            iou=ba["iou"],
                            sys_before=sys_before,
                            ob_before=ob_before,
                        )
                    )
        for i, ba in enumerate(breaks):
            if i in used_breaks:
                continue
            events.append(
                self._make_event(
                    frame_idx,
                    EventType.TEMPORAL_ID_BREAK,
                    user_identity_id=self.memory.records[ba["gid"]]["user_identity_id"],
                    dataset_gt_id=ba["gid"],
                    current_public_id=ba["tid"],
                    canonical_public_id=ba["canonical"],
                    pre_box=ba["box"],
                    gt_box=ba["gt_box"],
                    seen_before=True,
                    matched_prediction=True,
                    interaction_required=True,
                    action_type="AUTHORITATIVE_REASSIGN",
                    target_auto_tid=ba["tid"],
                    iou=ba["iou"],
                    sys_before=sys_before,
                    ob_before=ob_before,
                )
            )
        # unmatched raw rows -> false positives (statistics only, no budget)
        for pi, (tid, box) in enumerate(raw):
            if pi in matched_pi:
                continue
            events.append(
                self._make_event(
                    frame_idx,
                    EventType.FALSE_POSITIVE,
                    user_identity_id=None,
                    current_public_id=int(tid),
                    pre_box=np.asarray(box, dtype=float),
                    matched_prediction=False,
                    sys_before=sys_before,
                    ob_before=ob_before,
                )
            )
        # audit trail: one line per GT identity and per unmatched row
        for gi, gid in enumerate(gt.gt_ids):
            rec = self.memory.records.get(gid)
            self.observer_audit.append(
                {
                    "frame": frame_idx,
                    "sequence": self.sequence,
                    "dataset_gt_id": gid,
                    "event_type": next(
                        (e["event_type"] for e in events if e.get("dataset_gt_id") == gid),
                        "NO_EVENT",
                    ),
                    "canonical_public_id": None if rec is None else rec["canonical_public_id"],
                    "last_observed_public_id": None if rec is None else rec["last_observed_public_id"],
                    "memory_hash_before": ob_before,
                    "memory_hash_after": self.memory.hash(),
                }
            )
        # set observer hashes to the frame-level before/after values
        ob_after = self.memory.hash()
        for e in events:
            e["observer_memory_hash_before"] = ob_before
            e["observer_memory_hash_after"] = ob_after
        self.verified_errors.extend(events)
        return events

    def _budget_left(self) -> int:
        if self.config.budget < 0:
            return 10**9
        return max(0, self.config.budget - self.accepted_count)

    # ------------------------------------------------------------------
    def _apply_events(self, frame_idx: int, events: List[dict]) -> None:
        required = [e for e in events if e.get("interaction_required")]
        required.sort(
            key=lambda e: (
                ERROR_PRIORITY.get(e["event_type"], 99),
                e.get("user_identity_id") or 0,
            )
        )
        for e in required:
            sys_before = self.system_state_hash()
            ob_before = self.memory.hash()
            if self._budget_left() > 0:
                try:
                    self._apply_one(e)
                except Exception as exc:
                    e["accepted"] = False
                    e["budget_available"] = True
                    e["reason"] = f"FAILED_PRECONDITION: {exc}"
                    self.invariant_violations.append(
                        f"frame {frame_idx}: {e['action_type']} rejected: {exc}"
                    )
                else:
                    e["accepted"] = True
                    e["budget_available"] = True
                    self.accepted_count += 1
                    self.interaction_events.append(e)
            else:
                e["accepted"] = False
                e["budget_available"] = False
                e["reason"] = "BUDGET_EXHAUSTED"
            e["system_state_hash_before"] = sys_before
            e["system_state_hash_after"] = self.system_state_hash()
            e["observer_memory_hash_before"] = ob_before
            e["observer_memory_hash_after"] = self.memory.hash()
            if not e["accepted"] and e["system_state_hash_before"] != e["system_state_hash_after"]:
                self.gt_audit["system_mutation_without_accepted_action"] += 1
                self.invariant_violations.append(
                    f"frame {frame_idx}: state changed with zero accepted actions"
                )
        self.state_hashes.append(
            {
                "frame": frame_idx,
                "sequence": self.sequence,
                "observer_memory_hash": self.memory.hash(),
                "system_state_hash": self.system_state_hash(),
                "accepted_in_frame": sum(1 for e in required if e.get("accepted")),
            }
        )

    def _apply_one(self, e: dict) -> None:
        action = e["action_type"]
        gid = e.get("dataset_gt_id")
        if action == "AUTHORITATIVE_REASSIGN":
            self.canonical_map[int(e["target_auto_tid"])] = int(e["canonical_public_id"])
        elif action == "ATOMIC_ID_SWAP":
            self.canonical_map[int(e["target_auto_tid"])] = int(e["canonical_public_id"])
            self.canonical_map[int(e["other_auto_tid"])] = int(e["other_canonical_public_id"])
        elif action == "ADD_NEW_IDENTITY":
            uid, _lid, pid = self.ns.create_user(e["frame"] - 1)
            e["public_mot_id"] = pid
            e["canonical_public_id"] = pid
            e["namespace_user_identity_id"] = uid
        elif action == "RECOVER_IDENTITY":
            e["public_mot_id"] = e["canonical_public_id"]
        else:
            raise ValueError(f"unexpected costed action {action}")
        updated_gids = (
            [gid, e.get("other_dataset_gt_id")] if action == "ATOMIC_ID_SWAP" else [gid]
        )
        for rec_gid in updated_gids:
            rec = self.memory.records.get(rec_gid)
            if rec is None:
                continue
            rec["accepted_interactions"] += 1
            if rec_gid == gid:
                rec["last_correct_public_id"] = e.get("canonical_public_id")
            else:
                rec["last_correct_public_id"] = e.get("other_canonical_public_id")
        if action == "ADD_NEW_IDENTITY":
            rec = self.memory.records.get(gid)
            if rec is not None:
                rec["canonical_public_id"] = e["canonical_public_id"]
                rec["last_correct_public_id"] = e["canonical_public_id"]

    # ------------------------------------------------------------------
    def system_state_hash(self) -> str:
        payload = {
            "namespace": self.ns.mutable_state_hash(),
            "canonical_map": {str(k): v for k, v in sorted(self.canonical_map.items())},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _pid_for_auto(self, tid: int) -> int:
        if tid in self.canonical_map:
            return self.canonical_map[tid]
        uid = self.ns.user_for_auto(tid)
        if uid is not None:
            pid = self.ns.public_id_for(uid)
            if pid is not None:
                return pid
        return tid

    def _assemble_pre(self, raw) -> List[Tuple[int, np.ndarray]]:
        out: Dict[int, np.ndarray] = {}
        for tid, box in raw:
            box = np.asarray(box, dtype=float)
            if not self._valid_box(box):
                continue
            out[self._pid_for_auto(int(tid))] = box.copy()
        return [(pid, box) for pid, box in sorted(out.items())]

    def _assemble_post(self, frame_idx, raw, gt: GTFrame) -> List[Tuple[int, np.ndarray]]:
        rows: Dict[int, np.ndarray] = {}
        gt_boxes = [np.asarray(b, float) for b in gt.boxes]
        raw_boxes = [np.asarray(b, float) for _, b in raw]
        # Same Hungarian assignment as detection: every matched raw row is
        # carried into post (identity layer may remap its public id only).
        matches = match_boxes(gt_boxes, raw_boxes, self.config.match_iou_threshold)
        matched_pi = set()
        for gi, pi, _iou in matches:
            matched_pi.add(pi)
            tid, box = raw[pi]
            box = np.asarray(box, dtype=float)
            if not self._valid_box(box):
                continue
            pid = self._pid_for_auto(int(tid))
            rows[int(pid)] = box.copy()
        # authoritative current-frame rows (accepted ADD_NEW / RECOVER) win
        for e in self.interaction_events:
            if e["frame"] != frame_idx + 1:
                continue
            if e["action_type"] not in ("ADD_NEW_IDENTITY", "RECOVER_IDENTITY"):
                continue
            pid = e.get("public_mot_id") or e.get("canonical_public_id")
            gid = e.get("dataset_gt_id")
            gb = None
            for gi, gg in enumerate(gt.gt_ids):
                if gg == gid:
                    gb = gt_boxes[gi]
                    break
            if pid is not None and gb is not None:
                rows[int(pid)] = np.asarray(gb, dtype=float).copy()
        # unmatched raw rows (false positives kept, like P0)
        for pi, (tid, box) in enumerate(raw):
            if pi in matched_pi:
                continue
            box = np.asarray(box, dtype=float)
            if not self._valid_box(box):
                continue
            pid = self._pid_for_auto(int(tid))
            if pid not in rows:
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
        return sorted(
            ((r.public_mot_id, r.box_xyxy.copy()) for r in fr),
            key=lambda kv: kv[0],
        )

    def _valid_box(self, box: np.ndarray) -> bool:
        if box.size != 4:
            return False
        x1, y1, x2, y2 = box
        return bool(
            np.all(np.isfinite(box)) and x2 > x1 and y2 > y1
        )
