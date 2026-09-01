"""Persistent identity state for the N10/N11 online association layer."""

from typing import Dict, List, Optional, Set

import numpy as np


class IdentityState:
    """State of one persistent public identity.

    ACTIVE: matched recently.  LOST: not seen but within reactivation window.
    TERMINATED: lost longer than the allowed gap (no longer candidate).
    """

    ACTIVE = "ACTIVE"
    LOST = "LOST"
    TERMINATED = "TERMINATED"

    def __init__(
        self,
        pid: int,
        feat: np.ndarray,
        box: np.ndarray,
        frame: int,
        native_tid: int = -1,
    ) -> None:
        self.pid = int(pid)
        v = np.asarray(feat, dtype=np.float32)
        n = float(np.linalg.norm(v))
        self.prototype = v / n if n > 1e-6 else np.zeros(512, dtype=np.float32)
        self.anchors: List[np.ndarray] = []
        self.authority = 0.0
        self.state = self.ACTIVE
        self.last_box = np.asarray(box, dtype=float).copy()
        self.velocity = np.zeros(2, dtype=float)
        self.last_seen_frame = int(frame)
        self.birth_frame = int(frame)
        self.last_native_tid = int(native_tid)
        self.positive_native_tids: Set[int] = set()
        self.negative_native_tids: Set[int] = set()
        self.positive_expiry: Dict[int, Optional[int]] = {}
        self.negative_expiry: Dict[int, Optional[int]] = {}
        self.confidence = 1.0
        self.lost_age = 0
        self.matched_count = 1
        self.anchor_frame: Optional[int] = None
        self.last_match_score: Optional[float] = None

    def _anchor_weight(
        self,
        frame: int,
        authority_mode: str = "permanent",
        hard_frames: int = 1,
        decay_frames: int = 8,
        refresh_threshold: float = 0.5,
    ) -> float:
        if self.anchor_frame is None:
            return 0.0
        if authority_mode == "permanent":
            return 1.0
        if authority_mode == "evidence":
            if self.last_match_score is not None and self.last_match_score >= refresh_threshold:
                return 1.0
        age = max(0, frame - self.anchor_frame)
        if age <= hard_frames:
            return 1.0
        return max(0.0, 1.0 - (age - hard_frames) / max(1, decay_frames))

    def effective_feat(
        self,
        frame: Optional[int] = None,
        anchor_blend: float = 0.7,
        authority_mode: str = "permanent",
        hard_frames: int = 1,
        decay_frames: int = 8,
        refresh_threshold: float = 0.5,
    ) -> np.ndarray:
        if self.anchors:
            a = np.mean(self.anchors, axis=0)
            n = float(np.linalg.norm(a))
            a = a / n if n > 1e-6 else np.zeros(512, dtype=np.float32)
            w = self._anchor_weight(
                frame if frame is not None else self.anchor_frame,
                authority_mode,
                hard_frames,
                decay_frames,
                refresh_threshold,
            )
            if w >= 1.0:
                return a
            if w <= 0.0:
                return self.prototype
            blend = w * anchor_blend + (1 - anchor_blend)
            v = blend * a + (1 - blend) * self.prototype
            nv = float(np.linalg.norm(v))
            return v / nv if nv > 1e-6 else self.prototype
        return self.prototype

    def update_machine(
        self,
        feat: np.ndarray,
        box: np.ndarray,
        frame: int,
        native_tid: int,
        ema: float,
        update_prototype: bool = True,
    ) -> None:
        v = np.asarray(feat, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if update_prototype and n > 1e-6:
            v = v / n
            self.prototype = ema * self.prototype + (1 - ema) * v
            pn = float(np.linalg.norm(self.prototype))
            if pn > 1e-6:
                self.prototype /= pn
        if self.last_seen_frame is not None and frame > self.last_seen_frame:
            c_old = np.asarray(
                [(self.last_box[0] + self.last_box[2]) / 2, (self.last_box[1] + self.last_box[3]) / 2]
            )
            c_new = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            dt = max(1, frame - self.last_seen_frame)
            self.velocity = 0.8 * self.velocity + 0.2 * (c_new - c_old) / dt
        self.last_box = np.asarray(box, dtype=float).copy()
        self.last_seen_frame = int(frame)
        self.last_native_tid = int(native_tid)
        self.state = self.ACTIVE
        self.lost_age = 0
        self.confidence = min(1.0, self.confidence + 0.05)
        self.matched_count += 1

    def add_anchor(
        self,
        feat: np.ndarray,
        authority: float = 1.0,
        cap: int = 4,
        frame: Optional[int] = None,
    ) -> None:
        v = np.asarray(feat, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            v = v / n
            self.anchors.append(v.copy())
            if len(self.anchors) > cap:
                self.anchors.pop(0)
            self.authority = max(self.authority, float(authority))
            self.anchor_frame = frame if frame is not None else self.anchor_frame

    def has_positive(self, native_tid: int, frame: int) -> bool:
        if int(native_tid) not in self.positive_native_tids:
            return False
        exp = self.positive_expiry.get(int(native_tid))
        return exp is None or frame <= exp

    def has_negative(self, native_tid: int, frame: int) -> bool:
        if int(native_tid) not in self.negative_native_tids:
            return False
        exp = self.negative_expiry.get(int(native_tid))
        return exp is None or frame <= exp

    def add_positive(self, native_tid: int, expiry: Optional[int] = None) -> None:
        tid = int(native_tid)
        self.positive_native_tids.add(tid)
        self.positive_expiry[tid] = expiry

    def add_negative(self, native_tid: int, expiry: Optional[int] = None) -> None:
        tid = int(native_tid)
        self.negative_native_tids.add(tid)
        self.negative_expiry[tid] = expiry

    def prune_constraints(self, frame: int) -> None:
        for tid in list(self.positive_native_tids):
            exp = self.positive_expiry.get(tid)
            if exp is not None and frame > exp:
                self.positive_native_tids.remove(tid)
                self.positive_expiry.pop(tid, None)
        for tid in list(self.negative_native_tids):
            exp = self.negative_expiry.get(tid)
            if exp is not None and frame > exp:
                self.negative_native_tids.remove(tid)
                self.negative_expiry.pop(tid, None)

    def mark_lost(self, frame: int) -> None:
        self.state = self.LOST
        self.lost_age = max(1, frame - self.last_seen_frame)

    def advance_lost(self) -> None:
        if self.state == self.LOST:
            self.lost_age += 1

    def reactivate(self) -> None:
        self.state = self.ACTIVE
        self.lost_age = 0

    def terminate(self) -> None:
        self.state = self.TERMINATED

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "state": self.state,
            "authority": self.authority,
            "n_anchors": len(self.anchors),
            "last_seen_frame": self.last_seen_frame,
            "birth_frame": self.birth_frame,
            "lost_age": self.lost_age,
            "last_native_tid": self.last_native_tid,
            "pos_native": sorted(self.positive_native_tids),
            "neg_native": sorted(self.negative_native_tids),
            "anchor_frame": self.anchor_frame,
            "confidence": self.confidence,
            "matched_count": self.matched_count,
        }
