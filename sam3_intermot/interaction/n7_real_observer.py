"""N7 real-SAM observer: baseline-preserving sparse interaction (Route A v5).

Protocol v5 combines the two mechanisms that are each already proven on the
pinned SAM 3.1 API:

1. **P0 automatic backbone** (`ObjectIdentityRegistry` + 200-frame windows,
   no `reset_session`): exactly the frozen P0/A0_v2 pipeline, so
   zero-interaction output follows the canonical baseline.
2. **Interaction-triggered restart**: when a user action is accepted, the
   next frame performs the official `reset_session` and re-prompts the full
   active set (user identities with authoritative boxes; automatic
   identities with their latest system boxes), then returns to long windows.

Public MOT ids:
- automatic identities = registry track ids (1..N, same numbering as P0);
- user identities = allocator ids (>=1000), bound via ``track_to_uid``
  (registry track id -> user); the IdentityNamespace only binds SAM ext ids.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3_intermot.identity.registry import ObjectIdentityRegistry
from sam3_intermot.interaction.continuous_observer import match_boxes
from sam3_intermot.interaction.n6_observer import N6Event
from sam3_intermot.interaction.n6_real_observer import N6RealObserver
from sam3_intermot.tracking.association import center_distance as _center_distance


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class N7RealObserver(N6RealObserver):
    """P0-equivalent windows + interaction-triggered full rehydration."""

    def __init__(
        self,
        backend,
        video_source: str,
        gt_frames,
        num_frames: int,
        config,
        sequence: str = "",
        segment_len: int = 30,
        window_len: int = 200,
        hold_frames: int = 60,
        auto_iou_threshold: float = 0.2,
        user_det_iou: float = 0.2,
        rehydrate_autos: bool = True,
    ) -> None:
        super().__init__(
            backend,
            video_source,
            gt_frames,
            num_frames,
            config,
            sequence=sequence,
            segment_len=segment_len,
        )
        self.window_len = window_len
        self.hold_frames = hold_frames
        self.user_det_iou = user_det_iou
        self.rehydrate_autos = rehydrate_autos
        self.auto_registry = ObjectIdentityRegistry(iou_threshold=auto_iou_threshold)
        self.track_to_uid: Dict[int, int] = {}  # registry track id -> user id
        self._auto_last_frame: Dict[int, int] = {}
        self._user_native_ids = set()
        self._used_auto_tids = set()
        self._pending_restart: Optional[int] = None
        self.rehydrated_prompts = 0
        self.held_auto_rows = 0
        self.restart_snapshots = 0
        self.rollback_count = 0
        self.window_count = 0
        self.interaction_restarts = 0

    # ------------------------------------------------------------------
    # identity helpers
    # ------------------------------------------------------------------
    def _user_for_raw(self, tid: int) -> Optional[int]:
        uid = self.track_to_uid.get(tid)
        if uid is None:
            uid = self.ns.user_for_auto(tid)
        return uid

    def _pid_for_auto(self, tid: int) -> int:
        uid = self._user_for_raw(tid)
        pid = self.ns.public_id_for(uid) if uid is not None else None
        return pid if pid is not None else tid

    def _registry_tid_for_uid(self, uid: int) -> Optional[int]:
        for tid, u in self.track_to_uid.items():
            if u == uid:
                return tid
        return None

    def _match_user_box(self, box) -> Optional[int]:
        best = None
        best_iou = self.user_det_iou
        for uid, ub in self.active_user_box.items():
            iou = _iou(box, ub)
            if iou > best_iou:
                best_iou = iou
                best = uid
        return best

    # ------------------------------------------------------------------
    # restart transaction
    # ------------------------------------------------------------------
    def _ensure_segment(self, start: int) -> None:
        ns_snap = self.ns.snapshot()
        reg_snap = self.auto_registry.manager.snapshot()
        track_uid_snap = dict(self.track_to_uid)
        user_box_snap = {
            uid: np.asarray(box, dtype=float).copy()
            for uid, box in self.active_user_box.items()
        }
        prop_snap = {k: list(v) for k, v in self.propagated.items()}
        segment_id_snap = self.segment_id
        next_ext_snap = self._next_sam_ext
        last_frame_snap = dict(self._auto_last_frame)
        self.restart_snapshots += 1
        try:
            self._ensure_segment_impl(start)
        except Exception:
            self.ns.restore(ns_snap)
            self.auto_registry.manager.restore(reg_snap)
            self.track_to_uid = track_uid_snap
            self.active_user_box = user_box_snap
            self.propagated = prop_snap
            self.segment_id = segment_id_snap
            self._next_sam_ext = next_ext_snap
            self._auto_last_frame = last_frame_snap
            self.rollback_count += 1
            raise

    def _ensure_segment_impl(self, start: int) -> None:
        self.interaction_restarts += 1
        self.segment_id += 1
        self._user_native_ids = set()
        self.backend.reset_session()
        self.propagated = {}
        # 1) user anchors
        for uid, box in sorted(self.active_user_box.items()):
            sam_id = self._next_sam_ext
            self._next_sam_ext += 1
            self.backend.add_box(start, sam_id, box)
            self.ns.bind_sam(self.segment_id, sam_id, uid)
            self.ns.bind_auto(sam_id, uid)
        # 2) optionally rehydrate every active automatic identity
        self.auto_registry.unbind_all_for_window()
        if self.rehydrate_autos:
            for track in self.auto_registry.manager.active_tracks():
                if self._user_for_raw(track.mot_track_id) is not None or track.last_box is None:
                    continue
                sam_id = self._next_sam_ext
                self._next_sam_ext += 1
                obs = self.backend.add_box(
                    start, sam_id, np.asarray(track.last_box, dtype=float)
                )
                track2 = self._register_auto(start, obs)
                if track2 is not None:
                    self._auto_last_frame[track2.mot_track_id] = start
                self.rehydrated_prompts += 1
        # 3) discovery
        dets = self.backend.detect_concept(start, "person")
        self._used_auto_tids = set()
        for det in dets:
            if self.ns.user_for_auto(det.sam_object_id) is not None:
                continue
            if self._match_user_box(det.box_xyxy) is not None:
                self._user_native_ids.add(det.sam_object_id)
                continue
            track = self._register_auto(start, det)
            if track is not None:
                self._auto_last_frame[track.mot_track_id] = start
        self.propagated[start] = list(dets)
        end = min(start + self.window_len - 1, self.num_frames - 1)
        prop = self.backend.propagate(
            start, end, start_frame_index=start, keep_masks=False
        )
        for f, obs_list in prop.items():
            self.propagated[f] = list(obs_list)
        self.backend._output_cache.clear()
        self.window_count += 1

    def _ensure_window(self, start: int) -> None:
        """P0-style window: detect + propagate, no reset, registry association."""
        self.segment_id += 1
        self._user_native_ids = set()
        self.auto_registry.unbind_all_for_window()
        dets = self.backend.detect_concept(start, "person")
        self._used_auto_tids = set()
        for det in dets:
            if self.ns.user_for_auto(det.sam_object_id) is not None:
                continue
            if self._match_user_box(det.box_xyxy) is not None:
                self._user_native_ids.add(det.sam_object_id)
                continue
            track = self._register_auto(start, det)
            if track is not None:
                self._auto_last_frame[track.mot_track_id] = start
        self.propagated[start] = list(dets)
        end = min(start + self.window_len - 1, self.num_frames - 1)
        prop = self.backend.propagate(
            start, end, start_frame_index=start, keep_masks=False
        )
        for f, obs_list in prop.items():
            self.propagated[f] = list(obs_list)
        self.backend._output_cache.clear()
        self.window_count += 1

    def _consume_frame_obs(self, frame_idx: int) -> None:
        """Register this frame's automatic observations (per-frame timing)."""
        self._used_auto_tids = set()
        for obs in self.propagated.get(frame_idx, []):
            if self.ns.user_for_auto(obs.sam_object_id) is not None:
                continue
            if obs.sam_object_id in self._user_native_ids:
                continue
            track = self._register_auto(frame_idx, obs)
            if track is not None:
                self._auto_last_frame[track.mot_track_id] = frame_idx

    def _register_auto(self, frame_idx, obs):
        """Registry association with box-first fallback (A0_v2-compatible)."""
        manager = self.auto_registry.manager
        track = self.auto_registry.register_auto_object(frame_idx, obs)
        if track is not None:
            self._used_auto_tids.add(track.mot_track_id)
        return track

    def run(self) -> None:
        self.backend.start_video(self.video_source)
        try:
            self._ensure_window(0)
            for f in range(self.num_frames):
                if self._pending_restart == f:
                    self._ensure_segment(f)
                    self._pending_restart = None
                if f not in self.propagated:
                    self._ensure_window(f)
                self._consume_frame_obs(f)
                before = self.accepted_count
                self._process_frame(f)
                if self.accepted_count > before:
                    self._pending_restart = f + 1
                if (f + 1) % self.window_len == 0 and f + 1 < self.num_frames:
                    if self._pending_restart == f + 1:
                        self._ensure_segment(f + 1)
                        self._pending_restart = None
                    else:
                        self._ensure_window(f + 1)
                for pid, box in self.post_rows.get(f, []):
                    uid = self.ns.user_for_public(pid)
                    if uid is not None:
                        self.active_user_box[uid] = np.asarray(box, dtype=float).copy()
        finally:
            self.backend.close()

    # ------------------------------------------------------------------
    # frame assembly
    # ------------------------------------------------------------------
    def _raw(self, frame_idx: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        for obs in self.propagated.get(frame_idx, []):
            if obs.sam_object_id in self._user_native_ids:
                continue
            if self.ns.user_for_auto(obs.sam_object_id) is not None:
                out.append(
                    (obs.sam_object_id, np.asarray(obs.box_xyxy, dtype=float).copy())
                )
        for tid, obs in self.auto_registry.manager._outputs.get(frame_idx, {}).items():
            box = np.asarray(obs.box_xyxy, dtype=float)
            if not self._valid_box(box):
                continue
            uid = self._user_for_raw(tid)
            if uid is not None:
                if self.ns.public_id_for(uid) in {t for t, _ in out}:
                    continue
            out.append((int(tid), box.copy()))
        present = {tid for tid, _ in out}
        for track in self.auto_registry.manager.active_tracks():
            if track.mot_track_id in present or track.last_box is None:
                continue
            if self._user_for_raw(track.mot_track_id) is not None:
                continue
            if frame_idx - self._auto_last_frame.get(track.mot_track_id, -10**9) > self.hold_frames:
                continue
            box = np.asarray(track.last_box, dtype=float)
            if self._valid_box(box):
                out.append((track.mot_track_id, box.copy()))
                self.held_auto_rows += 1
        return out

    # ------------------------------------------------------------------
    # transactions (same online protocol as N6)
    # ------------------------------------------------------------------
    def _execute(self, cmd) -> None:
        if cmd.action_type == "AUTO_BIND":
            if cmd.target_auto_tid is None or cmd.user_identity_id is None:
                raise ValueError("AUTO_BIND requires target track and user")
            self.track_to_uid[cmd.target_auto_tid] = cmd.user_identity_id
        elif cmd.action_type == "ADD_NEW_IDENTITY":
            uid = self.gt_to_user.get(cmd.gt_id)
            if uid is None:
                uid = self.ns.create_user(cmd.frame)[0]
                self.gt_to_user[cmd.gt_id] = uid
            if cmd.target_auto_tid is not None:
                self.track_to_uid[cmd.target_auto_tid] = uid
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
            self.track_to_uid[cmd.target_auto_tid] = cmd.user_identity_id
            cmd.public_mot_id = self.ns.public_id_for(cmd.user_identity_id)
        elif cmd.action_type == "ATOMIC_ID_SWAP":
            ua = self.track_to_uid.get(cmd.target_auto_tid)
            ub = self.track_to_uid.get(cmd.other_auto_tid)
            if ua is not None:
                self.track_to_uid[cmd.other_auto_tid] = ua
            if ub is not None:
                self.track_to_uid[cmd.target_auto_tid] = ub
        elif cmd.action_type == "AUTHORITATIVE_CORRECT":
            if cmd.user_identity_id is None:
                raise ValueError("correct requires user identity")
            cmd.public_mot_id = self.ns.public_id_for(cmd.user_identity_id)
        elif cmd.action_type == "AUTHORITATIVE_DELETE":
            pass
        else:
            raise ValueError(f"unknown action {cmd.action_type}")

    def _generate_commands(self, frame_idx, raw, gt):
        commands = []
        gt_boxes = [np.asarray(b, float) for b in gt.boxes]
        matches = match_boxes(
            gt_boxes,
            [np.asarray(b, float) for _, b in raw],
            self.config.match_iou_threshold,
        )
        used_pi = set()
        matched_g = set()
        identity_errors = []
        pi_to_gi = {}
        for gi, pi, iou in matches:
            used_pi.add(pi)
            matched_g.add(gi)
            pi_to_gi[pi] = gi
            tid, box = raw[pi]
            gid = gt.gt_ids[gi]
            gt_uid = self.gt_to_user.get(gid)
            auto_uid = self._user_for_raw(tid)
            if gt_uid is None:
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
                if (
                    self._user_for_raw(tid_b) == uid_a
                    and self._user_for_raw(tid_a) == uid_b
                ):
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
