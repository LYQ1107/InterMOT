"""N9 observer: N8 temporal-error protocol + learned tracklet association.

The association overlay relinks unbound P0 rows to known identity memories
every frame (system-side, no GT).  Human corrections additionally store
authoritative anchors and negative constraints in the memory, which future
association uses.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch

from sam3_intermot.interaction.n8_temporal_observer import (
    EventType,
    N8Config,
    N8TemporalObserver,
)
from sam3_intermot.interaction.simulator import GTFrame
from sam3_intermot.n9.models import PairwiseMLP, SetAssociator


@dataclass
class N9Config(N8Config):
    variant: str = "proposed"  # n8 | reid | pairwise | auto | proposed
    model_path: Optional[str] = None
    w_motion: float = 1.0
    relink_threshold: float = 0.0
    min_similarity: float = 0.0
    max_gap: int = 200
    use_human_anchor: bool = True
    use_negative_constraints: bool = True
    adaptive_ema: float = 0.9
    assign_margin: float = 0.3
    stability_bonus: float = 1.5
    memory_cap: int = 12


class IdentityMemory:
    def __init__(self, pid: int, feat: np.ndarray, frame: int, box: np.ndarray) -> None:
        self.pid = int(pid)
        self.feat = np.asarray(feat, dtype=np.float32)
        n = np.linalg.norm(self.feat)
        if n > 0:
            self.feat /= n
        self.adaptive = self.feat.copy()
        self.anchors: List[np.ndarray] = []
        self.negative_tids: set = set()
        self.last_frame = int(frame)
        self.last_box = np.asarray(box, dtype=float).copy()
        self.velocity = np.zeros(2, dtype=float)
        self.first_frame = int(frame)
        self.assigned_tid: Optional[int] = None
        self.confidence = 1.0

    def update_adaptive(self, feat: np.ndarray, box: np.ndarray, frame: int, ema: float) -> None:
        v = np.asarray(feat, dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        self.adaptive = ema * self.adaptive + (1 - ema) * v
        n = np.linalg.norm(self.adaptive)
        if n > 0:
            self.adaptive /= n
        if self.last_box is not None and frame > self.last_frame:
            c_old = np.asarray([(self.last_box[0] + self.last_box[2]) / 2, (self.last_box[1] + self.last_box[3]) / 2])
            c_new = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            self.velocity = 0.8 * self.velocity + 0.2 * (c_new - c_old) / max(1, frame - self.last_frame)
        self.last_box = np.asarray(box, dtype=float).copy()
        self.last_frame = int(frame)
        self.confidence = min(1.0, self.confidence + 0.05)

    def add_anchor(self, feat: np.ndarray, cap: int = 4) -> None:
        v = np.asarray(feat, dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        self.anchors.append(v.copy())
        if len(self.anchors) > cap:
            self.anchors.pop(0)
        # recompute effective feature with anchor authority
        if self.anchors:
            self.feat = np.mean(self.anchors, axis=0)
            n = np.linalg.norm(self.feat)
            if n > 0:
                self.feat /= n

    def effective_feat(self, use_anchor: bool) -> np.ndarray:
        if use_anchor and self.anchors:
            return self.feat
        return self.adaptive


class N9Observer(N8TemporalObserver):
    def __init__(
        self,
        backbone,
        gt_frames,
        num_frames,
        config: N9Config,
        sequence: str = "",
        feat_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
        model=None,
    ) -> None:
        super().__init__(backbone, gt_frames, num_frames, config, sequence=sequence)
        self.cfg = config
        self.feat_cache = feat_cache or {}
        self.model = model
        self.id_memory: Dict[int, IdentityMemory] = {}
        self.frame_assignment: Dict[int, int] = {}
        self.prev_assignment: Dict[int, int] = {}
        self._pre_phase = False
        self.seen_tids: set = set()
        self.relink_events: List[dict] = []
        self.anchor_usage = {"anchor_usage_count": 0, "successful_anchor_relinks": 0}
        self.auto_relink_count = 0

    # ------------------------------------------------------------------
    def _feat(self, frame: int, tid: int) -> Optional[np.ndarray]:
        return self.feat_cache.get((int(frame), int(tid)))

    def _motion_score(self, mem: IdentityMemory, box: np.ndarray) -> float:
        if mem.last_box is None:
            return 0.0
        gap = max(0, self._current_frame - mem.last_frame)
        pred = np.asarray(mem.last_box, float).copy()
        if gap > 0:
            pred[0] += mem.velocity[0] * gap
            pred[2] += mem.velocity[0] * gap
            pred[1] += mem.velocity[1] * gap
            pred[3] += mem.velocity[1] * gap
        b = np.asarray(box, float)
        ix1, iy1 = max(pred[0], b[0]), max(pred[1], b[1])
        ix2, iy2 = min(pred[2], b[2]), min(pred[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (pred[2] - pred[0]) * (pred[3] - pred[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    def _motion_vec(self, mem: IdentityMemory, box: np.ndarray, tid_changed: float = 0.0) -> np.ndarray:
        b = np.asarray(box, float)
        gap = max(0, self._current_frame - mem.last_frame)
        dist = 1e9
        if mem.last_box is not None:
            c1 = np.asarray([(mem.last_box[0] + mem.last_box[2]) / 2, (mem.last_box[1] + mem.last_box[3]) / 2])
            c2 = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            dist = float(np.linalg.norm(c1 - c2))
        return np.asarray(
            [
                min(1.0, gap / 200.0),
                0.5,
                min(1.0, len(self.id_memory) / 20.0),
                self._motion_score(mem, b),
                min(1.0, dist / 1000.0),
                min(1.0, (b[2] - b[0]) / 2000.0),
                min(1.0, (b[3] - b[1]) / 1000.0),
                min(1.0, mem.last_frame / 2000.0),
                tid_changed,
                0.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def _process_frame(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        raw = self.backbone.get(frame_idx, [])
        # pre uses the previous frame's association (past knowledge only)
        self._pre_phase = True
        self.pre_rows[frame_idx] = self._assemble_pre(raw)
        self._pre_phase = False
        gt = self.gt_frames.get(frame_idx, GTFrame())
        self.gt_audit["gt_read_current_after_prediction"] += 1
        events = self._detect_errors(frame_idx, raw, gt)
        if self.cfg.variant != "n8":
            self._associate_frame(frame_idx, raw)
        self._apply_events(frame_idx, events)
        self.post_rows[frame_idx] = self._assemble_post(frame_idx, raw, gt)

    def _associate_frame(self, frame_idx: int, raw) -> None:
        self.prev_assignment = dict(self.frame_assignment)
        self.frame_assignment.clear()
        prev_tids = (
            {int(t) for t, _ in self.backbone.get(frame_idx - 1, [])}
            if frame_idx > 0
            else set()
        )
        unbound: List[Tuple[int, np.ndarray, np.ndarray]] = []  # tid, box, feat
        for tid, box in raw:
            tid = int(tid)
            box = np.asarray(box, float)
            if tid in self.canonical_map:
                self._refresh_memory_for_tid(tid, frame_idx, box)
                self.frame_assignment[tid] = self.canonical_map[tid]
                continue
            # conservative relinking: only genuinely new tids are candidates
            if tid in prev_tids:
                continue
            fv = self._feat(frame_idx, tid)
            if fv is None:
                continue
            unbound.append((tid, box, fv))
        if not unbound:
            return
        candidates = [
            m
            for m in self.id_memory.values()
            if frame_idx - m.last_frame <= self.cfg.max_gap
        ]
        if not candidates:
            for tid, box, fv in unbound:
                self._new_memory(tid, fv, frame_idx, box)
            return
        scores = self._score_matrix(unbound, candidates)
        # stability bonus: keep the previous frame's assignment by default
        for i, (tid, box, fv) in enumerate(unbound):
            prev_pid = self.prev_assignment.get(int(tid))
            if prev_pid is None:
                continue
            for j, mem in enumerate(candidates):
                if mem.pid == prev_pid:
                    scores[i, j] += self.cfg.stability_bonus
        assigned = self._assign(scores, unbound, candidates)
        confident = self._confident_pairs(scores, assigned, unbound, candidates)
        for row_i, mem_i in assigned:
            tid, box, fv = unbound[row_i]
            mem = candidates[mem_i]
            self.frame_assignment[tid] = mem.pid
            mem.assigned_tid = tid
            if confident.get((row_i, mem_i)):
                mem.update_adaptive(fv, box, frame_idx, self.cfg.adaptive_ema)
            self.auto_relink_count += 1
            self.relink_events.append(
                {
                    "sequence": self.sequence,
                    "frame": frame_idx + 1,
                    "tid": tid,
                    "pid": mem.pid,
                    "score": float(scores[row_i, mem_i]),
                    "anchor_used": bool(mem.anchors),
                    "via_anchor": bool(mem.anchors),
                }
            )
            if mem.anchors:
                self.anchor_usage["anchor_usage_count"] += 1
                self.anchor_usage["successful_anchor_relinks"] += 1
        assigned_rows = {i for i, _ in assigned}
        assigned_pids = {self.frame_assignment[unbound[i][0]] for i in assigned_rows}
        for i, (tid, box, fv) in enumerate(unbound):
            if i in assigned_rows:
                continue
            if int(tid) in assigned_pids:
                # natural tid collides with an auto-assigned memory pid: keep
                # the natural row and drop the auto assignment to avoid
                # duplicate-output rows (baseline preservation safety).
                for rt, mem_i in assigned:
                    if self.frame_assignment.get(unbound[rt][0]) == int(tid):
                        del self.frame_assignment[unbound[rt][0]]
                        break
                continue
            self._new_memory(tid, fv, frame_idx, box)

    def _pid_for_auto(self, tid: int) -> int:
        if tid in self.canonical_map:
            return self.canonical_map[tid]
        if self._pre_phase and tid in self.prev_assignment:
            return self.prev_assignment[tid]
        if tid in self.frame_assignment:
            return self.frame_assignment[tid]
        return tid

    def _new_memory(self, tid, fv, frame, box) -> None:
        pid = tid
        if pid not in self.id_memory:
            self.id_memory[pid] = IdentityMemory(pid, fv, frame, box)
        self.seen_tids.add(int(tid))

    def _refresh_memory_for_tid(self, tid, frame, box) -> None:
        pid = self.canonical_map.get(int(tid))
        mem = self.id_memory.get(pid)
        fv = self._feat(frame, int(tid))
        if mem is not None and fv is not None:
            mem.update_adaptive(fv, box, frame, self.cfg.adaptive_ema)

    def _score_matrix(self, unbound, candidates) -> np.ndarray:
        n, m = len(unbound), len(candidates)
        scores = np.zeros((n, m), dtype=np.float32)
        if self.model is not None and hasattr(self.model, "forward"):
            with torch.no_grad():
                if isinstance(self.model, PairwiseMLP):
                    for i, (tid, box, fv) in enumerate(unbound):
                        mem_feats = []
                        motions = []
                        for mem in candidates:
                            mem_feats.append(mem.effective_feat(self.cfg.use_human_anchor))
                            motions.append(self._motion_vec(mem, box))
                        if not mem_feats:
                            continue
                        mf = torch.as_tensor(np.stack(mem_feats))
                        rf = torch.as_tensor(np.repeat(fv[None, :], m, axis=0))
                        mv = torch.as_tensor(np.stack(motions))
                        logits = self.model(mf, rf, mv).numpy()
                        scores[i] = logits
                elif isinstance(self.model, SetAssociator):
                    mf = np.stack([mem.effective_feat(self.cfg.use_human_anchor) for mem in candidates])
                    rf = np.stack([fv for _, _, fv in unbound])
                    mem_mot = np.stack(
                        [
                            np.asarray(
                                [
                                    min(1.0, max(0, self._current_frame - mem.last_frame) / 200.0),
                                    0.5,
                                    min(1.0, len(self.id_memory) / 20.0),
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    min(1.0, mem.last_frame / 2000.0),
                                    0.0,
                                    0.0,
                                ],
                                dtype=np.float32,
                            )
                            for mem in candidates
                        ]
                    )
                    row_mot = np.stack(
                        [
                            np.asarray(
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    min(1.0, (box[2] - box[0]) / 2000.0),
                                    min(1.0, (box[3] - box[1]) / 1000.0),
                                    min(1.0, self._current_frame / 2000.0),
                                    0.0,
                                    0.0,
                                ],
                                dtype=np.float32,
                            )
                            for _, box, _ in unbound
                        ]
                    )
                    logits = self.model(
                        torch.as_tensor(mf[None]),
                        torch.as_tensor(rf[None]),
                        torch.as_tensor(mem_mot[None]),
                        torch.as_tensor(row_mot[None]),
                    )[0].numpy()
                    scores = logits
        else:
            for i, (tid, box, fv) in enumerate(unbound):
                for j, mem in enumerate(candidates):
                    mf = mem.effective_feat(self.cfg.use_human_anchor)
                    cos = float(np.dot(mf, fv))
                    scores[i, j] = cos + self.cfg.w_motion * self._motion_score(mem, box)
        # negative constraints
        if self.cfg.use_negative_constraints:
            for i, (tid, box, fv) in enumerate(unbound):
                for j, mem in enumerate(candidates):
                    if int(tid) in mem.negative_tids:
                        scores[i, j] -= 10.0
        return scores

    def _assign(self, scores, unbound, candidates):
        n, m = scores.shape
        if n == 0 or m == 0:
            return []
        thresh = self.cfg.relink_threshold
        valid = scores > thresh
        if not valid.any():
            return []
        # greedy one-to-one by best score
        used_rows = set()
        used_mems = set()
        out = []
        order = np.dstack(np.unravel_index(np.argsort(-scores, axis=None), scores.shape))[0]
        for ri, mi in order:
            if ri in used_rows or mi in used_mems:
                continue
            if scores[ri, mi] <= thresh:
                break
            out.append((int(ri), int(mi)))
            used_rows.add(int(ri))
            used_mems.add(int(mi))
        return out

    def _confident_pairs(self, scores, assigned, unbound, candidates):
        """Pairs whose score beats every alternative by assign_margin."""
        out = {}
        for ri, mi in assigned:
            row = scores[ri]
            best = row[mi]
            second = float("-inf")
            for j, v in enumerate(row):
                if j != mi and v > second:
                    second = v
            out[(ri, mi)] = best - second >= self.cfg.assign_margin
        return out

    # ------------------------------------------------------------------
    def _apply_one(self, e: dict) -> None:
        super()._apply_one(e)
        if not self.cfg.use_human_anchor or self.cfg.variant not in ("reid", "pairwise", "auto", "proposed"):
            return
        frame = e["frame"] - 1
        if e["action_type"] in ("AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"):
            if e["action_type"] == "ATOMIC_ID_SWAP":
                pairs = [
                    (e.get("dataset_gt_id"), e.get("target_auto_tid"), e.get("canonical_public_id")),
                    (e.get("other_dataset_gt_id"), e.get("other_auto_tid"), e.get("other_canonical_public_id")),
                ]
            else:
                pairs = [(e.get("dataset_gt_id"), e.get("target_auto_tid"), e.get("canonical_public_id"))]
            for gid, tid, canon in pairs:
                if tid is None:
                    continue
                fv = self._feat(frame, int(tid))
                if fv is None:
                    continue
                mem = self.id_memory.get(int(canon))
                if mem is None:
                    box = np.asarray(e.get("gt_box") or e.get("pre_box"), float)
                    if box.size != 4:
                        continue
                    mem = IdentityMemory(int(canon), fv, frame, box)
                    self.id_memory[int(canon)] = mem
                mem.add_anchor(fv, cap=4)
                # negative: corrected-away tid is not this identity
                if self.cfg.use_negative_constraints:
                    mem.negative_tids.add(int(tid))
