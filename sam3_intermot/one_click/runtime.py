"""GT-free immutable-anchor and bounded-bank reference runtime for N72R21.

Rule baselines are deliberately not named a learned ACIB. Learned scorers
plug into the same observation/state interface; offline labels live elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Sequence
import numpy as np


def unit(value):
    vector = np.asarray(value, np.float32).copy()
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-8:
        raise ValueError("finite nonzero identity vector required")
    vector /= np.linalg.norm(vector)
    vector.flags.writeable = False
    return vector


@dataclass(frozen=True)
class Candidate:
    uid: str
    embedding: np.ndarray
    box_xyxy: tuple
    quality: float
    native_tid: str | None = None

    def __post_init__(self):
        box = np.asarray(self.box_xyxy, float)
        if not self.uid or box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
            raise ValueError("invalid candidate UID/geometry")
        if not np.isfinite(self.quality) or not 0 <= self.quality <= 1: raise ValueError("quality must be in [0,1]")
        object.__setattr__(self, "embedding", unit(self.embedding))
        object.__setattr__(self, "box_xyxy", tuple(float(x) for x in box))


@dataclass(frozen=True)
class MemoryEntry:
    embedding: np.ndarray
    source_uid: str
    recording_id: str
    frame: int
    timestamp_seconds: float
    camera: str | None
    confidence: float
    quality: float
    anchor_consistency: float

    def __post_init__(self):
        object.__setattr__(self, "embedding", unit(self.embedding))


@dataclass(frozen=True)
class RuntimeConfig:
    bank_capacity: int = 4
    scoring: str = "anchor"
    memory_policy: str = "P0"
    accept_score: float = .8
    accept_margin: float = .03
    write_score: float = .9
    write_anchor_similarity: float = .8
    write_margin: float = .05
    write_quality: float = .5
    confirmation_frames: int = 3
    bank_anchor_weight: float = .5

    def __post_init__(self):
        if self.bank_capacity not in (1,4,8): raise ValueError("registered bank capacities: 1/4/8")
        if self.scoring not in {"anchor", "mean_prototype", "temporal_bank", "bank_only"}: raise ValueError("unknown rule scoring")
        if self.memory_policy not in {"P0", "P1", "SAFE_DELAYED", "SAFE_IMMEDIATE"}: raise ValueError("unknown causal write policy")
        if self.confirmation_frames < 1 or not 0 <= self.bank_anchor_weight <= 1: raise ValueError("invalid config")


class OneClickRecognizer:
    def __init__(self, config=RuntimeConfig(), *, scorer: Callable | None = None):
        self.config, self.scorer = config, scorer
        self._anchor = None
        self._bank = []
        self.recording_id = None
        self.last_frame = None
        self.last_box = None
        self.last_native_tid = None
        self._pending_embedding = None
        self._pending_count = 0
        self.initializations = 0

    @property
    def anchor(self):
        if self._anchor is None: raise RuntimeError("one click required")
        return self._anchor.copy()

    @property
    def bank(self):
        # Neither a returned array nor the returned tuple can mutate runtime memory.
        return tuple(MemoryEntry(e.embedding.copy(), e.source_uid, e.recording_id, e.frame, e.timestamp_seconds,
                                 e.camera, e.confidence, e.quality, e.anchor_consistency) for e in self._bank)

    def initialize(self, anchor, box_xyxy, *, recording_id, frame):
        if self.initializations: raise RuntimeError("additional click forbidden")
        # Validate the clicked geometry with the same contract as candidates.
        click = Candidate("anonymous-click", anchor, tuple(box_xyxy), 1.)
        self._anchor = click.embedding.copy()
        self._anchor.flags.writeable = False
        self.token = "target-" + hashlib.sha256(self._anchor.tobytes()).hexdigest()[:16]
        self.initializations = 1
        self.start_recording(recording_id)
        self.last_frame, self.last_box = int(frame), click.box_xyxy

    def start_recording(self, recording_id):
        if self._anchor is None: raise RuntimeError("cannot start identity without click")
        if not recording_id: raise ValueError("explicit independent recording identifier required")
        if recording_id == self.recording_id: raise ValueError("recording already active; do not reset within-video state")
        self.recording_id = str(recording_id)
        self.last_frame = self.last_box = self.last_native_tid = None
        self._pending_embedding, self._pending_count = None, 0
        # Preserve only the immutable anchor, anonymous token and committed bank.

    def _scores(self, candidates):
        features = np.stack([c.embedding for c in candidates])
        anchor_scores = features @ self._anchor
        if not self._bank or self.config.scoring == "anchor": return anchor_scores
        memory = np.stack([entry.embedding for entry in self._bank])
        if self.config.scoring == "mean_prototype":
            return features @ unit(np.mean(np.vstack([self._anchor, memory]), axis=0))
        bank_scores = (features @ memory.T).max(axis=1)
        if self.config.scoring == "bank_only": return bank_scores
        weight = self.config.bank_anchor_weight
        return weight * anchor_scores + (1-weight) * bank_scores

    def step(self, frame: int, candidates: Sequence[Candidate], *, fps: float, camera=None):
        if self._anchor is None: raise RuntimeError("one click required")
        if not np.isfinite(fps) or fps <= 0: raise ValueError("actual FPS required")
        if self.last_frame is not None and frame != self.last_frame + 1: raise ValueError("causal contiguous frame axis required")
        if any(not isinstance(c, Candidate) for c in candidates): raise TypeError("only GT-free Candidate values accepted")
        if len({c.uid for c in candidates}) != len(candidates): raise ValueError("duplicate candidate UID")
        if any(c.embedding.shape != self._anchor.shape for c in candidates): raise ValueError("embedding contract mismatch")
        scores = np.asarray(self._scores(candidates), float) if candidates else np.zeros(0)
        if not np.isfinite(scores).all(): raise ValueError("non-finite identity scores")
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].uid))
        top = order[0] if order else None
        margin = float(scores[top]-scores[order[1]]) if len(order)>1 else (2. if top is not None else 0.)
        selected = None
        probability = float(np.clip((scores[top]+1)/2,0,1)) if top is not None else 0.
        physical_presence_probability = probability
        write_probability = None
        if self.scorer is None:
            if top is not None and scores[top] >= self.config.accept_score and margin >= self.config.accept_margin: selected = candidates[top]
        else:
            evidence = self.scorer(self.anchor, self.bank, tuple(candidates), frame, fps, self.last_box, camera)
            learned_scores = np.asarray(evidence["identity_scores"], float)
            if learned_scores.shape != scores.shape or not np.isfinite(learned_scores).all(): raise ValueError("learned score axis mismatch")
            scores = learned_scores
            order = sorted(range(len(candidates)),key=lambda i:(-scores[i],candidates[i].uid))
            top = order[0] if order else None
            margin = float(scores[top]-scores[order[1]]) if len(order)>1 else (2. if top is not None else 0.)
            probability = float(evidence["candidate_available_probability"])
            physical_presence_probability = float(evidence["target_present_probability"])
            write_probability = float(evidence["safe_write_probability"])
            if any(not np.isfinite(p) or not 0<=p<=1 for p in (probability, physical_presence_probability, write_probability)): raise ValueError("invalid learned probability")
            if top is not None and probability >= self.config.accept_score: selected = candidates[top]
        wrote = False
        anchor_consistency = float(selected.embedding @ self._anchor) if selected is not None else None
        if selected is not None:
            consistent = self._pending_embedding is not None and float(selected.embedding @ self._pending_embedding) >= .9
            self._pending_count = self._pending_count+1 if consistent else 1
            self._pending_embedding = selected.embedding.copy()
            safe = (anchor_consistency >= self.config.write_anchor_similarity and selected.quality >= self.config.write_quality
                    and margin >= self.config.write_margin and (write_probability >= self.config.write_score if write_probability is not None else scores[top] >= self.config.write_score))
            policy = self.config.memory_policy
            confirmation = 1 if policy == "SAFE_IMMEDIATE" else self.config.confirmation_frames
            if policy == "P1" or policy.startswith("SAFE_") and safe and self._pending_count >= confirmation:
                entry = MemoryEntry(selected.embedding, selected.uid, self.recording_id, int(frame), frame/fps, camera,
                                    probability, selected.quality, anchor_consistency)
                self._bank = (self._bank + [entry])[-self.config.bank_capacity:]
                wrote = True
            self.last_box, self.last_native_tid = selected.box_xyxy, selected.native_tid
        else:
            self._pending_embedding, self._pending_count = None, 0
            self.last_box, self.last_native_tid = None, None
        self.last_frame = int(frame)
        return {"frame": int(frame), "recording_id": self.recording_id, "target_token": self.token,
                "target_present_probability": physical_presence_probability, "candidate_available_probability": probability,
                "presence_probability_semantics": "physical_presence_learned_three_class" if self.scorer else "UNCALIBRATED_REPRESENTED_TARGET_SCORE_NOT_PHYSICAL_PRESENCE",
                "selected_candidate_uid": selected.uid if selected else None, "rank1_candidate_uid": candidates[top].uid if top is not None else None,
                "predicted_box_xyxy": list(selected.box_xyxy) if selected else None,
                "identity_score": float(scores[top]) if top is not None else None, "identity_margin": margin,
                "memory_write": wrote, "memory_write_candidate_uid": selected.uid if wrote else None,
                "memory_size": len(self._bank), "anchor_sha256": hashlib.sha256(self._anchor.tobytes()).hexdigest(),
                "runtime_future_gt_used": False, "runtime_gt_used": False}
