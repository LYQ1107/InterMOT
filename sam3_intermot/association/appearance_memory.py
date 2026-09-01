"""Causal, target-scoped appearance memory for known-ID corrections.

The bank is deliberately non-parametric and CPU friendly.  A human write is
an intervention on one public ID and is only visible to calls made after its
source frame; callers are responsible for invoking it after current-frame
association has completed.  The class does not infer identity ownership: the
caller supplies the already-authoritative public ID.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


def _unit(x: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.size != dim:
        raise ValueError(f"feature dimension {v.size} != expected {dim}")
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(dim, dtype=np.float32)


@dataclass
class AppearanceAnchor:
    feature: np.ndarray
    source_frame: int
    quality: float
    source: str = "human"
    write_event_id: Optional[str] = None

    def age(self, frame: Optional[int]) -> int:
        if frame is None:
            return 0
        return max(0, int(frame) - int(self.source_frame))

    def to_dict(self, frame: Optional[int] = None) -> dict:
        return {
            "feature": self.feature.astype(np.float32).tolist(),
            "source_frame": int(self.source_frame),
            "age": self.age(frame),
            "quality": float(self.quality),
            "source": self.source,
            "write_event_id": self.write_event_id,
        }


@dataclass
class AppearanceRecord:
    prototype: Optional[np.ndarray] = None
    positive: List[AppearanceAnchor] = field(default_factory=list)
    negative: List[AppearanceAnchor] = field(default_factory=list)
    last_write_frame: Optional[int] = None
    write_count: int = 0
    last_human_frame: Optional[int] = None
    last_human_event_id: Optional[str] = None
    last_source: Optional[str] = None
    last_quality: float = 0.0
    reliability: float = 0.0


class AppearanceMemory:
    """Per-public-ID memory bank with deterministic scoring and persistence."""

    def __init__(
        self,
        feat_dim: int = 512,
        anchor_cap: int = 8,
        negative_cap: int = 16,
        ema: float = 0.9,
        decay_frames: float = 120.0,
        human_weight: float = 1.0,
        machine_weight: float = 0.35,
        min_machine_confidence: float = 0.5,
        reliability_threshold: float = 0.0,
    ) -> None:
        self.feat_dim = int(feat_dim)
        self.anchor_cap = max(0, int(anchor_cap))
        self.negative_cap = max(0, int(negative_cap))
        self.ema = float(ema)
        self.decay_frames = max(1.0, float(decay_frames))
        self.human_weight = float(human_weight)
        self.machine_weight = float(machine_weight)
        self.min_machine_confidence = float(min_machine_confidence)
        self.reliability_threshold = float(np.clip(reliability_threshold, 0.0, 1.0))
        self.records: Dict[int, AppearanceRecord] = {}

    def reset(self, public_id: Optional[int] = None) -> None:
        if public_id is None:
            self.records.clear()
        else:
            self.records.pop(int(public_id), None)

    def _record(self, public_id: int) -> AppearanceRecord:
        return self.records.setdefault(int(public_id), AppearanceRecord())

    def update_from_machine(
        self, public_id: int, frame: int, embedding: np.ndarray, confidence: float = 1.0
    ) -> bool:
        v = _unit(embedding, self.feat_dim)
        confidence = float(confidence)
        if not np.isfinite(confidence) or not np.any(v) or confidence < self.min_machine_confidence:
            return False
        r = self._record(public_id)
        if r.prototype is None or not np.any(r.prototype):
            r.prototype = v.copy()
        else:
            a = np.clip(self.ema + (1.0 - float(confidence)) * 0.08, 0.0, 0.999)
            r.prototype = _unit(a * r.prototype + (1.0 - a) * v, self.feat_dim)
        r.last_write_frame, r.write_count = int(frame), r.write_count + 1
        r.last_source = "machine"
        r.last_quality = float(np.clip(confidence, 0.0, 1.0))
        # Machine observations can establish a weak reliability signal, but
        # never lower reliability earned from a human-confirmed anchor.
        r.reliability = max(r.reliability, 0.5 * float(np.clip(confidence, 0.0, 1.0)))
        return True

    def update_from_human(
        self,
        public_id: int,
        frame: int,
        embedding: np.ndarray,
        quality: float = 1.0,
        competing_embeddings: Optional[Iterable[np.ndarray]] = None,
        write_event_id: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
    ) -> bool:
        """Write verified evidence for a known ID; returns False for invalid data.

        ``mask`` is metadata only here: feature extraction belongs to the
        caller, which must pool a supplied human mask or box crop.  It is never
        inferred from a native track.
        """
        del mask
        v = _unit(embedding, self.feat_dim)
        if not np.any(v):
            return False
        r = self._record(public_id)
        quality = float(quality)
        quality = quality if np.isfinite(quality) else 0.0
        quality = float(np.clip(quality, 0.0, 1.0))
        a = AppearanceAnchor(v.copy(), int(frame), quality, "human", write_event_id)
        if self.anchor_cap > 0:
            r.positive.append(a)
            r.positive = r.positive[-self.anchor_cap :]
        # Human evidence also seeds the long-term prototype, without allowing
        # a single correction to erase the existing machine history.
        if r.prototype is None or not np.any(r.prototype):
            r.prototype = v.copy()
        else:
            r.prototype = _unit(0.8 * r.prototype + 0.2 * v, self.feat_dim)
        if competing_embeddings is not None and self.negative_cap > 0:
            for c in competing_embeddings:
                cv = _unit(c, self.feat_dim)
                if np.any(cv):
                    r.negative.append(
                        AppearanceAnchor(cv.copy(), int(frame), float(np.clip(quality, 0.0, 1.0)), "human_competitor", write_event_id)
                    )
            r.negative = r.negative[-self.negative_cap :]
        r.last_write_frame, r.write_count = int(frame), r.write_count + 1
        r.last_human_frame = int(frame)
        r.last_human_event_id = write_event_id
        r.last_source = "human"
        r.last_quality = quality
        r.reliability = max(r.reliability, quality)
        return True

    def _weight(self, anchor: AppearanceAnchor, frame: int, source_weight: float) -> float:
        age = max(0, int(frame) - int(anchor.source_frame))
        return source_weight * float(anchor.quality) * np.exp(-age / self.decay_frames)

    def _score_components(
        self,
        public_id: int,
        candidate_embedding: np.ndarray,
        frame: int,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        gate_floor: float = 0.0,
    ) -> dict[str, float]:
        """Return decomposed additive evidence without mutating the bank."""
        v = _unit(candidate_embedding, self.feat_dim)
        r = self.records.get(int(public_id))
        if r is None or not np.any(v):
            return {"prototype": 0.0, "positive": 0.0, "negative": 0.0, "total": 0.0}
        if r.reliability < max(self.reliability_threshold, float(gate_floor)):
            return {"prototype": 0.0, "positive": 0.0, "negative": 0.0, "total": 0.0}
        prototype_term = 0.0
        positive_term = 0.0
        negative_term = 0.0
        # A write at frame t is deliberately invisible at t.  This makes the
        # causal boundary robust even if a caller audits score() after the
        # transaction; the first eligible frame is t+1.
        if r.prototype is not None and np.any(r.prototype) and (r.last_human_frame is None or int(frame) > r.last_human_frame):
            write_frame = r.last_write_frame if r.last_write_frame is not None else frame
            age = max(0, int(frame) - int(write_frame))
            gate = max(float(gate_floor), float(np.exp(-age / self.decay_frames)))
            prototype_term = gate * self.machine_weight * float(np.dot(v, r.prototype))
        if r.positive:
            eligible = [a for a in r.positive if int(frame) > int(a.source_frame)]
            if eligible:
                vals = [self._weight(a, frame, self.human_weight) * float(np.dot(v, a.feature)) for a in eligible]
                positive_term = float(positive_weight) * max(vals)
        if r.negative:
            eligible = [a for a in r.negative if int(frame) > int(a.source_frame)]
            if eligible:
                vals = [self._weight(a, frame, self.human_weight) * float(np.dot(v, a.feature)) for a in eligible]
                # A negative bank should penalize similarity, never reward a
                # candidate merely because all similarities are negative.
                negative_term = -float(negative_weight) * max(0.0, max(vals))
        return {
            "prototype": float(prototype_term),
            "positive": float(positive_term),
            "negative": float(negative_term),
            "total": float(prototype_term + positive_term + negative_term),
        }

    def score(
        self,
        public_id: int,
        candidate_embedding: np.ndarray,
        frame: int,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        gate_floor: float = 0.0,
    ) -> float:
        """Return additive identity evidence; zero means no usable memory."""
        return self._score_components(
            public_id,
            candidate_embedding,
            frame,
            positive_weight=positive_weight,
            negative_weight=negative_weight,
            gate_floor=gate_floor,
        )["total"]

    def serialize(self) -> dict:
        return {
            "feat_dim": self.feat_dim,
            "anchor_cap": self.anchor_cap,
            "negative_cap": self.negative_cap,
            "ema": self.ema,
            "decay_frames": self.decay_frames,
            "human_weight": self.human_weight,
            "machine_weight": self.machine_weight,
            "min_machine_confidence": self.min_machine_confidence,
            "reliability_threshold": self.reliability_threshold,
            "records": {
                str(pid): {
                    "prototype": None if r.prototype is None else r.prototype.astype(np.float32).tolist(),
                    "positive": [a.to_dict() for a in r.positive],
                    "negative": [a.to_dict() for a in r.negative],
                    "last_write_frame": r.last_write_frame,
                    "write_count": r.write_count,
                    "last_human_frame": r.last_human_frame,
                    "last_human_event_id": r.last_human_event_id,
                    "last_source": r.last_source,
                    "last_quality": r.last_quality,
                    "reliability": r.reliability,
                }
                for pid, r in self.records.items()
            },
        }

    @classmethod
    def deserialize(cls, payload: dict) -> "AppearanceMemory":
        obj = cls(
            **{
                k: payload[k]
                for k in (
                    "feat_dim",
                    "anchor_cap",
                    "negative_cap",
                    "ema",
                    "decay_frames",
                    "human_weight",
                    "machine_weight",
                    "min_machine_confidence",
                    "reliability_threshold",
                )
                if k in payload
            }
        )
        for pid_s, item in payload.get("records", {}).items():
            r = AppearanceRecord()
            p = item.get("prototype")
            r.prototype = None if p is None else _unit(np.asarray(p, dtype=np.float32), obj.feat_dim)
            for key, target in (("positive", r.positive), ("negative", r.negative)):
                for a in item.get(key, []):
                    target.append(
                        AppearanceAnchor(
                            _unit(np.asarray(a["feature"], dtype=np.float32), obj.feat_dim),
                            int(a["source_frame"]),
                            float(a["quality"]),
                            a.get("source", "human"),
                            a.get("write_event_id"),
                        )
                    )
            r.last_write_frame, r.write_count = item.get("last_write_frame"), int(item.get("write_count", 0))
            r.last_human_frame = item.get("last_human_frame")
            r.last_human_event_id = item.get("last_human_event_id")
            r.last_source = item.get("last_source")
            r.last_quality = float(item.get("last_quality", 0.0))
            r.reliability = float(np.clip(item.get("reliability", r.last_quality), 0.0, 1.0))
            r.positive = r.positive[-obj.anchor_cap :] if obj.anchor_cap > 0 else []
            r.negative = r.negative[-obj.negative_cap :] if obj.negative_cap > 0 else []
            obj.records[int(pid_s)] = r
        return obj
