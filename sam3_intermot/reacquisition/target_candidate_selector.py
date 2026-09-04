"""Deterministic causal target-candidate selector V0 for N72R7.

The selector produces evidence for one persistent target or explicit NONE.  It
never creates a public ID, reads GT, or makes a global assignment.  The fixed
weights are part of the N72R7 development protocol and are intentionally not
optimized from future outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .target_candidate_pool import MAIN_B0_CANDIDATE


def _unit(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        raise ValueError("selector feature contains non-finite values")
    norm = float(np.linalg.norm(array))
    return None if norm <= 1.0e-6 else array / norm


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    value = float(np.dot(left, right))
    return value if np.isfinite(value) else None


def box_iou(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size != 4 or b.size != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


@dataclass(frozen=True)
class SelectorConfig:
    """Frozen V0 coefficients; do not change during D1/D2 replay."""

    human_anchor_weight: float = 2.0
    trusted_target_weight: float = 1.0
    distractor_penalty_weight: float = 1.0
    motion_iou_weight: float = 0.75
    presence_weight: float = 0.25
    raw_continuity_weight: float = 0.25
    base_support_weight: float = 0.10
    none_score: float = 0.80
    admission_score: float = 1.60
    admission_margin: float = 0.20


@dataclass
class TargetSelectionContext:
    human_anchor: np.ndarray | None
    trusted_features: list[np.ndarray] = field(default_factory=list)
    distractor_features: list[np.ndarray] = field(default_factory=list)
    predicted_box: list[float] | None = None
    previous_raw_sam_id: int | None = None
    previous_native_scope: str | None = None
    frame: int = 0
    event_frame: int = 0
    memory_read: bool = False


class TargetCandidateSelector:
    def __init__(self, config: SelectorConfig | None = None):
        self.config = config or SelectorConfig()

    @staticmethod
    def _base_support(value: float | None) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        # The frozen solver score is an evidence feature, not an authority.
        # Clip only for numerical stability; the raw value remains audited.
        return float(np.clip(value, -10.0, 10.0))

    def score_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        context: TargetSelectionContext,
        base_target_score: float | None,
    ) -> dict[str, Any]:
        feature = _unit(candidate.get("feature"))
        anchor_similarity = _cosine(feature, context.human_anchor)
        target_similarities = [_cosine(feature, value) for value in context.trusted_features]
        target_similarities = [value for value in target_similarities if value is not None]
        distractor_similarities = [_cosine(feature, value) for value in context.distractor_features]
        distractor_similarities = [value for value in distractor_similarities if value is not None]
        human = 0.0 if anchor_similarity is None else float(anchor_similarity)
        trusted = max(target_similarities, default=0.0)
        distractor = max(distractor_similarities, default=0.0)
        motion = box_iou(candidate.get("box_xyxy"), context.predicted_box)
        raw = candidate.get("official_raw_sam_id")
        candidate_scope = candidate.get("native_scope")
        same_scope = (
            context.previous_native_scope is not None
            and candidate_scope is not None
            and str(candidate_scope) == str(context.previous_native_scope)
        )
        raw_continuity = float(
            same_scope
            and raw is not None
            and context.previous_raw_sam_id is not None
            and int(raw) == int(context.previous_raw_sam_id)
        )
        presence = float(np.clip(float(candidate.get("presence_score", candidate.get("confidence", 0.0))), 0.0, 1.0))
        base_support = self._base_support(base_target_score)
        score = (
            self.config.human_anchor_weight * human
            + self.config.trusted_target_weight * trusted
            - self.config.distractor_penalty_weight * distractor
            + self.config.motion_iou_weight * motion
            + self.config.presence_weight * presence
            + self.config.raw_continuity_weight * raw_continuity
            + self.config.base_support_weight * base_support
        )
        return {
            "candidate_uid": str(candidate["candidate_uid"]),
            "candidate_source": str(candidate.get("candidate_source", MAIN_B0_CANDIDATE)),
            "raw_sam_id": None if raw is None else int(raw),
            "human_anchor_similarity": anchor_similarity,
            "trusted_target_similarity": None if not target_similarities else trusted,
            "distractor_similarity": None if not distractor_similarities else distractor,
            "motion_iou": float(motion),
            "raw_continuity": raw_continuity,
            "native_scope_match": same_scope,
            "presence_score": presence,
            "base_target_score": None if base_target_score is None else float(base_target_score),
            "base_support": base_support,
            "selector_score": float(score),
            "runtime_future_gt_used": False,
        }

    def select(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        context: TargetSelectionContext,
        base_target_scores: Mapping[str, float | None],
    ) -> dict[str, Any]:
        if context.frame <= context.event_frame:
            raise ValueError("future selector may only run after the event frame")
        scored = [
            self.score_candidate(
                candidate,
                context=context,
                base_target_score=base_target_scores.get(str(candidate["candidate_uid"])),
            )
            for candidate in candidates
        ]
        ranked = sorted(scored, key=lambda item: (-float(item["selector_score"]), str(item["candidate_uid"])))
        best = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        best_score = None if best is None else float(best["selector_score"])
        second_score = None if second is None else float(second["selector_score"])
        margin = None if best is None else float(best_score - (second_score if second is not None else self.config.none_score))
        choose_candidate = bool(
            best is not None
            and best_score is not None
            and best_score >= self.config.none_score
            and margin is not None
            and margin >= 0.0
        )
        selected_uid = None if not choose_candidate else str(best["candidate_uid"])
        selected_score = None if best is None else best_score
        reliable = bool(
            selected_uid is not None
            and selected_score is not None
            and selected_score >= self.config.admission_score
            and margin is not None
            and margin >= self.config.admission_margin
        )
        return {
            "schema_version": "N72R7_TARGET_CANDIDATE_SELECTION_V1",
            "frame": int(context.frame),
            "event_frame": int(context.event_frame),
            "selected_candidate_uid": selected_uid,
            "selected_score": selected_score,
            "second_candidate_uid": None if second is None else str(second["candidate_uid"]),
            "second_score": second_score,
            "best_minus_second_margin": margin,
            "none_score": float(self.config.none_score),
            "none_selected": selected_uid is None,
            "reliable_for_memory_admission": reliable,
            "ranked_candidates": ranked,
            "candidate_count": len(candidates),
            "memory_read": bool(context.memory_read),
            "event_frame_memory_read": False,
            "runtime_future_gt_used": False,
            "public_id_inference": False,
        }


__all__ = [
    "SelectorConfig",
    "TargetCandidateSelector",
    "TargetSelectionContext",
    "box_iou",
]
