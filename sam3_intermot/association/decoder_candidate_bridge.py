"""Bridge official SAM3 decoder outputs into the existing global assignment.

The bridge is deliberately downstream of official ``track_step``.  It does
not make a public-ID decision from a multiplex slot, and it does not run an
independent row-wise argmax.  Decoder candidates and original candidates are
merged first, then one complete target-by-candidate-plus-NONE matrix is
solved by the existing Hungarian helper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from sam3_intermot.association.relational_challenger import (
    AssignmentResult,
    coupled_assignment,
)


@dataclass(frozen=True)
class DecoderCandidate:
    frame_idx: int
    mask_logits: np.ndarray
    mask: np.ndarray
    box_xyxy: tuple[float, float, float, float]
    presence: float
    iou_pred: float
    decoder_token: Optional[np.ndarray]
    clip_feature: Optional[np.ndarray]
    source: str = "sam3_lora_singleton"
    source_public_id: Any = None
    adapter_version: int = 0
    valid: bool = True
    reject_reason: Optional[str] = None


@dataclass(frozen=True)
class DecoderBridgeResult:
    candidates: tuple[DecoderCandidate, ...]
    matrix: np.ndarray
    assignment: AssignmentResult


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _one_mask(value: Any) -> Optional[np.ndarray]:
    array = _to_numpy(value)
    if array is None:
        return None
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    while array.ndim > 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 2:
        raise ValueError(f"expected one mask, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _one_scalar(value: Any, default: float) -> float:
    array = _to_numpy(value)
    if array is None or array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _one_token(output: Mapping[str, Any]) -> Optional[np.ndarray]:
    for key in ("decoder_token", "sam_output_token", "obj_ptr", "sam_tokens_out"):
        value = _to_numpy(output.get(key))
        if value is not None and value.size:
            while value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            while value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            return value.astype(np.float32, copy=True).reshape(-1)
    return None


def _mask_box(mask: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def official_output_to_decoder_candidate(
    output: Mapping[str, Any],
    *,
    frame_idx: int,
    source_public_id: Any,
    adapter_version: int = 0,
    clip_feature: Optional[np.ndarray] = None,
    mask_threshold: float = 0.0,
    min_presence: float = 0.05,
    source: str = "sam3_lora_singleton",
    keep_rejected: bool = False,
) -> Optional[DecoderCandidate]:
    """Convert one official SAM output into a candidate.

    Official SAM mask thresholding is applied only here, after the
    differentiable decoder update.  Empty/low-presence outputs are represented
    as rejected candidates (or omitted) and are therefore available to the
    target's explicit NONE option rather than being forced into an identity.
    """

    if source_public_id is None:
        raise ValueError("source_public_id is provenance only but must be explicit")
    logits = _one_mask(
        output.get("high_res_masks", output.get("low_res_masks", output.get("masks")))
    )
    if logits is None:
        raise ValueError("official output has no mask logits")
    binary = logits > float(mask_threshold)
    presence_logit = _one_scalar(output.get("object_score_logits"), 0.0)
    presence = float(1.0 / (1.0 + np.exp(-np.clip(presence_logit, -60.0, 60.0))))
    iou_pred = _one_scalar(output.get("iou_pred", output.get("ious")), 0.0)
    reason = None
    box = _mask_box(binary)
    if box is None:
        reason = "empty_official_mask"
    elif presence < min_presence:
        reason = "low_official_presence"
    candidate = DecoderCandidate(
        frame_idx=int(frame_idx),
        mask_logits=logits.copy(),
        mask=binary.astype(bool, copy=True),
        box_xyxy=box or (0.0, 0.0, 0.0, 0.0),
        presence=presence,
        iou_pred=iou_pred,
        decoder_token=_one_token(output),
        clip_feature=None if clip_feature is None else np.asarray(clip_feature).copy(),
        source=source,
        source_public_id=source_public_id,
        adapter_version=int(adapter_version),
        valid=reason is None,
        reject_reason=reason,
    )
    if reason is not None and not keep_rejected:
        return None
    return candidate


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def merge_decoder_candidates(
    original_candidates: Iterable[DecoderCandidate],
    decoder_candidates: Iterable[DecoderCandidate],
    *,
    dedupe_iou: float = 0.95,
) -> tuple[DecoderCandidate, ...]:
    """Merge candidate sources while retaining source provenance."""

    merged = [candidate for candidate in original_candidates if candidate.valid]
    for candidate in decoder_candidates:
        if not candidate.valid:
            continue
        duplicate = None
        for index, previous in enumerate(merged):
            # A decoder candidate may replace an equivalent candidate for the
            # same source identity.  Different explicit source identities are
            # never collapsed solely because their boxes overlap.
            same_source_id = (
                candidate.source_public_id is not None
                and candidate.source_public_id == previous.source_public_id
            )
            if same_source_id and box_iou(candidate.box_xyxy, previous.box_xyxy) >= dedupe_iou:
                duplicate = index
                break
        if duplicate is None:
            merged.append(candidate)
        else:
            previous = merged[duplicate]
            merged[duplicate] = replace(
                candidate,
                source=f"{previous.source}+{candidate.source}",
                clip_feature=(
                    candidate.clip_feature
                    if candidate.clip_feature is not None
                    else previous.clip_feature
                ),
            )
    return tuple(merged)


ScoreDeltaFn = Callable[[Any, DecoderCandidate], float]


def build_decoder_assignment(
    anchor_scores: np.ndarray,
    candidates: Sequence[DecoderCandidate],
    identity_ids: Sequence[Any],
    *,
    score_delta_fn: Optional[ScoreDeltaFn] = None,
    delta_scores: Optional[np.ndarray] = None,
    none_scores: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
) -> DecoderBridgeResult:
    """Build and solve one full matrix after candidate bridging."""

    anchor = np.asarray(anchor_scores, dtype=np.float64)
    if anchor.ndim != 2:
        raise ValueError("anchor_scores must be [identity, candidate]")
    if anchor.shape != (len(identity_ids), len(candidates)):
        raise ValueError("anchor_scores does not match identities/candidates")
    if delta_scores is None:
        delta = np.zeros_like(anchor)
        if score_delta_fn is not None:
            for row, identity_id in enumerate(identity_ids):
                for column, candidate in enumerate(candidates):
                    delta[row, column] = float(score_delta_fn(identity_id, candidate))
    else:
        delta = np.asarray(delta_scores, dtype=np.float64)
        if delta.shape != anchor.shape:
            raise ValueError("delta_scores does not match anchor_scores")
    result = coupled_assignment(
        anchor,
        delta,
        none_scores=none_scores,
        candidate_mask=candidate_mask,
    )
    return DecoderBridgeResult(tuple(candidates), result.matrix, result)


class DecoderCandidateBridge:
    """State-free bridge object used by the live and replay runners."""

    def __init__(self, *, dedupe_iou: float = 0.95):
        self.dedupe_iou = float(dedupe_iou)

    def merge(
        self,
        original_candidates: Iterable[DecoderCandidate],
        decoder_candidates: Iterable[DecoderCandidate],
    ) -> tuple[DecoderCandidate, ...]:
        return merge_decoder_candidates(
            original_candidates,
            decoder_candidates,
            dedupe_iou=self.dedupe_iou,
        )

    def assign(
        self,
        anchor_scores: np.ndarray,
        candidates: Sequence[DecoderCandidate],
        identity_ids: Sequence[Any],
        **kwargs: Any,
    ) -> DecoderBridgeResult:
        return build_decoder_assignment(
            anchor_scores,
            candidates,
            identity_ids,
            **kwargs,
        )

