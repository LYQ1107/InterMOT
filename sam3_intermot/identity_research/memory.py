"""Causal, frozen diagnostic memory baselines for N72R17."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .metrics import _cosine_scores, _spatial_index, summarize_records
from .protocol import iter_anchors

SINGLE_ANCHOR_LABEL = "SINGLE_ANCHOR"
ORACLE_MEMORY_LABEL = "POSTHOC ORACLE POSITIVE MEMORY DIAGNOSTIC"
CAUSAL_GT_LABEL = "OFFLINE CAUSAL GT-OBSERVATION DIAGNOSTIC"
EMA_ALPHAS = (0.90, 0.95, 0.99)
ATTENTION_TEMPERATURE = 0.07


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _mean_memory(vectors: Iterable[np.ndarray]) -> np.ndarray:
    return _normalize(np.mean(np.asarray(list(vectors), dtype=np.float32), axis=0))


def _attention_memory(anchor: np.ndarray, history: list[np.ndarray]) -> np.ndarray:
    vectors = np.asarray([anchor, *history], dtype=np.float32)
    query = _normalize(anchor)
    logits = (vectors @ query) / ATTENTION_TEMPERATURE
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    weights /= max(float(weights.sum()), 1e-12)
    return _normalize(np.sum(vectors * weights[:, None], axis=0))


def _record(
    *,
    variant: str,
    memory_label: str,
    anchor,
    future,
    memory: np.ndarray,
    target_embedding: np.ndarray,
    competitor_embeddings: list[np.ndarray],
    history_observations_used: int,
    update_rule: str,
    alpha: float | None = None,
) -> dict[str, object]:
    scores = _cosine_scores(memory, [target_embedding, *competitor_embeddings])
    positive = float(scores[0])
    hard_index = int(np.argmax(scores[1:]))
    hard_negative = float(scores[1 + hard_index])
    rank = 1 + int(np.sum(scores[1:] > positive + 1e-12))
    spatial_index = _spatial_index(future.target, future.competitors, "nearest_center")
    iou_index = _spatial_index(future.target, future.competitors, "highest_iou")
    return {
        "stage": "N72R17",
        "memory_variant": variant,
        "memory_label": memory_label,
        "sequence": anchor.sequence,
        "track_id": anchor.track_id,
        "anchor_frame": anchor.anchor_frame,
        "frame": future.frame,
        "gap": future.frame - anchor.anchor_frame,
        "positive_score": positive,
        "hard_negative_score": hard_negative,
        "margin": positive - hard_negative,
        "win": bool(positive > hard_negative),
        "rank": rank,
        "rank_1": rank == 1,
        "rank_2": rank <= 2,
        "rank_3": rank <= 3,
        "competitor_count": len(future.competitors),
        "appearance_hard_track_id": future.competitors[hard_index].track_id,
        "history_observations_used": history_observations_used,
        "observation_source": "same_identity_future_gt_crop",
        "update_rule": update_rule,
        "alpha": alpha,
        "spatial_hard_margin": positive - float(scores[1 + spatial_index]),
        "iou_hard_margin": positive - float(scores[1 + iou_index]),
    }


def memory_records(document: dict[str, object], store) -> dict[str, list[dict[str, object]]]:
    """Score before each causal update; future GT observations are diagnostics."""

    variants = ["M1", "M3", "M5", *(f"EMA_{alpha:.2f}" for alpha in EMA_ALPHAS), "ATTENTION_FIXED_ANCHOR_QUERY"]
    records = {variant: [] for variant in variants}
    for anchor in iter_anchors(document):
        anchor_embedding = _normalize(store.get(anchor.sequence, anchor.anchor_frame, anchor.track_id))
        history: list[np.ndarray] = []
        ema = {alpha: anchor_embedding.copy() for alpha in EMA_ALPHAS}
        for future in anchor.future:
            target_embedding = _normalize(store.get(anchor.sequence, future.frame, anchor.track_id))
            if future.competitors:
                competitor_embeddings = [store.get(anchor.sequence, future.frame, box.track_id) for box in future.competitors]
                if len(history) >= 2:
                    records["M3"].append(
                        _record(
                            variant="M3",
                            memory_label=ORACLE_MEMORY_LABEL,
                            anchor=anchor,
                            future=future,
                            memory=_mean_memory([anchor_embedding, *history[:2]]),
                            target_embedding=target_embedding,
                            competitor_embeddings=competitor_embeddings,
                            history_observations_used=2,
                            update_rule="mean(anchor, earliest_two_prior_same_identity_gt_observations)",
                        )
                    )
                if len(history) >= 4:
                    records["M5"].append(
                        _record(
                            variant="M5",
                            memory_label=ORACLE_MEMORY_LABEL,
                            anchor=anchor,
                            future=future,
                            memory=_mean_memory([anchor_embedding, *history[:4]]),
                            target_embedding=target_embedding,
                            competitor_embeddings=competitor_embeddings,
                            history_observations_used=4,
                            update_rule="mean(anchor, earliest_four_prior_same_identity_gt_observations)",
                        )
                    )
                records["M1"].append(
                    _record(
                        variant="M1",
                        memory_label=SINGLE_ANCHOR_LABEL,
                        anchor=anchor,
                        future=future,
                        memory=anchor_embedding,
                        target_embedding=target_embedding,
                        competitor_embeddings=competitor_embeddings,
                        history_observations_used=0,
                        update_rule="frozen_human_anchor",
                    )
                )
                for alpha in EMA_ALPHAS:
                    records[f"EMA_{alpha:.2f}"].append(
                        _record(
                            variant=f"EMA_{alpha:.2f}",
                            memory_label=CAUSAL_GT_LABEL,
                            anchor=anchor,
                            future=future,
                            memory=ema[alpha],
                            target_embedding=target_embedding,
                            competitor_embeddings=competitor_embeddings,
                            history_observations_used=len(history),
                            update_rule="score_then_ema_update_with_current_same_identity_gt_observation",
                            alpha=alpha,
                        )
                    )
                records["ATTENTION_FIXED_ANCHOR_QUERY"].append(
                    _record(
                        variant="ATTENTION_FIXED_ANCHOR_QUERY",
                        memory_label=CAUSAL_GT_LABEL,
                        anchor=anchor,
                        future=future,
                        memory=_attention_memory(anchor_embedding, history),
                        target_embedding=target_embedding,
                        competitor_embeddings=competitor_embeddings,
                        history_observations_used=len(history),
                        update_rule="fixed_anchor_query_softmax_pool_temperature_0.07",
                    )
                )
            for alpha in EMA_ALPHAS:
                ema[alpha] = _normalize(alpha * ema[alpha] + (1.0 - alpha) * target_embedding)
            history.append(target_embedding)
    return records


def summarize_memory(records: dict[str, list[dict[str, object]]], bootstrap_reps: int = 2000, seed: int = 7216) -> dict[str, object]:
    return {
        "stage": "N72R17",
        "memory_metric_version": "N72R17-memory-metrics-v1",
        "online_deployment_claim": False,
        "oracle_rule": "M3/M5 and EMA/attention use same-identity future GT crops in a causal offline replay; score precedes the current observation update and no current crop is included in its own score.",
        "attention_temperature": ATTENTION_TEMPERATURE,
        "variants": {
            variant: summarize_records(rows, bootstrap_reps=bootstrap_reps, seed=seed)
            for variant, rows in records.items()
        },
    }


__all__ = [
    "ATTENTION_TEMPERATURE",
    "EMA_ALPHAS",
    "memory_records",
    "summarize_memory",
]
