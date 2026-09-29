"""Posthoc oracle-memory diagnostics for N72R16."""

from __future__ import annotations

import numpy as np

from .metrics import _cosine_scores, _spatial_index, summarize_records
from .protocol import iter_anchors
from .storage import EmbeddingStore


MEMORY_LABEL = "POSTHOC ORACLE POSITIVE MEMORY DIAGNOSTIC"


def _memory_vector(vectors: list[np.ndarray]) -> np.ndarray:
    vector = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def memory_records(document: dict[str, object], store: EmbeddingStore) -> dict[str, list[dict[str, object]]]:
    records = {"M1": [], "M3": [], "M5": []}
    for anchor in iter_anchors(document):
        anchor_embedding = store.get(anchor.sequence, anchor.anchor_frame, anchor.track_id)
        history: list[np.ndarray] = []
        for future in anchor.future:
            # Scores are computed before appending the current target.  Thus the
            # oracle memory never self-includes the evaluation crop.
            if future.competitors:
                competitor_embeddings = [store.get(anchor.sequence, future.frame, box.track_id) for box in future.competitors]
                target_embedding = store.get(anchor.sequence, future.frame, anchor.track_id)
                for variant, needed in (("M1", 0), ("M3", 2), ("M5", 4)):
                    if len(history) < needed:
                        continue
                    memory = _memory_vector([anchor_embedding, *history[:needed]])
                    scores = _cosine_scores(memory, [target_embedding, *competitor_embeddings])
                    positive = float(scores[0])
                    hard_index = int(np.argmax(scores[1:]))
                    rank = 1 + int(np.sum(scores[1:] > positive + 1e-12))
                    spatial_index = _spatial_index(future.target, future.competitors, "nearest_center")
                    iou_index = _spatial_index(future.target, future.competitors, "highest_iou")
                    records[variant].append(
                        {
                            "memory_variant": variant,
                            "memory_label": MEMORY_LABEL if variant != "M1" else "SINGLE_ANCHOR",
                            "sequence": anchor.sequence,
                            "track_id": anchor.track_id,
                            "anchor_frame": anchor.anchor_frame,
                            "frame": future.frame,
                            "gap": future.frame - anchor.anchor_frame,
                            "positive_score": positive,
                            "hard_negative_score": float(scores[1 + hard_index]),
                            "margin": positive - float(scores[1 + hard_index]),
                            "win": bool(positive > float(scores[1 + hard_index])),
                            "rank": rank,
                            "rank_1": rank == 1,
                            "rank_2": rank <= 2,
                            "rank_3": rank <= 3,
                            "competitor_count": len(future.competitors),
                            "history_observations_used": needed,
                            "spatial_hard_margin": positive - float(scores[1 + spatial_index]),
                            "iou_hard_margin": positive - float(scores[1 + iou_index]),
                        }
                    )
            history.append(store.get(anchor.sequence, future.frame, anchor.track_id))
    return records


def summarize_memory(records: dict[str, list[dict[str, object]]], bootstrap_reps: int = 2000, seed: int = 7216) -> dict[str, object]:
    return {
        "memory_metric_version": "N72R16-memory-metrics-v1",
        "memory_label": MEMORY_LABEL,
        "variants": {
            variant: summarize_records(rows, bootstrap_reps=bootstrap_reps, seed=seed)
            for variant, rows in records.items()
        },
        "online_deployment_claim": False,
        "oracle_rule": "M3 uses anchor plus the earliest two prior same-ID future GT embeddings; M5 uses anchor plus the earliest four; current evaluation crop is never included.",
    }
