"""Causal N72R19 evaluation for EMA and learned memory variants."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from sam3_intermot.identity_research.metrics import _spatial_index, summarize_records

from .dataset import FrozenIdentityEpisodeDataset
from .encoder import normalize
from .noise import NoiseSpec, corrupt_observation
from .updater import HIIM

METHOD_EMA = "EMA_0.90"
METHOD_GRU = "GRU_MEMORY_N72R18"
METHOD_ROBUST = "GRU_RELIABILITY_N72R19"


def _episode_key(meta: Any, frame: int) -> str:
    anchor = meta.anchor
    return f"{anchor.sequence}|{anchor.track_id}|{anchor.anchor_frame}|{frame}"


def _scores(
    state: torch.Tensor,
    target: torch.Tensor,
    competitors: torch.Tensor,
    competitor_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = normalize(state)
    target = normalize(target)
    competitors = normalize(competitors)
    positive = (state * target).sum(dim=-1)
    scores = torch.einsum("bd,bcd->bc", state, competitors)
    scores = scores.masked_fill(~competitor_mask, -torch.inf)
    return positive, scores, scores.argmax(dim=-1)


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else None


def _drift_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    def group(items: list[dict[str, object]]) -> dict[str, object]:
        changes = [float(row["state_change_l2"]) for row in items]
        cosine = [float(row["state_change_cosine"]) for row in items]
        anchor = [float(row["anchor_drift_cosine"]) for row in items]
        return {
            "updates": len(items),
            "mean_state_change_l2": float(np.mean(changes)) if changes else None,
            "median_state_change_l2": float(np.median(changes)) if changes else None,
            "p95_state_change_l2": _percentile(changes, 95),
            "mean_state_change_cosine": float(np.mean(cosine)) if cosine else None,
            "mean_anchor_drift_cosine": float(np.mean(anchor)) if anchor else None,
        }

    return {
        "all_updates": group(rows),
        "observation_corrupted": group([row for row in rows if bool(row["noise_applied"])]),
        "missing": group([row for row in rows if not bool(row["observation_present"])]),
        "present_corrupted": group(
            [row for row in rows if bool(row["noise_applied"]) and bool(row["observation_present"])]
        ),
        "unmodified_or_clean": group([row for row in rows if not bool(row["noise_applied"])]),
    }


def _record(
    *,
    method: str,
    noise: NoiseSpec,
    anchor: Any,
    future: Any,
    positive_score: float,
    negative_scores: list[float],
    hard_index: int,
    corruption: dict[str, object],
    reliability: float | None,
    drift: dict[str, float],
) -> dict[str, object]:
    hard_negative_score = float(negative_scores[hard_index])
    margin = float(positive_score - hard_negative_score)
    rank = 1 + sum(score > positive_score + 1e-12 for score in negative_scores)
    spatial_index = _spatial_index(future.target, future.competitors, "nearest_center")
    iou_index = _spatial_index(future.target, future.competitors, "highest_iou")
    replacement_index = corruption.get("replacement_index")
    replacement_track_id = (
        future.competitors[int(replacement_index)].track_id
        if replacement_index is not None
        else None
    )
    return {
        "stage": "N72R19",
        "goal": "Robust Human Identity Memory Learning",
        "goal_reference": "outputs/N72R19/FINAL_GOAL.json",
        "method": method,
        "noise_type": noise.kind,
        "noise_rate": float(noise.rate),
        "sequence": anchor.sequence,
        "track_id": anchor.track_id,
        "anchor_frame": anchor.anchor_frame,
        "frame": future.frame,
        "gap": future.frame - anchor.anchor_frame,
        "positive_score": float(positive_score),
        "hard_negative_score": hard_negative_score,
        "margin": margin,
        "win": bool(positive_score > hard_negative_score),
        "rank": rank,
        "rank_1": rank == 1,
        "rank_2": rank <= 2,
        "rank_3": rank <= 3,
        "competitor_count": len(future.competitors),
        "appearance_hard_track_id": future.competitors[hard_index].track_id,
        "noise_applied": bool(corruption["applied"]),
        "observation_present": bool(corruption["present"]),
        "observation_source": corruption["source"],
        "replacement_competitor_index": replacement_index,
        "replacement_track_id": replacement_track_id,
        "reliability": reliability,
        "state_change_l2": drift["state_change_l2"],
        "state_change_cosine": drift["state_change_cosine"],
        "anchor_drift_cosine": drift["anchor_drift_cosine"],
        "spatial_hard_margin": float(positive_score - negative_scores[spatial_index]),
        "iou_hard_margin": float(positive_score - negative_scores[iou_index]),
    }


@torch.no_grad()
def evaluate_condition(
    dataset: FrozenIdentityEpisodeDataset,
    method: str,
    noise: NoiseSpec,
    device: torch.device | str,
    model: HIIM | None = None,
    batch_size: int = 32,
    bootstrap_reps: int = 2000,
    bootstrap_seed: int = 7219,
    ema_alpha: float = 0.90,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Evaluate one method/condition with score-before-update semantics."""

    if method == METHOD_EMA and model is not None:
        raise ValueError("EMA must not receive a learned model")
    if method != METHOD_EMA and model is None:
        raise ValueError("learned method requires a model")
    if method != METHOD_EMA:
        model.eval()
    device = torch.device(device)
    records: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []

    for indices in dataset.batch_indices(batch_size=batch_size, shuffle=False):
        batch = dataset.make_batch(indices, device=device)
        state = normalize(batch["anchor_embeddings"])
        anchor_embeddings = batch["anchor_embeddings"]
        targets = batch["target_embeddings"]
        target_mask = batch["target_mask"]
        competitors = batch["competitor_embeddings"]
        competitor_mask = batch["competitor_mask"]
        metadata = batch["metadata"]
        frames = batch["frames"]
        batch_size_actual, max_steps = target_mask.shape

        for step in range(max_steps):
            valid = target_mask[:, step]
            positive, negative_scores, hard_indices = _scores(
                state,
                targets[:, step],
                competitors[:, step],
                competitor_mask[:, step],
            )

            observations = targets[:, step].clone()
            present = torch.zeros(valid.shape, dtype=torch.bool, device=device)
            corruption_details: list[dict[str, object]] = []
            for batch_index, meta in enumerate(metadata):
                if not bool(valid[batch_index]):
                    corruption_details.append(
                        {"applied": False, "present": False, "replacement_index": None, "source": "padding"}
                    )
                    continue
                frame = int(frames[batch_index, step].item())
                key = _episode_key(meta, frame)
                corruption = corrupt_observation(
                    state[batch_index],
                    targets[batch_index, step],
                    competitors[batch_index, step],
                    competitor_mask[batch_index, step],
                    noise,
                    key,
                    step,
                )
                observations[batch_index] = corruption.observation
                present[batch_index] = corruption.present
                corruption_details.append(
                    {
                        "applied": corruption.applied,
                        "present": corruption.present,
                        "replacement_index": corruption.replacement_index,
                        "source": corruption.source,
                    }
                )

            update_mask = valid & present
            if method == METHOD_EMA:
                candidate_state = normalize(ema_alpha * state + (1.0 - ema_alpha) * observations)
                reliability_tensor = None
            else:
                candidate_state, reliability_tensor, _ = model(state, observations)
            new_state = torch.where(update_mask.unsqueeze(-1), candidate_state, state)

            for batch_index in range(batch_size_actual):
                if not bool(valid[batch_index]):
                    continue
                anchor = metadata[batch_index].anchor
                future = anchor.future[step]
                comp_count = int(competitor_mask[batch_index, step].sum().item())
                previous = state[batch_index]
                current = new_state[batch_index]
                state_change_l2 = float(torch.linalg.vector_norm(current - previous).item())
                state_change_cosine = float(1.0 - torch.dot(previous, current).item())
                anchor_drift_cosine = float(1.0 - torch.dot(anchor_embeddings[batch_index], current).item())
                drift = {
                    "state_change_l2": state_change_l2,
                    "state_change_cosine": state_change_cosine,
                    "anchor_drift_cosine": anchor_drift_cosine,
                }
                details = corruption_details[batch_index]
                reliability = (
                    float(reliability_tensor[batch_index].item())
                    if reliability_tensor is not None and bool(update_mask[batch_index])
                    else None
                )
                update_row = {
                    "stage": "N72R19",
                    "goal_reference": "outputs/N72R19/FINAL_GOAL.json",
                    "method": method,
                    "noise_type": noise.kind,
                    "noise_rate": float(noise.rate),
                    "sequence": anchor.sequence,
                    "track_id": anchor.track_id,
                    "anchor_frame": anchor.anchor_frame,
                    "frame": future.frame,
                    "gap": future.frame - anchor.anchor_frame,
                    "noise_applied": bool(details["applied"]),
                    "observation_present": bool(details["present"]),
                    "observation_source": details["source"],
                    "replacement_competitor_index": details["replacement_index"],
                    "replacement_track_id": (
                        future.competitors[int(details["replacement_index"])].track_id
                        if details["replacement_index"] is not None
                        else None
                    ),
                    "reliability": reliability,
                    **drift,
                }
                updates.append(update_row)

                if comp_count:
                    values = negative_scores[batch_index, :comp_count].cpu().tolist()
                    hard_index = int(hard_indices[batch_index].item())
                    records.append(
                        _record(
                            method=method,
                            noise=noise,
                            anchor=anchor,
                            future=future,
                            positive_score=float(positive[batch_index].item()),
                            negative_scores=values,
                            hard_index=hard_index,
                            corruption=details,
                            reliability=reliability,
                            drift=drift,
                        )
                    )
            state = new_state

    summary = summarize_records(records, bootstrap_reps=bootstrap_reps, seed=bootstrap_seed)
    summary.update(
        {
            "stage": "N72R19",
            "goal": "Robust Human Identity Memory Learning",
            "goal_reference": "outputs/N72R19/FINAL_GOAL.json",
            "method": method,
            "noise_type": noise.kind,
            "noise_rate": float(noise.rate),
            "metric_version": "N72R19-hard-negative-metrics-v1",
            "score_then_update": True,
            "observation_encoder": "frozen N72R17 OSNet x1.0 Market1501 512-D cache",
            "online_deployment_claim": False,
            "update_count": len(updates),
            "noise_applied_count": sum(bool(row["noise_applied"]) for row in updates),
            "missing_count": sum(not bool(row["observation_present"]) for row in updates),
        }
    )
    return records, updates, summary, _drift_summary(updates)


__all__ = [
    "METHOD_EMA",
    "METHOD_GRU",
    "METHOD_ROBUST",
    "evaluate_condition",
]
