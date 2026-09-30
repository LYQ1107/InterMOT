"""Frozen hard-negative evaluation for N72R18 identity memory models."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from sam3_intermot.identity_research.metrics import _spatial_index, summarize_records

from .dataset import FrozenIdentityEpisodeDataset
from .encoder import normalize
from .updater import HIIM

TIME_BINS = ((1, 5, "1_5"), (6, 20, "6_20"), (21, 50, "21_50"), (51, 100, "51_100"))


def _scores(
    state: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    negative_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive scores, padded negative scores, and rank-1 indices."""

    state = normalize(state)
    positive = normalize(positive)
    negatives = normalize(negatives)
    positive_score = (state * positive).sum(dim=-1)
    if negatives.shape[1] == 0:
        empty = torch.empty((state.shape[0], 0), device=state.device, dtype=state.dtype)
        return positive_score, empty, torch.empty((state.shape[0],), dtype=torch.long, device=state.device)
    negative_scores = torch.einsum("bd,bcd->bc", state, negatives)
    masked_scores = negative_scores.masked_fill(~negative_mask, -torch.inf)
    hard_index = masked_scores.argmax(dim=1)
    return positive_score, masked_scores, hard_index


def _record(
    *,
    variant: str,
    anchor,
    future,
    positive_score: float,
    negative_scores: list[float],
    hard_index: int,
    gate_value: float,
    post_update_margin: float,
    history_observations_used: int,
) -> dict[str, object]:
    hard_negative_score = float(negative_scores[hard_index])
    margin = float(positive_score - hard_negative_score)
    rank = 1 + sum(score > positive_score + 1e-12 for score in negative_scores)
    spatial_index = _spatial_index(future.target, future.competitors, "nearest_center")
    iou_index = _spatial_index(future.target, future.competitors, "highest_iou")
    return {
        "stage": "N72R18",
        "memory_variant": variant,
        "memory_label": "LEARNED_IDENTITY_MEMORY_OFFLINE_CAUSAL_DIAGNOSTIC",
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
        "history_observations_used": history_observations_used,
        "observation_source": "same_identity_future_gt_crop",
        "update_rule": "score_then_gru_update" if variant == "GRU_MEMORY" else "score_then_gated_gru_update",
        "gate_value": float(gate_value),
        "update_rejected": bool(gate_value < 0.5) if variant == "GRU_RELIABILITY_GATE" else False,
        "post_update_margin_same_observation": float(post_update_margin),
        "spatial_hard_margin": float(positive_score - negative_scores[spatial_index]),
        "iou_hard_margin": float(positive_score - negative_scores[iou_index]),
    }


def _update_diagnostics(updates: list[dict[str, object]], use_gate: bool) -> dict[str, object]:
    total = len(updates)
    rejected = sum(bool(row["update_rejected"]) for row in updates)
    result: dict[str, object] = {
        "gate_enabled": use_gate,
        "threshold_for_rejection": 0.5,
        "updates": total,
        "rejected_updates": rejected,
        "rejection_rate": rejected / total if total else None,
        "mean_reliability": (
            sum(float(row["gate_value"]) for row in updates) / total if total else None
        ),
    }
    by_gap: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_prior_result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in updates:
        gap = int(row["gap"])
        label = next((label for low, high, label in TIME_BINS if low <= gap <= high), "outside")
        by_gap[label].append(row)
        prior = row.get("pre_update_win")
        by_prior_result["pre_win" if prior is True else "pre_loss" if prior is False else "noncompetitive"].append(row)

    def group(rows: list[dict[str, object]]) -> dict[str, object]:
        count = len(rows)
        reject_count = sum(bool(row["update_rejected"]) for row in rows)
        return {
            "updates": count,
            "mean_reliability": sum(float(row["gate_value"]) for row in rows) / count if count else None,
            "rejected_updates": reject_count,
            "rejection_rate": reject_count / count if count else None,
        }

    result["time_gap_bins"] = {label: group(by_gap[label]) for _, _, label in TIME_BINS}
    result["pre_update_result"] = {label: group(rows) for label, rows in sorted(by_prior_result.items())}
    return result


@torch.no_grad()
def evaluate_model(
    dataset: FrozenIdentityEpisodeDataset,
    model: HIIM,
    device: torch.device | str,
    batch_size: int = 32,
    bootstrap_reps: int = 2000,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Score before each update using the exact inherited hard-negative rule."""

    model.eval()
    records: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    device = torch.device(device)

    for indices in dataset.batch_indices(batch_size=batch_size, shuffle=False):
        batch = dataset.make_batch(indices, device=device)
        state = normalize(batch["anchor_embeddings"])
        target_embeddings = batch["target_embeddings"]
        target_mask = batch["target_mask"]
        competitor_embeddings = batch["competitor_embeddings"]
        competitor_mask = batch["competitor_mask"]
        batch_size_actual, max_steps = target_mask.shape

        for step in range(max_steps):
            valid_targets = target_mask[:, step]
            has_competitors = competitor_mask[:, step].any(dim=1)
            positive_score, negative_scores, hard_indices = _scores(
                state,
                target_embeddings[:, step],
                competitor_embeddings[:, step],
                competitor_mask[:, step],
            )

            new_state, reliability, _ = model(state, target_embeddings[:, step])
            new_state = torch.where(valid_targets.unsqueeze(-1), new_state, state)
            post_positive, post_negative, _ = _scores(
                new_state,
                target_embeddings[:, step],
                competitor_embeddings[:, step],
                competitor_mask[:, step],
            )

            for batch_index in range(batch_size_actual):
                if not bool(valid_targets[batch_index]):
                    continue
                comp_count = int(competitor_mask[batch_index, step].sum().item())
                pre_margin = None
                pre_win = None
                post_margin = None
                if comp_count:
                    pre_values = negative_scores[batch_index, :comp_count].detach().cpu().tolist()
                    hard_index = int(hard_indices[batch_index].item())
                    pre_margin = float(positive_score[batch_index].item() - pre_values[hard_index])
                    pre_win = bool(pre_margin > 0.0)
                    post_values = post_negative[batch_index, :comp_count].detach().cpu().tolist()
                    post_hard = max(post_values)
                    post_margin = float(post_positive[batch_index].item() - post_hard)
                    anchor = dataset.anchors[indices[batch_index]]
                    future = anchor.future[step]
                    records.append(
                        _record(
                            variant=model.variant,
                            anchor=anchor,
                            future=future,
                            positive_score=float(positive_score[batch_index].item()),
                            negative_scores=pre_values,
                            hard_index=hard_index,
                            gate_value=float(reliability[batch_index].item()),
                            post_update_margin=post_margin,
                            history_observations_used=step,
                        )
                    )
                else:
                    post_margin = None

                gap = int(batch["gaps"][batch_index, step].item())
                updates.append(
                    {
                        "stage": "N72R18",
                        "memory_variant": model.variant,
                        "sequence": dataset.anchors[indices[batch_index]].sequence,
                        "track_id": dataset.anchors[indices[batch_index]].track_id,
                        "anchor_frame": dataset.anchors[indices[batch_index]].anchor_frame,
                        "frame": int(batch["frames"][batch_index, step].item()),
                        "gap": gap,
                        "competitor_count": comp_count,
                        "gate_value": float(reliability[batch_index].item()),
                        "update_rejected": bool(float(reliability[batch_index].item()) < 0.5)
                        if model.use_gate
                        else False,
                        "pre_update_margin": pre_margin,
                        "pre_update_win": pre_win,
                        "post_update_margin_same_observation": post_margin,
                    }
                )
            state = new_state

    summary = summarize_records(records, bootstrap_reps=bootstrap_reps, seed=7218)
    summary.update(
        {
            "stage": "N72R18",
            "metric_version": "N72R18-hard-negative-metrics-v1-inherited-N72R16",
            "memory_variant": model.variant,
            "online_deployment_claim": False,
            "oracle_rule": "score precedes the current same-identity GT observation update; this is an offline causal replay, not an online association claim",
            "feature_encoder": "frozen N72R17 OSNet x1.0 Market1501 cache",
            "gate_diagnostics": _update_diagnostics(updates, model.use_gate),
        }
    )
    return records, updates, summary


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


__all__ = ["evaluate_model", "write_jsonl"]
