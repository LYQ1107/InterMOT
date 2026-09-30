"""Causal evaluation utilities for N72R19R1."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
import torch

from sam3_intermot.identity_research.metrics import _spatial_index
from sam3_intermot.identity_probe.metrics import TIME_BINS

from .dataset import FrozenIdentityEpisodeDataset
from .encoder import normalize
from .r1_protocol import manifest_key
from .selective import ObservationSelector, build_evidence_batch
from .updater import HIIM

METHOD_SINGLE_ANCHOR = "B0_SINGLE_ANCHOR"
METHOD_EMA = "B1_EMA_0.90"
METHOD_FROZEN_GRU = "B2_FROZEN_GRU"
METHOD_SIMILARITY = "B3_SIMILARITY_THRESHOLD"
METHOD_CORRECTNESS = "B4_CORRECTNESS_SELECTOR"
METHOD_FUTURE_UTILITY = "B5_FUTURE_UTILITY_SELECTOR"
SELECTOR_METHODS = (METHOD_SIMILARITY, METHOD_CORRECTNESS, METHOD_FUTURE_UTILITY)
ALL_METHODS = (METHOD_SINGLE_ANCHOR, METHOD_EMA, METHOD_FROZEN_GRU, *SELECTOR_METHODS)
HORIZONS = (20, 50, 100)


def _safe_percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else None


def _empty_ci(reps: int, seed: int) -> dict[str, Any]:
    return {"reps": reps, "seed": seed, "lower": None, "upper": None, "clusters": 0}


def _sequence_cluster_ci_fast(
    records: list[dict[str, Any]],
    field: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    """Vectorized equivalent of the inherited sequence-cluster bootstrap."""

    if not records:
        return _empty_ci(reps, seed)
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_sequence[str(row["sequence"])].append(float(bool(row[field])) if field == "win" else float(row[field]))
    names = sorted(by_sequence)
    if not names or reps < 1:
        return _empty_ci(reps, seed) | {"clusters": len(names)}
    sums = np.asarray([sum(by_sequence[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(by_sequence[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(names), size=(reps, len(names)))
    numer = sums[selected].sum(axis=1)
    denom = counts[selected].sum(axis=1)
    values = numer / np.maximum(denom, 1.0)
    return {
        "reps": reps,
        "seed": seed,
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
        "clusters": len(names),
    }


def summarize_r1_records(records: list[dict[str, Any]], bootstrap_reps: int = 2000, seed: int = 72191) -> dict[str, Any]:
    """Summarize R1 points and bootstrap only the required H100 endpoint.

    The inherited metric helper bootstraps every horizon and every time bin.
    R1 requires the H100 sequence-cluster interval; keeping the other
    secondary breakdowns point-only makes dev scans and the frozen val run
    substantially cheaper without changing any reported endpoint.
    """

    def summarize(subset: list[dict[str, Any]], label: str, include_ci: bool) -> dict[str, Any]:
        wins = np.asarray([bool(row["win"]) for row in subset], dtype=np.float64)
        margins = np.asarray([float(row["margin"]) for row in subset], dtype=np.float64)
        ranks = np.asarray([int(row["rank"]) for row in subset], dtype=np.float64)
        return {
            "label": label,
            "samples": int(len(subset)),
            "sequences": int(len({str(row["sequence"]) for row in subset})),
            "hard_negative_win_rate": float(wins.mean()) if wins.size else None,
            "hard_negative_win_rate_sequence_cluster_ci95": (
                _sequence_cluster_ci_fast(subset, "win", bootstrap_reps, seed)
                if include_ci
                else _empty_ci(0, seed)
            ),
            "rank_1_accuracy": float(np.mean(ranks == 1)) if ranks.size else None,
            "rank_2_accuracy": float(np.mean(ranks <= 2)) if ranks.size else None,
            "rank_3_accuracy": float(np.mean(ranks <= 3)) if ranks.size else None,
            "mrr": float(np.mean(1.0 / ranks)) if ranks.size else None,
            "mean_margin": float(margins.mean()) if margins.size else None,
            "median_margin": float(np.median(margins)) if margins.size else None,
            "p10_margin": _safe_percentile(margins.tolist(), 10),
            "p25_margin": _safe_percentile(margins.tolist(), 25),
            "p75_margin": _safe_percentile(margins.tolist(), 75),
            "mean_spatial_hard_margin": float(np.mean([float(row["spatial_hard_margin"]) for row in subset])) if subset else None,
            "mean_iou_hard_margin": float(np.mean([float(row["iou_hard_margin"]) for row in subset])) if subset else None,
        }

    horizons = {
        f"H{horizon}": summarize(
            [row for row in records if int(row["gap"]) <= horizon],
            f"cumulative_H{horizon}",
            include_ci=horizon == 100,
        )
        for horizon in HORIZONS
    }
    bins = {
        label: summarize(
            [row for row in records if low <= int(row["gap"]) <= high],
            label,
            include_ci=False,
        )
        for low, high, label in TIME_BINS
    }
    return {
        "metric_version": "N72R19R1-hard-negative-metrics-v1",
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "sample_count": len(records),
        "horizons": horizons,
        "time_gap_bins": bins,
        "tie_policy": "strict positive_score > hard_negative_score; ties are losses for the primary win rate",
        "rank_tie_policy": "rank is one plus the number of competitors with strictly greater score",
        "bootstrap": {"unit": "sequence", "reps": bootstrap_reps, "seed": seed, "ci_computed_for": ["H100"]},
    }


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    sorted_labels = np.asarray(labels, dtype=np.int64)[order]
    ranks = np.arange(1, len(sorted_labels) + 1, dtype=np.float64)
    positive_rank_sum = float(ranks[sorted_labels == 1].sum())
    # ``order`` ranks scores from high to low, so a positive receiving a
    # smaller rank is evidence for the positive class.  The usual ascending
    # rank-sum formula therefore needs its complement here.
    return 1.0 - (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _pr_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    sorted_labels = np.asarray(labels, dtype=np.int64)[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(sorted_labels) + 1)
    recall = cumulative / float(positives)
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous_recall) * precision))


def summarize_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if row["present"]]
    good = [row for row in present if not row["corrupted"]]
    bad = [row for row in present if row["corrupted"]]
    labels = [int(not row["corrupted"]) for row in present]
    scores = [float(row["selection_score"]) for row in present]
    accepted = [bool(row["accepted"]) for row in present]
    tp = sum(label == 1 and value for label, value in zip(labels, accepted))
    fp = sum(label == 0 and value for label, value in zip(labels, accepted))
    fn = sum(label == 1 and not value for label, value in zip(labels, accepted))
    return {
        "present_observations": len(present),
        "good_observation_count": len(good),
        "bad_observation_count": len(bad),
        "good_observation_acceptance_rate": float(np.mean([row["accepted"] for row in good])) if good else None,
        "bad_observation_acceptance_rate": float(np.mean([row["accepted"] for row in bad])) if bad else None,
        "bad_observation_rejection_rate": float(np.mean([not row["accepted"] for row in bad])) if bad else None,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "roc_auc": _roc_auc(scores, labels),
        "pr_auc": _pr_auc(scores, labels),
        "accepted_by_source": {
            source: {
                "count": sum(row["source"] == source for row in present),
                "acceptance_rate": float(np.mean([row["accepted"] for row in present if row["source"] == source]))
                if any(row["source"] == source for row in present)
                else None,
            }
            for source in sorted({str(row["source"]) for row in present})
        },
    }


def summarize_memory_damage(rows: list[dict[str, Any]], catastrophic_threshold: float = 0.05) -> dict[str, Any]:
    damages = [float(row["damage"]) for row in rows if row.get("damage") is not None]
    harmful = [value for value in damages if value > 0.0]
    corrupted = [row for row in rows if row.get("corrupted") and row.get("present")]
    corrupted_harmful = [float(row["damage"]) for row in corrupted if row.get("damage") is not None and float(row["damage"]) > 0.0]
    return {
        "future_comparison_count": len(damages),
        "harmful_update_count": len(harmful),
        "mean_damage": float(np.mean(damages)) if damages else None,
        "median_damage": float(np.median(damages)) if damages else None,
        "p90_damage": _safe_percentile(damages, 90),
        "p95_damage": _safe_percentile(damages, 95),
        "catastrophic_threshold": float(catastrophic_threshold),
        "catastrophic_damage_rate": float(np.mean([value > catastrophic_threshold for value in damages])) if damages else None,
        "corrupted_update_count": len(corrupted),
        "corrupted_harmful_update_count": len(corrupted_harmful),
        "corrupted_mean_damage": float(np.mean(corrupted_harmful)) if corrupted_harmful else 0.0,
    }


def _summarize_scalar(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p90": _safe_percentile(values, 90),
        "p95": _safe_percentile(values, 95),
    }


def summarize_state_drift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize causal write drift and immutable-anchor drift by update class."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("observation_present"):
            group = "missing_observation"
        elif row.get("noise_applied") and row.get("accepted"):
            group = "corrupted_accepted"
        elif row.get("noise_applied"):
            group = "corrupted_rejected"
        elif row.get("accepted"):
            group = "clean_accepted"
        else:
            group = "clean_rejected"
        groups[group].append(row)

    def values(group_rows: list[dict[str, Any]], field: str) -> list[float]:
        return [float(row[field]) for row in group_rows if row.get(field) is not None]

    return {
        group: {
            "state_drift_1_minus_cosine": _summarize_scalar(values(group_rows, "state_change_cosine")),
            "anchor_drift_1_minus_cosine": _summarize_scalar(values(group_rows, "anchor_drift_cosine")),
        }
        for group, group_rows in sorted(groups.items())
    }


def _source_from_condition(condition: dict[str, Any]) -> str:
    if not condition["present"]:
        return "missing"
    if not condition["applied"]:
        return "same_identity"
    if condition["kind"] == "wrong_identity":
        return "wrong_visible_identity"
    if condition["kind"] == "hard_negative":
        return "hard_negative_visible_identity"
    return "missing"


@torch.no_grad()
def evaluate_condition(
    dataset: FrozenIdentityEpisodeDataset,
    manifest: dict[str, Any],
    *,
    method: str,
    condition: str,
    frozen_gru: HIIM,
    selector: ObservationSelector | None = None,
    threshold: float = 0.5,
    device: torch.device | str,
    batch_size: int = 16,
    ema_alpha: float = 0.90,
    bootstrap_reps: int = 2000,
    bootstrap_seed: int = 72191,
    catastrophic_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate one cell with score-before-selection-before-update semantics."""

    if method in (METHOD_CORRECTNESS, METHOD_FUTURE_UTILITY) and selector is None:
        raise ValueError(f"{method} requires an ObservationSelector")
    if method == METHOD_SIMILARITY and selector is not None:
        raise ValueError("similarity threshold baseline does not use a selector")
    device = torch.device(device)
    frozen_gru.eval()
    if selector is not None:
        selector.eval()
    records: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    damage_rows: list[dict[str, Any]] = []

    for indices in dataset.batch_indices(batch_size=batch_size, shuffle=False):
        batch = dataset.make_batch(indices, device=device)
        state = normalize(batch["anchor_embeddings"])
        human_anchor = state.clone()
        targets = batch["target_embeddings"]
        target_mask = batch["target_mask"]
        competitors = batch["competitor_embeddings"]
        competitor_mask = batch["competitor_mask"]
        frames = batch["frames"]
        metadata = batch["metadata"]
        batch_count = state.shape[0]
        anchor_frames = torch.tensor(
            [int(meta.anchor.anchor_frame) for meta in metadata], dtype=torch.long, device=device
        )
        last_update_frames = anchor_frames.clone()
        trusted_counts = torch.zeros(batch_count, dtype=torch.long, device=device)
        recent_stability = torch.zeros(batch_count, dtype=state.dtype, device=device)

        for step in range(target_mask.shape[1]):
            valid = target_mask[:, step]
            if not bool(valid.any()):
                continue
            observations = targets[:, step].clone()
            valid_cpu = valid.detach().cpu().tolist()
            frames_cpu = frames[:, step].detach().cpu().tolist()
            present_values = [False] * batch_count
            condition_rows: list[dict[str, Any] | None] = [None] * batch_count
            for batch_index, meta in enumerate(metadata):
                if not valid_cpu[batch_index]:
                    continue
                frame = int(frames_cpu[batch_index])
                key = manifest_key(meta.anchor.sequence, meta.anchor.track_id, meta.anchor.anchor_frame, frame)
                condition_row = manifest["rows"][key]["conditions"][condition]
                condition_rows[batch_index] = condition_row
                present_values[batch_index] = bool(condition_row["present"])
                replacement_index = condition_row.get("replacement_index")
                if replacement_index is not None:
                    observations[batch_index] = competitors[batch_index, step, int(replacement_index)]
            present = torch.tensor(present_values, dtype=torch.bool, device=device)

            # Primary score is deliberately computed from the state before any update.
            state_before = state.clone()
            positive_scores = (normalize(state_before) * normalize(targets[:, step])).sum(dim=-1)
            negative_scores = torch.einsum("bd,bcd->bc", normalize(state_before), normalize(competitors[:, step]))
            negative_scores = negative_scores.masked_fill(~competitor_mask[:, step], -torch.inf)

            candidate_state = state_before
            selection_scores = torch.ones(batch_count, dtype=state.dtype, device=device)
            accepted = valid & present
            evidence_info: dict[str, torch.Tensor] = {}
            if method in (METHOD_FROZEN_GRU, METHOD_SIMILARITY, METHOD_CORRECTNESS, METHOD_FUTURE_UTILITY):
                features, candidate_state, evidence_info, _ = build_evidence_batch(
                    state_before,
                    human_anchor,
                    observations,
                    competitors[:, step],
                    competitor_mask[:, step],
                    frames=frames[:, step],
                    last_trusted_update_frames=last_update_frames,
                    trusted_update_counts=trusted_counts,
                    recent_state_stability=recent_stability,
                    frozen_gru=frozen_gru,
                )
                if method == METHOD_SIMILARITY:
                    selection_scores = features[:, 0]
                    accepted = accepted & (selection_scores >= threshold)
                elif method in (METHOD_CORRECTNESS, METHOD_FUTURE_UTILITY):
                    selection_scores = selector(features)  # type: ignore[union-attr]
                    accepted = accepted & (selection_scores >= threshold)
                elif method == METHOD_FROZEN_GRU:
                    accepted = valid & present
            elif method == METHOD_EMA:
                selection_scores = torch.ones(batch_count, dtype=state.dtype, device=device)
            elif method == METHOD_SINGLE_ANCHOR:
                accepted = torch.zeros(batch_count, dtype=torch.bool, device=device)
                selection_scores = torch.zeros(batch_count, dtype=state.dtype, device=device)
            else:
                raise ValueError(f"unknown R1 evaluation method: {method}")

            if method == METHOD_EMA:
                candidate_state = normalize(ema_alpha * state_before + (1.0 - ema_alpha) * normalize(observations))
            new_state = torch.where(accepted.unsqueeze(-1), candidate_state, state_before)

            # All per-observation diagnostics are computed in one tensor pass.
            # The old scalar implementation synchronized with the GPU for every
            # row, which made the 25-sequence val sweep needlessly slow.
            state_before_normalized = normalize(state_before)
            new_state_normalized = normalize(new_state)
            state_change_l2_values = torch.linalg.vector_norm(new_state - state_before, dim=-1).detach().cpu().tolist()
            state_change_cosine_values = (
                1.0 - (state_before_normalized * new_state_normalized).sum(dim=-1)
            ).detach().cpu().tolist()
            anchor_drift_values = (
                1.0 - (human_anchor * new_state_normalized).sum(dim=-1)
            ).detach().cpu().tolist()
            if step + 1 < targets.shape[1]:
                future_target_normalized = normalize(targets[:, step + 1])
                damage_values_tensor = (
                    (state_before_normalized * future_target_normalized).sum(dim=-1)
                    - (new_state_normalized * future_target_normalized).sum(dim=-1)
                )
                damage_values = damage_values_tensor.detach().cpu().tolist()
                next_valid_cpu = target_mask[:, step + 1].detach().cpu().tolist()
            else:
                damage_values = [None] * batch_count
                next_valid_cpu = [False] * batch_count
            selection_score_values = selection_scores.detach().cpu().tolist()
            accepted_values = accepted.detach().cpu().tolist()
            positive_score_values = positive_scores.detach().cpu().tolist()
            negative_score_values = negative_scores.detach().cpu()
            competitor_count_values = competitor_mask[:, step].sum(dim=-1).detach().cpu().tolist()
            evidence_values = {
                name: value.detach().cpu().tolist()
                for name, value in evidence_info.items()
            }

            for batch_index, meta in enumerate(metadata):
                if not valid_cpu[batch_index]:
                    continue
                anchor = meta.anchor
                future = anchor.future[step]
                condition_row = condition_rows[batch_index]
                assert condition_row is not None
                comp_count = int(competitor_count_values[batch_index])
                accepted_value = bool(accepted_values[batch_index])
                source = _source_from_condition(condition_row)
                corrupted = bool(condition_row["applied"] and condition_row["present"])
                feature_row = {
                    name: float(values[batch_index])
                    for name, values in evidence_values.items()
                    if name != "assigned_rank"
                }
                if evidence_values:
                    feature_row["assigned_rank"] = int(evidence_values["assigned_rank"][batch_index])
                selection_rows.append(
                    {
                        "method": method,
                        "condition": condition,
                        "sequence": anchor.sequence,
                        "frame": int(future.frame),
                        "present": bool(condition_row["present"]),
                        "corrupted": corrupted,
                        "source": source,
                        "selection_score": float(selection_score_values[batch_index]),
                        "accepted": accepted_value,
                        **feature_row,
                    }
                )

                state_change_l2 = float(state_change_l2_values[batch_index])
                state_change_cosine = float(state_change_cosine_values[batch_index])
                anchor_drift = float(anchor_drift_values[batch_index])
                damage = float(damage_values[batch_index]) if next_valid_cpu[batch_index] else None
                damage_row = {
                    "method": method,
                    "condition": condition,
                    "sequence": anchor.sequence,
                    "frame": int(future.frame),
                    "present": bool(condition_row["present"]),
                    "corrupted": corrupted,
                    "accepted": accepted_value,
                    "damage": damage,
                }
                damage_rows.append(damage_row)
                update_rows.append(
                    {
                        "method": method,
                        "condition": condition,
                        "sequence": anchor.sequence,
                        "track_id": int(anchor.track_id),
                        "anchor_frame": int(anchor.anchor_frame),
                        "frame": int(future.frame),
                        "gap": int(future.frame - anchor.anchor_frame),
                        "observation_present": bool(condition_row["present"]),
                        "noise_applied": bool(condition_row["applied"]),
                        "observation_source": source,
                        "replacement_track_id": condition_row.get("replacement_track_id"),
                        "selection_score": float(selection_score_values[batch_index]),
                        "accepted": accepted_value,
                        "state_change_l2": state_change_l2,
                        "state_change_cosine": state_change_cosine,
                        "anchor_drift_cosine": anchor_drift,
                        "memory_damage": damage,
                    }
                )
                if comp_count:
                    values = negative_score_values[batch_index, :comp_count].tolist()
                    hard_index = max(range(comp_count), key=lambda index: values[index])
                    positive = float(positive_score_values[batch_index])
                    hard_negative = float(values[hard_index])
                    rank = 1 + sum(value > positive + 1e-12 for value in values)
                    spatial_index = _spatial_index(future.target, future.competitors, "nearest_center")
                    iou_index = _spatial_index(future.target, future.competitors, "highest_iou")
                    records.append(
                        {
                            "stage": "N72R19R1",
                            "goal": "Selective Identity Memory Update",
                            "goal_reference": "outputs/N72R19R1/FINAL_GOAL.json",
                            "method": method,
                            "condition": condition,
                            "sequence": anchor.sequence,
                            "track_id": int(anchor.track_id),
                            "anchor_frame": int(anchor.anchor_frame),
                            "frame": int(future.frame),
                            "gap": int(future.frame - anchor.anchor_frame),
                            "positive_score": positive,
                            "hard_negative_score": hard_negative,
                            "margin": positive - hard_negative,
                            "win": bool(positive > hard_negative),
                            "rank": rank,
                            "rank_1": rank == 1,
                            "rank_2": rank <= 2,
                            "rank_3": rank <= 3,
                            "competitor_count": comp_count,
                            "appearance_hard_track_id": int(future.competitors[hard_index].track_id),
                            "spatial_hard_margin": positive - float(values[spatial_index]),
                            "iou_hard_margin": positive - float(values[iou_index]),
                            "noise_applied": bool(condition_row["applied"]),
                            "observation_present": bool(condition_row["present"]),
                            "observation_source": source,
                            "replacement_track_id": condition_row.get("replacement_track_id"),
                            "selection_score": float(selection_score_values[batch_index]),
                            "accepted": accepted_value,
                        }
                    )

            last_update_frames = torch.where(
                accepted,
                frames[:, step],
                last_update_frames,
            )
            trusted_counts = trusted_counts + accepted.to(trusted_counts.dtype)
            state_change_cosine_tensor = torch.as_tensor(
                state_change_cosine_values, dtype=state.dtype, device=device
            )
            recent_stability = torch.where(
                accepted,
                0.8 * recent_stability + 0.2 * state_change_cosine_tensor,
                recent_stability,
            )
            state = new_state

    summary = summarize_r1_records(records, bootstrap_reps=bootstrap_reps, seed=bootstrap_seed)
    summary.update(
        {
            "stage": "N72R19R1",
            "goal": "Selective Identity Memory Update",
            "goal_reference": "outputs/N72R19R1/FINAL_GOAL.json",
            "method": method,
            "condition": condition,
            "score_then_select_then_update": True,
            "observation_encoder": "frozen N72R17 OSNet x1.0 Market1501 512-D cache",
            "online_deployment_claim": False,
            "update_count": len(update_rows),
            "noise_applied_count": sum(bool(row["noise_applied"]) for row in update_rows),
            "missing_count": sum(not bool(row["observation_present"]) for row in update_rows),
            "accepted_update_count": sum(bool(row["accepted"]) for row in update_rows),
            "selection": summarize_selection(selection_rows) if method in SELECTOR_METHODS else None,
            "memory_damage": summarize_memory_damage(damage_rows, catastrophic_threshold),
            "state_drift": summarize_state_drift(update_rows),
        }
    )
    return {
        "records": records,
        "updates": update_rows,
        "selection_rows": selection_rows,
        "damage_rows": damage_rows,
        "summary": summary,
    }


def paired_sequence_bootstrap_delta(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    reps: int = 2000,
    seed: int = 72191,
) -> dict[str, Any]:
    """Paired sequence-cluster bootstrap for method A minus method B."""

    key_fields = ("sequence", "track_id", "anchor_frame", "frame")
    keyed_b = {tuple(row[field] for field in key_fields): row for row in records_b}
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for row in records_a:
        key = tuple(row[field] for field in key_fields)
        other = keyed_b.get(key)
        if other is None:
            continue
        by_sequence[str(row["sequence"])].append(float(bool(row["win"])) - float(bool(other["win"])))
    names = sorted(by_sequence)
    if not names:
        return {"reps": reps, "seed": seed, "sequences": 0, "point_delta": None, "lower": None, "upper": None}
    point_values = [value for values in by_sequence.values() for value in values]
    rng = np.random.default_rng(seed)
    sums = np.asarray([sum(by_sequence[name]) for name in names], dtype=np.float64)
    counts = np.asarray([len(by_sequence[name]) for name in names], dtype=np.float64)
    selected = rng.integers(0, len(names), size=(reps, len(names)))
    bootstrap_values = sums[selected].sum(axis=1) / np.maximum(counts[selected].sum(axis=1), 1.0)
    return {
        "reps": reps,
        "seed": seed,
        "sequences": len(names),
        "paired_samples": len(point_values),
        "point_delta": float(np.mean(point_values)),
        "lower": float(np.percentile(bootstrap_values, 2.5)),
        "upper": float(np.percentile(bootstrap_values, 97.5)),
    }


def paired_sequence_bootstrap_macro_delta(
    records_a_by_condition: dict[str, list[dict[str, Any]]],
    records_b_by_condition: dict[str, list[dict[str, Any]]],
    *,
    reps: int = 2000,
    seed: int = 72191,
) -> dict[str, Any]:
    """Bootstrap the equal-weight macro delta across several conditions.

    Sequence is the resampling unit.  Each bootstrap draw resamples complete
    sequences, computes a delta for every available condition, and then takes
    the unweighted mean over conditions.  This keeps the four-cell primary
    endpoint from being dominated by a cell with more frames.
    """

    key_fields = ("sequence", "track_id", "anchor_frame", "frame")
    by_sequence: dict[str, dict[str, list[float]]] = {}
    by_condition: dict[str, dict[str, list[float]]] = {}
    sequences: set[str] = set()
    cell_deltas: dict[str, float | None] = {}
    for condition, records_a in records_a_by_condition.items():
        keyed_b = {
            tuple(row[field] for field in key_fields): row
            for row in records_b_by_condition.get(condition, [])
        }
        by_sequence[condition] = defaultdict(list)
        for row in records_a:
            key = tuple(row[field] for field in key_fields)
            other = keyed_b.get(key)
            if other is None:
                continue
            sequence = str(row["sequence"])
            sequences.add(sequence)
            by_sequence[condition][sequence].append(
                float(bool(row["win"])) - float(bool(other["win"]))
            )
        values = [value for values in by_sequence[condition].values() for value in values]
        cell_deltas[condition] = float(np.mean(values)) if values else None

    names = sorted(sequences)
    available_conditions = [condition for condition in records_a_by_condition if cell_deltas[condition] is not None]
    if not names or not available_conditions:
        return {
            "reps": reps,
            "seed": seed,
            "sequences": 0,
            "conditions": available_conditions,
            "cell_point_deltas": cell_deltas,
            "point_delta": None,
            "lower": None,
            "upper": None,
        }

    point_delta = float(np.mean([cell_deltas[condition] for condition in available_conditions]))
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(names), size=(reps, len(names)))
    bootstrap_condition_values: list[np.ndarray] = []
    for condition in available_conditions:
        sums = np.asarray(
            [sum(by_sequence[condition].get(name, [])) for name in names], dtype=np.float64
        )
        counts = np.asarray(
            [len(by_sequence[condition].get(name, [])) for name in names], dtype=np.float64
        )
        numer = sums[selected].sum(axis=1)
        denom = counts[selected].sum(axis=1)
        bootstrap_condition_values.append(numer / np.maximum(denom, 1.0))
    bootstrap_values = np.mean(np.stack(bootstrap_condition_values, axis=1), axis=1)
    return {
        "reps": reps,
        "seed": seed,
        "sequences": len(names),
        "conditions": available_conditions,
        "cell_point_deltas": cell_deltas,
        "point_delta": point_delta,
        "lower": float(np.percentile(bootstrap_values, 2.5)),
        "upper": float(np.percentile(bootstrap_values, 97.5)),
    }


__all__ = [
    "ALL_METHODS",
    "METHOD_CORRECTNESS",
    "METHOD_EMA",
    "METHOD_FROZEN_GRU",
    "METHOD_FUTURE_UTILITY",
    "METHOD_SIMILARITY",
    "METHOD_SINGLE_ANCHOR",
    "evaluate_condition",
    "paired_sequence_bootstrap_delta",
    "paired_sequence_bootstrap_macro_delta",
    "summarize_memory_damage",
    "summarize_selection",
    "summarize_state_drift",
]
