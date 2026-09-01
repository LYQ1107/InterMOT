#!/usr/bin/env python3
"""Ranking and correction-response tables for N27's predeclared controls."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from n27_apcr_model import APCRConfig, APCRS, feature_tensors


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"


def rank_metrics(scores: np.ndarray, mask: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    ranks, margins = [], []
    wins = pairs = 0
    present = target < 5
    for row in np.flatnonzero(present):
        valid = np.flatnonzero(mask[row])
        if int(target[row]) not in valid:
            continue
        ordered = sorted(valid.tolist(), key=lambda index: (-float(scores[row, index]), index))
        rank = ordered.index(int(target[row])) + 1
        ranks.append(rank)
        negatives = [index for index in valid if index != target[row]]
        if negatives:
            margins.append(float(scores[row, target[row]] - max(scores[row, index] for index in negatives)))
        for index in negatives:
            pairs += 1
            wins += 1 if scores[row, target[row]] > scores[row, index] else 0.5 if scores[row, target[row]] == scores[row, index] else 0
    rank_array = np.asarray(ranks, dtype=np.float32)
    return {
        "events": len(target), "candidate_present_events": int(present.sum()), "evaluable": len(rank_array),
        "top1": float(np.mean(rank_array == 1)) if len(rank_array) else math.nan,
        "top3": float(np.mean(rank_array <= 3)) if len(rank_array) else math.nan,
        "mrr": float(np.mean(1.0 / rank_array)) if len(rank_array) else math.nan,
        "pair_auc": float(wins / pairs) if pairs else math.nan, "pair_count": pairs,
        "hardest_negative_margin": float(np.mean(margins)) if margins else math.nan,
    }


@torch.no_grad()
def model_scores(model: APCRS, arrays: dict[str, np.ndarray], device: torch.device, mode: str, batch_size: int = 2048) -> np.ndarray:
    output = np.full_like(arrays["b10_score"].astype(np.float32), -1e4)
    needed = {"candidate_mask", "b10_score", "positive_similarity", "negative_similarity", "hard_similarity", "detector_score", "candidate_count", "has_positive", "has_negative", "has_hard", "positive_count", "negative_count", "hard_count", "positive_age", "negative_age", "hard_age"}
    for start in range(0, len(output), batch_size):
        end = min(len(output), start + batch_size)
        batch = {key: torch.from_numpy(arrays[key][start:end]).to(device) for key in needed}
        values = model(feature_tensors(batch), mode=mode)["scores"].float().cpu().numpy()
        output[start:end] = values
    return output


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def response_rows(path: Path, method: str) -> dict[str, Any]:
    arrays = load(path)
    target = arrays["target"].astype(np.int64)
    present = target < 5
    pair = arrays["pair_valid"].astype(bool) & present
    rejected = arrays["rejected_index"].astype(np.int64)
    pair &= (rejected >= 0) & (rejected < 5) & (rejected != target)
    score = arrays["apcr_score"].astype(np.float32)
    cf_score = arrays["cf_apcr_score"].astype(np.float32)
    mask = arrays["candidate_mask"].astype(bool)
    selected = arrays["selected_apcr"].astype(np.int64)
    cf_selected = cf_score.argmax(axis=1)
    if pair.any():
        prob = np.exp(score - np.max(score, axis=1, keepdims=True)) * mask
        prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-9)
        cf_prob = np.exp(cf_score - np.max(cf_score, axis=1, keepdims=True)) * mask
        cf_prob /= np.maximum(cf_prob.sum(axis=1, keepdims=True), 1e-9)
        target_gain = float(np.mean(prob[pair, target[pair]] - cf_prob[pair, target[pair]]))
        rejected_delta = float(np.mean((selected[pair] == rejected[pair]).astype(np.int8) - (cf_selected[pair] == rejected[pair]).astype(np.int8)))
        target_selection_delta = float(np.mean((selected[pair] == target[pair]).astype(np.int8) - (cf_selected[pair] == target[pair]).astype(np.int8)))
    else:
        target_gain = rejected_delta = target_selection_delta = math.nan
    return {"method": method, "dataset_role": path.stem, "pair_events": int(pair.sum()), "target_probability_gain": target_gain, "rejected_selection_delta": rejected_delta, "target_selection_delta": target_selection_delta, "current_apcr_correction_events": int(arrays["apcr_correction_event"].sum()), "candidate_present_top1": float(arrays["apcr_correct"][present].mean()) if present.any() else math.nan}


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "checkpoints/apcr_s_p2_best.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = APCRS(APCRConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = []
    for role, path in (("external_heldout", DATA / "external_heldout_b10_round0.npz"), ("dance_train_real_p2", DATA / "dance_train_real_b10_round0.npz")):
        arrays = load(path)
        for name, mode in (("B10_residual_off", None), ("APCR_positive_only", "positive_only"), ("APCR_explicit_negative_only", "negative_only"), ("APCR_positive_plus_explicit_negative", "both"), ("APCR_ordinary_hard_negative_control", "hard_negative")):
            scores = arrays["b10_score"].astype(np.float32) if mode is None else model_scores(model, arrays, device, mode)
            metrics = rank_metrics(scores, arrays["candidate_mask"], arrays["target"])
            rows.append({"evaluation": "static_precomputed_causal_B10_memory", "role": role, "method": name, "checkpoint": "frozen B10" if mode is None else str(checkpoint_path.relative_to(ROOT)), **metrics, "val25_read": False})
    dynamic_rows = []
    for role, path in (("external_heldout_dynamic", DATA / "apcr_rollout_external_heldout.npz"), ("dance_train_dynamic", DATA / "apcr_rollout_dance_train.npz"), ("historical_cal10_ranking_only_dynamic", DATA / "apcr_rollout_cal10_ranking_only.npz")):
        arrays = load(path)
        for method in ("b10", "apcr"):
            scores = arrays[f"{method}_score"].astype(np.float32)
            metrics = rank_metrics(scores, arrays["candidate_mask"], arrays["target"])
            dynamic_rows.append({"evaluation": "dynamic_causal_rollout", "role": role, "method": method.upper(), "checkpoint": "frozen B10" if method == "b10" else str(checkpoint_path.relative_to(ROOT)), **metrics, "val25_read": False})
    write_csv(OUT / "ablation_results.csv", rows + dynamic_rows)
    responses = [response_rows(DATA / "apcr_rollout_external_heldout.npz", "APCR_dynamic_external_heldout"), response_rows(DATA / "apcr_rollout_dance_train.npz", "APCR_dynamic_dance_train")]
    cal = load(DATA / "apcr_rollout_cal10_ranking_only.npz")
    cal_present = cal["target"] < 5
    responses.append({"method": "APCR_dynamic_historical_cal10", "dataset_role": "historical_cal10", "pair_events": int(cal["pair_valid"].sum()), "target_probability_gain": math.nan, "rejected_selection_delta": math.nan, "target_selection_delta": math.nan, "current_apcr_correction_events": int(cal["apcr_correction_event"].sum()), "candidate_present_top1": float(cal["apcr_correct"][cal_present].mean())})
    write_csv(OUT / "correction_response.csv", responses)
    summary = {"static_ablations": len(rows), "dynamic_policy_rows": len(dynamic_rows), "correction_response_rows": len(responses), "not_run": ["N26_CCSAM_free_absolute_logit_retraining", "new_tracker", "candidate_union", "cal10_safety_threshold_selection", "FULL_LOOP", "TrackEval"], "val25_read": False}
    temporary = (OUT / "ablation_summary.json").with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, OUT / "ablation_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("N27_ABLATIONS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
