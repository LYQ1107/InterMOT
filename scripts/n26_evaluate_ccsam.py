#!/usr/bin/env python3
"""Strict cal10 evaluation, ablations, counterfactuals and N26-B gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(".")
OUT = ROOT / "outputs/n26"
DENSE = OUT / "dense_dataset"
MAX_K = 5
NONE = 5
SEED = 26
sys.path.insert(0, str(ROOT / "scripts"))
from n26_ccsam_model import CCSAM, CCSAMConfig  # noqa: E402


MODEL_MODES = {
    "CCSAM_MEMORY_OFF": ("off", False),
    "CCSAM_POSITIVE_ONLY": ("positive", False),
    "CCSAM_EXPLICIT_NEGATIVE_ONLY": ("negative", False),
    "CCSAM_POSITIVE_AND_NEGATIVE": ("positive_negative", False),
    "CCSAM_HARD_NEGATIVE_CONTROL": ("hard_negative", False),
    "CCSAM_NO_NONE_EXISTENCE": ("positive_negative", True),
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40, 40)
    return 1.0 / (1.0 + np.exp(-value))


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.maximum(1e-12, exponential.sum(axis=1, keepdims=True))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@torch.no_grad()
def infer(
    model: CCSAM,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    memory_mode: str,
    disable_none: bool = False,
    memory_mask_name: str = "memory_mask",
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    output = {
        "logits": np.empty((len(indices), 6), dtype=np.float32),
        "candidate_logits": np.empty((len(indices), 5), dtype=np.float32),
        "risk_logits": np.empty((len(indices), 5), dtype=np.float32),
        "existence_logit": np.empty(len(indices), dtype=np.float32),
    }
    for start in range(0, len(indices), batch_size):
        state = indices[start : start + batch_size]
        parent = arrays["parent"][state].astype(int)
        kwargs = {
            "candidate_clip": torch.from_numpy(arrays["candidate_clip"][state]).to(device),
            "candidate_scalar": torch.from_numpy(arrays["candidate_scalar"][state]).to(device),
            "candidate_mask": torch.from_numpy(arrays["candidate_mask"][state]).to(device),
            "memory_clip": torch.from_numpy(arrays["memory_clip"][parent]).to(device),
            "memory_meta": torch.from_numpy(arrays["memory_meta"][parent]).to(device),
            "memory_mask": torch.from_numpy(arrays[memory_mask_name][parent]).to(device),
            "memory_kind": torch.from_numpy(arrays["memory_kind"][parent]).to(device),
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(**kwargs, memory_mode=memory_mode, disable_none=disable_none)
        end = start + len(state)
        for name in output:
            output[name][start:end] = values[name].float().cpu().numpy()
    return output


def ranking_metrics(scores: np.ndarray, candidate_mask: np.ndarray, target: np.ndarray, population: np.ndarray) -> dict[str, Any]:
    ranks: list[int] = []
    margins: list[float] = []
    wins = pairs = 0.0
    positive_present = int(np.sum(population & (target < MAX_K)))
    for index in np.flatnonzero(population & (target < MAX_K)):
        truth = int(target[index])
        valid = np.flatnonzero(candidate_mask[index] & np.isfinite(scores[index]) & (scores[index] > -9999))
        if truth not in valid:
            continue
        ordered = sorted(valid, key=lambda candidate: (-float(scores[index, candidate]), int(candidate)))
        rank = ordered.index(truth) + 1
        ranks.append(rank)
        negatives = [candidate for candidate in valid if candidate != truth]
        if negatives:
            margins.append(float(scores[index, truth] - max(scores[index, candidate] for candidate in negatives)))
        for candidate in negatives:
            pairs += 1
            wins += 1.0 if scores[index, truth] > scores[index, candidate] else 0.5 if scores[index, truth] == scores[index, candidate] else 0.0
    rank_array = np.asarray(ranks, dtype=np.int32)
    return {
        "rank_population_parents": int(population.sum()),
        "candidate_positive_present": positive_present,
        "positive_evaluable": len(ranks),
        "top1": float(np.mean(rank_array == 1)) if len(rank_array) else math.nan,
        "top3": float(np.mean(rank_array <= 3)) if len(rank_array) else math.nan,
        "mrr": float(np.mean(1.0 / rank_array)) if len(rank_array) else math.nan,
        "pair_auc": float(wins / pairs) if pairs else math.nan,
        "pair_count": int(pairs),
        "hardest_negative_margin": float(np.mean(margins)) if margins else math.nan,
    }


def existence_metrics(logit: np.ndarray, target: np.ndarray) -> dict[str, float]:
    truth = target < MAX_K
    prediction = sigmoid(logit) >= 0.5
    return {
        "existence_accuracy": float(np.mean(prediction == truth)),
        "existence_recall": float(np.mean(prediction[truth])) if truth.any() else math.nan,
        "existence_specificity": float(np.mean(~prediction[~truth])) if (~truth).any() else math.nan,
    }


def operation(prediction: np.ndarray, accepted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    commits = accepted & (prediction < MAX_K)
    correct = commits & (prediction == target)
    absent = target == NONE
    count = int(commits.sum())
    return {
        "events": len(target), "commits": count, "correct_commits": int(correct.sum()),
        "false_commits": int(count - correct.sum()),
        "commit_precision": float(correct.sum() / count) if count else math.nan,
        "coverage": float(count / len(target)),
        "candidate_set_absent_events": int(absent.sum()),
        "candidate_set_absent_false_accept": float((commits & absent).sum() / max(1, absent.sum())),
    }


def select_threshold(score: np.ndarray, prediction: np.ndarray, target: np.ndarray, calibration: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidate = calibration & (prediction < MAX_K)
    values = np.unique(score[candidate])
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in np.concatenate(([math.inf], values[::-1])):
        accepted = calibration & (prediction < MAX_K) & (score >= threshold)
        metrics = operation(prediction[calibration], accepted[calibration], target[calibration])
        precision = metrics["commit_precision"]
        if metrics["commits"] and precision >= 0.90 and metrics["candidate_set_absent_false_accept"] <= 0.0726:
            key = (metrics["coverage"], precision, float(threshold))
            if best is None or key > best[0]:
                best = (key, float(threshold), metrics)
    if best is None:
        return math.inf, {"status": "NO_NONEMPTY_FEASIBLE_THRESHOLD", **operation(prediction[calibration], np.zeros(int(calibration.sum()), dtype=bool), target[calibration])}
    return best[1], {"status": "FEASIBLE_ON_OTHER_SEQUENCES", **best[2]}


def effort_metrics(prediction: np.ndarray, target: np.ndarray, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray, dict[tuple[str, int], int]]:
    prior = np.zeros(len(rows), dtype=bool)
    corrected: set[tuple[str, int]] = set()
    correction_frames: dict[tuple[str, int], list[int]] = defaultdict(list)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    errors = prediction != target
    for index, row in enumerate(rows):
        key = (str(row["sequence"]), int(row["public_identity_id"]))
        prior[index] = key in corrected
        if errors[index]:
            corrected.add(key)
            counts[key] += 1
            correction_frames[key].append(int(row["frame"]))
    targets = {(str(row["sequence"]), int(row["public_identity_id"])) for row in rows}
    intervals = [b - a for frames in correction_frames.values() for a, b in zip(frames, frames[1:])]
    eligible = prior
    return {
        "parent_accuracy": float(np.mean(~errors)),
        "corrections": int(errors.sum()),
        "targets": len(targets),
        "corrections_per_target": float(errors.sum() / max(1, len(targets))),
        "re_correction_eligible_events": int(eligible.sum()),
        "re_correction_rate": float(errors[eligible].mean()) if eligible.any() else math.nan,
        "mean_time_to_next_correction_frames": float(np.mean(intervals)) if intervals else math.nan,
        "median_time_to_next_correction_frames": float(np.median(intervals)) if intervals else math.nan,
        "none_prediction_rate": float(np.mean(prediction == NONE)),
        "raw_candidate_absent_false_accept": float(np.mean(prediction[target == NONE] < MAX_K)) if np.any(target == NONE) else math.nan,
    }, prior, counts


def budget_rows(method: str, counts: dict[tuple[str, int], int], events: int, targets: int) -> list[dict[str, Any]]:
    result = []
    values = list(counts.values()) + [0] * max(0, targets - len(counts))
    total_corrections = sum(values)
    for budget in (0, 1, 2, 4, 8):
        served = sum(min(value, budget) for value in values)
        unserved = total_corrections - served
        result.append({
            "method": method, "budget_per_target": budget, "targets": targets,
            "total_model_errors": total_corrections, "served_corrections": served,
            "unserved_errors": unserved, "targets_within_budget_fraction": float(np.mean(np.asarray(values) <= budget)),
            "offline_parent_decision_accuracy_after_human_budget": float(1.0 - unserved / max(1, events)),
            "interpretation": "offline parent-event proxy; not TrackEval",
        })
    return result


def b10_scores(arrays: dict[str, np.ndarray], canonical: np.ndarray) -> np.ndarray:
    output = np.full((len(canonical), MAX_K), -1e4, dtype=np.float32)
    for out_index, state in enumerate(canonical):
        parent = int(arrays["parent"][state])
        valid = np.flatnonzero(arrays["candidate_mask"][state])
        candidate = arrays["candidate_clip"][state].astype(np.float32)
        memory = arrays["memory_clip"][parent].astype(np.float32)
        kind = arrays["memory_kind"][parent]
        mask = arrays["memory_mask"][parent]
        positive = memory[mask & ((kind == 0) | (kind == 1))]
        negative = memory[mask & (kind == 2)]
        for index in valid:
            pos = float(np.max(positive @ candidate[index]))
            penalty = max(0.0, float(np.max(negative @ candidate[index])) - pos + 0.02) if len(negative) else 0.0
            output[out_index, index] = pos - 0.8 * penalty
    return output


def bootstrap(
    sequences: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    accepted: np.ndarray,
    rank_correct: np.ndarray,
    rank_eligible: np.ndarray,
    pair_mask: np.ndarray,
    rejected_delta: np.ndarray,
    error_delta: np.ndarray,
    recorrection_mask: np.ndarray,
    recorrection_delta: np.ndarray,
    replicates: int = 2000,
) -> dict[str, Any]:
    names = sorted(set(sequences.tolist()))
    indices = {name: np.flatnonzero(sequences == name) for name in names}
    rng = np.random.default_rng(SEED)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = rng.choice(names, len(names), replace=True)
        block = np.concatenate([indices[str(name)] for name in sampled])
        metrics = operation(prediction[block], accepted[block], target[block])
        for name in ("commit_precision", "coverage", "candidate_set_absent_false_accept"):
            values[name].append(metrics[name])
        eligible = rank_eligible[block]
        values["h5_top1"].append(float(rank_correct[block][eligible].mean()) if eligible.any() else math.nan)
        pair = pair_mask[block]
        values["rejected_selection_delta"].append(float(rejected_delta[block][pair].mean()) if pair.any() else math.nan)
        values["counterfactual_error_delta"].append(float(error_delta[block][pair].mean()) if pair.any() else math.nan)
        re = recorrection_mask[block]
        values["memory_recorrection_delta"].append(float(recorrection_delta[block][re].mean()) if re.any() else math.nan)
    output: dict[str, Any] = {"unit": "sequence", "replicates": replicates, "seed": SEED}
    for name, block in values.items():
        array = np.asarray(block, dtype=np.float64)
        array = array[np.isfinite(array)]
        output[name] = {
            "mean": float(array.mean()) if len(array) else math.nan,
            "ci95_low": float(np.quantile(array, 0.025)) if len(array) else math.nan,
            "ci95_high": float(np.quantile(array, 0.975)) if len(array) else math.nan,
            "valid_replicates": len(array),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "checkpoints/n26_round1_final.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    evaluation_protocol = json.loads((OUT / "evaluation_protocol.json").read_text(encoding="utf-8"))
    if evaluation_protocol["val25_read"]:
        raise RuntimeError("evaluation protocol unexpectedly authorizes val25")

    with np.load(DENSE / "round1_cal10.npz", allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    with np.load(DENSE / "round0_cal10.npz", allow_pickle=False) as z:
        round0 = {name: z[name].copy() for name in z.files}
    parent_rows = [json.loads(line) for line in (DENSE / "round1_cal10_parents.jsonl").open(encoding="utf-8") if line.strip()]
    round0_rows = [json.loads(line) for line in (DENSE / "round0_cal10_parents.jsonl").open(encoding="utf-8") if line.strip()]
    canonical = np.asarray([int(row["canonical_state_index"]) for row in parent_rows], dtype=np.int64)
    if len(canonical) != 1700 or len(set(canonical.tolist())) != 1700:
        raise RuntimeError("canonical parent mapping is incomplete")
    target = arrays["target"][canonical].astype(int)
    candidate_mask = arrays["candidate_mask"][canonical]
    sequences = np.asarray([row["sequence"] for row in parent_rows])
    materialized = np.asarray([not bool(row["extra_target_present_zero_attempt"]) for row in parent_rows])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CCSAM(CCSAMConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    mode_outputs: dict[str, dict[str, np.ndarray]] = {}
    for method, (mode, disable_none) in MODEL_MODES.items():
        print(f"INFER {method}", flush=True)
        mode_outputs[method] = infer(model, arrays, canonical, device, mode, disable_none)
    mode_outputs["CCSAM_NO_COMMIT_RISK"] = {name: value.copy() for name, value in mode_outputs["CCSAM_POSITIVE_AND_NEGATIVE"].items()}
    round0_checkpoint_path = OUT / "checkpoints/n26_round0_final.pt"
    round0_checkpoint = torch.load(round0_checkpoint_path, map_location="cpu", weights_only=False)
    round0_model = CCSAM(CCSAMConfig(**round0_checkpoint["model_config"]))
    round0_model.load_state_dict(round0_checkpoint["model_state"], strict=True)
    round0_model.to(device).eval()
    round0_diagnostic = "CCSAM_ROUND0_CHECKPOINT_FIXED_FINAL_HISTORY_DIAGNOSTIC"
    mode_outputs[round0_diagnostic] = infer(round0_model, arrays, canonical, device, "positive_negative", False)
    del round0_model
    pre_output = infer(model, arrays, canonical, device, "positive_negative", False, "memory_pre_mask")

    prediction_payload: dict[str, np.ndarray] = {"canonical_state": canonical, "target": target, "sequence": sequences}
    for method, values in mode_outputs.items():
        prediction_payload[f"{method}_logits"] = values["logits"].astype(np.float16)
        prediction_payload[f"{method}_risk"] = values["risk_logits"].astype(np.float16)
        prediction_payload[f"{method}_existence"] = values["existence_logit"].astype(np.float16)
    prediction_payload["CCSAM_PRE_LATEST_CORRECTION_logits"] = pre_output["logits"].astype(np.float16)
    temporary = OUT / "evaluation_predictions.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **prediction_payload)
    os.replace(temporary, OUT / "evaluation_predictions.npz")

    scalar = arrays["candidate_scalar"][canonical].astype(np.float32)
    baseline_scores = {
        "SAM3_EXECUTOR_BOX_ROI_IDENTITY_PROXY": np.where((scalar[:, :, 37] < 0.5) & candidate_mask, scalar[:, :, 4], -1e4),
        "B2_GFN_R0_DENSE": np.where((scalar[:, :, 35] < 0.5) & (scalar[:, :, 36] < 0.5) & candidate_mask, scalar[:, :, 2], -1e4),
        "B10_EXPLICIT_NEGATIVE_DENSE": b10_scores(round0, canonical),
    }
    model_scores = {method: values["candidate_logits"] for method, values in mode_outputs.items()}
    all_scores = {**baseline_scores, **model_scores}
    ranking = {method: ranking_metrics(scores, candidate_mask, target, materialized) for method, scores in all_scores.items()}

    predictions: dict[str, np.ndarray] = {}
    effort: dict[str, dict[str, Any]] = {}
    own_prior: dict[str, np.ndarray] = {}
    correction_counts: dict[str, dict[tuple[str, int], int]] = {}
    for method, scores in baseline_scores.items():
        pred = np.where(np.any(scores > -9999, axis=1), np.argmax(scores, axis=1), NONE)
        predictions[method] = pred
        effort[method], own_prior[method], correction_counts[method] = effort_metrics(pred, target, parent_rows)
    for method, values in mode_outputs.items():
        pred = values["logits"].argmax(axis=1)
        predictions[method] = pred
        effort[method], own_prior[method], correction_counts[method] = effort_metrics(pred, target, parent_rows)

    main_method = "CCSAM_POSITIVE_AND_NEGATIVE"
    main_output = mode_outputs[main_method]
    main_prediction = predictions[main_method]
    probability = softmax(main_output["logits"])
    selected_index = np.minimum(main_prediction, 4)
    selected_risk = sigmoid(main_output["risk_logits"][np.arange(len(target)), selected_index])
    selected_probability = probability[np.arange(len(target)), np.minimum(main_prediction, 5)]
    commit_score = sigmoid(main_output["existence_logit"]) * selected_risk * selected_probability
    commit_score[main_prediction == NONE] = 0.0
    no_risk_score = sigmoid(main_output["existence_logit"]) * selected_probability
    no_risk_score[main_prediction == NONE] = 0.0

    oof_accepted = np.zeros(len(target), dtype=bool)
    fold_policies: dict[str, Any] = {}
    for held in sorted(set(sequences.tolist())):
        held_mask = sequences == held
        calibration = ~held_mask
        threshold, calibration_metrics = select_threshold(commit_score, main_prediction, target, calibration)
        oof_accepted[held_mask] = (main_prediction[held_mask] < MAX_K) & (commit_score[held_mask] >= threshold)
        fold_policies[held] = {
            "held_sequence": held, "calibration_sequences": sorted(set(sequences[calibration].tolist())),
            "held_labels_used_for_threshold": False, "threshold": threshold, "calibration": calibration_metrics,
        }
    oof_operation = operation(main_prediction, oof_accepted, target)
    commit_sequences = {sequence: int((oof_accepted & (sequences == sequence)).sum()) for sequence in sorted(set(sequences.tolist())) if (oof_accepted & (sequences == sequence)).any()}

    ablation_oof: dict[str, dict[str, Any]] = {main_method: oof_operation}
    ablation_oof_accept: dict[str, np.ndarray] = {main_method: oof_accepted}
    no_none_method = "CCSAM_NO_NONE_EXISTENCE"
    no_none_output = mode_outputs[no_none_method]
    no_none_prediction = predictions[no_none_method]
    no_none_probability = softmax(no_none_output["logits"])
    no_none_selected = np.minimum(no_none_prediction, 4)
    no_none_score = sigmoid(no_none_output["risk_logits"][np.arange(len(target)), no_none_selected]) * no_none_probability[np.arange(len(target)), no_none_selected]
    for method, score_values, prediction_values in (
        ("CCSAM_NO_COMMIT_RISK", no_risk_score, main_prediction),
        (no_none_method, no_none_score, no_none_prediction),
    ):
        accepted_values = np.zeros(len(target), dtype=bool)
        for held in sorted(set(sequences.tolist())):
            held_mask = sequences == held
            threshold, _ = select_threshold(score_values, prediction_values, target, ~held_mask)
            accepted_values[held_mask] = (prediction_values[held_mask] < MAX_K) & (score_values[held_mask] >= threshold)
        ablation_oof[method] = operation(prediction_values, accepted_values, target)
        ablation_oof_accept[method] = accepted_values

    oof_rows = []
    for index, row in enumerate(parent_rows):
        oof_rows.append({
            "parent_event_id": index, "event_key": row["event_key"], "sequence": row["sequence"],
            "frame": row["frame"], "gid": row["gid"], "target": int(target[index]),
            "prediction": int(main_prediction[index]), "prediction_correct": bool(main_prediction[index] == target[index]),
            "candidate_set_absent": bool(target[index] == NONE), "commit_score": float(commit_score[index]),
            "oof_threshold": fold_policies[row["sequence"]]["threshold"], "oof_commit": bool(oof_accepted[index]),
            "held_sequence_labels_used_for_threshold": False,
        })
    write_csv(OUT / "n26b_oof_predictions.csv", oof_rows)

    curve_rows = []
    candidate_values = commit_score[main_prediction < MAX_K]
    thresholds = np.unique(np.concatenate(([math.inf], np.quantile(candidate_values, np.linspace(0, 1, 101)) if len(candidate_values) else np.asarray([]))))[::-1]
    for threshold in thresholds:
        accepted = (main_prediction < MAX_K) & (commit_score >= threshold)
        curve_rows.append({"phase": "N26B", "method": main_method, "policy": "POSTHOC_DIAGNOSTIC_NOT_DEPLOYABLE", "threshold": threshold, **operation(main_prediction, accepted, target)})
    curve_rows.append({"phase": "N26B", "method": main_method, "policy": "SEQUENCE_OOF_PRIMARY", "threshold": "PER_HELD_SEQUENCE", **oof_operation})
    for method in ("CCSAM_NO_COMMIT_RISK", no_none_method):
        curve_rows.append({"phase": "N26B", "method": method, "policy": "SEQUENCE_OOF_ABLATION", "threshold": "PER_HELD_SEQUENCE", **ablation_oof[method]})
    n26a = json.loads((OUT / "n26a_gate.json").read_text(encoding="utf-8"))
    curve_rows.append({"phase": "N26A", "method": "N26A_FACTORIZED_SAFE_B10", "policy": "SEQUENCE_OOF_PRIMARY", "threshold": "PER_HELD_SEQUENCE", **n26a["cal10_sequence_oof"]})
    write_csv(OUT / "risk_coverage.csv", curve_rows)

    pre_probability = softmax(pre_output["logits"])
    pair = arrays["pair_valid"][canonical].astype(bool) & (arrays["rejected_index"][canonical] >= 0)
    rejected = arrays["rejected_index"][canonical].astype(int).clip(min=0)
    pair &= (target == NONE) | (target != rejected)
    rejected_post_logit = main_output["candidate_logits"][np.arange(len(target)), rejected]
    rejected_pre_logit = pre_output["candidate_logits"][np.arange(len(target)), rejected]
    rejected_post_probability = probability[np.arange(len(target)), rejected]
    rejected_pre_probability = pre_probability[np.arange(len(target)), rejected]
    target_post_probability = probability[np.arange(len(target)), target]
    target_pre_probability = pre_probability[np.arange(len(target)), target]
    post_rejected_selected = main_prediction == rejected
    pre_prediction = pre_output["logits"].argmax(axis=1)
    pre_rejected_selected = pre_prediction == rejected
    post_error = main_prediction != target
    pre_error = pre_prediction != target
    correction_rows = []
    for index in np.flatnonzero(pair):
        correction_rows.append({
            "parent_event_id": index, "event_key": parent_rows[index]["event_key"], "sequence": sequences[index],
            "frame": parent_rows[index]["frame"], "gid": parent_rows[index]["gid"], "rejected_candidate": int(rejected[index]),
            "rejected_logit_pre": float(rejected_pre_logit[index]), "rejected_logit_post": float(rejected_post_logit[index]),
            "rejected_logit_delta_post_minus_pre": float(rejected_post_logit[index] - rejected_pre_logit[index]),
            "rejected_probability_pre": float(rejected_pre_probability[index]), "rejected_probability_post": float(rejected_post_probability[index]),
            "rejected_probability_delta_post_minus_pre": float(rejected_post_probability[index] - rejected_pre_probability[index]),
            "rejected_selected_pre": bool(pre_rejected_selected[index]), "rejected_selected_post": bool(post_rejected_selected[index]),
            "target_probability_pre": float(target_pre_probability[index]), "target_probability_post": float(target_post_probability[index]),
            "target_probability_delta_post_minus_pre": float(target_post_probability[index] - target_pre_probability[index]),
            "error_pre": bool(pre_error[index]), "error_post": bool(post_error[index]),
            "same_checkpoint_candidates_observation": True, "only_latest_legal_past_correction_added": True,
        })
    write_csv(OUT / "correction_response.csv", correction_rows)

    fixed_recorrection = own_prior[main_method]
    off_error = predictions["CCSAM_MEMORY_OFF"] != target
    main_error = main_prediction != target
    recorrection_delta = main_error.astype(float) - off_error.astype(float)
    rejected_selection_delta = post_rejected_selected.astype(float) - pre_rejected_selected.astype(float)
    counterfactual_error_delta = post_error.astype(float) - pre_error.astype(float)
    main_rank_prediction = np.argmax(main_output["candidate_logits"], axis=1)
    rank_eligible = materialized & (target < MAX_K) & candidate_mask[np.arange(len(target)), np.minimum(target, 4)]
    rank_correct = main_rank_prediction == target
    boot = bootstrap(sequences, target, main_prediction, oof_accepted, rank_correct, rank_eligible, pair, rejected_selection_delta, counterfactual_error_delta, fixed_recorrection, recorrection_delta)
    bootstrap_file = json.loads((OUT / "bootstrap_results.json").read_text(encoding="utf-8")) if (OUT / "bootstrap_results.json").is_file() else {}
    bootstrap_file["N26B"] = boot
    (OUT / "bootstrap_results.json").write_text(json.dumps(bootstrap_file, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    response_summary = {
        "eligible_parent_events": int(pair.sum()),
        "rejected_logit_delta_post_minus_pre": float(np.mean(rejected_post_logit[pair] - rejected_pre_logit[pair])) if pair.any() else math.nan,
        "rejected_probability_delta_post_minus_pre": float(np.mean(rejected_post_probability[pair] - rejected_pre_probability[pair])) if pair.any() else math.nan,
        "rejected_selection_rate_pre": float(pre_rejected_selected[pair].mean()) if pair.any() else math.nan,
        "rejected_selection_rate_post": float(post_rejected_selected[pair].mean()) if pair.any() else math.nan,
        "target_probability_delta_post_minus_pre": float(np.mean(target_post_probability[pair] - target_pre_probability[pair])) if pair.any() else math.nan,
        "future_error_rate_pre": float(pre_error[pair].mean()) if pair.any() else math.nan,
        "future_error_rate_post": float(post_error[pair].mean()) if pair.any() else math.nan,
        "sequence_bootstrap_rejected_selection_delta": boot["rejected_selection_delta"],
        "sequence_bootstrap_error_delta": boot["counterfactual_error_delta"],
    }
    (OUT / "correction_response_summary.json").write_text(json.dumps(response_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ablation_rows: list[dict[str, Any]] = []
    budget: list[dict[str, Any]] = []
    for method in all_scores:
        row = {"phase": "N26B", "method": method, **ranking[method], **effort[method]}
        if method in mode_outputs and method != "CCSAM_NO_NONE_EXISTENCE":
            row.update(existence_metrics(mode_outputs[method]["existence_logit"], target))
        elif method == "CCSAM_NO_NONE_EXISTENCE":
            row.update({"existence_accuracy": "NOT_APPLICABLE", "existence_recall": "NOT_APPLICABLE", "existence_specificity": "NOT_APPLICABLE"})
        if method in ablation_oof:
            row.update({f"oof_{name}": value for name, value in ablation_oof[method].items() if name in ("commit_precision", "coverage", "candidate_set_absent_false_accept", "commits")})
        row["ablation_type"] = "round0_checkpoint_on_fixed_final_history_not_for_selection" if method == round0_diagnostic else ("frozen_weight_inference_intervention" if method.startswith("CCSAM") else "same_candidate_stream_baseline")
        ablation_rows.append(row)
        budget.extend(budget_rows(method, correction_counts[method], len(target), effort[method]["targets"]))
    n25_calibration = [row for row in csv.DictReader((ROOT / "outputs/n25r/calibration.csv").open(newline="", encoding="utf-8")) if row["split"] == "cal10" and row["method"] == "B10_EXPLICIT_NEGATIVE" and row["history"] == "5"][0]
    ablation_rows.append({
        "phase": "N25R_HISTORY", "method": "B10_N25R_FROZEN_COMMIT", "top1": 0.545817,
        "mrr": 0.7069, "pair_auc": 0.7480, "hardest_negative_margin": 0.0092,
        "oof_commit_precision": float(n25_calibration["commit_precision"]), "oof_coverage": float(n25_calibration["commit_coverage"]),
        "oof_candidate_set_absent_false_accept": float(n25_calibration["target_absent_false_acceptance"]),
        "ablation_type": "historical N25-R frozen train-threshold baseline",
    })
    ablation_rows.append({
        "phase": "N26A", "method": "N26A_FACTORIZED_SAFE_B10_NOT_DEPLOYABLE", **n26a["same_policy_b10_rank"]["cal10"],
        "oof_commit_precision": n26a["cal10_sequence_oof"]["commit_precision"], "oof_coverage": n26a["cal10_sequence_oof"]["coverage"],
        "oof_candidate_set_absent_false_accept": n26a["cal10_sequence_oof"]["candidate_set_absent_false_accept"],
        "ablation_type": "failed scientific gate baseline", "deployable": False,
    })
    write_csv(OUT / "ablation_results.csv", ablation_rows)
    write_csv(OUT / "human_effort_budget.csv", budget)

    per_sequence_rows: list[dict[str, Any]] = []
    ranking_methods = ["B2_GFN_R0_DENSE", "B10_EXPLICIT_NEGATIVE_DENSE", "CCSAM_MEMORY_OFF", main_method, "CCSAM_HARD_NEGATIVE_CONTROL", round0_diagnostic]
    for sequence in sorted(set(sequences.tolist())):
        sequence_mask = sequences == sequence
        for method in ranking_methods:
            metrics = ranking_metrics(all_scores[method], candidate_mask, target, materialized & sequence_mask)
            pred = predictions[method]
            local_effort, _, _ = effort_metrics(pred[sequence_mask], target[sequence_mask], [row for row, keep in zip(parent_rows, sequence_mask) if keep])
            per_sequence_rows.append({"phase": "N26B", "sequence": sequence, "method": method, **metrics, **local_effort})
        accepted = oof_accepted & sequence_mask
        per_sequence_rows.append({"phase": "N26B_SAFETY", "sequence": sequence, "method": main_method, **operation(main_prediction[sequence_mask], accepted[sequence_mask], target[sequence_mask])})
    write_csv(OUT / "per_sequence.csv", per_sequence_rows)

    sequence_rank = defaultdict(dict)
    for row in per_sequence_rows:
        if row["phase"] == "N26B" and row["method"] in (main_method, "B10_EXPLICIT_NEGATIVE_DENSE") and math.isfinite(float(row["top1"])):
            sequence_rank[row["sequence"]][row["method"]] = float(row["top1"])
    rank_support = sum(values.get(main_method, -math.inf) >= values.get("B10_EXPLICIT_NEGATIVE_DENSE", math.inf) for values in sequence_rank.values())
    main_top1 = ranking[main_method]["top1"]
    b10_top1 = ranking["B10_EXPLICIT_NEGATIVE_DENSE"]["top1"]
    recorrection_main = float(main_error[fixed_recorrection].mean()) if fixed_recorrection.any() else math.nan
    recorrection_off = float(off_error[fixed_recorrection].mean()) if fixed_recorrection.any() else math.nan
    sequence_recorrection_support = 0
    sequence_recorrection_eligible = 0
    for sequence in sorted(set(sequences.tolist())):
        mask = fixed_recorrection & (sequences == sequence)
        if mask.any():
            sequence_recorrection_eligible += 1
            sequence_recorrection_support += float(main_error[mask].mean()) < float(off_error[mask].mean())
    criteria = {
        "oof_commit_precision_at_least_90": bool(oof_operation["commit_precision"] >= 0.90),
        "oof_coverage_at_least_5": bool(oof_operation["coverage"] >= 0.05),
        "candidate_set_absent_false_accept_at_most_7_26": bool(oof_operation["candidate_set_absent_false_accept"] <= 0.0726),
        "not_reject_all": bool(oof_operation["commits"] > 0),
        "commits_from_at_least_5_sequences": len(commit_sequences) >= 5,
        "no_sequence_above_50_percent_commits": max(commit_sequences.values(), default=0) / max(1, oof_operation["commits"]) <= 0.5,
        "h5_top1_not_more_than_1pp_below_dense_b10": bool(main_top1 >= b10_top1 - 0.01),
        "memory_lowers_fixed_history_recorrection": bool(recorrection_main < recorrection_off),
        "memory_recorrection_improves_majority_sequences": bool(sequence_recorrection_support > sequence_recorrection_eligible / 2),
        "rank_nonnegative_vs_b10_in_majority_materialized_sequences": bool(rank_support > len(sequence_rank) / 2),
        "latest_correction_lowers_rejected_selection_rate": bool(response_summary["rejected_selection_rate_post"] < response_summary["rejected_selection_rate_pre"]),
        "rejected_selection_reduction_sequence_bootstrap_upper_below_zero": bool(boot["rejected_selection_delta"]["ci95_high"] < 0),
        "latest_correction_lowers_future_error_rate": bool(response_summary["future_error_rate_post"] < response_summary["future_error_rate_pre"]),
        "held_sequence_labels_not_used_for_threshold": True,
        "val25_not_read": True,
    }
    gate = {
        "phase": "N26B", "status": "SCIENTIFIC_GATE_PASS" if all(criteria.values()) else "SCIENTIFIC_GATE_FAIL",
        "pass": all(criteria.values()), "criteria": criteria, "cal10_sequence_oof": oof_operation,
        "fold_policies": fold_policies, "commit_sequences": commit_sequences,
        "max_single_sequence_commit_fraction": max(commit_sequences.values(), default=0) / max(1, oof_operation["commits"]),
        "ranking": {"main": ranking[main_method], "dense_round0_b10": ranking["B10_EXPLICIT_NEGATIVE_DENSE"], "rank_sequence_support": rank_support, "eligible_sequences": len(sequence_rank)},
        "memory_recorrection": {"fixed_history_eligible_events": int(fixed_recorrection.sum()), "enabled": recorrection_main, "disabled": recorrection_off, "support_sequences": sequence_recorrection_support, "eligible_sequences": sequence_recorrection_eligible},
        "correction_response": response_summary, "bootstrap": boot,
        "full_loop_authorized": all(criteria.values()), "failure_class": None if all(criteria.values()) else "SCIENTIFIC_FAILURE_NO_THIRD_ROUTE",
        "val25_read": False,
    }
    (OUT / "n26b_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_stage": checkpoint["stage"],
        "parents": len(parent_rows), "materialized_parents": int(materialized.sum()), "additional_attempts": int((~materialized).sum()),
        "ranking": ranking, "existence": existence_metrics(main_output["existence_logit"], target),
        "effort": effort, "oof_safety": oof_operation, "correction_response": response_summary,
        "gate_status": gate["status"], "full_loop_authorized": gate["full_loop_authorized"], "val25_read": False,
    }
    (OUT / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate["status"], "oof": oof_operation, "main_top1": main_top1, "b10_top1": b10_top1, "response": response_summary}, sort_keys=True), flush=True)
    print("N26_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
