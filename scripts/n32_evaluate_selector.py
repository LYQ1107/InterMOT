#!/usr/bin/env python3
"""N32-G: frozen selection, calibration, and sequence-disjoint selector gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INDEX = ROOT / "outputs/n32/policy_rollout_index.json"
TRAINING = ROOT / "outputs/n32/selector_training.json"
ORACLE = ROOT / "outputs/n32/policy_oracle_689.json"
AUDIT = ROOT / "outputs/n32/selector_feature_audit.json"
FROZEN = ROOT / "outputs/n32/frozen_protocol.json"
OUT_DIR = ROOT / "outputs/n32"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")
SEEDS = (3201, 3202, 3203)
THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10)

from sam3_intermot.adaptation.correction_application_selector import CorrectionApplicationSelector  # noqa: E402
from sam3_intermot.adaptation.correction_selector_features import FEATURE_NAMES  # noqa: E402
from scripts.n32_train_selector import _arrays, _load_groups, _ordered_groups, _train_model  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _h20(row: Mapping[str, Any], key: str) -> float | None:
    return _finite(row.get("metrics", {}).get("20", {}).get(key))


def _cluster_ci(values: Sequence[Mapping[str, Any]], *, seed: int, draws: int = 2000) -> list[float] | None:
    grouped: dict[str, list[float]] = {}
    for item in values:
        value = _finite(item.get("value"))
        if value is not None:
            grouped.setdefault(str(item["sequence"]), []).append(value)
    if not grouped:
        return None
    means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if len(means) == 1:
        return [float(means[0]), float(means[0])]
    rng = np.random.default_rng(int(seed))
    bootstrap = means[rng.integers(0, len(means), size=(int(draws), len(means)))].mean(axis=1)
    return [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]


def _load_checkpoint(path: Path) -> tuple[CorrectionApplicationSelector, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    names = list(checkpoint.get("feature_names", []))
    if names != list(FEATURE_NAMES):
        raise RuntimeError(f"checkpoint feature names mismatch: {path}")
    model = CorrectionApplicationSelector(int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    if mean.shape != (len(FEATURE_NAMES),) or std.shape != mean.shape:
        raise RuntimeError(f"checkpoint normalization mismatch: {path}")
    return model, mean, std, checkpoint


def _model_actions(model: CorrectionApplicationSelector, groups: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> tuple[dict[str, str], dict[str, float], dict[str, list[float]]]:
    x, _, _ = _arrays(groups, mean, std)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).cpu().numpy()
    actions: dict[str, str] = {}
    margins: dict[str, float] = {}
    logits_by_id: dict[str, list[float]] = {}
    for group, row_logits in zip(groups, logits):
        order = np.argsort(-row_logits, kind="stable")
        top = int(order[0])
        second = int(order[1])
        actions[group["episode_id"]] = POLICIES[top]
        margins[group["episode_id"]] = float(row_logits[top] - row_logits[second])
        logits_by_id[group["episode_id"]] = [float(value) for value in row_logits]
    return actions, margins, logits_by_id


def _best_fixed(groups: list[dict[str, Any]]) -> str:
    means = {
        policy: float(np.mean([float(group["rows"][policy]["reward"]) for group in groups]))
        for policy in POLICIES
    }
    return max(POLICIES, key=lambda policy: (means[policy], -POLICIES.index(policy)))


def _action_accuracy(groups: list[dict[str, Any]], actions: Mapping[str, str]) -> float | None:
    values = []
    for group in groups:
        rewards = [float(group["rows"][policy]["reward"]) for policy in POLICIES]
        selected = float(group["rows"][actions[group["episode_id"]]]["reward"])
        values.append(selected >= max(rewards) - 1.0e-6)
    return float(np.mean(values)) if values else None


def _heuristic_action(group: Mapping[str, Any]) -> str:
    values = np.asarray(group["rows"][POLICIES[0]]["feature_vector"], dtype=np.float32)
    # The rule is frozen before looking at future rewards: use the existing
    # ensure path when the correction disagrees with the current prediction or
    # when the official singleton/mapping is not currently healthy.
    if values[0] < 0.5 or values[1] < 0.5 or values[12] < 0.5 or values[13] < 0.5:
        return "K1_APPLY_ENSURE"
    return "K0_KEEP_OLD"


def _random_action(group: Mapping[str, Any], seed: int = 3201) -> str:
    digest = hashlib.sha256(f"{seed}:{group['episode_id']}".encode("utf-8")).digest()
    return POLICIES[int.from_bytes(digest[:8], "big") % len(POLICIES)]


def _summarize(
    groups: list[dict[str, Any]],
    actions: Mapping[str, str],
    *,
    method: str,
    base_policy: str,
    action_accuracy: float | None = None,
) -> dict[str, Any]:
    base_actions = {group["episode_id"]: base_policy for group in groups}
    records: list[dict[str, Any]] = []
    base_records: list[dict[str, Any]] = []
    for group in groups:
        episode_id = group["episode_id"]
        chosen = group["rows"][actions[episode_id]]
        base = group["rows"][base_policy]
        chosen_iou = _h20(chosen, "mean_box_iou_visible")
        base_iou = _h20(base, "mean_box_iou_visible")
        chosen_reward = _finite(chosen.get("reward"))
        base_reward = _finite(base.get("reward"))
        records.append({"episode_id": episode_id, "sequence": group["sequence"], "value": (chosen_iou - base_iou) if chosen_iou is not None and base_iou is not None else None})
        base_records.append({"episode_id": episode_id, "sequence": group["sequence"], "value": (chosen_reward - base_reward) if chosen_reward is not None and base_reward is not None else None})

    def mean_metric(key: str) -> float | None:
        values = [_h20(group["rows"][actions[group["episode_id"]]], key) for group in groups]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    rewards = [_finite(group["rows"][actions[group["episode_id"]]].get("reward")) for group in groups]
    rewards = [value for value in rewards if value is not None]
    gains = [item["value"] for item in records if item["value"] is not None]
    sequence_gains: dict[str, float] = {}
    sequence_reward_gains: dict[str, float] = {}
    for sequence in sorted({group["sequence"] for group in groups}):
        seq_values = [item["value"] for item in records if item["sequence"] == sequence and item["value"] is not None]
        seq_rewards = [item["value"] for item in base_records if item["sequence"] == sequence and item["value"] is not None]
        if seq_values:
            sequence_gains[sequence] = float(np.mean(seq_values))
        if seq_rewards:
            sequence_reward_gains[sequence] = float(np.mean(seq_rewards))
    positive = [value for value in sequence_gains.values() if value > 0.0]
    contribution_fraction = None
    if positive and sum(positive) > 0.0:
        contribution_fraction = float(max(positive) / sum(positive))
    negative_sequences = [sequence for sequence, value in sequence_reward_gains.items() if value < -1.0e-8]
    raw_action_counts = {policy: int(sum(actions[group["episode_id"]] == policy for group in groups)) for policy in POLICIES}
    selected_success = mean_metric("success_at_0_5_visible")
    selected_missing = mean_metric("missing_prediction_rate_visible")
    selected_iou = mean_metric("mean_box_iou_visible")
    selected_drift = mean_metric("mask_area_drift")
    base_success = float(np.mean([value for value in (_h20(group["rows"][base_policy], "success_at_0_5_visible") for group in groups) if value is not None])) if groups else None
    base_missing = float(np.mean([value for value in (_h20(group["rows"][base_policy], "missing_prediction_rate_visible") for group in groups) if value is not None])) if groups else None
    base_iou = float(np.mean([value for value in (_h20(group["rows"][base_policy], "mean_box_iou_visible") for group in groups) if value is not None])) if groups else None
    base_reward = float(np.mean([value for value in (_finite(group["rows"][base_policy].get("reward")) for group in groups) if value is not None])) if groups else None
    return {
        "method": method,
        "episode_count": len(groups),
        "sequence_count": len({group["sequence"] for group in groups}),
        "h20_iou": selected_iou,
        "h20_success": selected_success,
        "h20_missing": selected_missing,
        "h20_mask_area_drift": selected_drift,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "base_policy": base_policy,
        "base_h20_iou": base_iou,
        "base_h20_success": base_success,
        "base_h20_missing": base_missing,
        "base_mean_reward": base_reward,
        "h20_iou_gain_vs_base": float(np.mean(gains)) if gains else None,
        "reward_gain_vs_base": float(np.mean([item["value"] for item in base_records if item["value"] is not None])) if any(item["value"] is not None for item in base_records) else None,
        "sequence_cluster_ci95_h20_iou_gain": _cluster_ci(records, seed=320320 + len(groups)),
        "sequence_cluster_ci95_reward_gain": _cluster_ci(base_records, seed=320321 + len(groups)),
        "positive_sequence_count_h20_iou": int(sum(value > 0.0 for value in sequence_gains.values())),
        "negative_transfer_sequence_count": len(negative_sequences),
        "negative_transfer_sequence_rate": float(len(negative_sequences) / max(1, len(sequence_reward_gains))),
        "sequence_gains_h20_iou": sequence_gains,
        "sequence_reward_gains": sequence_reward_gains,
        "single_sequence_positive_contribution_fraction": contribution_fraction,
        "action_counts": raw_action_counts,
        "action_accuracy_to_reward_oracle": action_accuracy,
        "actions": dict(actions),
    }


def _fixed_methods(groups: list[dict[str, Any]], base_policy: str) -> dict[str, dict[str, str]]:
    methods = {policy: {group["episode_id"]: policy for group in groups} for policy in POLICIES}
    methods["BEST_FIXED"] = {group["episode_id"]: base_policy for group in groups}
    methods["HEURISTIC"] = {group["episode_id"]: _heuristic_action(group) for group in groups}
    methods["RANDOM"] = {group["episode_id"]: _random_action(group) for group in groups}
    return methods


def _oracle_actions(groups: list[dict[str, Any]], raw_iou: bool = False) -> dict[str, str]:
    actions = {}
    for group in groups:
        if raw_iou:
            actions[group["episode_id"]] = max(POLICIES, key=lambda policy: ((_h20(group["rows"][policy], "mean_box_iou_visible") or -1.0), -POLICIES.index(policy)))
        else:
            actions[group["episode_id"]] = max(POLICIES, key=lambda policy: (float(group["rows"][policy]["reward"]), -POLICIES.index(policy)))
    return actions


def _selector_actions(
    groups: list[dict[str, Any]],
    model: CorrectionApplicationSelector,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float | None = None,
    base_policy: str = "K1_APPLY_ENSURE",
) -> tuple[dict[str, str], dict[str, float], dict[str, list[float]]]:
    actions, margins, logits = _model_actions(model, groups, mean, std)
    if threshold is not None:
        actions = {episode_id: (base_policy if margins[episode_id] <= float(threshold) else policy) for episode_id, policy in actions.items()}
    return actions, margins, logits


def _select_threshold(groups: list[dict[str, Any]], model: CorrectionApplicationSelector, mean: np.ndarray, std: np.ndarray, base_policy: str) -> tuple[float, list[dict[str, Any]]]:
    records = []
    for threshold in THRESHOLDS:
        actions, _, _ = _selector_actions(groups, model, mean, std, threshold=threshold, base_policy=base_policy)
        summary = _summarize(groups, actions, method=f"SELECTOR_MARGIN_{threshold:g}", base_policy=base_policy, action_accuracy=_action_accuracy(groups, actions))
        records.append({"threshold": float(threshold), "h20_iou": summary["h20_iou"], "mean_reward": summary["mean_reward"], "h20_missing": summary["h20_missing"], "summary": summary})
    chosen = max(records, key=lambda item: ((item["mean_reward"] if item["mean_reward"] is not None else -1.0e9), (item["h20_iou"] if item["h20_iou"] is not None else -1.0e9), -(item["h20_missing"] if item["h20_missing"] is not None else 1.0e9), item["threshold"]))
    return float(chosen["threshold"]), records


def _seed_eval(groups: list[dict[str, Any]], seed: int, base_policy: str) -> dict[str, Any]:
    path = CHECKPOINT_DIR / f"selector_seed{seed}.pt"
    model, mean, std, checkpoint = _load_checkpoint(path)
    raw_actions, raw_margins, raw_logits = _selector_actions(groups, model, mean, std, threshold=None, base_policy=base_policy)
    threshold, threshold_records = _select_threshold(groups, model, mean, std, base_policy)
    margin_actions, margins, _ = _selector_actions(groups, model, mean, std, threshold=threshold, base_policy=base_policy)
    return {
        "seed": int(seed),
        "checkpoint": str(path),
        "checkpoint_sha256": _sha(path),
        "best_fixed_policy": base_policy,
        "selected_threshold": threshold,
        "threshold_candidates": list(THRESHOLDS),
        "threshold_selection_records": threshold_records,
        "raw_actions": raw_actions,
        "raw_margins": raw_margins,
        "raw_logits": raw_logits,
        "margin_actions": margin_actions,
        "margin_margins": margins,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_metadata": {key: value for key, value in checkpoint.items() if key not in {"state_dict", "normalization_mean", "normalization_std"}},
    }


def _loso(groups: list[dict[str, Any]], base_policy: str) -> dict[str, Any]:
    train_groups = _ordered_groups({group["episode_id"]: group for group in groups}, "train")
    sequences = sorted({group["sequence"] for group in train_groups})
    records = []
    for index, sequence in enumerate(sequences):
        held = [group for group in train_groups if group["sequence"] == sequence]
        fit = [group for group in train_groups if group["sequence"] != sequence]
        raw_x = np.asarray([group["rows"][POLICIES[0]]["feature_vector"] for group in fit], dtype=np.float32)
        mean = raw_x.mean(axis=0)
        std = np.where(raw_x.std(axis=0) < 1.0e-6, 1.0, raw_x.std(axis=0)).astype(np.float32)
        x, rewards, weights = _arrays(fit, mean, std)
        model, _ = _train_model(x, rewards, weights, seed=3201 + index, epochs=100, batch_size=64)
        raw_actions, margins, _ = _selector_actions(held, model, mean, std, threshold=None, base_policy=base_policy)
        threshold, _ = _select_threshold(fit, model, mean, std, base_policy)
        margin_actions, _, _ = _selector_actions(held, model, mean, std, threshold=threshold, base_policy=base_policy)
        raw_summary = _summarize(held, raw_actions, method="LOSO_RAW", base_policy=base_policy, action_accuracy=_action_accuracy(held, raw_actions))
        margin_summary = _summarize(held, margin_actions, method="LOSO_MARGIN", base_policy=base_policy, action_accuracy=_action_accuracy(held, margin_actions))
        records.append({"held_out_sequence": sequence, "fit_episode_count": len(fit), "held_out_episode_count": len(held), "threshold": threshold, "raw": raw_summary, "margin": margin_summary})
    margin_gains = [record["margin"]["h20_iou_gain_vs_base"] for record in records if record["margin"]["h20_iou_gain_vs_base"] is not None]
    return {
        "protocol": "N32-G-LOSO",
        "sequence_count": len(records),
        "records": records,
        "margin_gain_min": min(margin_gains) if margin_gains else None,
        "margin_gain_mean": float(np.mean(margin_gains)) if margin_gains else None,
        "worst_direction_not_reversed": bool(margin_gains and min(margin_gains) >= 0.0),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selector_input": False,
    }


def _gate(summary: Mapping[str, Any], *, oracle: bool = False) -> dict[str, Any]:
    gain = summary.get("h20_iou_gain_vs_base")
    ci = summary.get("sequence_cluster_ci95_h20_iou_gain") or [None, None]
    base_success = summary.get("base_h20_success")
    base_missing = summary.get("base_h20_missing")
    success = summary.get("h20_success")
    missing = summary.get("h20_missing")
    checks = {
        "h20_gain": gain is not None and float(gain) >= (0.005 if not oracle else 0.01),
        "sequence_cluster_ci_lower_gt_zero": ci[0] is not None and float(ci[0]) > 0.0,
        "success_not_decreased": success is not None and base_success is not None and float(success) >= float(base_success),
        "missing_not_increased": missing is not None and base_missing is not None and float(missing) <= float(base_missing),
        "negative_transfer_sequence_rate_lt_020": float(summary.get("negative_transfer_sequence_rate", 1.0)) < 0.20,
        "single_sequence_positive_contribution_le_050": summary.get("single_sequence_positive_contribution_fraction") is not None and float(summary["single_sequence_positive_contribution_fraction"]) <= 0.50,
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def _mean_of_summaries(summaries: Sequence[Mapping[str, Any]], *, method: str, base_policy: str) -> dict[str, Any]:
    """Average the three independently trained seed evaluations.

    This is a reporting aggregate, not a fourth model and not a post-hoc
    action ensemble.  Sequence gains are averaged before the cluster CI is
    recomputed, which keeps the stated unit of inference unchanged.
    """
    if not summaries:
        raise ValueError("cannot average an empty summary list")
    numeric_keys = (
        "h20_iou", "h20_success", "h20_missing", "h20_mask_area_drift",
        "mean_reward", "base_h20_iou", "base_h20_success", "base_h20_missing",
        "base_mean_reward", "h20_iou_gain_vs_base", "reward_gain_vs_base",
        "negative_transfer_sequence_rate",
    )
    result: dict[str, Any] = {"method": method, "base_policy": base_policy, "episode_count": summaries[0].get("episode_count"), "sequence_count": summaries[0].get("sequence_count")}
    for key in numeric_keys:
        values = [float(item[key]) for item in summaries if item.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    sequences = sorted({sequence for item in summaries for sequence in item.get("sequence_gains_h20_iou", {})})
    seq_gains = {
        sequence: float(np.mean([float(item["sequence_gains_h20_iou"][sequence]) for item in summaries if sequence in item.get("sequence_gains_h20_iou", {})]))
        for sequence in sequences
    }
    seq_reward = {
        sequence: float(np.mean([float(item["sequence_reward_gains"][sequence]) for item in summaries if sequence in item.get("sequence_reward_gains", {})]))
        for sequence in sorted({sequence for item in summaries for sequence in item.get("sequence_reward_gains", {})})
    }
    positive = [value for value in seq_gains.values() if value > 0.0]
    result["sequence_gains_h20_iou"] = seq_gains
    result["sequence_reward_gains"] = seq_reward
    result["positive_sequence_count_h20_iou"] = int(sum(value > 0.0 for value in seq_gains.values()))
    result["negative_transfer_sequence_count"] = int(sum(value < -1.0e-8 for value in seq_reward.values()))
    result["negative_transfer_sequence_rate"] = float(result["negative_transfer_sequence_count"] / max(1, len(seq_reward)))
    result["single_sequence_positive_contribution_fraction"] = float(max(positive) / sum(positive)) if positive and sum(positive) > 0.0 else None
    result["sequence_cluster_ci95_h20_iou_gain"] = _cluster_ci([{"sequence": sequence, "value": value} for sequence, value in seq_gains.items()], seed=320998)
    result["sequence_cluster_ci95_reward_gain"] = _cluster_ci([{"sequence": sequence, "value": value} for sequence, value in seq_reward.items()], seed=320999)
    result["action_counts"] = {
        policy: float(np.mean([item.get("action_counts", {}).get(policy, 0) for item in summaries]))
        for policy in POLICIES
    }
    result["action_accuracy_to_reward_oracle"] = float(np.mean([float(item["action_accuracy_to_reward_oracle"]) for item in summaries if item.get("action_accuracy_to_reward_oracle") is not None])) if any(item.get("action_accuracy_to_reward_oracle") is not None for item in summaries) else None
    result["actions"] = None
    return result


def run(*, index_path: Path = INDEX, output_dir: Path = OUT_DIR) -> dict[str, Any]:
    groups = _load_groups(index_path)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("feature audit is not PASS")
    train_groups = _ordered_groups(groups, "train")
    selection_groups = _ordered_groups(groups, "selection")
    calibration_groups = _ordered_groups(groups, "calibration")
    base_policy = _best_fixed(train_groups)
    full_oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    all_seed_evals = {
        str(seed): {
            "selection": _seed_eval(selection_groups, seed, base_policy),
            "calibration": None,
        }
        for seed in SEEDS
    }
    selection_seed_summaries = []
    for seed in SEEDS:
        payload = all_seed_evals[str(seed)]["selection"]
        raw_summary = _summarize(selection_groups, payload["raw_actions"], method="SELECTOR_RAW", base_policy=base_policy, action_accuracy=_action_accuracy(selection_groups, payload["raw_actions"]))
        margin_summary = _summarize(selection_groups, payload["margin_actions"], method="SELECTOR_MARGIN", base_policy=base_policy, action_accuracy=_action_accuracy(selection_groups, payload["margin_actions"]))
        selection_seed_summaries.append({"seed": seed, "raw": raw_summary, "margin": margin_summary, "threshold": payload["selected_threshold"]})
        model, mean, std, _ = _load_checkpoint(CHECKPOINT_DIR / f"selector_seed{seed}.pt")
        cal_raw, _, _ = _selector_actions(calibration_groups, model, mean, std, threshold=None, base_policy=base_policy)
        cal_margin, _, _ = _selector_actions(calibration_groups, model, mean, std, threshold=payload["selected_threshold"], base_policy=base_policy)
        all_seed_evals[str(seed)]["calibration"] = {
            "raw_actions": cal_raw,
            "margin_actions": cal_margin,
            "raw": _summarize(calibration_groups, cal_raw, method="SELECTOR_RAW", base_policy=base_policy, action_accuracy=_action_accuracy(calibration_groups, cal_raw)),
            "margin": _summarize(calibration_groups, cal_margin, method="SELECTOR_MARGIN", base_policy=base_policy, action_accuracy=_action_accuracy(calibration_groups, cal_margin)),
        }
    selected = max(selection_seed_summaries, key=lambda item: ((item["margin"]["mean_reward"] if item["margin"]["mean_reward"] is not None else -1.0e9), (item["margin"]["h20_iou"] if item["margin"]["h20_iou"] is not None else -1.0e9), -int(item["seed"])))
    selected_seed = int(selected["seed"])
    selected_threshold = float(selected["threshold"])

    def split_methods(split_groups: list[dict[str, Any]], *, include_selector: bool, selector_seed: int | None = None) -> dict[str, dict[str, str]]:
        methods = _fixed_methods(split_groups, base_policy)
        methods["ORACLE_REWARD"] = _oracle_actions(split_groups, raw_iou=False)
        methods["ORACLE_RAW_IOU"] = _oracle_actions(split_groups, raw_iou=True)
        if include_selector and selector_seed is not None:
            payload = all_seed_evals[str(selector_seed)]
            if split_groups is selection_groups:
                methods["SELECTOR_RAW"] = payload["selection"]["raw_actions"]
                methods["SELECTOR_MARGIN"] = payload["selection"]["margin_actions"]
            else:
                methods["SELECTOR_RAW"] = payload["calibration"]["raw_actions"]
                methods["SELECTOR_MARGIN"] = payload["calibration"]["margin_actions"]
        return methods

    selection_methods = split_methods(selection_groups, include_selector=True, selector_seed=selected_seed)
    calibration_methods = split_methods(calibration_groups, include_selector=True, selector_seed=selected_seed)
    selection_summaries = {
        name: _summarize(selection_groups, actions, method=name, base_policy=base_policy, action_accuracy=_action_accuracy(selection_groups, actions))
        for name, actions in selection_methods.items()
    }
    calibration_summaries = {
        name: _summarize(calibration_groups, actions, method=name, base_policy=base_policy, action_accuracy=_action_accuracy(calibration_groups, actions))
        for name, actions in calibration_methods.items()
    }
    seed_calibration_summaries = {
        str(seed): {
            "raw": all_seed_evals[str(seed)]["calibration"]["raw"],
            "margin": all_seed_evals[str(seed)]["calibration"]["margin"],
            "threshold": all_seed_evals[str(seed)]["selection"]["selected_threshold"],
        }
        for seed in SEEDS
    }
    mean_three_summary = _mean_of_summaries(
        [seed_calibration_summaries[str(seed)]["margin"] for seed in SEEDS],
        method="SELECTOR_MARGIN_MEAN_OF_THREE",
        base_policy=base_policy,
    )
    independent_seed_gates = {}
    for seed in SEEDS:
        independent_seed_gates[str(seed)] = {
            "selection": _gate(selection_seed_summaries[SEEDS.index(seed)]["margin"]),
            "calibration": _gate(seed_calibration_summaries[str(seed)]["margin"]),
            "pass": bool(_gate(selection_seed_summaries[SEEDS.index(seed)]["margin"])["pass"] and _gate(seed_calibration_summaries[str(seed)]["margin"])["pass"]),
        }
    mean_cal = mean_three_summary
    mean_gate = _gate(mean_cal)
    loso = _loso(list(groups.values()), base_policy)
    learn_checks = {
        "oracle_gate_pass": full_oracle.get("status") == "PASS",
        "overfit_gate_pass": bool(json.loads((output_dir / "overfit_gate.json").read_text(encoding="utf-8")).get("status") == "PASS"),
        "at_least_two_of_three_seeds_pass": sum(bool(independent_seed_gates[str(seed)]["pass"]) for seed in SEEDS) >= 2,
        "mean_of_three_seeds_pass": bool(mean_gate["pass"]),
        "loso_worst_direction_not_reversed": bool(loso.get("worst_direction_not_reversed")),
    }
    learn_result = {
        "protocol": "N32-G-LEARN-GATE",
        "status": "PASS" if all(learn_checks.values()) else "FAIL",
        "checks": learn_checks,
        "independent_seed_gates": independent_seed_gates,
        "mean_of_three_seed_calibration": mean_cal,
        "mean_gate": mean_gate,
        "selected_seed": selected_seed,
        "selected_threshold": selected_threshold,
        "best_fixed_policy_train": base_policy,
        "loso": loso,
        "threshold_rule": "maximize selection mean reward; tie-break h20 IoU, lower missing, then larger safety threshold",
        "threshold_candidates": list(THRESHOLDS),
        "future_gt_used_for_selector_input": False,
        "future_gt_used_for_training_labels": True,
        "val25_read": False,
        "test_labels_used": False,
        "frozen_learn_gate": frozen["learn_gate"],
    }
    selection_result = {
        "protocol": "N32-G-SELECTION",
        "status": "PASS",
        "split": "selection",
        "episode_count": len(selection_groups),
        "sequence_count": len({group["sequence"] for group in selection_groups}),
        "best_fixed_policy_train": base_policy,
        "selected_seed": selected_seed,
        "selected_threshold": selected_threshold,
        "seed_summaries": selection_seed_summaries,
        "methods": selection_summaries,
        "future_gt_used_for_selection": False,
        "val25_read": False,
        "test_labels_used": False,
    }
    calibration_result = {
        "protocol": "N32-G-CALIBRATION",
        "status": "PASS",
        "split": "calibration",
        "episode_count": len(calibration_groups),
        "sequence_count": len({group["sequence"] for group in calibration_groups}),
        "best_fixed_policy_train": base_policy,
        "selected_seed": selected_seed,
        "selected_threshold": selected_threshold,
        "all_seed_summaries": seed_calibration_summaries,
        "methods": calibration_summaries,
        "future_gt_used_for_selection": False,
        "future_gt_used_for_calibration_metrics": True,
        "val25_read": False,
        "test_labels_used": False,
    }
    _write(output_dir / "selection_results.json", selection_result)
    _write(output_dir / "calibration_results.json", calibration_result)
    _write(output_dir / "leave_one_sequence_out.json", loso)
    _write(output_dir / "learn_gate.json", learn_result)
    return {"selection": selection_result, "calibration": calibration_result, "learn_gate": learn_result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    result = run(index_path=args.index, output_dir=args.output_dir)
    print(json.dumps({"selection": {"selected_seed": result["selection"]["selected_seed"], "selected_threshold": result["selection"]["selected_threshold"]}, "learn_gate": result["learn_gate"]}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
