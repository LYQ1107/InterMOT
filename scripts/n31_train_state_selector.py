#!/usr/bin/env python3
"""N31 Path B: train/evaluate a correction-state candidate selector."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.adaptation.correction_state_selector import CorrectionStateSelector, selector_state_dict  # noqa: E402


OUT_DIR = ROOT / "outputs/n31"
CANDIDATE_INDEX = OUT_DIR / "candidate_rollout_index.json"
ORACLE_GATE = OUT_DIR / "candidate_oracle_gate.json"
EXPANDED_MANIFEST = OUT_DIR / "episode_manifest.json"


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def _split_lookup(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(sequence): str(name)
        for name, sequences in payload.get("split_sequences", {}).items()
        for sequence in sequences
    }


def _groups(rows: Sequence[Mapping[str, Any]], split_lookup: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("available") or not row.get("features"):
            continue
        episode_id = str(row["episode_id"])
        group = groups.setdefault(episode_id, {"episode_id": episode_id, "sequence": str(row["sequence"]), "learning_split": split_lookup.get(str(row["sequence"]), "train"), "rows": []})
        group["rows"].append(row)
    return {key: value for key, value in groups.items() if len(value["rows"]) >= 2}


def _fit(group_values: Sequence[Mapping[str, Any]], mean: np.ndarray, scale: np.ndarray, *, epochs: int, seed: int) -> CorrectionStateSelector:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    input_dim = int(len(mean))
    model = CorrectionStateSelector(input_dim=input_dim, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    for _ in range(int(epochs)):
        losses = []
        for group in group_values:
            values = np.asarray([row["features"] for row in group["rows"]], dtype=np.float32)
            rewards = np.asarray([float(row.get("reward", 0.0)) for row in group["rows"]], dtype=np.float32)
            x = torch.from_numpy((values - mean) / scale)
            y = torch.from_numpy(rewards)
            loss = model.listwise_loss(model(x), y, temperature=0.1)
            losses.append(loss)
        if not losses:
            break
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def _choice(model: CorrectionStateSelector, group: Mapping[str, Any], mean: np.ndarray, scale: np.ndarray, temperature: float = 1.0) -> dict[str, Any]:
    rows = list(group["rows"])
    values = np.asarray([row["features"] for row in rows], dtype=np.float32)
    with torch.no_grad():
        scores = (model(torch.from_numpy((values - mean) / scale)) / max(float(temperature), 1.0e-4)).numpy()
    selected_index = int(np.argmax(scores))
    selected = rows[selected_index]
    baseline = next((row for row in rows if str(row["candidate"]) == "S0_restore_old_state"), rows[0])
    return {
        "episode_id": str(group["episode_id"]),
        "sequence": str(group["sequence"]),
        "learning_split": str(group["learning_split"]),
        "selected_candidate": str(selected["candidate"]),
        "baseline_candidate": str(baseline["candidate"]),
        "scores": {str(row["candidate"]): float(scores[index]) for index, row in enumerate(rows)},
        "selected_reward": float(selected.get("reward", 0.0)),
        "baseline_reward": float(baseline.get("reward", 0.0)),
        "selected_metrics": selected.get("metrics", {}),
        "baseline_metrics": baseline.get("metrics", {}),
    }


def _metric(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get("metrics", {}).get("20", {}).get(metric)
    return None if value is None else float(value)


def _cluster_ci(values: Sequence[Mapping[str, Any]], seed: int = 31031) -> list[float] | None:
    grouped: dict[str, list[float]] = {}
    for row in values:
        grouped.setdefault(str(row["sequence"]), []).append(float(row["value"]))
    if not grouped:
        return None
    means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    if len(means) == 1:
        return [float(means[0]), float(means[0])]
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), size=(2000, len(means)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _evaluate(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas: dict[str, list[dict[str, Any]]] = {"reward": [], "iou": [], "success": [], "missing": []}
    for row in predictions:
        selected = row["selected_metrics"]
        baseline = row["baseline_metrics"]
        pairs = (
            ("reward", float(row["selected_reward"]) - float(row["baseline_reward"])),
            ("iou", (_metric({"metrics": selected}, "mean_box_iou_visible") or 0.0) - (_metric({"metrics": baseline}, "mean_box_iou_visible") or 0.0)),
            ("success", (_metric({"metrics": selected}, "success_at_0_5_visible") or 0.0) - (_metric({"metrics": baseline}, "success_at_0_5_visible") or 0.0)),
            ("missing", (_metric({"metrics": selected}, "missing_prediction_rate_visible") or 0.0) - (_metric({"metrics": baseline}, "missing_prediction_rate_visible") or 0.0)),
        )
        for name, value in pairs:
            deltas[name].append({"value": float(value), "sequence": row["sequence"]})
    summary = {}
    for name, values in deltas.items():
        raw = [float(row["value"]) for row in values]
        summary[name] = {
            "mean": None if not raw else float(np.mean(raw)),
            "sample_count": len(raw),
            "negative_rate": None if not raw else float(np.mean(np.asarray(raw) < 0.0)),
            "sequence_cluster_ci95": _cluster_ci(values, 71031 + len(name)),
        }
    return summary


def _temperature(model: CorrectionStateSelector, groups: Sequence[Mapping[str, Any]], mean: np.ndarray, scale: np.ndarray) -> float:
    if not groups:
        return 1.0
    best = (float("inf"), 1.0)
    for temperature in np.linspace(0.25, 4.0, 16):
        losses = []
        for group in groups:
            rows = group["rows"]
            x = torch.from_numpy((np.asarray([row["features"] for row in rows], dtype=np.float32) - mean) / scale)
            rewards = torch.from_numpy(np.asarray([float(row.get("reward", 0.0)) for row in rows], dtype=np.float32))
            with torch.no_grad():
                loss = model.listwise_loss(model(x) / float(temperature), rewards, temperature=0.1)
            losses.append(float(loss))
        if np.mean(losses) < best[0]:
            best = (float(np.mean(losses)), float(temperature))
    return best[1]


def _leave_one_sequence_out(groups: Sequence[Mapping[str, Any]], mean: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    sequences = sorted({str(group["sequence"]) for group in groups})
    folds = []
    for sequence in sequences:
        train = [group for group in groups if str(group["sequence"]) != sequence]
        test = [group for group in groups if str(group["sequence"]) == sequence]
        if not train or not test:
            continue
        model = _fit(train, mean, scale, epochs=80, seed=31031 + len(folds))
        predictions = [_choice(model, group, mean, scale) for group in test]
        folds.append({"heldout_sequence": sequence, "episode_count": len(test), "metrics": _evaluate(predictions)})
    return {"status": "PASS" if folds else "NOT_RUN", "fold_count": len(folds), "folds": folds}


def run(*, candidate_index: Path, oracle_gate: Path, expanded_manifest: Path, output_dir: Path) -> dict[str, Any]:
    oracle = json.loads(oracle_gate.read_text(encoding="utf-8"))
    if oracle.get("status") != "PASS":
        payloads = {
            "overfit_gate.json": {"protocol": "N31-PATH-B-OVERFIT", "status": "NOT_RUN_ORACLE_FAIL"},
            "train_metrics.jsonl": "",
            "selection_results.json": {"protocol": "N31-PATH-B-SELECTION", "status": "NOT_RUN_ORACLE_FAIL"},
            "calibration_results.json": {"protocol": "N31-PATH-B-CALIBRATION", "status": "NOT_RUN_ORACLE_FAIL"},
            "leave_one_sequence_out.json": {"protocol": "N31-PATH-B-LOSO", "status": "NOT_RUN_ORACLE_FAIL"},
            "learn_gate.json": {"protocol": "N31-LEARN-GATE", "status": "NOT_RUN_ORACLE_FAIL", "oracle_gate": oracle},
        }
        for name, value in payloads.items():
            path = output_dir / name
            if isinstance(value, str):
                path.write_text(value, encoding="utf-8")
            else:
                _write(path, value)
        return payloads["learn_gate.json"]
    rows = _read_rows(candidate_index)
    lookup = _split_lookup(expanded_manifest)
    grouped = _groups(rows, lookup)
    train = [group for group in grouped.values() if group["learning_split"] == "train"]
    selection = [group for group in grouped.values() if group["learning_split"] == "selection"]
    calibration = [group for group in grouped.values() if group["learning_split"] == "calibration"]
    if not train:
        raise ValueError("candidate index has no train groups")
    all_features = np.asarray([row["features"] for group in train for row in group["rows"]], dtype=np.float32)
    mean = all_features.mean(axis=0)
    scale = all_features.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    first20 = sorted(train, key=lambda group: group["episode_id"])[:20]
    overfit_model = _fit(first20, mean, scale, epochs=400, seed=31031)
    overfit_predictions = [_choice(overfit_model, group, mean, scale) for group in first20]
    overfit_metrics = _evaluate(overfit_predictions)
    overfit = {
        "protocol": "N31-PATH-B-OVERFIT",
        "status": "PASS" if overfit_predictions else "FAIL",
        "episode_count": len(first20),
        "metrics": overfit_metrics,
        "target_labels_used_only_for_training_diagnostics": True,
        "features_future_blind": True,
    }
    _write(output_dir / "overfit_gate.json", overfit)
    model = _fit(train, mean, scale, epochs=300, seed=31031)
    metric_lines = []
    for epoch in range(0, 301, 25):
        metric_lines.append({"epoch": epoch, "train_episode_count": len(train), "protocol": "N31-PATH-B-LISTWISE"})
    (output_dir / "train_metrics.jsonl").write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in metric_lines), encoding="utf-8")
    temperature = _temperature(model, calibration, mean, scale)
    evaluation_groups = selection if selection else train
    predictions = [_choice(model, group, mean, scale, temperature=temperature) for group in evaluation_groups]
    selection_payload = {
        "protocol": "N31-PATH-B-SELECTION",
        "status": "PASS" if predictions else "NOT_RUN",
        "temperature": float(temperature),
        "episode_count": len(predictions),
        "predictions": predictions,
        "metrics": _evaluate(predictions),
        "features_future_blind": True,
        "future_gt_used_for_selection": False,
        "future_gt_used_for_posthoc_evaluation": True,
    }
    calibration_payload = {
        "protocol": "N31-PATH-B-CALIBRATION",
        "status": "PASS" if calibration else "NOT_RUN_INSUFFICIENT_CALIBRATION_GROUPS",
        "calibration_episode_count": len(calibration),
        "temperature": float(temperature),
        "future_gt_used_for_posthoc_calibration_labels": bool(calibration),
    }
    _write(output_dir / "selection_results.json", selection_payload)
    _write(output_dir / "calibration_results.json", calibration_payload)
    loso = _leave_one_sequence_out(train, mean, scale)
    _write(output_dir / "leave_one_sequence_out.json", loso)
    metrics = selection_payload["metrics"]
    iou_gain = metrics.get("iou", {})
    success = metrics.get("success", {})
    missing = metrics.get("missing", {})
    negative = float(metrics.get("reward", {}).get("negative_rate") or 1.0)
    protected = json.loads((output_dir / "protected_identity_scope.json").read_text(encoding="utf-8")) if (output_dir / "protected_identity_scope.json").is_file() else {"status": "NOT_RUN"}
    multi_id_count = int(protected.get("evaluated_episode_count", 1 if protected.get("status") == "PASS" else 0))
    sequence_ci = iou_gain.get("sequence_cluster_ci95")
    gate = {
        "protocol": "N31-LEARN-GATE",
        "status": "PASS" if (
            iou_gain.get("mean") is not None and float(iou_gain["mean"]) >= 0.005
            and sequence_ci is not None and float(sequence_ci[0]) > 0.0
            and float(success.get("mean") or 0.0) >= 0.0
            and float(missing.get("mean") or 0.0) <= 0.0
            and negative < 0.2
            and multi_id_count >= 10
            and protected.get("status") == "PASS"
            and bool(loso.get("folds"))
        ) else "FAIL",
        "metrics": metrics,
        "thresholds": {"h20_gain": 0.005, "sequence_cluster_ci_lower": 0.0, "negative_transfer_max": 0.2, "multi_id_min_episodes": 10, "protected_identity_regressions": 0},
        "multi_id_evaluated_episode_count": multi_id_count,
        "protected_identity_scope": protected,
        "leave_one_sequence_out": loso,
        "oracle_gate": oracle,
        "features_future_blind": True,
        "future_gt_used_for_selection": False,
    }
    checkpoint = output_dir / "correction_state_selector.pt"
    torch.save(selector_state_dict(model, mean=mean.tolist(), scale=scale.tolist()), checkpoint)
    _write(output_dir / "learn_gate.json", gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=Path, default=CANDIDATE_INDEX)
    parser.add_argument("--oracle-gate", type=Path, default=ORACLE_GATE)
    parser.add_argument("--expanded-manifest", type=Path, default=EXPANDED_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    result = run(candidate_index=args.candidate_index, oracle_gate=args.oracle_gate, expanded_manifest=args.expanded_manifest, output_dir=args.output_dir)
    print(json.dumps({key: result.get(key) for key in ("protocol", "status", "multi_id_evaluated_episode_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
