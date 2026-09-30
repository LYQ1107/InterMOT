#!/usr/bin/env python
"""Run dev tuning or the frozen N72R19R1 identity-memory evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from sam3_intermot.identity_memory.dataset import FrozenIdentityEpisodeDataset
from sam3_intermot.identity_memory.r1_eval import (
    ALL_METHODS,
    METHOD_CORRECTNESS,
    METHOD_EMA,
    METHOD_FROZEN_GRU,
    METHOD_FUTURE_UTILITY,
    METHOD_SIMILARITY,
    METHOD_SINGLE_ANCHOR,
    evaluate_condition,
    paired_sequence_bootstrap_delta,
    paired_sequence_bootstrap_macro_delta,
)
from sam3_intermot.identity_memory.r1_protocol import (
    all_condition_slugs,
    load_corruption_manifest,
    primary_condition_slugs,
)
from sam3_intermot.identity_memory.selective import ObservationSelector, load_frozen_n72r18_gru


DEFAULT_PROTOCOL = "outputs/N72R19R1/protocol.json"
DEFAULT_MANIFEST = "outputs/N72R19R1/corruption_manifest.json"
DEFAULT_BASE_CHECKPOINT = "outputs/N72R18/checkpoints/identity_memory_gru.pt"
DEFAULT_TRAIN_PROTOCOL = "/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_train.json"
DEFAULT_VAL_PROTOCOL = "/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_val.json"
DEFAULT_TRAIN_STORE = "/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_train"
DEFAULT_VAL_STORE = "/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/embeddings_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "val"), required=True)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--train-protocol", default=DEFAULT_TRAIN_PROTOCOL)
    parser.add_argument("--val-protocol", default=DEFAULT_VAL_PROTOCOL)
    parser.add_argument("--train-store", default=DEFAULT_TRAIN_STORE)
    parser.add_argument("--val-store", default=DEFAULT_VAL_STORE)
    parser.add_argument("--checkpoint-dir", default="outputs/N72R19R1/checkpoints")
    parser.add_argument("--cache-dir", default="outputs/N72R19R1/metrics/eval_cache")
    parser.add_argument("--selector-seeds", default="72191,72192,72193")
    parser.add_argument("--include-correctness", action="store_true")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--ablation-seed", type=int, default=72191)
    parser.add_argument("--methods", default="B0_SINGLE_ANCHOR,B1_EMA_0.90,B2_FROZEN_GRU,B3_SIMILARITY_THRESHOLD,B4_CORRECTNESS_SELECTOR,B5_FUTURE_UTILITY_SELECTOR")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--b3-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--b3-final-threshold", type=float, default=None)
    parser.add_argument("--selector-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=72191)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_selector(path: Path, device: torch.device) -> ObservationSelector:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    selector = ObservationSelector(
        feature_indices=tuple(int(value) for value in config.get("feature_indices", range(13)))
    ).to(device)
    selector.load_state_dict(checkpoint["selector_state_dict"], strict=True)
    selector.eval()
    return selector


def _dataset(args: argparse.Namespace, protocol: dict[str, Any]) -> FrozenIdentityEpisodeDataset:
    if args.split == "dev":
        return FrozenIdentityEpisodeDataset(
            args.train_protocol,
            args.train_store,
            split="train",
            sequence_filter=protocol["train_dev_split"]["dev_sequences"],
        )
    return FrozenIdentityEpisodeDataset(
        args.val_protocol,
        args.val_store,
        split="val",
        sequence_filter=protocol["train_dev_split"]["val_sequences"],
    )


def _h100(summary: dict[str, Any]) -> float | None:
    value = summary.get("horizons", {}).get("H100", {}).get("hard_negative_win_rate")
    return None if value is None else float(value)


def _clean_ok(value: float | None, baseline: float | None) -> bool:
    if value is None:
        return False
    if value >= 0.90:
        return True
    return baseline is not None and value >= baseline - 0.015


def _method_for_checkpoint(mode: str) -> str:
    return METHOD_CORRECTNESS if mode == "correctness" else METHOD_FUTURE_UTILITY


def _run_cell(
    dataset: FrozenIdentityEpisodeDataset,
    manifest: dict[str, Any],
    frozen_gru,
    *,
    method: str,
    condition: str,
    selector: ObservationSelector | None,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return evaluate_condition(
        dataset,
        manifest,
        method=method,
        condition=condition,
        frozen_gru=frozen_gru,
        selector=selector,
        threshold=threshold,
        device=args.device,
        batch_size=args.batch_size,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )


def _cached_cell(
    cache_dir: Path,
    run_key: str,
    condition: str,
    runner,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_key = run_key.replace("/", "_").replace("@", "_at_").replace(".", "p")
    cache_path = cache_dir / f"{safe_key}__{condition}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = runner()
    cache_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _summary_table(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        key: {
            "H20": value["summary"]["horizons"]["H20"]["hard_negative_win_rate"],
            "H50": value["summary"]["horizons"]["H50"]["hard_negative_win_rate"],
            "H100": value["summary"]["horizons"]["H100"]["hard_negative_win_rate"],
            "H100_ci95": value["summary"]["horizons"]["H100"]["hard_negative_win_rate_sequence_cluster_ci95"],
            "median_margin": value["summary"]["horizons"]["H100"]["median_margin"],
            "mean_margin": value["summary"]["horizons"]["H100"]["mean_margin"],
            "p10_margin": value["summary"]["horizons"]["H100"]["p10_margin"],
            "p25_margin": value["summary"]["horizons"]["H100"]["p25_margin"],
            "p75_margin": value["summary"]["horizons"]["H100"]["p75_margin"],
            "rank1": value["summary"]["horizons"]["H100"]["rank_1_accuracy"],
            "rank2": value["summary"]["horizons"]["H100"]["rank_2_accuracy"],
            "rank3": value["summary"]["horizons"]["H100"]["rank_3_accuracy"],
            "mrr": value["summary"]["horizons"]["H100"]["mrr"],
            "time_gap_bins": value["summary"].get("time_gap_bins"),
            "selection": value["summary"].get("selection"),
            "memory_damage": value["summary"].get("memory_damage"),
            "state_drift": value["summary"].get("state_drift"),
        }
        for key, value in cells.items()
    }


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def _selector_seed_aggregates(
    cells: dict[str, dict[str, dict[str, Any]]],
    conditions: tuple[str, ...],
) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for method in (METHOD_CORRECTNESS, METHOD_FUTURE_UTILITY):
        prefix = method + "_seed"
        keys = sorted(key for key in cells if key.startswith(prefix))
        if not keys:
            continue
        by_condition: dict[str, Any] = {}
        for condition in conditions:
            by_condition[condition] = {
                horizon: _mean_std(
                    [
                        float(cells[key][condition]["summary"]["horizons"][horizon]["hard_negative_win_rate"])
                        for key in keys
                        if condition in cells[key]
                    ]
                )
                for horizon in ("H20", "H50", "H100")
            }
        macro_by_seed = []
        for key in keys:
            values = [
                float(cells[key][condition]["summary"]["horizons"]["H100"]["hard_negative_win_rate"])
                for condition in primary_condition_slugs()
                if condition in cells[key]
            ]
            if values:
                macro_by_seed.append(float(sum(values) / len(values)))
        aggregates[method] = {
            "seed_keys": keys,
            "by_condition": by_condition,
            "primary_H100_macro": _mean_std(macro_by_seed),
        }
    return aggregates


def main() -> None:
    args = parse_args()
    if args.split == "val" and args.b3_thresholds:
        # A val run must receive the already frozen threshold, not scan it.
        pass
    device = torch.device(args.device)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    manifest = load_corruption_manifest(args.manifest)
    dataset = _dataset(args, protocol)
    frozen_gru = load_frozen_n72r18_gru(args.base_checkpoint, device)
    requested_methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if args.conditions == "all":
        conditions = all_condition_slugs()
    elif args.conditions == "primary":
        conditions = ("clean_0p00", *primary_condition_slugs())
    else:
        conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    selector_seeds = tuple(int(item) for item in args.selector_seeds.split(",") if item.strip())
    checkpoint_dir = Path(args.checkpoint_dir)
    cache_dir = Path(args.cache_dir) / args.split
    cells: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    record_cells: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(dict)

    # Baselines are evaluated once and shared by all selector seeds.
    baseline_methods = [
        method
        for method in requested_methods
        if method in {METHOD_SINGLE_ANCHOR, METHOD_EMA, METHOD_FROZEN_GRU, METHOD_SIMILARITY}
    ]
    b3_thresholds = [float(item) for item in args.b3_thresholds.split(",") if item.strip()]
    for method in baseline_methods:
        if method == METHOD_SIMILARITY and args.split == "dev":
            thresholds = b3_thresholds
        elif method == METHOD_SIMILARITY and args.b3_final_threshold is not None:
            thresholds = [args.b3_final_threshold]
        else:
            thresholds = [args.selector_threshold]
        for threshold in thresholds:
            run_key = method if method != METHOD_SIMILARITY else f"{method}@{threshold:.2f}"
            for condition in conditions:
                result = _cached_cell(
                    cache_dir,
                    run_key,
                    condition,
                    lambda: _run_cell(
                        dataset,
                        manifest,
                        frozen_gru,
                        method=method,
                        condition=condition,
                        selector=None,
                        threshold=threshold,
                        args=args,
                    ),
                )
                cells[run_key][condition] = result
                record_cells[run_key][condition] = result["records"]

    selector_jobs: list[tuple[str, int, Path]] = []
    if METHOD_CORRECTNESS in requested_methods or args.include_correctness:
        selector_jobs.extend(
            (METHOD_CORRECTNESS, seed, checkpoint_dir / f"selector_correctness_seed{seed}.pt")
            for seed in selector_seeds
        )
    if METHOD_FUTURE_UTILITY in requested_methods:
        selector_jobs.extend(
            (METHOD_FUTURE_UTILITY, seed, checkpoint_dir / f"selector_future_utility_seed{seed}.pt")
            for seed in selector_seeds
        )
    if args.include_ablations:
        for ablation in ("A1_memory_similarity_only", "A2_memory_plus_human_anchor", "A3_plus_competition_margin", "A4_plus_prospective_state_drift"):
            selector_jobs.append(
                (f"ABLATION_{ablation}", args.ablation_seed, checkpoint_dir / f"selector_{ablation}_seed{args.ablation_seed}.pt")
            )
    for method, seed, checkpoint_path in selector_jobs:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing frozen selector checkpoint: {checkpoint_path}")
        selector = _load_selector(checkpoint_path, device)
        run_key = f"{method}_seed{seed}"
        for condition in conditions:
            actual_method = method if method in {METHOD_CORRECTNESS, METHOD_FUTURE_UTILITY} else METHOD_FUTURE_UTILITY
            result = _cached_cell(
                cache_dir,
                run_key,
                condition,
                lambda: _run_cell(
                    dataset,
                    manifest,
                    frozen_gru,
                    method=actual_method,
                    condition=condition,
                    selector=selector,
                    threshold=args.selector_threshold,
                    args=args,
                ),
            )
            result["reported_method"] = method
            result["selector_seed"] = seed
            cells[run_key][condition] = result
            record_cells[run_key][condition] = result["records"]

    baseline_key = METHOD_FROZEN_GRU
    paired: dict[str, Any] = {}
    for run_key, by_condition in record_cells.items():
        if run_key == baseline_key or run_key.startswith(METHOD_SIMILARITY + "@"):
            continue
        if not by_condition or baseline_key not in record_cells:
            continue
        cell_delta = {}
        for condition in conditions:
            if condition in by_condition and condition in record_cells[baseline_key]:
                cell_delta[condition] = paired_sequence_bootstrap_delta(
                    by_condition[condition], record_cells[baseline_key][condition], reps=args.bootstrap_reps, seed=args.bootstrap_seed
                )
        macro_conditions = [condition for condition in primary_condition_slugs() if condition in by_condition]
        if macro_conditions:
            paired[run_key] = {
                "cells": cell_delta,
                "primary_macro": paired_sequence_bootstrap_macro_delta(
                    {condition: by_condition[condition] for condition in macro_conditions},
                    {condition: record_cells[baseline_key][condition] for condition in macro_conditions},
                    reps=args.bootstrap_reps,
                    seed=args.bootstrap_seed,
                ),
            }

    baseline_h100 = (
        _h100(cells[baseline_key]["clean_0p00"]["summary"])
        if baseline_key in cells and "clean_0p00" in cells[baseline_key]
        else None
    )
    dev_candidates: list[dict[str, Any]] = []
    if args.split == "dev":
        for run_key, by_condition in cells.items():
            if "clean_0p00" not in by_condition:
                continue
            primary_values = [
                _h100(by_condition[condition]["summary"])
                for condition in primary_condition_slugs()
                if condition in by_condition
            ]
            clean_value = _h100(by_condition["clean_0p00"]["summary"])
            if primary_values and all(value is not None for value in primary_values):
                dev_candidates.append(
                    {
                        "run_key": run_key,
                        "dev_clean_H100": clean_value,
                        "dev_robust_H100_macro": sum(primary_values) / len(primary_values),
                        "clean_ok": _clean_ok(clean_value, baseline_h100),
                    }
                )
        dev_candidates.sort(key=lambda item: (item["clean_ok"], item["dev_robust_H100_macro"]), reverse=True)

    output = {
        "stage": "N72R19R1",
        "goal": "Selective Identity Memory Update",
        "goal_reference": "outputs/N72R19R1/FINAL_GOAL.json",
        "central_question": "Can selective observation updates preserve the strong learned identity memory while preventing corrupted observations from damaging future identity recognition?",
        "split": args.split,
        "sequence_count": len({anchor.sequence for anchor in dataset.anchors}),
        "episode_count": len(dataset),
        "conditions": list(conditions),
        "threshold_policy": {
            "selector_threshold": args.selector_threshold,
            "b3_dev_scan": b3_thresholds if args.split == "dev" else "not scanned on val",
            "b3_final_threshold": args.b3_final_threshold,
            "val_thresholds_frozen_from_dev": args.split == "val",
        },
        "summaries": {run_key: _summary_table(by_condition) for run_key, by_condition in cells.items()},
        "selector_seed_aggregates": _selector_seed_aggregates(cells, conditions),
        "paired_deltas_vs_b2": paired,
        "dev_selection_candidates": dev_candidates,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path.resolve()), "runs": len(cells), "conditions": len(conditions)}, sort_keys=True))


if __name__ == "__main__":
    main()
