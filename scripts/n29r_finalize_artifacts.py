#!/usr/bin/env python3
"""Materialize N29-R audit summaries without loading model weights.

The script consumes only completed/recoverable N29-R artifacts.  It performs
episode-level aggregation (never frame-level pseudo-replication), writes the
frozen protocol and supervision decomposition, and records the exact
optimization diagnostic used by the mechanism gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n29r"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def bootstrap_ci(values: list[float], seed: int = 2901, draws: int = 2000) -> dict[str, Any]:
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "ci95": [None, None]}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "bootstrap_seed": int(seed),
        "bootstrap_draws": int(draws),
        "unit": "episode",
    }


def metric_summary(rows: list[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    result = bootstrap_ci(values)
    result["missing_count"] = int(sum(row.get(metric) is None for row in rows))
    return result


def aggregate_paired(payload: Mapping[str, Any]) -> dict[str, Any]:
    episode_rows = [row for row in payload.get("episode_results", []) if row.get("status") == "PASS"]
    horizons = ("5", "10", "20")
    metrics = (
        "mean_box_iou_visible",
        "success_at_0_5_visible",
        "missing_prediction_rate_visible",
        "error_count_visible",
        "mask_area_drift",
    )
    result: dict[str, Any] = {
        "protocol": "N29-R3-PAIRED-REPLAY-AGGREGATE",
        "source_status": payload.get("status"),
        "episode_count_requested": payload.get("episode_count_requested"),
        "episode_count_processed": payload.get("episode_count_processed"),
        "episode_count_pass": len(episode_rows),
        "episode_count_failed": int(sum(row.get("status") != "PASS" for row in payload.get("episode_results", []))),
        "val25_read": bool(payload.get("val25_read", False)),
        "future_gt_used_for_selection": bool(payload.get("future_gt_used_for_selection", False)),
        "branches": {},
        "lora_vs_write_only": {},
        "update": {
            "commit_count": 0,
            "rollback_count": 0,
            "not_run_count": 0,
            "finite_diagnostic_count": 0,
            "deterministic_support_decrease_count": 0,
        },
    }
    for branch_name in ("anchor_no_correction", "correction_write_only", "correction_plus_lora", "correction_plus_zero_update", "remove_latest_correction"):
        branch_rows = []
        for episode in episode_rows:
            branch = episode.get("branches", {}).get(branch_name, {})
            if branch.get("metrics"):
                branch_rows.append(branch["metrics"])
        result["branches"][branch_name] = {
            horizon: {metric: metric_summary([row[horizon] for row in branch_rows], metric) for metric in metrics}
            for horizon in horizons
        }
    for horizon in horizons:
        deltas = [episode.get("paired_delta", {}).get(horizon, {}) for episode in episode_rows]
        result["lora_vs_write_only"][horizon] = {
            metric: metric_summary(deltas, metric)
            for metric in metrics
        }
        iou_values = [float(row["mean_box_iou_visible"]) for row in deltas if row.get("mean_box_iou_visible") is not None]
        result["lora_vs_write_only"][horizon]["negative_transfer_rate_iou"] = (
            None if not iou_values else float(np.mean(np.asarray(iou_values) < 0.0))
        )
        result["lora_vs_write_only"][horizon]["positive_strict_ci_lower_iou"] = bool(
            result["lora_vs_write_only"][horizon]["mean_box_iou_visible"]["ci95"][0] is not None
            and result["lora_vs_write_only"][horizon]["mean_box_iou_visible"]["ci95"][0] > 0.0
        )
    for episode in episode_rows:
        update = episode.get("branches", {}).get("correction_plus_lora", {}).get("update", {})
        status = str(update.get("status", "NOT_RUN"))
        result["update"][{"COMMIT": "commit_count", "ROLLBACK": "rollback_count"}.get(status, "not_run_count")] += 1
        diagnostic = update.get("optimization_diagnostic", {})
        result["update"]["finite_diagnostic_count"] += int(diagnostic.get("finite", False) is True)
        result["update"]["deterministic_support_decrease_count"] += int(
            diagnostic.get("deterministic_support_loss_after") is not None
            and diagnostic.get("deterministic_support_loss_before") is not None
            and float(diagnostic["deterministic_support_loss_after"]) < float(diagnostic["deterministic_support_loss_before"])
        )
    result["r3_mechanism_gate"] = bool(
        result["source_status"] == "PASS"
        and result["episode_count_pass"] == result["episode_count_requested"]
        and result["update"]["commit_count"] == result["episode_count_pass"]
        and result["update"]["finite_diagnostic_count"] == result["episode_count_pass"]
        and result["update"]["deterministic_support_decrease_count"] == result["episode_count_pass"]
    )
    result["future_benefit_gate"] = {
        horizon: bool(result["lora_vs_write_only"][horizon]["positive_strict_ci_lower_iou"])
        for horizon in horizons
    }
    result["r3_future_benefit_gate"] = bool(result["r3_mechanism_gate"] and any(result["future_benefit_gate"].values()))
    return result


def optimization_summary(n29b: Mapping[str, Any], paired: Mapping[str, Any]) -> dict[str, Any]:
    sequences = n29b.get("sequence_results", [])
    official = [
        {
            "sequence": row.get("sequence"),
            "status": row.get("status"),
            "update_status": row.get("update", {}).get("status"),
            "diagnostic": row.get("update", {}).get("optimization_diagnostic", {}),
            "forward_tensor_diagnostic": row.get("update", {}).get("forward_tensor_diagnostic", {}),
        }
        for row in sequences
    ]
    return {
        "protocol": "N29-R2-OPTIMIZATION-DIAGNOSTIC",
        "official_pilot": official,
        "official_pilot_status": n29b.get("status"),
        "official_pilot_update_statuses": [row.get("update", {}).get("status") for row in sequences],
        "paired_episode_diagnostic_source": "outputs/n29r/paired_replay_results.json",
        "paired_episode_count": len([row for row in paired.get("episode_results", []) if row.get("status") == "PASS"]),
        "validator_requirements": {
            "finite": True,
            "parameter_delta_l2_positive": True,
            "logit_delta_linf_positive": True,
            "deterministic_support_loss_after_lt_before": True,
            "rollback_on_failure": True,
        },
        "val25_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n29b", type=Path, default=OUT / "n29b_result.json")
    parser.add_argument("--manifest", type=Path, default=OUT / "hard_episode_manifest.json")
    parser.add_argument("--paired", type=Path, default=OUT / "paired_replay_results.json")
    parser.add_argument("--association", type=Path, default=OUT / "association_results.json")
    args = parser.parse_args()

    n29b = json.loads(args.n29b.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paired = json.loads(args.paired.read_text(encoding="utf-8")) if args.paired.is_file() else {"status": "NOT_RUN", "episode_results": []}
    association = json.loads(args.association.read_text(encoding="utf-8")) if args.association.is_file() else {"status": "NOT_RUN"}
    paired_summary = aggregate_paired(paired)
    optimization = optimization_summary(n29b, paired)
    supervision = {
        "protocol": "N29-R-SUPERVISION-DECOMPOSITION",
        "main_train_fold_source": "BOX_DERIVED_PSEUDO_MASK",
        "box_derived_pseudo_mask": {
            "status": "USED",
            "count": int(len([row for row in paired.get("episode_results", []) if row.get("status") == "PASS"])),
            "definition": "explicit rectangle generated from the current verified box; used as decoder target",
        },
        "accepted_mask_correction": {"status": "NOT_AVAILABLE", "reason": "pinned official multiplex endpoint exposes box prompts only"},
        "confirmed_mask_correction": {"status": "NOT_AVAILABLE", "reason": "no confirmed mask ledger in the authorized box-only train protocol"},
        "oracle_gt_mask": {"status": "NOT_AVAILABLE", "reason": "DanceTrack train annotations provide boxes, not segmentation masks"},
        "oracle_separate_from_main": True,
        "val25_read": False,
    }
    checkpoint = ROOT / "checkpoints" / "sam3.1_mirror" / "sam3.1_multiplex.pt"
    protocol = {
        "protocol": "N29-R",
        "frozen_at": "2026-08-26T00:00:00+08:00",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "sam3_commit": git_commit(ROOT / "third_party" / "sam3"),
        "dataset": {
            "name": "DanceTrack",
            "fold": "train/train_fold",
            "hard_manifest": str(args.manifest.relative_to(ROOT)),
            "hard_manifest_sha256": sha256(args.manifest),
            "sequence_count": manifest.get("sequence_count"),
            "episode_count": manifest.get("episode_count"),
            "selection_frozen_before_paired": manifest.get("selection_frozen_before_paired"),
            "future_gt_used_for_selection": manifest.get("future_gt_used_for_selection"),
        },
        "identity_evaluation": {
            "binding": ["dataset_identity", "public_id", "sam_object_id"],
            "target_namespace": "dataset_identity",
            "missing_target": "None; excluded from visible-only means",
            "missing_prediction_on_visible": "zero IoU and counted as missing",
            "val25_read": False,
            "test_labels_used": False,
        },
        "paired_branches": [
            "anchor_no_correction",
            "correction_write_only",
            "correction_plus_lora",
            "correction_plus_zero_update",
            "remove_latest_correction_alias_to_anchor",
        ],
        "future_horizons": [5, 10, 20],
        "statistics": {"unit": "episode", "bootstrap_draws": 2000, "seed": 2901, "ci": "95% percentile"},
        "supervision": supervision,
        "association": {
            "candidate_union": "original+official_decoder",
            "score": "frozen CLIP-ReID B10 with audited positive/explicit-negative memory",
            "assignment": "one global Hungarian with per-identity NONE columns",
            "relation_cache_used": False,
            "trackeval_authorized": False,
            "delivery_status": association.get("delivery_status"),
            "correction_conditioned_pairing": "NOT_RUN_R3_FUTURE_BENEFIT_GATE_FALSE",
            "full_loop_delivered": False,
        },
        "resource_policy": {"max_simultaneous_gpus": 4, "minimum_data1_free_gib": 40},
    }
    d_gate = {
        "trigger_condition": "mechanism fits AND paired future benefit absent AND confirmed/oracle supervision hints generalization",
        "r3_mechanism_gate": paired_summary.get("r3_mechanism_gate", False),
        "r3_future_benefit_gate": paired_summary.get("r3_future_benefit_gate", False),
        "confirmed_oracle_hint": False,
        "status": "NOT_TRIGGERED_NO_CONFIRMED_ORACLE_SUPERVISION",
        "reason": "box-only DanceTrack train protocol has neither accepted/confirmed mask nor legal oracle mask supervision",
        "d_training": "NOT_RUN",
        "val25_read": False,
    }
    write_json(OUT / "frozen_protocol.json", protocol)
    write_json(OUT / "optimization_diagnostic.json", optimization)
    write_json(OUT / "paired_replay_summary.json", paired_summary)
    write_json(OUT / "supervision_decomposition.json", supervision)
    write_json(OUT / "n29d_gate.json", d_gate)
    write_json(OUT / "association_summary.json", {
        "status": association.get("status"),
        "delivery_status": association.get("delivery_status"),
        "cases": association.get("case_count"),
        "unaffected_identity_regression": association.get("unaffected_identity_regression"),
        "correction_conditioned_pairing": "NOT_RUN_R3_FUTURE_BENEFIT_GATE_FALSE",
        "full_loop_delivered": False,
        "trackeval_authorized": False,
        "relation_cache_used": association.get("relation_cache_used", False),
        "val25_read": False,
    })
    print(json.dumps({
        "paired_status": paired_summary.get("source_status"),
        "paired_pass": paired_summary.get("episode_count_pass"),
        "r3_mechanism_gate": paired_summary.get("r3_mechanism_gate"),
        "r3_future_benefit_gate": paired_summary.get("r3_future_benefit_gate"),
        "association_status": association.get("status"),
        "d_status": d_gate["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
