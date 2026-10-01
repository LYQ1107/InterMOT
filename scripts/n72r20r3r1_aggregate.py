#!/usr/bin/env python3
"""Aggregate independent N72R20R3R1 LOSO workers without retraining."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from scripts.n72r20r3r1_train_eval import (
    MODEL_SPECS,
    OUTPUT_DIR,
    SEEDS,
    SEQUENCES,
    _aggregate_report,
    _baseline_report,
    _bootstrap,
    _json,
    _write_json,
    aggregate_seed_predictions,
    per_sequence_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    worker_paths = sorted(Path(path) for path in glob.glob(str(args.root / "workers" / "*/worker_result.json")))
    expected = {(name, seed) for name, _, _ in MODEL_SPECS for seed in SEEDS}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for path in worker_paths:
        payload = _json(path)
        models = payload.get("models", [])
        seeds = payload.get("seeds", [])
        if len(models) != 1 or len(seeds) != 1:
            raise RuntimeError(f"worker must contain exactly one model/seed: {path}")
        key = (str(models[0]), int(seeds[0]))
        if key in found:
            raise RuntimeError(f"duplicate worker: {key}")
        found[key] = payload
    missing = sorted(expected.difference(found))
    if missing:
        raise RuntimeError(f"missing workers: {missing}")

    prediction_archive: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for (name, seed), payload in sorted(found.items()):
        prediction_archive.setdefault(name, {})[str(seed)] = payload["prediction_archive"][name][str(seed)]
        all_records.extend(payload["records"])
        checkpoints.extend(payload.get("checkpoint_records", []))
        worker_summaries.append({"model": name, "seed": seed, "source_commit": payload.get("source_commit"), "records": len(payload.get("records", []))})

    primary_reports: dict[str, dict[str, Any]] = {}
    seed_summary: dict[str, Any] = {}
    for name, _, _ in MODEL_SPECS:
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            predictions = prediction_archive[name][str(seed)]
            report = _aggregate_report(predictions)
            report.update({"model": name, "seed": seed})
            per_seed.append(report)
        combined = aggregate_seed_predictions([prediction_archive[name][str(seed)] for seed in SEEDS])
        primary_reports[name] = _aggregate_report(combined)
        fields = (
            "false_present_rate",
            "open_set_correct_identification_recall",
            "macro_false_present_rate",
            "macro_open_set_correct_identification_recall",
            "none_recall_on_absent",
            "nll",
            "brier",
            "ece",
            "auroc_present",
            "auprc_present",
        )
        seed_summary[name] = {
            "seeds": list(SEEDS),
            "per_seed": per_seed,
            "mean_std": {
                field: {
                    "mean": float(np.mean([float(item[field]) for item in per_seed if item.get(field) is not None])),
                    "std": float(np.std([float(item[field]) for item in per_seed if item.get(field) is not None])),
                }
                for field in fields
                if any(item.get(field) is not None for item in per_seed)
            },
            "three_seed_aggregate": primary_reports[name],
            "mean_seed_gate": all(bool(item.get("static_gate", {}).get("pass")) for item in per_seed),
        }

    selection = _json(args.root / "training" / "model_selection.json")
    selected_name = str(selection["selected_model"])
    selected_predictions = aggregate_seed_predictions([prediction_archive[selected_name][str(seed)] for seed in SEEDS])
    selected_report = primary_reports[selected_name]
    _write_json(
        args.root / "training" / "seed_results.json",
        {
            "stage": "N72R20R3R1",
            "status": "PASS_FORMAL_LOSO_COMPLETE",
            "source_commit": worker_summaries[0]["source_commit"],
            "folds": len(SEQUENCES),
            "heldout_sequences": list(SEQUENCES),
            "seeds": list(SEEDS),
            "models": [name for name, _, _ in MODEL_SPECS],
            "model_selection": selection,
            "seed_summary": seed_summary,
            "checkpoint_records": checkpoints,
            "worker_summaries": worker_summaries,
            "runtime_future_gt_used": False,
            "candidate_created": False,
        },
    )
    _write_json(
        args.root / "static_eval" / "loso_results.json",
        {
            "stage": "N72R20R3R1",
            "status": "PASS_FORMAL_LOSO_COMPLETE",
            "reports": primary_reports,
            "seed_summary": seed_summary,
            "records": all_records,
            "runtime_future_gt_used": False,
            "candidate_created": False,
        },
    )
    _write_json(args.root / "static_eval" / "per_sequence_results.json", {"stage": "N72R20R3R1", "model": selected_name, "records": per_sequence_metrics(selected_predictions)})
    _write_json(args.root / "static_eval" / "bootstrap.json", _bootstrap(selected_predictions))
    _write_json(args.root / "static_eval" / "calibration.json", {"model": selected_name, "metrics": selected_report, "calibration": "V1 threshold fitted on internal sequences only; V2 NONE is learned without held-out tuning"})
    _write_json(args.root / "static_eval" / "selected_report.json", selected_report)
    _write_json(args.root / "static_eval" / "baseline_v0_r3_b4.json", _baseline_report())

    # Keep one canonical small-checkpoint directory in the final output tree.
    # Worker directories remain outside Git and contain the full provenance.
    final_models = args.root / "models"
    final_models.mkdir(parents=True, exist_ok=True)
    for record in checkpoints:
        source = Path(record["path"])
        destination = final_models / source.name
        if source.exists():
            shutil.copy2(source, destination)
            record["canonical_path"] = str(destination.relative_to(args.root.parent.parent))
            record.pop("path", None)
    _write_json(args.root / "training" / "checkpoint_manifest.json", {"stage": "N72R20R3R1", "records": checkpoints, "runtime_future_gt_used": False})
    print(json.dumps({"status": "PASS_FORMAL_LOSO_COMPLETE", "selected_model": selected_name, "static_gate": selected_report["static_gate"], "metrics": {key: selected_report.get(key) for key in ("false_present_rate", "open_set_correct_identification_recall", "macro_false_present_rate", "macro_open_set_correct_identification_recall")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
