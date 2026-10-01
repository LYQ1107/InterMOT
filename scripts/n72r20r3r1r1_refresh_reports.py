#!/usr/bin/env python3
"""Refresh derived R3R1R1 reports after metric-schema-only changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sam3_intermot.identity_verification.explicit_none_verifier import ExplicitNoneVerifier
from scripts.n72r20r3r1r1_protocol import (
    EXPECTED_FRAMES,
    OUT,
    STATIC,
    STAGE,
    SEEDS,
    SEQUENCES,
    _json,
    _write_json,
    aggregate_report,
    _raw_predictions,
    load_dataset,
)
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-summary", default="PENDING")
    args = parser.parse_args()
    # Reconstruct the primary per-seed prediction tables from the sealed
    # outer-fold checkpoints.  This is evaluation-only; no fitting occurs.
    device = torch.device("cpu")
    dataset = load_dataset()
    checkpoint_manifest = _json(OUT / "training/checkpoint_manifest.json")
    seed_prediction_files = {}
    for seed in SEEDS:
        rows = []
        for record in checkpoint_manifest["records"]:
            if record["model"] != "V2_DUAL_STATE" or int(record["seed"]) != int(seed):
                continue
            payload = torch.load(record["path"], map_location=device, weights_only=False)
            model = ExplicitNoneVerifier(
                mode="V2_EXPLICIT_NONE",
                state_variant=str(payload["state_variant"]),
                projection_dim=int(payload.get("projection_dim", 32)),
                hidden_dim=int(payload.get("hidden_dim", 48)),
            ).to(device)
            model.load_state_dict(payload["state_dict"])
            examples = [item for item in dataset.examples if item.sequence == record["heldout_sequence"]]
            rows.extend(_raw_predictions(model, dataset, examples, device))
        if len(rows) != EXPECTED_FRAMES or len({(row["sequence"], row["frame"]) for row in rows}) != EXPECTED_FRAMES:
            raise RuntimeError(f"reconstructed primary seed table is not {EXPECTED_FRAMES} rows: {seed}")
        path = STATIC / f"seed_predictions_V2_DUAL_STATE_seed{seed}.jsonl.zst"
        from scripts.n72r20r3_common import write_zstd_jsonl

        write_zstd_jsonl(path, rows)
        seed_prediction_files[str(seed)] = str(path)
    formal_manifest = _json(STATIC / "formal_predictions_manifest.json")
    formal_manifest["seed_prediction_files"] = seed_prediction_files
    _write_json(STATIC / "formal_predictions_manifest.json", formal_manifest)
    split = _json(OUT / "training/split_manifest.json")
    split_by_heldout = {item["heldout_sequence"]: item for item in split["folds"]}
    formal_records = []
    for record in checkpoint_manifest["records"]:
        fold = split_by_heldout[record["heldout_sequence"]]
        formal_records.append(
            {
                "model": record["model"],
                "seed": record["seed"],
                "heldout_sequence": record["heldout_sequence"],
                "parameter_fit_sequences": fold["parameter_fit_sequences"],
                "internal_validation_sequence": fold["internal_validation_sequence"],
                "final_training_sequences": fold["all_nonheldout_sequences"],
                "outer_heldout_absent_from_fit": fold["outer_heldout_absent_from_fit"],
                "outer_heldout_absent_from_validation": fold["outer_heldout_absent_from_validation"],
                "selected_epoch": record["selected_epoch"],
                "v1_threshold_calibration_source": "internal_validation_sequence_only" if record["model"] == "V1_PAIRWISE_THRESHOLD_NONE" else "not_applicable",
                "architecture_selection_from_outer_scores": False,
            }
        )
    _write_json(OUT / "training/formal_records.json", {"stage": STAGE, "records": formal_records})
    reports = {}
    for path in sorted(STATIC.glob("predictions_*.jsonl.zst")):
        name = path.name.removeprefix("predictions_").removesuffix(".jsonl.zst")
        reports[name] = aggregate_report(read_zstd_jsonl(path))
    primary = reports["V2_DUAL_STATE"]
    primary_result = _json(STATIC / "primary_result.json")
    primary_result["metrics"] = primary
    primary_result["static_gate"] = primary["static_gate"]
    _write_json(STATIC / "primary_result.json", primary_result)
    ablations = _json(STATIC / "ablations.json")
    ablations["reports"] = reports
    _write_json(STATIC / "ablations.json", ablations)
    ranking = _json(STATIC / "ranking_diagnostics.json")
    ranking.update(
        {
            "rank_1_identity_accuracy": primary["rank_1_identity_accuracy"],
            "rank_2_identity_accuracy": primary["rank_2_identity_accuracy"],
            "rank_3_identity_accuracy": primary["rank_3_identity_accuracy"],
            "closed_set_MRR": primary["closed_set_MRR"],
        }
    )
    _write_json(STATIC / "ranking_diagnostics.json", ranking)
    result = _json(OUT / "FINAL_RESULT.json")
    result.update(
        {
            "closed_set_candidate_top1": primary["closed_set_candidate_top1_accuracy"],
            "closed_set_MRR": primary["closed_set_MRR"],
            "rank_1_identity_accuracy": primary["rank_1_identity_accuracy"],
            "rank_2_identity_accuracy": primary["rank_2_identity_accuracy"],
            "rank_3_identity_accuracy": primary["rank_3_identity_accuracy"],
            "tests_summary": args.tests_summary,
        }
    )
    _write_json(OUT / "FINAL_RESULT.json", result)
    status = _json(OUT / "stage_status.json")
    status["tests"] = args.tests_summary
    _write_json(OUT / "stage_status.json", status)
    report_path = OUT / "FINAL_REPORT.md"
    report = report_path.read_text()
    report = report.replace(
        f"- closed-set candidate top-1: `{result['closed_set_candidate_top1']}`\n- closed-set MRR: `{result['closed_set_MRR']}`",
        f"- closed-set candidate top-1 / Rank-1: `{result['closed_set_candidate_top1']}`\n- Rank-2: `{result['rank_2_identity_accuracy']}`\n- Rank-3: `{result['rank_3_identity_accuracy']}`\n- closed-set MRR: `{result['closed_set_MRR']}`",
    )
    report_lines = report.splitlines()
    for index, line in enumerate(report_lines):
        if line.startswith("Focused and full test summary:"):
            report_lines[index] = f"Focused and full test summary: {args.tests_summary}."
    report = "\n".join(report_lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"stage": STAGE, "primary_model": "V2_DUAL_STATE", "rank_2": result["rank_2_identity_accuracy"], "rank_3": result["rank_3_identity_accuracy"], "tests_summary": args.tests_summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
