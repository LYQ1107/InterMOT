#!/usr/bin/env python3
"""Validate the completed N25-R state and write a checksummed manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(".")
OUT = ROOT / "outputs/n25r"
MANIFEST = OUT / "manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    repair = load_json(OUT / "protocol_repair/repair_validation.json")
    feature = load_json(OUT / "feature_alignment.json")
    negative = load_json(OUT / "negative_ledger_summary.json")
    simulated = load_json(OUT / "human_negative_ledger/simulated_train_negative_summary.json")
    gate = load_json(OUT / "r5_summary.json")

    assert repair["status"] == "PASS"
    assert repair["actual_groups"] == repair["expected_groups"] == 1000
    assert repair["duplicate_candidate_keys"] == 0
    assert feature["status"] == "PASS_F1_DIRECT_CACHE__FAIL_OBJECT_CONDITIONED_F2_F4"
    assert feature["global_checks"]["duplicate_candidate_row_keys"] == 0
    assert feature["global_checks"]["future_gt_used_in_features"] is False
    assert negative["admissible_unique_human_negative_keys"] == 53
    assert simulated["human_explicit_negative_writes"] == 267
    assert simulated["current_correction_never_scores_current_event"] is True
    assert gate["status"] == "PARTIAL_N25R_FEATURE_SIGNAL"
    assert gate["passed_methods"] == []
    assert gate["rank_signal_methods"] == ["B10_EXPLICIT_NEGATIVE"]
    assert gate["downstream_authorization"] == "STOP_BEFORE_CCRIM_UNION_FULL_LOOP"
    assert gate["val25_read"] is False

    expected_sequences = {"train30": 27, "cal10": 9}
    done_counts: dict[str, dict[str, int]] = {}
    for backbone in ("clipreid", "sam3_f1"):
        done_counts[backbone] = {}
        for split, expected in expected_sequences.items():
            count = len(list((OUT / "candidate_aligned_features" / backbone / split).glob("*.done")))
            assert count == expected, (backbone, split, count, expected)
            done_counts[backbone][split] = count

    required_docs = [
        ROOT / "docs/N25R_INITIAL_ANALYSIS.md",
        ROOT / "docs/N25R_PROTOCOL_AUDIT.md",
        ROOT / "docs/N25R_TARGETED_OFFICIAL_CODE_AUDIT.md",
        ROOT / "docs/N25R_CANDIDATE_ALIGNMENT_AUDIT.md",
        ROOT / "docs/N25R_NEGATIVE_PROVENANCE_AUDIT.md",
        ROOT / "docs/N25R_FINAL_REPORT.md",
    ]
    required_outputs = [
        OUT / "commands.md",
        OUT / "RESUME.md",
        OUT / "information_gate.csv",
        OUT / "per_sequence.csv",
        OUT / "feature_alignment.json",
        OUT / "negative_ledger_summary.json",
        OUT / "frozen_config.json",
        OUT / "calibration.csv",
        OUT / "precision_risk_coverage.csv",
        OUT / "stratified_metrics.csv",
        OUT / "bootstrap_sequence.json",
        OUT / "b10_memory_audit.json",
        OUT / "r5_summary.json",
    ]
    required_scripts = [
        ROOT / "scripts/n25r_protocol_and_negative_audit.py",
        ROOT / "scripts/n25r_alignment_smoke.py",
        ROOT / "scripts/n25r_validate_repaired_cache.py",
        ROOT / "scripts/n25r_extract_candidate_features.py",
        ROOT / "scripts/n25r_validate_feature_cache.py",
        ROOT / "scripts/n25r_build_train_negative_ledger.py",
        ROOT / "scripts/n25r_r5_gate.py",
        ROOT / "scripts/n25r_finalize_manifest.py",
        ROOT / "scripts/run_n20_all_candidate_shadow.py",
        ROOT / "scripts/build_n25_dataset_and_gate.py",
    ]
    for path in required_docs + required_outputs + required_scripts:
        assert path.is_file() and path.stat().st_size > 0, path

    all_paths = set(required_docs + required_scripts)
    all_paths.update(
        path
        for path in OUT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and not path.name.endswith(".tmp")
        and "__pycache__" not in path.parts
    )
    artifacts = [artifact_record(path) for path in sorted(all_paths)]

    manifest = {
        "schema_version": 1,
        "phase": "N25-R",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "repository": {"is_git_repository": False},
        "python": platform.python_version(),
        "status": gate["status"],
        "downstream_authorization": gate["downstream_authorization"],
        "val25_read": False,
        "protocol": {
            "old_error": "INCOMPLETE_TRAIN_SHARD_0_335_OF_500",
            "repair_status": repair["status"],
            "train_groups": repair["actual_groups"],
            "train_candidate_rows": repair["rows"],
            "cal_groups": 1200,
            "cal_candidate_rows": 5781,
            "old_files_preserved": True,
        },
        "alignment": {
            "status": feature["status"],
            "object_conditioned_F2_F4_authorized": False,
            "candidate_box_F1_authorized": True,
            "global_checks": feature["global_checks"],
        },
        "feature_done_counts": done_counts,
        "negative_ledger": {
            "raw_occurrences": negative["occurrences"],
            "admissible_existing_keys": negative["admissible_unique_human_negative_keys"],
            "admissible_by_split": negative["admissible_by_split"],
            "simulated_train_corrections": simulated["simulated_human_corrections"],
            "simulated_train_explicit_negatives": simulated["human_explicit_negative_writes"],
            "causal_current_event_exclusion": simulated["current_correction_never_scores_current_event"],
        },
        "r5": {
            "primary_history": gate["primary_history"],
            "passed_methods": gate["passed_methods"],
            "rank_signal_methods": gate["rank_signal_methods"],
            "method_gate": gate["gate"],
            "information_rows": csv_rows(OUT / "information_gate.csv"),
            "per_sequence_rows": csv_rows(OUT / "per_sequence.csv"),
            "calibration_rows": csv_rows(OUT / "calibration.csv"),
            "risk_curve_rows": csv_rows(OUT / "precision_risk_coverage.csv"),
            "stratified_rows": csv_rows(OUT / "stratified_metrics.csv"),
        },
        "conditional_stages": {
            "CCRIM_C0_C1": "NOT_RUN_R5_GATE_FAILED",
            "candidate_union": "NOT_RUN_R5_GATE_FAILED",
            "full_loop": "NOT_RUN_R5_GATE_FAILED",
            "trackeval": "NOT_RUN_NO_FULL_LOOP_OUTPUT",
            "val25": "NOT_READ",
        },
        "resources": {
            "maximum_concurrent_gpus": 4,
            "measured_core_gpu_hours_approx": 1.24,
            "protocol_repair_gpu_seconds": 3022.0,
            "alignment_smoke_gpu_seconds": 157.50900220870972,
            "feature_sequence_gpu_seconds": 1218.798169851303,
        },
        "warnings_and_repairs": [
            "original train shard incomplete; repaired in a new directory",
            "object-conditioned alignment failed; F2/F3/F4 frozen",
            "first cal feature launch used wrong filesystem split and wrote no admitted cal artifact; logs preserved",
            "empty-slice warnings in R5 correspond to invalid zero-observation rows retained as NaN",
            "B10 rank signal failed frozen commit safety; downstream stages stopped",
        ],
        "artifact_count_excluding_manifest": len(artifacts),
        "artifacts": artifacts,
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifacts": manifest["artifact_count_excluding_manifest"],
                "manifest": relative(MANIFEST),
                "validation": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
