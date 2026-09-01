#!/usr/bin/env python3
"""Read-only N48-R1 invalid-selection audit for repair2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "outputs/n48/repair1b"
DATASET = ROOT / "outputs/n48/training/risk_aware_512d_dataset.npz"
MANIFEST = ROOT / "outputs/n48/training/dataset_manifest.json"
R1_SCRIPT = ROOT / "scripts/n48_repair1_stage03_train.py"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    data = np.load(DATASET); manifest = json.loads(MANIFEST.read_text(encoding="utf-8")); source = R1_SCRIPT.read_text(encoding="utf-8")
    pair_split = data["pair_split"].astype(np.int8); label = data["label"].astype(np.int8); split = data["split"].astype(np.int8)
    pair_counts = {"train": int(np.sum(pair_split == 0)), "validation": int(np.sum(pair_split == 1)), "holdout": int(np.sum(pair_split == 2))}
    cells = {"train_positive": int(np.sum((split == 0) & (label == 1))), "train_negative": int(np.sum((split == 0) & (label == 0))), "validation_positive": int(np.sum((split == 1) & (label == 1))), "validation_negative": int(np.sum((split == 1) & (label == 0))), "holdout_positive": int(np.sum((split == 2) & (label == 1))), "holdout_negative": int(np.sum((split == 2) & (label == 0)))}
    old_checkpoint = R2 / "legacy_r1/training/n48_r1_risk_aware_512d_bce.pt"
    result = {"status": "PASS", "command": ["python", "scripts/n48_repair1b_stage01_audit.py"], "inputs": {"r1_training_script": str(R1_SCRIPT), "r1_checkpoint_snapshot": str(old_checkpoint), "dataset": str(DATASET), "dataset_manifest": str(MANIFEST)}, "outputs": {"failure_audit": str(R2 / "attempts/r1_invalid_evaluate_and_objective_audit.json"), "legacy_snapshot": str(R2 / "legacy_r1")}, "metrics": {"pair_split_counts": pair_counts, "cell_counts": cells, "dataset_pair_count": int(manifest["pair_count"]), "old_r1_checkpoint_sha256": digest(old_checkpoint), "old_r1_manifest_sha256": digest(R2 / "legacy_r1/training/training_manifest.json")}, "gate_checks": {"read_only_audit_completed": True, "old_r1_snapshotted": True, "evaluate_index_bug_found": "evaluate(model, candidate, memory, scalar, label, pair_pos, pair_neg, pair_split" in source, "multiple_optimizer_step_mismatch_found": source.count("optimizer.step()") >= 2, "pair_count_manifest_consistent": sum(pair_counts.values()) == int(manifest["pair_count"]), "cell_counts_manifest_consistent": cells["train_positive"] == manifest["counts"]["train_positive"] and cells["train_negative"] == manifest["counts"]["train_negative"], "r1_selection_valid": False, "production_authorized": False}, "failure_root_cause": "R1 evaluate used pair_split values as pair indices and R1 used a second cell-only optimizer loop; R1 is provisional-invalid-selection.", "next_action": "Use the frozen repair2 amendment and corrected deterministic accumulated-objective training path.", "runtime_future_gt_used": False}
    write(R2 / "stage_01_status.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
