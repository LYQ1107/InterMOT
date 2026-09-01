#!/usr/bin/env python3
"""Targeted regression for repair2 split indices and single-objective estimator."""

from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n48_repair1b_stage03_train import DATASET, DATASET_MANIFEST, AMENDMENT, accumulate_one_epoch, evaluate, split_indices

SCRIPT = ROOT / "scripts/n48_repair1b_stage03_train.py"


def main() -> None:
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8")); amendment = json.loads(AMENDMENT.read_text(encoding="utf-8")); data = np.load(DATASET); indices = split_indices(data, manifest)
    source = SCRIPT.read_text(encoding="utf-8"); tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "evaluate"]
    call_map = {node.args[7].id: node.args[8].id for node in calls if len(node.args) > 8 and isinstance(node.args[7], ast.Name) and isinstance(node.args[8], ast.Name)}
    checks = {
        "evaluate_train_pair_argument_is_train_pairs": call_map.get("train_pairs") == "train_cells",
        "evaluate_validation_pair_argument_is_val_pairs": call_map.get("val_pairs") == "val_cells",
        "evaluate_pair_indices_not_split_labels": "pair_split" not in call_map,
        "train_pairs_exact": np.array_equal(indices["train_pairs"], np.flatnonzero(data["pair_split"] == 0)),
        "validation_pairs_exact": np.array_equal(indices["validation_pairs"], np.flatnonzero(data["pair_split"] == 1)),
        "holdout_pairs_exact": np.array_equal(indices["holdout_pairs"], np.flatnonzero(data["pair_split"] == 2)),
        "pair_index_union_exact": len(np.unique(np.concatenate([indices["train_pairs"], indices["validation_pairs"], indices["holdout_pairs"]]))) == manifest["pair_count"],
        "train_validation_holdout_disjoint": not (set(indices["train_pairs"]) & set(indices["validation_pairs"]) or set(indices["train_pairs"]) & set(indices["holdout_pairs"]) or set(indices["validation_pairs"]) & set(indices["holdout_pairs"])),
        "manifest_pair_count": sum(len(indices[key]) for key in ("train_pairs", "validation_pairs", "holdout_pairs")) == manifest["pair_count"],
        "manifest_cell_counts": len(indices["train_cells"]) == manifest["counts"]["train_positive"] + manifest["counts"]["train_negative"] and len(indices["validation_cells"]) == manifest["counts"]["validation_positive"] + manifest["counts"]["validation_negative"] and len(indices["holdout_cells"]) == manifest["counts"]["holdout_positive"] + manifest["counts"]["holdout_negative"],
        "evaluate_signature_requires_pair_ids": list(inspect.signature(evaluate).parameters)[7] == "pair_ids",
        "one_optimizer_step_per_epoch": inspect.getsource(accumulate_one_epoch).count("optimizer.step()") == 1,
        "pair_and_cell_gradients_accumulated_before_step": "optimizer.zero_grad(set_to_none=True)" in inspect.getsource(accumulate_one_epoch) and inspect.getsource(accumulate_one_epoch).find("optimizer.step()") > inspect.getsource(accumulate_one_epoch).find("cell_batch"),
        "no_holdout_selection": amendment["evaluation"]["holdout_used_for_selection"] is False and amendment["selection"]["holdout_used_for_selection"] is False if "selection" in amendment else amendment["evaluation"]["holdout_used_for_selection"] is False,
        "fixed_coefficients": amendment["objective"]["coefficients"] == {"rank": 1.0, "cell_bce": 0.25, "uncertainty_bce": 0.25, "residual_l2": 0.001},
        "production_false": amendment["runtime"]["production_authorized"] is False,
        "runtime_gt_false": amendment["runtime"]["runtime_future_gt_used"] is False,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N48_R1_REPAIR2_STAGE_03_TARGETED_REGRESSION_V1", "command": ["python", "scripts/n48_repair1b_stage03_targeted_regression.py"], "inputs": {"amendment": str(AMENDMENT), "dataset": str(DATASET), "dataset_manifest": str(DATASET_MANIFEST), "training_script": str(SCRIPT)}, "outputs": {}, "metrics": {"pair_counts": {key: int(len(indices[key])) for key in ("train_pairs", "validation_pairs", "holdout_pairs")}, "cell_counts": {key: int(len(indices[key])) for key in ("train_cells", "validation_cells", "holdout_cells")}, "evaluate_call_map": call_map}, "gate_checks": checks, "failure_root_cause": "Targeted regression proves evaluate receives complete split-specific pair index sets and the objective is accumulated with one optimizer step per epoch." if all(checks.values()) else "Split or objective implementation regression failed; preserve and repair before training.", "next_action": "Run actual GPU0 repair2 training." if all(checks.values()) else "Preserve failure and repair first.", "runtime_future_gt_used": False}
    (ROOT / "outputs/n48/repair1b/stage_03_targeted_regression.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
