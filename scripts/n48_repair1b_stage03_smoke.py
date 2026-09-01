#!/usr/bin/env python3
"""Deterministic pre-training smoke for N48-R1 repair2 contracts."""

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

from scripts.n48_repair1b_stage03_train import DATASET, DATASET_MANIFEST, AMENDMENT, split_indices, accumulate_one_epoch

SCRIPT = ROOT / "scripts/n48_repair1b_stage03_train.py"


def main() -> None:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    data = np.load(DATASET)
    indices = split_indices(data, manifest)
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    evaluate_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "evaluate"]
    pair_index_names = [node.args[7].id for node in evaluate_calls if len(node.args) > 7 and isinstance(node.args[7], ast.Name)]
    checks = {
        "amendment_frozen": amendment["status"] == "FROZEN_BEFORE_RETRAINING" and amendment["frozen_before_result"] is True,
        "dataset_hash_frozen": manifest["dataset_sha256"] == amendment["dataset_sha256"],
        "pair_indices_are_split_sets": pair_index_names.count("train_pairs") == 1 and pair_index_names.count("val_pairs") == 1 and "pair_split" not in pair_index_names,
        "pair_sets_complete": len(indices["train_pairs"]) + len(indices["validation_pairs"]) + len(indices["holdout_pairs"]) == manifest["pair_count"],
        "pair_sets_disjoint": len(set(indices["train_pairs"]) & set(indices["validation_pairs"])) == 0 and len(set(indices["train_pairs"]) & set(indices["holdout_pairs"])) == 0 and len(set(indices["validation_pairs"]) & set(indices["holdout_pairs"])) == 0,
        "cell_counts_manifest": len(indices["train_cells"]) == manifest["counts"]["train_positive"] + manifest["counts"]["train_negative"] and len(indices["validation_cells"]) == manifest["counts"]["validation_positive"] + manifest["counts"]["validation_negative"],
        "one_optimizer_step_source": inspect.getsource(accumulate_one_epoch).count("optimizer.step()") == 1,
        "no_separate_cell_optimizer_loop": "cell_order" not in inspect.getsource(accumulate_one_epoch),
        "fixed_normalization_recorded": amendment["objective"]["normalization"].startswith("pair terms divided by N_pair_train"),
        "holdout_not_selection": amendment["evaluation"]["holdout_used_for_selection"] is False,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "protocol": "N48_R1_REPAIR2_STAGE_03_SMOKE_V1", "command": ["python", "scripts/n48_repair1b_stage03_smoke.py"], "inputs": {"amendment": str(AMENDMENT), "dataset": str(DATASET), "dataset_manifest": str(DATASET_MANIFEST)}, "outputs": {}, "metrics": {"pair_counts": {key: int(len(indices[key])) for key in ("train_pairs", "validation_pairs", "holdout_pairs")}, "cell_counts": {key: int(len(indices[key])) for key in ("train_cells", "validation_cells", "holdout_cells")}}, "gate_checks": checks, "failure_root_cause": "Repair2 smoke verifies true split index sets and one accumulated optimizer step before training; it is not a training or efficacy result." if all(checks.values()) else "Repair2 contract smoke failed; do not train.", "next_action": "Run targeted split/objective regression." if all(checks.values()) else "Preserve failure and repair first.", "runtime_future_gt_used": False}
    (ROOT / "outputs/n48/repair1b/stage_03_smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
