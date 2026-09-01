#!/usr/bin/env python3
"""Targeted post-training checks for the isolated N47 checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import (
    CHECKPOINT,
    DATASET,
    DATASET_MANIFEST,
    N42_RUNTIME,
    PROTOCOL,
    SEED,
    TRAIN_MANIFEST,
    VARIANTS,
    event_map,
    feature_matrix,
    load,
    load_checkpoint,
    predict_logit,
    score_matrix,
    sha256,
    write_json,
)


def main() -> None:
    train = load(TRAIN_MANIFEST)
    data = load(DATASET_MANIFEST)
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    event_id = sorted(event_map())[0]
    source = load(N42_RUNTIME / f"{event_id}.json")
    audit = source["variants"]["M2"]["branches"]["memory_write=True"]["future_trace"][0]["candidate_audit"]
    features = feature_matrix(audit, int(audit["frame"]) - int(event_map()[event_id]["frame"]))
    logits = predict_logit(model, features)
    checks = {
        "training_manifest_pass": train.get("status") == "PASS" and train.get("actual_full_training") is True,
        "checkpoint_hash_matches_manifest": sha256(CHECKPOINT) == train.get("checkpoint_sha256"),
        "dataset_hash_matches_manifest": sha256(DATASET) == data.get("dataset_sha256"),
        "checkpoint_production_authorized_false": checkpoint.get("production_authorized") is False,
        "checkpoint_seed_frozen": checkpoint.get("seed") == SEED and train.get("seed") == SEED,
        "checkpoint_reload": model is not None,
        "real_feature_shape": features.shape[1] == 8,
        "real_prediction_finite": bool(np.all(np.isfinite(logits))),
        "runtime_source_future_gt_false": audit.get("runtime_future_gt_used") is False,
        "all_variants_declared": tuple(VARIANTS) == ("M0", "M1", "M2", "M3", "M4"),
        "n44_checkpoint_untouched_hash": sha256(ROOT / "outputs/n44/training/n44_assignment_aware.pt") == "0b5e750f5d9569f71ae887595c1d88d4d625f120f8a3811f2598a852cf82348f",
        "n47_protocol_exists": load(PROTOCOL).get("status") == "FROZEN",
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": "N47_STAGE_03_TARGETED_REGRESSION_V1",
        "command": ["python", "scripts/n47_stage03_targeted_regression.py"],
        "inputs": {"checkpoint": str(CHECKPOINT), "training_manifest": str(TRAIN_MANIFEST), "dataset_manifest": str(DATASET_MANIFEST), "frozen_n44_checkpoint": str(ROOT / "outputs/n44/training/n44_assignment_aware.pt")},
        "outputs": {"regression": str(ROOT / "outputs/n47_global_probe/stage_03_targeted_regression.json")},
        "metrics": {"checkpoint_sha256": sha256(CHECKPOINT), "dataset_sha256": sha256(DATASET), "sample_event": event_id, "sample_feature_rows": int(features.shape[0]), "sample_logit_min": float(np.min(logits)), "sample_logit_max": float(np.max(logits))},
        "gate_checks": checks,
        "failure_root_cause": "Checkpoint and causal runtime feature contract are validated before the full replay; any failure is a training-stage failure, not a PASS.",
        "next_action": "Run the complete GT-free same-source global-assignment replay only if all checks pass.",
        "runtime_future_gt_used": False,
        "gt_loaded_posthoc": False,
    }
    write_json(ROOT / "outputs/n47_global_probe/stage_03_targeted_regression.json", result)
    print(json.dumps({"status": result["status"], "checkpoint_sha256": result["metrics"]["checkpoint_sha256"]}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
