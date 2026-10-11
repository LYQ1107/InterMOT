"""Read sealed state-source support; record only measured no-optimizer failures.

Never starts a fit or rollout below the resource floor. The original frozen
training function has an explicit supervision-failure return BEFORE its
storage/model/optimizer path. Only that already-measured zero-benefit branch
is executed here, guarded against constructing models/optimizers or weights.
This is not a trained-model failure, on-policy result, or complete MOT group.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts import n72r21r2_train_joint_state as study
from scripts.n72r21r2_common import (
    ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, utcnow,
    append_log, update_status,
)
from scripts.n72r21r2_main_terminal_recovery_v1 import no_live_scoped_workers
from scripts.n72r21r2_storage_guard_recovery_v1 import dead_idle_storage_owner
from sam3_intermot.one_click.event_authority_learning import correlated_sequence_weights

SELF = "scripts/n72r21r2_state_source_support_v1.py"
PREFIX = "on_policy/state_source_support_v1"
RESULT = PREFIX + "/ACTUAL_READBACK.json"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def row_digest(row):
    return hashlib.sha256(json.dumps(jsonable(row), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def record_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row_digest(row).encode()); digest.update(b"\n")
    return digest.hexdigest()


def describe(rows):
    weights = correlated_sequence_weights(rows)
    per_sequence = Counter()
    for row, weight in zip(rows, weights, strict=True):
        per_sequence[row["sequence"]] += float(weight)
    return {"rows": len(rows), "target_counts": {
        name: float(sum(r["target"][name] for r in rows)) for name in ("benefit", "risk")},
        "state_source_rows": dict(Counter(r["state_source"] for r in rows)),
        "actual_nonempty_sequences": sorted(per_sequence),
        "hierarchical_sequence_weight_mass": dict(per_sequence),
        "rows_features_labels_dtype_shape_order_sha256": record_digest(rows),
        "weight_sum": float(weights.sum()),
        "weight_array_float64_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "correlation_controls_NOT_independent_causal_roots": True}


def no_fit_supported(summary):
    return (summary["FIT"]["rows"] > 0 and summary["INNER"]["rows"] > 0
        and summary["FIT"]["target_counts"]["benefit"] == 0
        and summary["FIT"]["target_counts"]["risk"] > 0)


def seal_original_no_optimizer_failure(seed, expected):
    """Execute the real frozen failure branch, never fabricate a fit receipt."""
    if not no_fit_supported(expected):
        raise ValueError("Only measured zero-benefit FIT supervision qualifies")
    relative = "on_policy/state_source_fits_v1/BASELINE_STATE__seed" + str(seed) + ".json"
    path = OUT / relative
    destination = ASSETS / "training/state_source_fits_v1" / ("BASELINE_STATE__seed" + str(seed))
    if path.exists() or destination.exists():
        raise FileExistsError("Never overwrite/adopt an existing attempt")
    saved = study.EventAuthorityHead, study.torch.optim.AdamW, study.torch.save
    def forbidden(*args, **kwargs):
        raise RuntimeError("No model, optimizer or checkpoint allowed in support-failure closure")
    try:
        study.EventAuthorityHead = study.torch.optim.AdamW = study.torch.save = forbidden
        study.run("BASELINE_STATE", seed)
    finally:
        study.EventAuthorityHead, study.torch.optim.AdamW, study.torch.save = saved
    row = read_json(path)
    assert row["status"] == "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY"
    assert row["mode"] == "BASELINE_STATE" and row["seed"] == seed
    assert row["optimizer_steps"] == 0 and row["FIT_target_counts"] == expected["FIT"]["target_counts"]
    assert row["FIT_rows"] == expected["FIT"]["rows"] and row["INNER_rows"] == expected["INNER"]["rows"]
    assert row["manifest"] == expected["manifest"] and row["protocol_sha256"] == sha256(study.PROTOCOL)
    assert not destination.exists()
    return {"path": str(path), "sha256": sha256(path), "status": row["status"],
        "seed": seed, "optimizer_steps": 0, "actual_model_or_checkpoint_created": False}


def run():
    if (OUT / RESULT).exists() or (OUT / PREFIX / "PREPARED.json").exists():
        raise FileExistsError("Retain prior audit/attempt; no duplicate ownership")
    no_live_scoped_workers(("scripts.n72r21r2_train_joint_state",
        "scripts.n72r21r2_joint_state_fit_driver"))
    marker = OUT / "audit/idle_driver_terminal_recovery_v2/owners/STATE_FIT.json"
    old_log = ASSETS / "idle_driver_terminal_recovery_v2_logs/STATE_FIT.log"
    observation = dead_idle_storage_owner(read_json(marker), old_log)
    protocol = read_json(study.PROTOCOL)
    assert protocol["source_sha256"] == {name: sha256(ROOT / name) for name in study.CODE}
    source_path = OUT / "on_policy/ALL24_CONTROLLED_SOURCE_ACTUAL_CANONICAL_READBACK_V1.json"
    source = read_json(source_path); assert source["actual_verified_sequences"] == 24
    refs = {**source["evidence_sha256"], str(source_path): sha256(source_path),
        str(study.PROTOCOL): sha256(study.PROTOCOL), str(marker): sha256(marker), str(old_log): sha256(old_log)}
    assert all(sha256(path) == digest for path, digest in refs.items())
    summaries, datasets = {}, {}
    for mode in protocol["modes"]:
        fit, inner, manifest = study.load_records(mode)
        summaries[mode] = {"FIT": describe(fit), "INNER": describe(inner), "manifest": manifest}
        datasets[mode] = [row_digest(row) for row in fit]
        print({"mode": mode, "FIT_rows": len(fit), "FIT_targets": summaries[mode]["FIT"]["target_counts"],
            "COMMON_INNER_rows": len(inner)}, flush=True)
    inner_values = [s["INNER"] for s in summaries.values()]
    assert all(value == inner_values[0] for value in inner_values)
    assert sorted(datasets["MIXED_STATE"]) == sorted(datasets["BASELINE_STATE"] + datasets["TREATMENT_STATE"])
    assert no_fit_supported(summaries["BASELINE_STATE"])
    prepared = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "source_sha256": sha256(ROOT / SELF),
        "original_protocol_sha256": sha256(study.PROTOCOL), "summaries": summaries,
        "old_storage_stopped_owner": observation, "preserved_inputs_and_old_owner_sha256": refs,
        "only_operations": "Original BASELINE_STATE zero-benefit supervision-failure return for all3 seeds; model/optimizer/checkpoint constructors forbidden. No treatment/mixed fits or runtime.",
        "all_seeds": protocol["seeds"], "scientific_goal_complete": False, "next_stage_authorized": False}
    prepared_path = write_json(PREFIX + "/PREPARED.json", prepared)
    failures = [seal_original_no_optimizer_failure(seed, summaries["BASELINE_STATE"]) for seed in protocol["seeds"]]
    assert all(sha256(path) == digest for path, digest in refs.items())
    row = {**prepared, "utc": utcnow(), "status": "COMPLETE_ACTUAL_ALL24_STATE_SOURCE_SUPPORT_AND_ALL3_BASELINE_MEASURED_NO_OPTIMIZER_FAILURES",
        "prepared": {"path": str(prepared_path), "sha256": sha256(prepared_path)}, "original_baseline_failure_receipts": failures,
        "COMMON_INNER_features_past_KEEP_labels_dtype_order_and_weights_identical_all_modes": True,
        "MIXED_FIT_is_exact_BASELINE_plus_TREATMENT_row_union": True,
        "INNER_positive_labels_never_moved_into_FIT": True,
        "zero_baseline_FIT_benefit_NOT_trained_model_failure": True,
        "controlled_rows_NOT_model_generated_on_policy_or_independent_causal_events": True,
        "remaining_TREATMENT_and_MIXED_six_seed_fits_require_original_storage_guard": True,
        "storage_floor_GiB_unchanged": 60, "new_optimizer_steps": 0, "new_models_or_weights": 0,
        "new_runtime_extraction_or_candidate_policy_changes": 0,
        "full17_MOT_group_closure_or_whole_goal_complete": False, "scientific_decision": "PENDING"}
    path = write_json(RESULT, row)
    update_status(latest_actual_state_source_support={"path": str(path), "sha256": sha256(path)},
        actual_baseline_state_no_optimizer_supervision_failures=3, actual_state_models=0,
        scientific_decision="PENDING", next_stage_authorized=False, all_current_working_changes_published=False)
    append_log("ACTUAL_STATE_SOURCE_SUPPORT_AND_BASELINE_MEASURED_NO_OPTIMIZER_FAILURES", path=str(path), sha256=sha256(path))
    print({"path": str(path), "sha256": sha256(path), "actual_baseline_no_optimizer_failures": len(failures)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    run()
