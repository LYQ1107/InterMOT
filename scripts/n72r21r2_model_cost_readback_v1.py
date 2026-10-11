"""Verify all42 actual frozen MAIN heads' capacity and original timing evidence.

No optimization, model selection, synthetic performance benchmark or rollout.
Legacy 'single event' timing actually covers one model-forward branch row,
not the whole online MOT/SAM3/feature/global-assignment pipeline.
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path
import statistics

import torch

from scripts.n72r21r2_common import (
    ROOT, OUT, GOAL, read_json, write_json, sha256, utcnow, preregistration,
    append_log, update_status, storage,
)
from sam3_intermot.one_click.event_authority_models import EventAuthorityHead
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES

RESULT = "training/ALL42_MAIN_MODEL_CAPACITY_AND_TIMING_READBACK_V1.json"
CODE = ("scripts/n72r21r2_model_cost_readback_v1.py",
        "sam3_intermot/one_click/event_authority_models.py",
        "sam3_intermot/one_click/event_authority_learning.py",
        "scripts/n72r21r2_train_event_authority.py")


def verified_capacity(checkpoint, initial, record):
    for key in ("family", "objective", "seed"):
        if checkpoint[key] != record[key]: raise ValueError("Wrong actual model identity")
    if checkpoint["schema"] != "N72R21R2_EVENT_AUTHORITY_V1" or checkpoint["pilot"] or checkpoint["epoch"] != record["best_epoch"]:
        raise ValueError("Only the original selected actual MAIN checkpoint")
    if checkpoint["feature_names"] != list(FEATURE_NAMES): raise ValueError("Changed feature axes")
    model = EventAuthorityHead(checkpoint["family"], hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["model"], strict=True)
    parameters = dict(model.named_parameters())
    actual_count = sum(t.numel() for t in parameters.values())
    if type(record["parameter_count"]) is not int or actual_count != record["parameter_count"]:
        raise ValueError("Parameter count must be recomputed from the actual loaded architecture")
    state = model.state_dict()
    if set(state) != set(initial): raise ValueError("Initial and selected weight axes differ")
    changed = {}
    for name, value in state.items():
        if value.shape != initial[name].shape or value.dtype != initial[name].dtype:
            raise ValueError("Initial and selected tensor layout differs")
        changed[name] = int(torch.count_nonzero(value != initial[name]))
    if changed != record["changed_weight_elements_by_tensor"] or sum(changed.values()) == 0:
        raise ValueError("Selected model is not the recorded actually changed model")
    if record["nonzero_gradient_steps"] <= 0 or record["optimizer_steps"] < record["nonzero_gradient_steps"]:
        raise ValueError("No actual nonzero optimization evidence")
    latency = record["CPU_single_event_latency_ms"]
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
        raise ValueError("Missing finite positive original CPU timing")
    return {"parameter_count": actual_count,
            "tensor_layout": {name: {"shape": list(p.shape), "dtype": str(p.dtype), "trainable": p.requires_grad}
                              for name, p in parameters.items()},
            "actual_changed_weight_elements": sum(changed.values()),
            "original_CPU_single_branch_forward_latency_ms": float(latency)}


def complete_groups(rows, seeds):
    grouped = defaultdict(list)
    for row in rows: grouped[row["family"] + "__" + row["objective"]].append(row)
    result = {}
    for name, values in grouped.items():
        if len(values) != len(seeds) or {r["seed"] for r in values} != set(seeds):
            raise ValueError("Every original seed is required; no best/ready subset")
        count, layout = values[0]["parameter_count"], values[0]["tensor_layout"]
        if any(r["parameter_count"] != count or r["tensor_layout"] != layout for r in values):
            raise ValueError("A registered architecture cannot change across seeds")
        times = [r["original_CPU_single_branch_forward_latency_ms"] for r in values]
        result[name] = {"seeds": list(seeds), "parameter_count": count,
            "tensor_layout": layout, "original_forward_ms_median": statistics.median(times),
            "original_forward_ms_min": min(times), "original_forward_ms_max": max(times),
            "timing_NOT_new_controlled_latency_experiment_or_statistical_family_ranking": True,
            "FIT_rows_by_seed": {str(r["seed"]): r["FIT_rows"] for r in values},
            "INNER_rows_by_seed": {str(r["seed"]): r["INNER_rows"] for r in values},
            "actual_models": values}
    return result


def run():
    if (OUT / RESULT).exists(): raise FileExistsError("Keep actual cost readback; version a later audit")
    storage(8 << 20); torch.set_num_threads(1)
    prereg = preregistration()
    policy_path = OUT / "mot/MAIN_POLICY_PROTOCOL_V1.json"
    fit_protocol_path = OUT / "training/EVENT_AUTHORITY_PROTOCOL_V1.json"
    policy, fit_protocol = read_json(policy_path), read_json(fit_protocol_path)
    required = prereg["split"]["fit"] + prereg["split"]["inner"]
    refs = {}
    def attest(path, digest=None):
        actual = sha256(path)
        if digest is not None: assert actual == digest
        refs[str(path)] = actual
    for path in (policy_path, fit_protocol_path, OUT / "PREREGISTRATION.json"): attest(path)
    for name, value in policy["source_sha256"].items(): attest(ROOT / name, value)
    rows = []
    for job in policy["jobs"]:
        uid = "MAIN__" + job["family"] + "__" + job["objective"] + "__seed" + str(job["seed"])
        path = OUT / "training/event_authority" / (uid + ".json")
        record = read_json(path); attest(path)
        assert record["experiment_uid"] == uid and record["protocol_sha256"] == refs[str(fit_protocol_path)]
        assert not record["pilot_not_main_fit_or_independent_confirmation"]
        assert record["FIT_normalizer_only"] and not record["optimizer_validation_or_confirmation_usage"]
        assert record["manifest"]["required_sequences"] == required
        assert set(record["manifest"]["actual_fit_videos"]) <= set(prereg["split"]["fit"])
        assert set(record["manifest"]["actual_validation_videos"]) <= set(prereg["split"]["inner"])
        for name, value in record["source_freeze"].items(): attest(ROOT / name, value)
        attest(record["checkpoint_path"], record["checkpoint_sha256"])
        initial_path = Path(record["checkpoint_path"]).parent / "initial.pt"
        attest(initial_path, record["initial_sha256"])
        checkpoint = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=True)
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        assert checkpoint["protocol_sha256"] == record["protocol_sha256"]
        assert checkpoint["source_freeze"] == record["source_freeze"]
        assert checkpoint["hidden"] == fit_protocol["training"]["hidden"]
        attest(record["validation_predictions_path"], record["validation_predictions_sha256"])
        for item in record["epochs"]: attest(item["checkpoint_path"], item["checkpoint_sha256"])
        rows.append({**job, "experiment_uid": uid, **verified_capacity(checkpoint, initial, record),
            "FIT_rows": record["FIT_rows"], "INNER_rows": record["validation_rows"],
            "FIT_target_counts": record["FIT_target_counts"],
            "objective_specific_excluded_support": record["manifest"]["excluded"],
            "fit_record": {"path": str(path), "sha256": refs[str(path)]},
            "checkpoint": {"path": record["checkpoint_path"], "sha256": record["checkpoint_sha256"]}})
    assert len(rows) == len(policy["jobs"]) == 42
    groups = complete_groups(rows, policy["seeds"])
    assert len(groups) == 14
    mlps = [r for r in rows if r["family"] == "SMALL_MLP"]
    assert len(mlps) == 21 and len({r["objective"] for r in mlps}) == 7
    assert all(r["tensor_layout"] == mlps[0]["tensor_layout"] for r in mlps)
    result = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
        "status": "COMPLETE_ALL42_ACTUAL_MAIN_CAPACITY_AND_ORIGINAL_TIMING_READBACK_NOT_TABLE4_CLOSURE",
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "models": 42, "complete_three_seed_groups": 14, "groups": groups,
        "all_seven_MLP_objectives_same_actual_tensor_layout_and_capacity": True,
        "same_eligible_training_rows_across_objectives_claimed": False,
        "objective_specific_valid_label_support_and_exclusions_must_be_reported": True,
        "legacy_latency_field_actual_scope": "100 CPU single-thread inference-mode forward calls of one normalized real INNER branch row, its3-step past and same-prestate KEEP row; milliseconds per call. Sigmoid/output conversion, feature extraction, solver/tracker, SAM3 and OSNet excluded. Original timing records, not newly controlled benchmarks or population timing CI.",
        "STATE_fits_costs_and_effects_pending_original_all24_sources": True,
        "RELATIONAL_training_requires_original_measured_trigger_not_automatically_authorized": True,
        "Table3_Table4_effect_event_generalization_and_whole_goal_complete": False,
        "new_training_weight_selection_or_runtime_policy_changes": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "next_stage_authorized": False,
        "evidence_sha256": refs}
    path = write_json(RESULT, result)
    update_status(latest_actual_main_capacity_and_timing_readback={"path": str(path), "sha256": sha256(path)},
                  all_current_working_changes_published=False)
    append_log("ALL42_MAIN_MODEL_CAPACITY_TIMING_ACTUAL_READBACK", path=str(path), sha256=sha256(path), models=42, groups=14)
    print({"verified_models": 42, "groups": 14, "model_cost_readback": str(path), "sha256": sha256(path),
           "primary_L3_parameters_and_forward_ms_median": {k: {n: v[n] for n in ("parameter_count", "original_forward_ms_median")}
                for k, v in groups.items() if k.endswith("__L3_HARM_AWARE")}}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    run()
