"""Rebuild the seven actually fitted MLP supervision conditions, without fits.

Checks all original FIT16/INNER8 source receipts, exact validation targets and
FIT-only normalizers against all21 selected checkpoints. Reports limitations,
not pending full-MOT comparisons as measured loss superiority.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, GOAL, write_json, preregistration, storage
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts import n72r21r2_train_event_authority as original
from sam3_intermot.one_click.event_authority_learning import (
    OBJECTIVES, correlated_sequence_weights, fit_normalizer)
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES


SELF = "scripts/n72r21r2_objective_delivery_v1.py"
RESULT = "objectives/ALL7_ACTUAL_MLP_SUPERVISION_READBACK_V1.json"
NAMED = {"CURRENT_CORRECTNESS": [OBJECTIVES[0]], "EVENT_CORRECTION": [OBJECTIVES[1]],
    "H20": [OBJECTIVES[2]], "H50": [OBJECTIVES[2]], "H100": [OBJECTIVES[2]],
    "GLOBAL_ASSOCIATION": [OBJECTIVES[4], OBJECTIVES[5]],
    "RISK_CONSTRAINED": [OBJECTIVES[3], OBJECTIVES[6]]}
KEYS = ("sequence", "episode_uid", "event_uid", "branch")


def row_key(row):
    return tuple(row[k] for k in KEYS)


def input_signature(row):
    h = hashlib.sha256()
    h.update(json.dumps(row_key(row), separators=(",", ":")).encode())
    for name in ("x", "past", "keep"):
        value = np.asarray(row[name])
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Exact finite float32 original current/past/KEEP axes")
        h.update(name.encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def verify_predictions(lines, rows):
    if len(lines) != len(rows):
        raise ValueError("Validation prediction row count differs from original support")
    for line, row in zip(lines, rows, strict=True):
        if row_key(line) != row_key(row) or line["offline_target"] != row["target"]:
            raise ValueError("Selected checkpoint validation labels/order did not reproduce")
        if line["not_own_policy_outcome"] is not True:
            raise ValueError("Teacher-state prediction is not a deployed policy outcome")
        if set(line["prediction"]) != {"beneficial", "harmful", "value"}:
            raise ValueError("Exact original independent heads required")
        if not all(np.isfinite(v) for v in line["prediction"].values()):
            raise ValueError("Nonfinite recorded predictions")
        if any(not 0. <= line["prediction"][k] <= 1. for k in ("beneficial", "harmful")):
            raise ValueError("Recorded probabilities must be bounded")


def summarize(rows):
    weights = correlated_sequence_weights(rows)
    target = np.asarray([[r["target"][k] for k in ("benefit", "risk", "value")] for r in rows], np.float32)
    if not np.isfinite(target).all() or not np.isin(target[:, :2], [0., 1.]).all():
        raise ValueError("Actual finite binary supervision and value required")
    videos = sorted({r["sequence"] for r in rows})
    masses = {s: float(sum(w for r, w in zip(rows, weights, strict=True) if r["sequence"] == s)) for s in videos}
    if not all(np.isclose(w, 1. / len(videos)) for w in masses.values()):
        raise ValueError("Equal-video hierarchical training weights did not reproduce")
    return {"rows": len(rows), "nonempty_videos": videos,
        "benefit_rows": int(target[:, 0].sum()), "risk_rows": int(target[:, 1].sum()),
        "value_quantiles_P10_P25_P50_P75_P90": np.quantile(target[:, 2], [.1, .25, .5, .75, .9]).tolist(),
        "scope_counts": dict(Counter(r["target"]["scope"] for r in rows)),
        "observed_events_NOT_independent_roots": len({r["event_uid"] for r in rows}),
        "observed_correlation_groups_NOT_independent_roots": len({(r["sequence"], r["episode_uid"], r["correlation_group_not_proven_causal_origin"]) for r in rows}),
        "weight_sum": float(weights.sum()), "per_video_weight_mass": masses,
        "actual_float32_target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "independent_causal_root_count": None}


def support_contrast(conditions):
    supports = {name: {row_key(r): r for r in rows} for name, rows in conditions.items()}
    for name, rows in conditions.items():
        if len(supports[name]) != len(rows):
            raise ValueError("Duplicate same-event arm escaped original eligibility")
    shared = set.intersection(*(set(rows) for rows in supports.values()))
    for key in shared:
        if len({input_signature(rows[key]) for rows in supports.values()}) != 1:
            raise ValueError("Shared observations have different current/past/KEEP inputs")
    l1, l3 = supports[OBJECTIVES[1]], supports[OBJECTIVES[3]]
    identical = set(l1) == set(l3) and all(l1[k]["target"] == l3[k]["target"] for k in l1)
    return {"eligible_rows": {name: len(rows) for name, rows in supports.items()},
        "all7_common_rows_same_runtime_input": len(shared),
        "same_eligible_support_across_all7": all(set(rows) == shared for rows in supports.values()),
        "L1_L3_same_eligible_rows_and_targets": identical,
        "loss_only_comparison_on_identical_support_across_all7_claimed": False}


def run():
    storage(16 << 20); torch.set_num_threads(1)
    ev = Evidence()
    protocol = ev.json(original.PROTOCOL)
    ev.code({name: ev.digest(ROOT / name) for name in original.CODE})
    split = preregistration()["split"]
    seeds = preregistration()["hyperparameters"]["seeds"]
    assert protocol["frozen"] and protocol["seeds"] == seeds
    assert protocol["same_architecture_loss_contrast"] == "SMALL_MLP for " + ", ".join(OBJECTIVES)
    fitted = ev.json(OUT / "training/MAIN_AND_STATE_CONTROLLER_INVENTORY_V1.json")
    assert fitted["actual_optimized_controller_models"] == 48
    conditions, actual, layouts = {}, {}, []
    for objective in OBJECTIVES:
        fit, inner, manifest = original.load_records(objective, pilot=False)
        assert manifest["required_sequences"] == split["fit"] + split["inner"]
        assert {r["optimization_role"] for r in fit} == {"FIT"}
        assert {r["optimization_role"] for r in inner} == {"INNER"}
        mean, scale = fit_normalizer(np.stack([r["x"] for r in fit]), correlated_sequence_weights(fit))
        models = []
        for seed in seeds:
            path = OUT / "training/event_authority" / ("MAIN__SMALL_MLP__" + objective + "__seed" + str(seed) + ".json")
            record = ev.json(path); ev.code(record["source_freeze"])
            assert record["protocol_sha256"] == ev.digest(original.PROTOCOL)
            assert record["manifest"] == manifest and record["FIT_rows"] == len(fit) and record["validation_rows"] == len(inner)
            assert record["FIT_target_counts"] == {k: sum(r["target"][k] for r in fit) for k in ("benefit", "risk")}
            assert record["nonzero_gradient_steps"] > 0 and sum(record["changed_weight_elements_by_tensor"].values()) > 0
            assert not record["pilot_not_main_fit_or_independent_confirmation"] and not record["optimizer_validation_or_confirmation_usage"]
            assert record["FIT_normalizer_only"] and not record["association_deployment_authorized"]
            ev.artifact({"path": record["checkpoint_path"], "sha256": record["checkpoint_sha256"]})
            saved = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=True)
            assert saved["schema"] == "N72R21R2_EVENT_AUTHORITY_V1" and saved["family"] == "SMALL_MLP"
            assert saved["seed"] == seed and saved["objective"] == objective and saved["epoch"] == record["best_epoch"]
            assert saved["feature_names"] == list(FEATURE_NAMES) and not saved["pilot"]
            assert saved["protocol_sha256"] == ev.digest(original.PROTOCOL)
            assert np.array_equal(np.asarray(saved["FIT_mean"], np.float32), mean)
            assert np.array_equal(np.asarray(saved["FIT_scale"], np.float32), scale)
            layouts.append({k: (list(v.shape), str(v.dtype)) for k, v in saved["model"].items()})
            ev.artifact({"path": record["validation_predictions_path"], "sha256": record["validation_predictions_sha256"]})
            with Path(record["validation_predictions_path"]).open() as handle:
                verify_predictions([json.loads(line) for line in handle], inner)
            models.append({"seed": seed, "fit_record": {"path": str(path), "sha256": ev.digest(path)},
                "checkpoint": {"path": record["checkpoint_path"], "sha256": record["checkpoint_sha256"]},
                "selected_epoch": record["best_epoch"], "parameter_count": record["parameter_count"],
                "recorded_teacher_validation_loss_NOT_MOT": next(e["validation_correlated_video_macro_loss"] for e in record["epochs"] if e["epoch"] == record["best_epoch"])})
        for receipt in manifest["receipts"]:
            seq = receipt["sequence"]
            label = ev.json(OUT / "events/label_audit" / (seq + ".json"))
            ev.artifact(label["artifact"])
            if receipt["utility_audit_sha256"]:
                utility = ev.json(OUT / "events/trajectory_utility_audit" / (seq + ".json"))
                ev.artifact(utility["artifact"])
        conditions[objective] = {"fit": fit, "inner": inner}
        actual[objective] = {"all3_seed_models": models, "harm_weight": protocol["loss_harm_weights"][objective],
            "FIT": summarize(fit), "INNER": summarize(inner), "excluded": manifest["excluded"],
            "required24_including_failed_initializations": manifest["required_sequences"],
            "actual_full96_cell_MOT_loss_comparison": "PENDING_ORIGINAL_QUEUE_NOT_TEACHER_LOSS_OR_READY_SUBSET",
            "MOT_delta": None, "scientific_success": None}
        print({"actual_objective_source_readback": objective, "FIT_rows": len(fit), "INNER_rows": len(inner), "seeds": seeds}, flush=True)
    assert all(layout == layouts[0] for layout in layouts) and len(layouts) == 21
    ev.digest(ROOT / SELF)
    report = {"stage": "N72R21R2", "goal": GOAL,
        "status": "ACTUAL_ALL7_TARGET_NORMALIZER_PREDICTION_READBACK_EFFECT_COMPARISON_PENDING",
        "objectives": actual, "same_architecture_all21_MLP_checkpoints": True,
        "support_contrast": {role: support_contrast({name: rows[role] for name, rows in conditions.items()}) for role in ("fit", "inner")},
        "L1_requested_independent_event_correction_implemented": False,
        "L1_limitation": "Actual L1 fits use one-shot H100 component labels and interval correlation controls; no independent causal-origin correction target has been proved. Observed events/arms are not independent roots.",
        "L1_L3_distinction": "The original targets/support are checked, not assumed distinct; original risk-positive loss weights1 versus5 distinguish the fits.",
        "L2_horizons": [20, 50, 100], "L2_three_independent_horizon_fits": False,
        "L2_training_target": "One original mean-H20/H50/H100 continuity condition with all3 original seeds.",
        "no_new_optimizer_or_model_scores_or_threshold_selection": True,
        "all3_seeds_retained_no_best_seed_or_validation_loss_superiority_claim": True,
        "evidence_sha256": ev.refs, "scientific_goal_complete": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "next_association_stage_authorized": False}
    write_json(RESULT, report)
    reference = {"path": str(OUT / RESULT), "sha256": ev.digest(OUT / RESULT)}
    for name, names in NAMED.items():
        write_json("objectives/" + name + ".json", {"stage": "N72R21R2", "goal": GOAL,
            "status": report["status"], "actual_conditions": {k: actual[k] for k in names},
            "horizon": int(name[1:]) if name in ("H20", "H50", "H100") else None,
            "separate_horizon_training_condition": False, "all7_supervision_readback": reference,
            "independent_event_correction_proved": False, "MOT_superiority_proved": False,
            "scientific_goal_complete": False, "next_association_stage_authorized": False})
    print({"actual_seven_objectives_named_delivery": len(NAMED), "actual_selected_models": 21,
           "full_MOT_loss_superiority": "PENDING"}, flush=True)


if __name__ == "__main__":
    run()
