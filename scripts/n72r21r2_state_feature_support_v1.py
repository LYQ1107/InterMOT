"""Retrospective read-only STATE training/own-deployment support audit.

Same fixed sampling/support/reproduction recipe as original MAIN M7. No fits,
rollouts, new score thresholds or retrospective edits to original protocols.
Every point requires all3 seeds and INNER8 before reading its FIT source.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, GOAL, read_json, write_json, sha256, storage, preregistration
from scripts import n72r21r2_main_feature_support_v1 as original
from scripts import n72r21r2_state_policy as state
from scripts import n72r21r2_train_joint_state as source
from scripts.n72r21r2_state_inner_readback_v1 import checked_cell
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence
from scripts.n72r21r2_training_delivery_v1 import state_capacity
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.event_authority_learning import encode_runtime_input, correlated_sequence_weights, fit_normalizer
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.current_feature_support import support_comparison, unpadded_history, prediction_error


MODES = ("TREATMENT_STATE", "MIXED_STATE")
SELF = "scripts/n72r21r2_state_feature_support_v1.py"
PROTOCOL = OUT / "on_policy/STATE_FEATURE_SUPPORT_AUDIT_PROTOCOL_V1.json"
PREFIX = "on_policy/state_feature_support_audit_v1"
CODE = tuple(dict.fromkeys((SELF, "scripts/n72r21r2_state_inner_readback_v1.py",
    "scripts/n72r21r2_training_delivery_v1.py", *original.CODE, *state.CODE, *source.CODE)))


def freeze():
    p, recipe = read_json(state.PROTOCOL), read_json(original.PROTOCOL)
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in state.CODE}
    assert recipe["source_sha256"] == {n: sha256(ROOT / n) for n in original.CODE}
    assert p["seeds"] == recipe["seeds"] and p["split"]["inner"] == recipe["required_sequences"]
    assert list(p["points"]) == recipe["points"] and p["family"] == "SMALL_MLP"
    assert not list((OUT / PREFIX).glob("*.json")), "Freeze before these diagnostic outputs"
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "design": "RETROSPECTIVE_DIAGNOSIS_NOT_PROSPECTIVE_FIT_OR_MOT_SELECTION", "source_sha256": {n: sha256(ROOT / n) for n in CODE},
        "state_policy_protocol_sha256": sha256(state.PROTOCOL), "state_fit_protocol_sha256": sha256(source.PROTOCOL),
        "original_MAIN_support_protocol_sha256": sha256(original.PROTOCOL),
        "fitted_modes": list(MODES), "registered_points": [{"mode": mode, "point": point} for mode in MODES for point in p["points"]],
        "split": p["split"], "seeds": p["seeds"], "points": p["points"],
        "sampling": "Unchanged every16 original postclick frames plus every effective action; all current choices with actual own H3 and same-prestate KEEP. Failed slots/videos retained.",
        "FIT_source": "Exact original matched state-source load_records(mode), mode-specific FIT-only hierarchical normalizer; original identical COMMON_INNER is not used for support/normalization/optimization.",
        "prediction_absolute_tolerance": recipe["prediction_absolute_tolerance"],
        "support_metrics": "Original exact float32 marginal min/max/outside counts, encoded quantiles, frozen-normalized magnitude; own past counted once per sampled frame with exact padding removed.",
        "BASELINE_STATE": "Three actual original no-benefit/no-optimizer receipts; no fitted weight, normalizer or own-model rollout. Not imputed as zero performance or successful support alignment.",
        "proxies_and_missing_measurements": recipe["M7_proxies_not_exact_measurements"],
        "no_new_model_training_rollout_threshold_clipping_score_ablation_or_selection": True,
        "actual_model_generated_on_policy_training_proved": False, "full_M7_or_scientific_goal_complete": False,
        "CONFIRM_VAL_TEST_SOT_authorized": False, "next_association_stage_authorized": False, "CPU_workers": 1})


def required_point_paths(mode, point, protocol):
    if mode not in protocol["fitted_modes"] or point not in protocol["points"]:
        raise ValueError("Only original fitted STATE modes/points, never BASELINE or new points")
    paths = []
    for seed in protocol["seeds"]:
        uid = "STATE__" + mode + "__" + source.OBJECTIVE + "__seed" + str(seed) + "__" + point
        for sequence in protocol["split"]["inner"]:
            for folder in ("runtime", "onset_runtime", "results"):
                paths.append(OUT / state.PREFIX / folder / uid / (sequence + ".json"))
    return paths


def sampled_choices(row, event):
    if row["runtime_GT_read"] or row["runtime_future_GT_used"]:
        raise ValueError("Only actually GT-free own-runtime traces")
    authority = row["authority"]
    if type(authority["causal_feature_sample"]) is not bool or type(authority["effective_assignment_change"]) is not bool:
        raise ValueError("Original sampling/effect flags must be booleans")
    expected = row["frame"] > event["frame"] and (row["frame"] - event["frame"] - 1) % 16 == 0
    if authority["causal_feature_sample"] != bool(expected or authority["effective_assignment_change"]):
        raise ValueError("Original sampled-frame schedule cannot be changed")
    if not authority["causal_feature_sample"]:
        return None
    choices = authority["all_current_choices"]
    if not choices or len({c["branch"] for c in choices}) != len(choices) or sum(c["branch"] == "KEEP" for c in choices) != 1:
        raise ValueError("Unique original choices and actual same-prestate KEEP required")
    primary = np.asarray(authority["own_primary_feature_vector"], np.float32)
    past = np.asarray(authority["causal_previous_feature_vectors_H3"], np.float32)
    if primary.shape != (32,) or past.shape != (3, 32) or not np.isfinite(primary).all() or not np.isfinite(past).all():
        raise ValueError("Exact finite own current/past3 feature axes required")
    return choices, authority["causal_previous_feature_vectors_H3"]


def bound_predictor(saved, initial, fit, mode, seed, mean, scale):
    if mode not in MODES or fit["mode"] != mode or type(seed) is not int:
        raise ValueError("Exact original controlled state source and seed")
    if fit["seed"] != seed or fit["family"] != "SMALL_MLP" or fit["objective"] != source.OBJECTIVE:
        raise ValueError("Actual architecture is SMALL_MLP, not the source mode name")
    if fit["actual_model_generated_on_policy"] or fit["association_authority"] or fit["nonzero_gradient_steps"] <= 0:
        raise ValueError("Actual unqualified controlled fit required; not oracle on-policy")
    if fit["status"] != "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE" or saved["authority_status"] != "TRAINED_UNCALIBRATED_NOT_DEPLOYABLE":
        raise ValueError("Original controlled unqualified model status required")
    state_capacity(saved, initial, fit)
    if saved["feature_names"] != list(FEATURE_NAMES) or not np.array_equal(np.asarray(saved["FIT_mean"], np.float32), mean) or not np.array_equal(np.asarray(saved["FIT_scale"], np.float32), scale):
        raise ValueError("Original mode's exact FIT-only normalizer and feature axis required")
    return EventAuthorityPredictor(saved, development_diagnostic=True)


def run(mode, point):
    ev = Evidence(); p = ev.json(PROTOCOL); ev.code(p["source_sha256"])
    assert p["frozen"] and p["goal"] == GOAL
    assert p["state_policy_protocol_sha256"] == ev.digest(state.PROTOCOL)
    assert p["state_fit_protocol_sha256"] == ev.digest(source.PROTOCOL)
    assert p["original_MAIN_support_protocol_sha256"] == ev.digest(original.PROTOCOL)
    split = preregistration()["split"]
    assert p["split"] == {k: split[k] for k in ("fit", "inner")}
    paths = required_point_paths(mode, point, p)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("Whole original STATE point is not ready; no source labels opened: " + str(len(missing)))
    relative = PREFIX + "/" + mode + "__" + point + ".json"
    if (OUT / relative).exists():
        raise FileExistsError("Keep original diagnostic outputs; no rerun/overwrite")
    storage(8 << 20); torch.set_num_threads(1)
    for seed in p["seeds"]:
        for sequence in p["split"]["inner"]:
            checked_cell(mode, seed, point, sequence, ev)
    fit_rows, _, manifest = source.load_records(mode)
    assert manifest["required_all24_videos"] == split["fit"] + split["inner"]
    assert {r["role"] for r in fit_rows} == {"FIT"}
    raw = np.stack([r["x"][:32] for r in fit_rows])
    src_past, src_padding = unpadded_history(np.stack([r["past"] for r in fit_rows]), 32)
    mean, scale = fit_normalizer(np.stack([r["x"] for r in fit_rows]), correlated_sequence_weights(fit_rows))
    result = {"stage": "N72R21R2", "goal": GOAL, "mode": mode, "actual_architecture_family": "SMALL_MLP",
        "point": point, "protocol_sha256": ev.digest(PROTOCOL),
        "status": "COMPLETE_ACTUAL_READ_ONLY_SAMPLED_STATE_SUPPORT_NOT_FULL_M7_OR_SAFETY",
        "FIT_manifest": manifest, "FIT_rows_correlated": len(fit_rows), "FIT_past_padding_vectors": src_padding,
        "all_original_seeds": p["seeds"], "required_INNER_sequences": p["split"]["inner"], "per_seed": {},
        "actual_model_generated_on_policy_training_proved": False, "full_M7_or_scientific_goal_complete": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False, "next_association_stage_authorized": False}
    for seed in p["seeds"]:
        fit_path = OUT / "on_policy/state_source_fits_v1" / (mode + "__seed" + str(seed) + ".json")
        fit = ev.json(fit_path)
        saved = torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=True)
        initial_path = Path(fit["checkpoint_path"]).parent / "initial.pt"
        assert ev.digest(initial_path) == fit["initial_sha256"]
        predictor = bound_predictor(saved, torch.load(initial_path, map_location="cpu", weights_only=True), fit, mode, seed, mean, scale)
        current, histories, by_video = [], [], defaultdict(list)
        counts, branches, max_error = Counter(), Counter(), 0.
        uid = "STATE__" + mode + "__" + source.OBJECTIVE + "__seed" + str(seed) + "__" + point
        for sequence in p["split"]["inner"]:
            seal = ev.json(OUT / state.PREFIX / "runtime" / uid / (sequence + ".json"))
            counts["failed_initialization_slots_retained"] += seal["initialization_failures_not_replaced"]
            counts["all_failed_videos_retained_not_successes"] += not bool(seal["episodes"])
            for episode in seal["episodes"]:
                counts["valid_actual_clicks"] += 1
                ref = next(a for a in episode["artifacts"] if a["kind"] == "trace")
                with runtime_file_guard():
                    trace = read_zstd_jsonl(Path(ref["path"]))
                    assert len(trace) == episode["event"]["frames"]
                    for index, row in enumerate(trace):
                        assert row["frame"] == index
                        counts["all_actual_global_frames"] += 1
                        counts["effective_actions_all_frames"] += bool(row["authority"]["effective_assignment_change"])
                        item = sampled_choices(row, episode["event"])
                        if item is None:
                            continue
                        choices, past = item
                        histories.append(past); counts["sampled_current_frames_correlated"] += 1
                        keep = next(c for c in choices if c["branch"] == "KEEP")
                        keep_runtime = original.runtime_input(keep["features"], past)
                        for choice in choices:
                            runtime = original.runtime_input(choice["features"], past)
                            encoded, _ = encode_runtime_input(runtime, choice["branch"])
                            error = prediction_error(predictor.predict(runtime, choice["branch"], keep_runtime), choice["prediction"])
                            if error > p["prediction_absolute_tolerance"]:
                                raise AssertionError("Actual original STATE prediction did not reproduce")
                            max_error = max(max_error, error)
                            current.append(encoded[:32]); by_video[sequence].append(encoded[:32]); branches[choice["branch"]] += 1
                            counts["sampled_choices_reproduced"] += 1
                            if choice["branch"] != "KEEP":
                                pred, gate = choice["prediction"], p["points"][point]
                                counts["sampled_non_KEEP_choices_correlated"] += 1
                                counts["sampled_non_KEEP_score_only_pass_not_action"] += bool(pred["beneficial"] >= gate["claim_min"] and pred["harmful"] <= gate["risk_max"] and pred["value"] > 0)
                assert sha256(ref["path"]) == ref["sha256"]
        own = np.stack(current); history, padding = unpadded_history(np.asarray(histories), 32)
        result["per_seed"][str(seed)] = {"actual_counts": dict(counts), "checkpoint_sha256": fit["checkpoint_sha256"],
            "FIT_normalizer_exact_float32_match": True, "prediction_absolute_max_error": max_error,
            "prediction_reproduction_numeric_NOT_bitwise_AA": True, "sampled_branch_choice_counts_correlated": dict(branches),
            "current_support": support_comparison(raw, own, FEATURE_NAMES, mean, scale),
            "own_past_padding_vectors": padding, "past_support": support_comparison(src_past, history, FEATURE_NAMES, mean, scale) if len(src_past) else None,
            "past_rows_counted_once_per_sample_frame_NOT_per_branch": True,
            "per_video_current_support": {seq: support_comparison(raw, np.asarray(by_video[seq], np.float32).reshape(-1, 32), FEATURE_NAMES, mean, scale) for seq in p["split"]["inner"]}}
        print({"actual_STATE_support_seed": seed, "mode": mode, "point": point, "choice_predictions": counts["sampled_choices_reproduced"], "maximum_reproduction_error": max_error}, flush=True)
    for receipt in manifest["receipts"]:
        path = OUT / "on_policy/joint_state_v1/supervision" / (receipt["sequence"] + ".json")
        assert ev.digest(path) == receipt["source_receipt_sha256"]
        ev.artifact(ev.json(path)["artifact"])
    result["evidence_sha256"] = ev.refs
    write_json(relative, result)
    print({"actual_STATE_support_point_complete": mode + "__" + point, "scientific_goal_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--point")
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run(args.mode, args.point)
