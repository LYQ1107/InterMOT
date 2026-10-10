"""Actual fixed-source pure-margin calibration plus exact missed CF repairs.

No frozen source/threshold/model is modified. Existing temperatures are
reused; new pure controls use only the prospectively frozen INNER grid.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, preregistration, development_sequence, storage, append_log
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_joint_state_curriculum import writer
from scripts.n72r21r2_exact_cf_verifier import direct_repair_opportunities
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.pure_candidate_margin import CONTROLS, runtime_predictions, choose_control, offline_selector
from sam3_intermot.one_click.fresh_open_set_learning import assess_claims, select_point
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "availability/PURE_MARGIN_PROTOCOL_V1.json"
SELECTED = OUT / "availability/PURE_MARGIN_SELECTED_POINTS_V1.json"
PREFIX = "availability/pure_margin_v1"
CODE = ("scripts/n72r21r2_pure_margin_probe.py", "scripts/n72r21r2_pure_margin_driver.py",
        "sam3_intermot/one_click/pure_candidate_margin.py", "sam3_intermot/one_click/fresh_open_set_learning.py",
        "sam3_intermot/one_click/open_set_verifier.py", "sam3_intermot/one_click/intervention_features.py",
        "sam3_intermot/one_click/runtime_file_guard.py")


def freeze():
    original_path = OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json"
    original = read_json(original_path)
    simple_path = OUT / "availability/SIMPLE_CURRENT_AXIS_CALIBRATION_V1.json"
    simple = read_json(simple_path)
    assert simple["protocol_sha256"] == sha256(original_path)
    selection = simple["controls"]["CALIBRATED_SCORE"]["selection"]
    write_json("availability/PURE_MARGIN_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": preregistration()["goal"], "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "split": preregistration()["split"], "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "source_current_axis_protocol_sha256": sha256(OUT / "availability/CURRENT_AXIS_PROTOCOL_V1.json"),
        "original_simple_calibration_sha256": sha256(simple_path), "controls": list(CONTROLS),
        "temperature": selection["temperature"], "temperature_NOT_refitted": True,
        "score": "Exact E1 calibrated original joint UID+NONE scores. No relative-NONE bonus. REAL ranks/margins actual identity UIDs only (singleton runner0), retaining original NONE score mass; UID_PLUS_NONE uses the whole existing axis.",
        "cutoffs": original["cutoffs"], "margins": [.05, .15], "unknown_max": original["unknown_max"],
        "risk_max": original["risk_max"], "min_identity_claims": original["min_identity_claims"],
        "min_identity_claim_videos": original["min_identity_claim_videos"],
        "selection": "Same existing deterministic INNER nonvacuous current decision-risk selection. No feasible point=>CALIBRATION_ABSTAIN. All24 runtime sources are required, no failed-click replacement.",
        "diagnostic_point": {"status": "UNQUALIFIED_FIXED_CURRENT_IDENTITY_DIAGNOSTIC", "probability_min": .5, "margin_min": .05, "unknown_max": .5},
        "runtime": "Predictions/choices from sealed GT-free current axes; seal every prediction before joining existing offline labels. Sampled every32 axes remain sampled, not dense online policy.",
        "calibration": "Report GT-free scores' current correctness Brier/true-axis mass NLL with equal video/click/frame weighting. Simple scores do not have a trained UNKNOWN probability head; uncertainty remains actual offline contamination.",
        "CF": "Frozen selected points and fixed diagnostic at EVERY original CF frame, not nearest32. Join actual own-future safe direct/current and future-only repair opportunities offline. Correlated event-frame counts, NOT independent deployed corrections.",
        "bootstrap_seed": 730104, "bootstrap_repetitions": 2000, "all24_required_for_final": True,
        "new_training_or_association_authority_or_CONFIRM_VAL_TEST": False})


def protocol():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    return p


def current_groups():
    p = protocol()
    groups, receipts = [], []
    for sequence in p["split"]["fit"] + p["split"]["inner"]:
        path = OUT / "availability/current_axis_v1/runtime_sequences" / (sequence + ".json")
        source = read_json(path)
        assert source["protocol_sha256"] == p["source_current_axis_protocol_sha256"] and source["no_runtime_GT"]
        assert all(sha256(ROOT / name) == expected for name, expected in source["source_sha256"].items())
        init_path = OUT / "data/initialization" / (sequence + ".json")
        init = read_json(init_path)
        for event in init["inputs"]:
            if event["initialization_failure"]:
                continue
            episode_path = OUT / "availability/current_axis_v1/runtime" / (event["episode_uid"] + ".json")
            episode = read_json(episode_path)
            assert episode["source_sha256"] == source["source_sha256"] and episode["actual_GT_file_guard"]
            assert all(sha256(a["path"]) == a["sha256"] for a in episode["artifacts"])
            examples = read_zstd_jsonl(artifact(episode, "current_axis"))
            assert len(examples) == episode["counts"]["sampled_frame_groups"]
            groups.extend(examples)
            receipts.append({"path": str(episode_path), "sha256": sha256(episode_path)})
        receipts.append({"path": str(path), "sha256": sha256(path), "sequence": sequence,
                         "initialization_sha256": sha256(init_path), "failed_clicks_not_replaced": sum(bool(e["initialization_failure"]) for e in init["inputs"])})
    return groups, receipts


def score_current():
    p = protocol()
    storage(32 << 20)
    path = ASSETS / "open_set_pure_margin_v1/current_predictions.jsonl.zst"
    with runtime_file_guard():
        groups, receipts = current_groups()
        with writer(path) as emit:
            for group in groups:
                predicted = runtime_predictions(group["axis"], p["temperature"])
                emit({k: group[k] for k in ("sequence", "role", "episode_uid", "frame")} | {
                    "axis_uids": [r["candidate_uid"] for r in group["axis"]], "predictions": predicted,
                    "diagnostic_choices": {control: choose_control(group["axis"], predicted, p["diagnostic_point"], control) for control in p["controls"]},
                    "GT_or_future_features_used": False, "association_authority": False})
    write_json(PREFIX + "/current_predictions.json", {"stage": "N72R21R2", "protocol_sha256": sha256(PROTOCOL),
        "all24_source_receipts": receipts, "current_frame_groups": len(groups), "all_runtime_scores_sealed_before_labels": True,
        "artifact": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}})


def cluster_interval(values, p):
    if not values:
        return {"mean": None, "CI95": None, "video_clusters": 0}
    values = np.asarray(values, float)
    rng = np.random.default_rng(p["bootstrap_seed"])
    draws = values[rng.integers(0, len(values), size=(p["bootstrap_repetitions"], len(values)))].mean(1)
    return {"mean": float(values.mean()), "CI95": np.quantile(draws, [.025, .975]).tolist(), "video_clusters": len(values),
            "descriptive_selected_INNER_NOT_independent_generalization": True,
            "degenerate_zero_interval_NOT_population_two_percent_proof": bool(np.all(draws == draws[0]))}


def calibration(groups, predicted):
    videos, clicks, frames = {}, {}, Counter()
    for g in groups:
        videos.setdefault(g["sequence"], set()).add(g["episode_uid"])
        clicks[g["episode_uid"]] = g["sequence"]
        frames[g["episode_uid"]] += 1
    brier = nll = 0.
    for g, pred in zip(groups, predicted, strict=True):
        weight = 1. / len(videos) / len(videos[g["sequence"]]) / frames[g["episode_uid"]]
        brier += weight * np.mean([(s["correct"] - (a["class"] == 0)) ** 2 for a, s in zip(g["axis"], pred, strict=True)])
        nll -= weight * np.log(max(sum(s["correct"] for a, s in zip(g["axis"], pred, strict=True) if a["class"] == 0), 1e-12))
    return {"hierarchical_current_correctness_Brier": float(brier), "hierarchical_true_axis_mass_NLL": float(nll),
            "weighting": "Equal video -> click -> actual sampled current frame -> current rows for Brier",
            "simple_UNKNOWN_head_NOT_trained_or_calibrated": True}


def calibrate_current():
    p = protocol()
    runtime_path = OUT / PREFIX / "current_predictions.json"
    source = read_json(runtime_path)
    assert source["protocol_sha256"] == sha256(PROTOCOL) and source["all_runtime_scores_sealed_before_labels"]
    assert sha256(source["artifact"]["path"]) == source["artifact"]["sha256"]
    scores = read_zstd_jsonl(Path(source["artifact"]["path"]))
    truth, label_refs = {}, []
    for sequence in p["split"]["fit"] + p["split"]["inner"]:
        path = OUT / "availability/current_axis_v1/supervision" / (sequence + ".json")
        r = read_json(path)
        assert r["all_registered_video_runtime_sealed_before_GT"] and r["UNKNOWN_separate_class_not_verified_negative"]
        assert sha256(r["artifact"]["path"]) == r["artifact"]["sha256"]
        for g in read_zstd_jsonl(Path(r["artifact"]["path"])):
            truth[g["episode_uid"], g["frame"]] = g
        label_refs.append({"path": str(path), "sha256": sha256(path)})
    groups = [truth[s["episode_uid"], s["frame"]] for s in scores]
    assert len(groups) == len(truth) == source["current_frame_groups"]
    for group, score in zip(groups, scores, strict=True):
        assert [r["candidate_uid"] for r in group["axis"]] == score["axis_uids"]
        assert group["sequence"] == score["sequence"] and group["role"] == score["role"]
    inner_ids = [i for i, g in enumerate(groups) if g["role"] == "INNER"]
    inner = [groups[i] for i in inner_ids]
    inner_pred = [scores[i]["predictions"] for i in inner_ids]
    result, selections = {}, {}
    for control in p["controls"]:
        with offline_selector(control):
            selection, grid = select_point(inner, inner_pred, p)
            selection = {**selection, "temperature": p["temperature"], "pure_margin_control": control}
            selected = assess_claims(inner, inner_pred, selection)
            diagnostic = assess_claims(inner, inner_pred, p["diagnostic_point"])
        candidates = [r for r in grid if r["result"]["micro"]["verified_OTHER_row_FPR"] is not None and r["result"]["micro"]["verified_OTHER_row_FPR"] <= .02]
        intervals = {name: cluster_interval([r[name] for r in selected["per_sequence"].values() if r[name] is not None], p)
                     for name in ("coverage", "target_recall_given_available", "wrong_takeover_per_group", "false_presence_rate_no_positive_candidate")}
        result[control] = {"selection": selection, "all_frozen_INNER_points": grid, "selected_INNER": selected,
                           "fixed_unqualified_diagnostic": diagnostic, "video_macro_95CI": intervals,
                           "Recall_at_verified_OTHER_row_FPR2_measured_grid_ONLY": max((r["result"]["micro"]["target_recall_given_available"] or 0. for r in candidates), default=None),
                           "not_physically_absent_false_presence_FPR": True, "future_association_authority": False}
        selections[control] = selection
    output = write_json(PREFIX + "/current_calibration.json", {"stage": "N72R21R2", "protocol_sha256": sha256(PROTOCOL),
        "runtime_source_receipt_sha256": sha256(runtime_path), "offline_label_receipts": label_refs,
        "controls": result, "current_score_calibration": calibration(inner, inner_pred),
        "sampling_NOT_dense_full_video_policy": True, "scientific_success": None})
    write_json("availability/PURE_MARGIN_SELECTED_POINTS_V1.json", {"stage": "N72R21R2", "protocol_sha256": sha256(PROTOCOL),
        "current_calibration_receipt_sha256": sha256(output), "controls": selections,
        "frozen_before_exact_CF_score_or_repair_labels": True, "association_authority": False})
    print({"actual_pure_margin_calibration": {c: {"status": r["selection"]["status"], "selected_claims": r["selected_INNER"]["counts"].get("accepted_identity_claims", 0),
             "diagnostic_claims": r["fixed_unqualified_diagnostic"]["counts"].get("accepted_identity_claims", 0)} for c, r in result.items()}}, flush=True)


def score_cf(sequence):
    development_sequence(sequence)
    p, selected = protocol(), read_json(SELECTED)
    assert selected["protocol_sha256"] == sha256(PROTOCOL) and selected["frozen_before_exact_CF_score_or_repair_labels"]
    source_path = OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json")
    source = read_json(source_path)
    groups = []
    for r in source["episodes"]:
        assert sha256(r["path"]) == r["sha256"]
        seal = read_json(r["path"])
        assert sha256(seal["artifact"]["path"]) == seal["artifact"]["sha256"]
        groups.extend(g for g in read_zstd_jsonl(Path(seal["artifact"]["path"])) if g["record_type"] == "EXACT_CF_CURRENT_AXIS")
    assert len(groups) == source["counts"].get("exact_CF_current_axis_frames", 0)
    path = ASSETS / "open_set_pure_margin_v1/exact_CF" / (sequence + ".jsonl.zst")
    with runtime_file_guard(), writer(path) as emit:
        for g in groups:
            pred = runtime_predictions(g["axis"], p["temperature"])
            emit({"sequence": sequence, "episode_uid": g["episode_uid"], "frame": g["frame"],
                  "predictions": pred, "axis_uids": [a["candidate_uid"] for a in g["axis"]],
                  "choices": {c: {"selected": choose_control(g["axis"], pred, selected["controls"][c], c),
                                      "diagnostic": choose_control(g["axis"], pred, p["diagnostic_point"], c)} for c in p["controls"]},
                  "GT_or_future_used": False, "association_authority": False})
    write_json(PREFIX + "/CF_predictions/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence,
        "protocol_sha256": sha256(PROTOCOL), "selected_points_sha256": sha256(SELECTED), "source_sequence_sha256": sha256(source_path),
        "actual_exact_CF_frames": len(groups), "sealed_before_offline_repair_join": True,
        "artifact": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}})


def report_cf(sequence):
    development_sequence(sequence)
    p = protocol()
    score_path = OUT / PREFIX / "CF_predictions" / (sequence + ".json")
    seal = read_json(score_path)
    assert seal["protocol_sha256"] == sha256(PROTOCOL) and seal["selected_points_sha256"] == sha256(SELECTED)
    assert sha256(seal["artifact"]["path"]) == seal["artifact"]["sha256"] and seal["sealed_before_offline_repair_join"]
    labels_path = OUT / "events/label_audit" / (sequence + ".json")
    labels = read_json(labels_path)
    assert sha256(labels["artifact"]["path"]) == labels["artifact"]["sha256"]
    direct, future = direct_repair_opportunities(read_zstd_jsonl(Path(labels["artifact"]["path"])))
    predictions = read_zstd_jsonl(Path(seal["artifact"]["path"]))
    available_keys = {(r["episode_uid"], r["frame"]) for r in predictions}
    assert set(direct) <= available_keys and set(future) <= available_keys
    result = {}
    for c in p["controls"]:
        result[c] = {}
        for point in ("selected", "diagnostic"):
            outcomes = Counter()
            for group in predictions:
                key, choice = (group["episode_uid"], group["frame"]), group["choices"][c][point]
                hit = choice["accepted"] and choice["candidate_uid"] is not None
                outcomes["exact_CF_current_axes"] += 1
                outcomes["actual_accepted_identity_claims"] += hit
                for name, opportunities in (("direct_current", direct), ("future_only_or_current", future)):
                    if key in opportunities:
                        correct = hit and choice["candidate_uid"] in opportunities[key]
                        outcomes[name + "_safe_repair_event_frame_opportunities_NOT_independent"] += 1
                        outcomes[name + "_correctly_claimed_repair_UID"] += correct
                        outcomes[name + "_missed_due_rejection_or_wrong_UID"] += not correct
            result[c][point] = dict(outcomes)
    write_json(PREFIX + "/CF_results/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence,
        "protocol_sha256": sha256(PROTOCOL), "prediction_receipt_sha256": sha256(score_path), "CF_labels_receipt_sha256": sha256(labels_path),
        "controls": result, "not_nearest_frame_imputation_or_deployed_MOT_corrections": True,
        "all24_required_before_final_summary": True, "scientific_success": None})
    print({"actual_pure_margin_exact_CF": sequence, "controls": result}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "score_current", "calibrate_current", "score_cf", "report_cf"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    function = {"freeze": freeze, "score_current": score_current, "calibrate_current": calibrate_current,
                "score_cf": score_cf, "report_cf": report_cf}[args.action]
    function(args.sequence) if args.action.endswith("_cf") else function()
