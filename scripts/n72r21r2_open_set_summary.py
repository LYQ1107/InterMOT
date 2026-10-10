"""Strict all-model current runtime rescore, then offline clustered summary.

Every32 sampled C0 current axes are not dense model-policy trajectories.
All seeds are retained; observed zero errors do not establish population2%.
"""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, preregistration, storage
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r2_events import artifact
from sam3_intermot.one_click.fresh_open_set_runtime import FreshOpenSetPredictor
from sam3_intermot.one_click.fresh_open_set_learning import assess_claims
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard


def cluster_interval(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values): return {"mean": None, "CI95": None, "video_clusters": 0}
    rng = np.random.default_rng(730104)
    samples = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "CI95": np.quantile(samples, [.025, .975]).tolist(),
            "video_clusters": len(values), "resamples": 2000, "seed": 730104,
            "descriptive_selected_INNER_not_independent_confirmation": True}


def run():
    torch.set_num_threads(1)
    p = read_json(OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json")
    assert len(p["families"]) * len(p["seeds"]) == 12
    permitted = preregistration()["split"]["inner"]
    records = {}
    for family in p["families"]:
        for seed in p["seeds"]:
            uid = family + "__seed" + str(seed)
            path = OUT / "availability/current_axis_fits" / (uid + ".json")
            record = read_json(path)
            assert record["protocol_sha256"] == sha256(OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json")
            assert record["source_sha256"] == {f: sha256(ROOT / f) for f in record["source_sha256"]}
            assert sha256(record["checkpoint_path"]) == record["checkpoint_sha256"]
            records[uid] = (path, record)
    source, initialization = [], []
    for sequence in permitted:
        init = read_json(OUT / "data/initialization" / (sequence + ".json"))
        initialization.append({"sequence": sequence, "valid_clicks": sum(not e["initialization_failure"] for e in init["inputs"]),
                               "failed_clicks_not_replaced": sum(e["initialization_failure"] for e in init["inputs"])})
        for e in init["inputs"]:
            if e["initialization_failure"]: continue
            seal = read_json(OUT / "availability/current_axis_v1/runtime" / (e["episode_uid"] + ".json"))
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            source.extend(read_zstd_jsonl(artifact(seal, "current_axis")))
    storage(64 << 20)
    destination = ASSETS / "open_set_strict_runtime_rescore_v1/INNER_ALL12.jsonl.zst"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError("Preserve actual scored runtime attempt")
    scores = {}
    # Original current runtime files only. Offline outcome files are not open
    # and no taxonomy/truth field reaches the predictor during this phase.
    with runtime_file_guard(), destination.open("xb") as stream:
        proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=stream)
        try:
            for uid, (_, record) in records.items():
                predictor = FreshOpenSetPredictor(record["checkpoint_path"])
                predicted = []
                for group in source:
                    pred = predictor.predict_axis(group["axis"])
                    predicted.append(pred)
                    result = {"model": uid, "sequence": group["sequence"], "episode_uid": group["episode_uid"], "frame": group["frame"],
                              "current_axis_predictions": [{"candidate_uid": r["candidate_uid"], **s} for r, s in zip(group["axis"], pred, strict=True)],
                              "association_authority": False, "GT_or_future_input": False}
                    proc.stdin.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
                scores[uid] = predicted
            proc.stdin.close()
            assert proc.wait() == 0
        finally:
            if proc.poll() is None: proc.terminate(); proc.wait()
    seal_path = write_json("availability/STRICT_CURRENT_RUNTIME_RESCORE_V1.json", {
        "stage": "N72R21R2", "status": "COMPLETE_ACTUAL_STRICT_LOADER_ALL12_CURRENT_RUNTIME_PREDICTIONS_BEFORE_LABELS",
        "models": {uid: {"fit_record_sha256": sha256(path), "checkpoint_sha256": r["checkpoint_sha256"]} for uid, (path, r) in records.items()},
        "original_current_sampled_groups": len(source), "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size},
        "source_sha256": sha256(__file__), "runtime_predictor_sha256": sha256(ROOT / "sam3_intermot/one_click/fresh_open_set_runtime.py"),
        "runtime_GT_truth_files_opened": False, "sampled_C0_current_axes_not_dense_MOT_policy": True})
    # Offline labels only after every predictor run is sealed above.
    truths = {}
    for sequence in permitted:
        receipt = read_json(OUT / "availability/current_axis_v1/supervision" / (sequence + ".json"))
        assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
        for group in read_zstd_jsonl(Path(receipt["artifact"]["path"])):
            truths[group["episode_uid"], group["frame"]] = group
    groups = [truths[g["episode_uid"], g["frame"]] for g in source]
    models, family_results = {}, defaultdict(list)
    for uid, (path, record) in records.items():
        result = assess_claims(groups, scores[uid], record["selection"])
        assert result["counts"] == record["selected_INNER_current_claims"]["counts"], "Reloaded current inference changed accepted claims"
        points = record["all_INNER_operating_points"]
        fpr_points = [point for point in points if point["result"]["micro"]["verified_OTHER_row_FPR"] is not None and point["result"]["micro"]["verified_OTHER_row_FPR"] <= .02]
        recall_fpr2 = max((point["result"]["micro"]["target_recall_given_available"] or 0. for point in fpr_points), default=None)
        models[uid] = {"fit_record_path": str(path), "fit_record_sha256": sha256(path), "seed": record["seed"], "family": record["family"],
                       "selection": record["selection"], "actual_strict_rescore_selected_claims_AA": True,
                       "selected_current_metrics": {k: v for k, v in result.items() if k != "choices"},
                       "Recall_at_verified_OTHER_row_FPR2_measured_grid_ONLY": recall_fpr2,
                       "not_physically_absent_false_presence_FPR": True,
                       "INDEPENDENT_SAFE_CF_REPAIRS_MISSED": "PENDING_EXACT_CF_AXIS_DIAGNOSTIC_NOT_IMPUTED_FROM_EVERY32_SAMPLES",
                       "empirical_zero_errors_not_population_2percent_guarantee": True}
        family_results[record["family"]].append(result)
    families = {}
    for family, results in family_results.items():
        clusters = sorted({s for result in results for s in result["per_sequence"]})
        intervals = {}
        for metric in ("target_recall_given_available", "coverage", "wrong_takeover_per_group", "false_presence_rate_no_positive_candidate"):
            video_values = []
            for sequence in clusters:
                values = [r["per_sequence"][sequence][metric] for r in results if r["per_sequence"][sequence][metric] is not None]
                if values: video_values.append(float(np.mean(values)))
            intervals[metric] = cluster_interval(video_values)
        families[family] = {"all_three_seeds_retained_not_best_seed": True, "video_macro_seed_averaged_cluster_CI95": intervals,
                            "claim_risk_CI_not_population_guarantee_from_zero_observed_errors": True,
                            "all_init_failure_video_not_imputed_as_correct_or_successful": True}
    write_json("availability/OPEN_SET_CURRENT_AXIS_PROGRESS_V1.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "status": "ACTUAL_OPEN_SET_DIAGNOSTIC_NOT_FULL_MOT_OR_RESEARCH_CLOSURE",
        "runtime_rescore_seal_sha256": sha256(seal_path), "models": models, "families": families, "INNER_initialization_census": initialization,
        "all24_new_video_sources_required_before_fits": True, "UNKNOWN_separate_from_verified_OTHER": True,
        "sampling_period": 32, "current_sampled_frame_groups": len(groups), "CONFIRM_VAL_TEST_or_SOT_used": False,
        "current_identity_nonvacuity_not_event_correction_gate_G1": True, "association_stage_authorized": False,
        "scientific_final_decision": None, "whole_goal_complete": False})
    print({"actual_strict_open_set_rescore": len(records), "groups_each": len(groups), "AA": True,
           "INDEPENDENT_CF_missed_repairs_still_required": True}, flush=True)


if __name__ == "__main__": run()
