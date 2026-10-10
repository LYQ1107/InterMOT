"""M7 actual MAIN training/deployment support audit, no new rollout or fit.

Every registered INNER video and seed is required for a requested point.
Compact current-choice/past3 samples are diagnostic, NOT all-frame evidence.
This audit cannot change a model, operating point, action or qualification.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from scripts.n72r21r2_common import (ROOT, OUT, GOAL, read_json, write_json,
                                    sha256, append_log, storage)
from scripts import n72r21r2_main_policy as main
from scripts.n72r21r2_train_event_authority import load_records
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r21r2_events import artifact
from sam3_intermot.one_click.event_authority_learning import (
    encode_runtime_input, correlated_sequence_weights, fit_normalizer)
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.current_feature_support import (
    support_comparison, unpadded_history, prediction_error)

PROTOCOL = OUT / "training/MAIN_FEATURE_SUPPORT_AUDIT_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_main_feature_support_v1.py",
        "sam3_intermot/evaluation/current_feature_support.py",
        "scripts/n72r21r2_train_event_authority.py",
        "sam3_intermot/one_click/event_authority_learning.py",
        "sam3_intermot/one_click/event_authority_runtime.py",
        "sam3_intermot/one_click/event_authority_models.py",
        "sam3_intermot/one_click/intervention_features.py",
        "sam3_intermot/evaluation/learned_policy_evidence.py",
        "sam3_intermot/one_click/authority_support_ablation.py",
        "sam3_intermot/one_click/matched_event_observer.py")


def freeze():
    p = read_json(main.PROTOCOL)
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in main.CODE}
    write_json("training/MAIN_FEATURE_SUPPORT_AUDIT_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "main_protocol_sha256": sha256(main.PROTOCOL),
        "source_sha256": {n: sha256(ROOT / n) for n in CODE},
        "jobs": p["jobs"], "points": list(p["points"]), "seeds": p["seeds"],
        "required_sequences": p["split"]["inner"],
        "scope": "Every actual compact sampled current choice at every16 postclick frames and every effective action, all INNER8 and all3 seeds for each requested completed family/objective/point. Failed clicks and all-failed videos retained, not imputed as successful zero observations.",
        "source_support": "Exact original objective-specific accepted FIT16 optimizer rows from load_records(pilot=False), including original deduplication/incomplete-future exclusions. INNER optimizer rows are not used as FIT support. Exact hierarchical FIT normalizer reproduced and checked against every checkpoint.",
        "measures": "Exact float32 min/max and marginal outside counts; scaled/clipped encoded current and unpadded past3 distributions; own-video counts; frozen predictor replay for every sampled choice using actual same-prestate KEEP and actual own past3.",
        "prediction_absolute_tolerance": .0001,
        "not_joint_support_or_calibration_or_independent_roots_or_G1_G2_G4": True,
        "no_refit_or_feature_zeroing_or_clipping_or_score_counterfactual_or_runtime_change": True,
        "no_best_seed_video_branch_threshold_checkpoint_selection": True,
        "M7_proxies_not_exact_measurements": {
            "last_trusted_gap": "trusted_gap scaled/clipped feature, not original frame units",
            "native_streak": "native_same only binary continuity proxy; actual streak not in this frozen32-feature axis",
            "anchor_state_agreement": "prototype_anchor_agreement",
            "association_margin": "base_KEEP_margin and candidate_margin",
            "identity_uncertainty": "joint_entropy and NONE_probability",
            "bank_drift": "actor_bank_size and prototype_anchor_agreement only proxies; no bank-vector drift measurement in this frozen axis"},
        "source_and_deployment_extractors": "MAIN SupportAblationBridge inherits LearnedEventBridge; prospective controlled source MatchedEventObserver reuses LearnedEventBridge on isolated always-KEEP core view, installs only causal metadata after actual own commit. This audit measures actual support differences; shared extractor code alone is not distribution alignment.",
        "CPU_workers": 1, "reserve_GiB": 60, "next_stage_authorized": False})


def runtime_input(features, past):
    return {"features": features, "feature_vector": feature_vector(features).tolist(),
            "causal_previous_feature_vectors": {"H3": past}}


def run(family, objective, point):
    p, mp = read_json(PROTOCOL), read_json(main.PROTOCOL)
    assert p["goal"] == GOAL and p["frozen"]
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in CODE}
    assert p["main_protocol_sha256"] == sha256(main.PROTOCOL)
    assert point in p["points"] and any(j["family"] == family and j["objective"] == objective for j in p["jobs"])
    uid = family + "__" + objective + "__" + point
    destination = "training/main_feature_support_audit_v1/" + uid + ".json"
    if (OUT / destination).exists():
        raise FileExistsError("Preserve completed/partial source-support audits")
    storage(8 << 20)
    torch.set_num_threads(1)
    # REQUIRE the whole point BEFORE opening optimizer labels or reading effects.
    cells = {seed: [main.verified_runtime(seq, family, objective, seed, point)
                   for seq in p["required_sequences"]] for seed in p["seeds"]}
    source, _, manifest = load_records(objective, pilot=False)
    raw = np.stack([r["x"][:32] for r in source])
    src_past, src_padding = unpadded_history(np.stack([r["past"] for r in source]), 32)
    mean, scale = fit_normalizer(np.stack([r["x"] for r in source]), correlated_sequence_weights(source))
    result = {"stage": "N72R21R2", "goal": GOAL, "protocol_sha256": sha256(PROTOCOL),
        "family": family, "objective": objective, "point": point,
        "status": "COMPLETE_ACTUAL_READ_ONLY_SAMPLED_MAIN_FEATURE_SUPPORT_AND_PREDICTION_REPLAY",
        "FIT_manifest": manifest, "FIT_optimizer_rows_correlated": len(source),
        "FIT_branch_rows_correlated": dict(Counter(r["branch"] for r in source)),
        "FIT_past_padding_vectors": src_padding,
        "required_INNER_sequences": p["required_sequences"], "all_registered_seeds": p["seeds"],
        "per_seed": {}, "on_policy_optimizer_or_staged_training_executed_by_this_audit": False,
        "not_all_frame_feature_audit_or_OOD_causal_failure_or_scientific_closure": True,
        "unchanged_runtime_models_policies_candidates_labels": True, "next_stage_authorized": False}
    for seed in p["seeds"]:
        _, _, fit, _, _ = cells[seed][0]
        saved = torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=True)
        assert saved["feature_names"] == list(FEATURE_NAMES)
        assert saved["family"] == family and saved["objective"] == objective and saved["seed"] == seed
        assert np.array_equal(mean, np.asarray(saved["FIT_mean"], np.float32))
        assert np.array_equal(scale, np.asarray(saved["FIT_scale"], np.float32))
        predictor = EventAuthorityPredictor(saved, development_diagnostic=True)
        current, own_past, rows_by_video, refs = [], [], defaultdict(list), []
        branch_counts, counters, max_error = Counter(), Counter(), 0.
        for _, _, same_fit, seal_path, seal in cells[seed]:
            assert same_fit["checkpoint_sha256"] == fit["checkpoint_sha256"]
            sequence = seal["sequence"]
            refs.append({"sequence": sequence, "runtime_seal_path": str(seal_path),
                         "runtime_seal_sha256": sha256(seal_path),
                         "trace_artifacts": [next(a for a in e["artifacts"] if a["kind"] == "trace") for e in seal["episodes"]]})
            counters["failed_initialization_slots_retained"] += seal["initialization_failures_not_replaced"]
            counters["all_failed_videos_retained_not_successes"] += not bool(seal["episodes"])
            for episode in seal["episodes"]:
                counters["valid_actual_clicks"] += 1
                trace_path = artifact(episode, "trace")
                with runtime_file_guard():
                    for row in read_zstd_jsonl(trace_path):
                        authority = row["authority"]
                        counters["all_actual_global_frames"] += 1
                        counters["effective_actions_all_frames"] += bool(authority["effective_assignment_change"])
                        expected_sample = row["frame"] > episode["event"]["frame"] and (row["frame"] - episode["event"]["frame"] - 1) % 16 == 0
                        assert authority["causal_feature_sample"] == bool(expected_sample or authority["effective_assignment_change"])
                        if not authority["causal_feature_sample"]:
                            continue
                        counters["sampled_current_frames_correlated"] += 1
                        choices = authority["all_current_choices"]
                        assert choices and len({c["branch"] for c in choices}) == len(choices)
                        keep = next(c for c in choices if c["branch"] == "KEEP")
                        past = authority["causal_previous_feature_vectors_H3"]
                        primary = np.asarray(authority["own_primary_feature_vector"], np.float32)
                        assert primary.shape == (32,) and np.isfinite(primary).all()
                        own_past.append(past)
                        keep_runtime = runtime_input(keep["features"], past)
                        for choice in choices:
                            runtime = runtime_input(choice["features"], past)
                            encoded, _ = encode_runtime_input(runtime, choice["branch"])
                            reproduced = predictor.predict(runtime, choice["branch"], keep_runtime)
                            error = prediction_error(reproduced, choice["prediction"])
                            if error > p["prediction_absolute_tolerance"]:
                                raise AssertionError("Actual stored prediction cannot be replayed")
                            max_error = max(max_error, error)
                            current.append(encoded[:32])
                            rows_by_video[sequence].append(encoded[:32])
                            branch_counts[choice["branch"]] += 1
                            counters["sampled_choices_reproduced"] += 1
                            pred = choice["prediction"]
                            if choice["branch"] != "KEEP":
                                counters["sampled_non_KEEP_choices_correlated"] += 1
                                gate = mp["points"][point]
                                counters["sampled_non_KEEP_score_only_pass_not_action"] += (pred["beneficial"] >= gate["claim_min"] and pred["harmful"] <= gate["risk_max"] and pred["value"] > 0)
                assert sha256(trace_path) == next(a["sha256"] for a in episode["artifacts"] if a["kind"] == "trace")
        own = np.stack(current)
        history, padding = unpadded_history(np.asarray(own_past), 32)
        # If a source has no actual past, record NOT_MEASURED rather than
        # treating padding as observations or fabricating a reference support.
        past_comparison = support_comparison(src_past, history, FEATURE_NAMES, mean, scale) if len(src_past) else None
        result["per_seed"][str(seed)] = {"actual_counts": dict(counters),
            "FIT_normalizer_exact_float32_match": True,
            "checkpoint_sha256": fit["checkpoint_sha256"], "runtime_seals": refs,
            "prediction_absolute_max_error": max_error,
            "prediction_reproduction_is_numeric_NOT_bitwise_AA": True,
            "sampled_branch_choice_counts_correlated": dict(branch_counts),
            "current_support": support_comparison(raw, own, FEATURE_NAMES, mean, scale),
            "own_past_padding_vectors": padding, "past_support": past_comparison,
            "past_rows_counted_once_per_sample_frame_NOT_per_branch": True,
            "per_video_current_support": {seq: support_comparison(raw, np.asarray(rows, np.float32).reshape(-1, 32), FEATURE_NAMES, mean, scale)
                                          for seq in p["required_sequences"] for rows in [rows_by_video[seq]]}}
        print({"actual_read_only_MAIN_support_seed": seed, "scope": uid,
               "sampled_choices_reproduced": counters["sampled_choices_reproduced"],
               "prediction_error": max_error,
               "current_FIT_constant_features": [n for n, d in result["per_seed"][str(seed)]["current_support"]["features"].items() if d["FIT_exactly_constant"]]}, flush=True)
    write_json(destination, result)
    append_log("ACTUAL_READ_ONLY_MAIN_FEATURE_SUPPORT", family=family, objective=objective, point=point,
               output=destination, output_sha256=sha256(OUT / destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--family")
    parser.add_argument("--objective")
    parser.add_argument("--point")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        run(args.family, args.objective, args.point)
