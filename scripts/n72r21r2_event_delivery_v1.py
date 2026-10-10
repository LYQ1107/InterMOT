"""Recompute all24 event labels from sealed actual arms, then report M2/M3.

Offline FIT/INNER audit only: no new intervention, training, threshold,
candidate generation or confirmation access. Old failed evidence stays intact.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import torch
from scripts.n72r21r2_common import (
    ROOT, OUT, ASSETS, TRAIN, GOAL, read_json, write_json, sha256,
    preregistration, development_sequence, append_log, storage,
)
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.one_click.causal_state_fingerprint import fingerprint
from sam3_intermot.one_click.event_authority_learning import encode_runtime_input
from sam3_intermot.evaluation.safe_intervention_events import assignment_map
from sam3_intermot.evaluation.event_causal_labels import label_actual_branch
from sam3_intermot.evaluation.event_delivery_evidence import component_audit, original_counts, overlap_components, HORIZONS
from sam3_intermot.evaluation.pinned_event_trajectory import METRIC_NAMES

PREFIX = "events/delivery_audit_v1"
PROTOCOL = "events/EVENT_DELIVERY_AUDIT_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_event_delivery_v1.py", "sam3_intermot/evaluation/event_delivery_evidence.py",
        "scripts/n72r21r2_events.py", "scripts/n72r21r2_events_future_v2.py",
        "scripts/n72r21r2_label_counterfactual.py", "sam3_intermot/evaluation/event_causal_labels.py",
        "sam3_intermot/one_click/event_authority_learning.py")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def reference(path, expected=None, cache=None):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in (ROOT, ASSETS, TRAIN)):
        raise ValueError("Event audit reference escaped existing scoped inputs")
    key = str(path)
    if cache is None or key not in cache:
        value = {"path": key, "sha256": sha256(path)}
        if cache is not None:
            cache[key] = value
    else:
        value = cache[key]
    if expected is not None and value["sha256"] != expected:
        raise ValueError("Sealed event input changed: " + key)
    return value


def prepare():
    p = preregistration()
    seqs = p["split"]["fit"] + p["split"]["inner"]
    write_json(PROTOCOL, {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "frozen": True, "sequences": seqs, "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "audit": "All actual original same-prestate arms, including harmful, duplicate and incomplete windows. Recompute exact original labels from raw trajectories and original offline FIT/INNER GT; separately audit t versus k=1..H components.",
        "selection_or_scientific_method_changed": False, "new_runtime_rollouts_or_optimizer_steps": 0,
        "independent_causal_roots": "UNPROVEN: chronological runs, arm counts and overlap-union components are NOT independent roots",
        "resources": {"CPU_threads": 1, "GPU_required": False, "floor_GiB": 60},
        "confirmation_VAL_TEST_SOT_allowed": False})


def checked_protocol():
    protocol = read_json(OUT / PROTOCOL)
    p = preregistration()
    assert protocol["frozen"] and protocol["goal"] == GOAL
    assert protocol["sequences"] == p["split"]["fit"] + p["split"]["inner"]
    assert protocol["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    assert protocol["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    return protocol


def runtime_inputs(sequence, cache):
    """Verify the ENTIRE original registered runtime before opening raw GT."""
    cf_path = OUT / "events/counterfactual_sequences" / (sequence + ".json")
    cf = read_json(cf_path)
    assert cf["sequence"] == sequence and cf["actual_forbidden_GT_file_guard"] and cf["source_C0_shadow_states_outputs_AA_after_all_future_branches"]
    assert all(sha256(ROOT / name) == digest for name, digest in cf["source_freeze"].items())
    reference(cf_path, cache=cache)
    plan_path = OUT / "events/counterfactual_plans" / (sequence + ".json")
    reference(plan_path, cf["plan_sha256"], cache)
    plan = read_json(plan_path)
    assert not plan["runtime_GT_or_future_label_fields"] and plan["sampling_frozen_before_branch_effects"]
    assert plan["protocol_sha256"] == cf["protocol_sha256"] == sha256(OUT / "events/COUNTERFACTUAL_PROTOCOL_V1.json")
    corpus_path = OUT / "events/corpus_v2" / (sequence + ".json")
    reference(corpus_path, plan["offline_sampling_source_sha256"], cache)
    corpus = read_json(corpus_path)
    assert corpus["event_sampling_class_strata_unchanged"] and corpus["scientific_gate_unchanged"]
    reference(OUT / "events/corpus" / (sequence + ".json"), corpus["original_V1_corpus_sha256"], cache)
    assert corpus["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_events_future_v2.py")
    episodes = {e["episode_uid"]: e for e in corpus["episodes"]}
    seals = {}
    for episode in plan["episodes"]:
        uid = episode["episode_uid"]
        assert set(episode) == {"episode_uid", "frames"}
        assert episode["frames"] == episodes[uid].get("preregistered_long_branch_frames", [])
        for f in episode["frames"]:
            path = OUT / "events/counterfactual_seals" / uid / ("frame" + str(f) + ".json")
            seal = read_json(path)
            reference(path, cache=cache)
            assert seal["source_freeze"] == cf["source_freeze"] and seal["plan_sha256"] == cf["plan_sha256"]
            assert seal["candidate_index_sha256"] == cf["candidate_index_sha256"]
            assert seal["protocol_sha256"] == cf["protocol_sha256"]
            assert seal["all_branches_same_starting_tracker_state"] and seal["future_state_from_own_branch_only"] and seal["actual_forbidden_GT_file_guard"]
            for arm in seal["artifacts"]:
                reference(arm["path"], arm["sha256"], cache)
            assert len({a["starting_tracker_state_sha256"] for a in seal["artifacts"]}) == 1
            assert len({a["branch"] for a in seal["artifacts"]}) == len(seal["artifacts"])
            assert "KEEP" in {a["branch"] for a in seal["artifacts"]}
            seals[uid + "__f" + str(f)] = (path, seal)
    assert len(seals) == cf["actual_event_positions"]
    context_path = OUT / "events/current_context_sequences" / (sequence + ".json")
    context_seq = read_json(context_path)
    reference(context_path, cache=cache)
    assert context_seq["CF_sequence_seal_sha256"] == sha256(cf_path)
    assert context_seq["all_original_full_prefix_states_outputs_AA"] and context_seq["runtime_actual_GT_file_guard"]
    assert context_seq["actual_event_contexts"] == len(seals)
    audit_path = OUT / "events/label_audit" / (sequence + ".json")
    audit = read_json(audit_path)
    reference(audit_path, cache=cache)
    assert audit["sequence"] == sequence and audit["events"] == len(seals)
    assert audit["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_label_counterfactual.py")
    assert audit["label_component_source_sha256"] == sha256(ROOT / "sam3_intermot/evaluation/event_causal_labels.py")
    assert audit["corpus_sha256"] == sha256(corpus_path)
    if "successful_recovery_receipt_path" in audit:
        reference(audit["successful_recovery_receipt_path"], audit["successful_recovery_receipt_sha256"], cache)
        reference(audit["verified_tensor_context_receipt_path"], audit["verified_tensor_context_receipt_sha256"], cache)
        for key in ("original_failed_label_log", "original_failed_partial_preserved"):
            reference(audit[key]["path"], audit[key]["sha256"], cache)
        assert audit["canonicalization_source_sha256"] == sha256(ROOT / "scripts/n72r21r2_context_recovery_driver.py")
        assert audit["original_CF_tapes_and_sampling_not_rerun_or_changed"]
    reference(audit["artifact"]["path"], audit["artifact"]["sha256"], cache)
    rows = read_zstd_jsonl(Path(audit["artifact"]["path"]))
    grouped, contexts = defaultdict(list), {}
    for row in rows:
        assert row["sequence"] == sequence and row["state_source"] == "ACTUAL_FRESH_C0_SHADOW_P0"
        assert row["future_truth_not_runtime_feature"] and row["current_features_captured_before_branch_future"]
        path, seal = seals[row["event_uid"]]
        assert row["CF_runtime_seal_sha256"] == sha256(path)
        ref = next(a for a in seal["artifacts"] if a["branch"] == row["branch"])
        assert row["actual_branch_path"] == ref["path"] and row["actual_branch_sha256"] == ref["sha256"] and row["action"] == ref["current_action"]
        assert row["candidate_index_sha256"] == cf["candidate_index_sha256"]
        runtime = row["runtime_features"]
        encode_runtime_input(runtime, row["branch"], past_steps=3)
        encode_runtime_input(runtime, row["branch"], past_steps=8)
        cp = runtime["current_context_path"]
        reference(cp, runtime["current_context_sha256"], cache)
        if cp not in contexts:
            context = read_json(cp)
            assert context["CF_seal_sha256"] == row["CF_runtime_seal_sha256"] and not context["runtime_GT_or_future_truth_used"]
            assert context["actual_forbidden_GT_file_guard"] and context["candidate_index_sha256"] == cf["candidate_index_sha256"]
            assert context["source_sha256"] in {sha256(ROOT / "scripts/n72r21r2_event_context.py"), sha256(ROOT / "scripts/n72r21r2_event_context_v2.py")}
            contexts[cp] = context
        grouped[row["event_uid"]].append(row)
    assert set(grouped) == set(seals) and len(rows) == audit["counts"].get("actual_arm_records", 0)
    for uid, records in grouped.items():
        assert len(records) == len(seals[uid][1]["artifacts"])
        assert {r["branch"] for r in records} == {a["branch"] for a in seals[uid][1]["artifacts"]}
    return audit, corpus, plan, seals, grouped, contexts


def audit_sequence(sequence):
    p = development_sequence(sequence)
    cache = {}
    audit, corpus, plan, seals, grouped, contexts = runtime_inputs(sequence, cache)
    # Raw original truth opens ONLY after all registered runtime hashes/seals.
    truth_path = OUT / "data/initialization_truth" / (sequence + ".json")
    truth = read_json(truth_path)
    reference(truth_path, cache=cache)
    reference(TRAIN / sequence / "gt/gt.txt", truth["GT_sha256"], cache)
    assert truth["GT_sha256"] == audit["GT_sha256"]
    identities = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    frames, index_sha = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matching = {int(payload["frame"]): strict_candidate_matching(rows, gt.get(int(payload["frame"]), [])) for payload, rows in frames}
    init_path = OUT / "data/initialization" / (sequence + ".json")
    init = read_json(init_path)
    reference(init_path, cache=cache)
    assert index_sha == init["candidate_index_sha256"]
    prefixes, origins, cursors = {}, {}, {}
    for selected in plan["episodes"]:
        if not selected["frames"]:
            continue
        uid = selected["episode_uid"]
        path = OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (uid + ".json")
        seal = read_json(path)
        reference(path, cache=cache)
        artifact = next(a for a in seal["artifacts"] if a["kind"] == "trace")
        reference(artifact["path"], artifact["sha256"], cache)
        prefixes[uid] = read_zstd_jsonl(Path(artifact["path"]))
        origins[uid], cursors[uid] = {}, 0
    utilities_path = OUT / "events/trajectory_utility_audit" / (sequence + ".json")
    ua = read_json(utilities_path)
    reference(utilities_path, cache=cache)
    assert ua["label_audit_sha256"] == sha256(OUT / "events/label_audit" / (sequence + ".json"))
    assert ua["GT_sha256"] == truth["GT_sha256"] and ua["status"] == "COMPLETE_ACTUAL_PINNED_EVENT_TRAJECTORY_UTILITY"
    assert ua["source_freeze"] == {name: sha256(ROOT / name) for name in ua["source_freeze"]}
    reference(ua["artifact"]["path"], ua["artifact"]["sha256"], cache)
    utility_rows = read_zstd_jsonl(Path(ua["artifact"]["path"]))
    utilities = {(r["event_uid"], r["branch"]): r for r in utility_rows}
    assert len(utilities) == len(utility_rows) == ua["arms"]
    outcomes, by_branch, audits, intervals, feasibility = Counter(), defaultdict(Counter), [], [], defaultdict(Counter)
    raw_rows, tensor_prestate, semantic_only_prestate, noop_arms = 0, 0, 0, 0
    for uid, records in grouped.items():
        seal_path, seal = seals[uid]
        episode, frame = seal["episode_uid"], seal["frame"]
        context = contexts[records[0]["runtime_features"]["current_context_path"]]
        assert context["episode_uid"] == episode and context["frame"] == frame
        assert [r["candidate_uid"] for r in context["full_current_candidate_axis"]] == [r["candidate_uid"] for r in frames[frame][1]]
        while cursors[episode] < frame:
            f = cursors[episode]
            for output in prefixes[episode][f]["outputs"]:
                identity = matching[f][output["candidate_uid"]]
                if identity is not None:
                    origins[episode].setdefault(int(output["public_id"]), identity)
            cursors[episode] += 1
        active_origins = {public: identity for public, identity in origins[episode].items() if public in context["public_axis"]}
        branches = {a["branch"]: read_zstd_jsonl(Path(a["path"])) for a in seal["artifacts"]}
        keep = branches["KEEP"]
        visible = {f: any(a["identity"] == identities[episode] for a in gt.get(f, [])) for f in range(frame, min(frame + 101, len(frames)))}
        signatures = {}
        ordered = {r["branch"]: r for r in records}
        for arm in seal["artifacts"]:
            name, trace = arm["branch"], branches[arm["branch"]]
            row = ordered[name]
            assert row["role"] == ("FIT" if sequence in p["split"]["fit"] else "INNER")
            assert canonical(row["prestate_origin_proxy_only_offline"]) == canonical(active_origins)
            assert [r["frame"] for r in trace] == list(range(frame, min(frame + 101, len(frames))))
            assert len(trace) == arm["frames"] and trace[0]["state_before"] == arm["starting_tracker_state_sha256"] == keep[0]["state_before"]
            if trace[0].get("full_tracker_state_before_sha256") is not None:
                assert trace[0]["full_tracker_state_before_sha256"] == context["tensor_inclusive_prestate_sha256"]
                tensor_prestate += 1
            else:
                semantic_only_prestate += 1
            for actual, own_keep in zip(trace, keep, strict=True):
                am, bm = assignment_map(actual), assignment_map(own_keep)
                assert len(am) == len(actual["outputs"]) and len(set(am.values())) == len(am)
                assert set(am.values()) == set(bm.values()) == set(matching[actual["frame"]])
                assert not actual["runtime_GT_read"] and not actual["runtime_future_GT_used"]
                if name == "KEEP":
                    assert all(actual[key] == prefixes[episode][actual["frame"]][key] for key in ("outputs", "target_uid", "state_before", "state_after"))
            recomputed = label_actual_branch(trace, keep, matching, identities[episode], active_origins, visible)
            actions = [r["frame"] for r, b in zip(trace, keep, strict=True) if r["selected_action"]["family"] != "KEEP" and assignment_map(r) != assignment_map(b)]
            sig = fingerprint([r["selected_action"] for r in trace])
            first = signatures.setdefault(sig, name)
            recomputed.update(effective_direct_action_frames=actions, direct_action_onsets_in_this_one_shot_branch=len(actions),
                distinct_correlated_window_not_independent_causal_origin=True, same_executed_action_configuration_as=first if first != name else None)
            current = recomputed["current_t"]
            current_harm = bool(current["N10"] or current["non_target_damage"] or current["verified_OTHER_writes"] or current["UNKNOWN_writes"])
            full = recomputed["future"]["H100"]
            recomputed["H100_any_harm_including_current_t_label"] = bool(current_harm or full["risk_label"]) if full["complete"] else None
            assert canonical(recomputed) == row["offline_supervision_labels"], (sequence, uid, name, "RAW_TRAJECTORY_GT_LABEL_MISMATCH")
            result = component_audit(row["offline_supervision_labels"], frame, branch=name)
            audits.append(result)
            by_branch[name].update({k: int(v) for k, v in result.items() if isinstance(v, (int, bool))})
            by_branch[name]["actual_arm_records"] += 1
            outcomes.update(r["outcome"] for r in recomputed["raw_frame_components"])
            raw_rows += len(trace)
            noop_arms += not actions
            u = utilities[uid, name]
            assert u["actual_branch_sha256"] == arm["sha256"] and u["KEEP_branch_sha256"] == ordered["KEEP"]["actual_branch_sha256"]
            assert u["complete_H100"] == full["complete"] and u["original_future_axis"] == [frame + 1, frame + 100]
            if full["complete"]:
                assert u["L5_value_label"] == .5 * (u["paired_delta_vs_own_KEEP"]["HOTA"] + u["paired_delta_vs_own_KEEP"]["AssA"])
                assert all(u["paired_delta_vs_own_KEEP"][k] == u["actual_nine_metrics"][k] - utilities[uid, "KEEP"]["actual_nine_metrics"][k] for k in METRIC_NAMES)
            else:
                assert u["actual_nine_metrics"] is None and u["paired_delta_vs_own_KEEP"] is None and u["L5_value_label"] is None
            feasibility[name]["actually_executed_feasible_arms"] += 1
            feasibility[name]["effective_direct_decisions_not_roots"] += len(actions)
            feasibility[name]["current_solver_assignment_changed"] += bool(arm["current_feasibility"]["assignment_changed"])
            assert arm["current_feasibility"]["feasible"]
        intervals.append({"sequence": sequence, "anonymous_identity_scope": records[0]["anonymous_identity_scope"], "start": frame, "end": keep[-1]["frame"]})
        for item in seal["not_applicable_or_hard_infeasible"]:
            feasibility[item["branch"]][item["status"]] += 1
        registered = read_json(OUT / "events/COUNTERFACTUAL_PROTOCOL_V1.json")["branches"]
        assert set(branches) | {r["branch"] for r in seal["not_applicable_or_hard_infeasible"]} == set(registered)
        assert len(branches) + len(seal["not_applicable_or_hard_infeasible"]) == len(registered)
    rebuilt = original_counts(audits)
    assert rebuilt == {key: audit["counts"].get(key, 0) for key in rebuilt}
    assert set(utilities) == {(uid, row["branch"]) for uid, rows in grouped.items() for row in rows}
    assert sum(r["complete_H100"] for r in utility_rows) == ua["complete_H100_arms"]
    window_path = OUT / "events/window_trackeval_results" / (sequence + ".json")
    window = read_json(window_path)
    reference(window_path, cache=cache)
    if window["status"] == "COMPLETE_ACTUAL_PINNED_FULL_GLOBAL_WINDOW_METRICS":
        invocation_path = OUT / "events/window_trackeval_invocations" / (sequence + ".json")
        inv = read_json(invocation_path)
        reference(invocation_path, cache=cache)
        reference(inv["log_path"], inv["log_sha256"], cache)
        assert inv == window["receipt"] and inv["returncode"] == 0
        assert inv["TrackEval_commit"] == read_json(OUT / "events/WINDOW_TRACKEVAL_PROTOCOL_V1.json")["TrackEval_commit"]
        assert window["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_window_trackeval.py")
        assert window["protocol_sha256"] == sha256(OUT / "events/WINDOW_TRACKEVAL_PROTOCOL_V1.json")
    else:
        assert window["status"] == "NOT_RUN_NO_REGISTERED_COMPLETE_WINDOW" and not grouped
        assert all(r["initialization_failure"] for r in init["inputs"])
    observed = Counter()
    class_counts = Counter()
    for episode in corpus["episodes"]:
        class_counts.update(episode.get("frame_class_counts", {}))
        observed.update(r["category"] for r in episode.get("observational_intervals", []))
    sums = {key: sum(r[key] for r in audits) for key in audits[0] if isinstance(audits[0][key], (int, bool))} if audits else {}
    value = {"stage": "N72R21R2", "goal": GOAL, "sequence": sequence, "role": "FIT" if sequence in p["split"]["fit"] else "INNER",
        "status": "COMPLETE_ACTUAL_RAW_ARM_GT_LABEL_RECOMPUTATION_NOT_CAUSAL_ROOT_PROOF",
        "protocol_sha256": sha256(OUT / PROTOCOL), "source_sha256": checked_protocol()["source_sha256"],
        "counts": rebuilt, "current_vs_future_overlapping_arm_components": sums,
        "current_and_future_outcomes_overlapping_arm_rows": dict(outcomes), "by_branch": {k: dict(v) for k, v in by_branch.items()},
        "registered_CF_positions_not_independent_events": len(grouped), "raw_current_plus_future_arm_rows": raw_rows,
        "CF_window_intervals_for_correlation_only": intervals,
        "actual_direct_tensor_inclusive_prestate_matching_arms": tensor_prestate,
        "legacy_semantic_only_prestate_arms_NO_retrospective_tensor_proof": semantic_only_prestate,
        "no_effective_action_arms_retained": noop_arms, "action_feasibility": {k: dict(v) for k, v in feasibility.items()},
        "C0_episode_frame_class_counts_not_independent_roots": dict(class_counts),
        "observational_interval_counts_not_independent_roots": dict(observed),
        "observed_interval_total": sum(observed.values()),
        "all_registered_probe_rows_not_independent_events": sum(len(e.get("event_probe_rows", [])) for e in corpus["episodes"]),
        "sole_click_attempts": len(init["inputs"]), "failed_initializations_retained": sum(e["initialization_failure"] for e in init["inputs"]),
        "window_TrackEval_status": window["status"], "window_TrackEval_arms": len(window.get("arms", {})),
        "all_actual_label_fields_recomputed_exactly_from_raw_branches_and_original_GT": True,
        "all_current_runtime_vectors_accept_frozen_label_excluding_encoder": True,
        "own_KEEP_all_raw_frame_outputs_states_AA_vs_sealed_source_prefix": True,
        "new_runtime_rollouts_or_optimizer_steps": 0, "raw_GT_only_after_entire_registered_runtime_verified": True,
        "original_failed_logs_partials_reverified_NOT_overwritten": "successful_recovery_receipt_path" in audit,
        "source_artifacts": list(cache.values()), "independent_beneficial_correction_roots": None, "independent_N10_roots": None,
        "scientific_G1_PASS": False, "next_stage_authorized": False}
    write_json(PREFIX + "/sequences/" + sequence + ".json", value)
    append_log("ALL24_EVENT_RAW_ARM_LABEL_AUDIT_SEQUENCE_COMPLETE", sequence=sequence, counts=rebuilt)
    print({"event_label_recomputed": sequence, "arms": rebuilt["actual_arm_records"], "roots": "UNPROVEN"}, flush=True)
    return value


def reports(rows):
    protocol = checked_protocol()
    assert [row["sequence"] for row in rows] == protocol["sequences"]
    common = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "protocol_sha256": sha256(OUT / PROTOCOL), "preregistration_sha256": protocol["preregistration_sha256"],
        "source_sha256": protocol["source_sha256"], "verified_sequences": protocol["sequences"],
        "source_receipts": [reference(OUT / PREFIX / "sequences" / (row["sequence"] + ".json")) for row in rows],
        "scope": "Actual original FIT16+INNER8, including two all-failed initializer videos; no historical substitution",
        "new_runtime_rollouts_or_optimizer_steps": 0, "CONFIRM_VAL_TEST_SOT_accessed": False,
        "independent_beneficial_correction_roots": None, "independent_N10_roots": None,
        "scientific_G1_PASS": False, "full_goal_complete": False, "next_stage_authorized": False}
    total = Counter()
    for row in rows:
        total.update(row["counts"])
    by_role = {role: {"sequences": [r["sequence"] for r in rows if r["role"] == role],
        "counts": dict(sum((Counter(r["counts"]) for r in rows if r["role"] == role), Counter()))} for role in ("FIT", "INNER")}
    union = overlap_components([i for r in rows for i in r["CF_window_intervals_for_correlation_only"]])
    write_json("events/EVENT_SCHEMA.json", {**common, "status": "ACTUAL_EVENT_SCHEMA_AND_EVIDENCE_FIELD_ROUTING_AUDITED",
        "training_unit": "Pre-state -> proposed current/delayed full-global action -> own future trajectory; observed correlation groups are not proven roots",
        "runtime_input": "Frozen32 causal feature values plus registered branch one-hot; past H3/H8 only. Full context is evidence, NOT all model inputs.",
        "offline_supervision": "Original GT strict matching; current t separate from future t+1..t+H. Raw labels/components in hashed compressed supervision tapes.",
        "field_routes": {"sequence_anonymous_identity": "supervision.sequence / anonymous_identity_scope (video-local; not cross-scene people)",
            "first_divergent_frame": "offline_supervision_labels.first_global_ownership_divergent_frame / first_semantic_state_divergent_frame / first_tensor_state_divergent_frame",
            "pre_intervention_tracker_state": "runtime_features.current_context_path -> pre_intervention_tracker_state / tensor_inclusive_prestate_sha256",
            "baseline_challenger_action": "context.baseline_action / challenger_action; branch action and actual selected_action tape",
            "candidate_UID_full_current_axis": "context.full_current_candidate_axis; original candidate index SHA; actual trajectory output UID",
            "global_scores_identity_scores": "context.base_global_score_matrix / identity_joint_candidate_plus_NONE_scores",
            "global_assignment_cost_competing_public_IDs": "CF_seal.artifacts.current_feasibility.global_regret / displaced_public_ids; context.public_axis/base_assignments",
            "current_GT_correctness": "offline_supervision_labels.current_t ONLY; never encoder input",
            "H1_H5_H20_H50_H100": "offline_supervision_labels.future, k=1..H inclusive; incomplete labels=null",
            "target_gain_other_track_damage": "future.N01/N10/non_target_damage/non_target_benefit plus raw harmed/helped public IDs",
            "future_state_motion_prototype": "raw_frame_components tracker semantic/tensor divergence, motion L2 and prototype hashes; original V1 unavailable fields remain null",
            "takeover_duration": "verified_OTHER_takeover_intervals / max_observed_verified_OTHER_takeover_duration, UNKNOWN separate",
            "action_feasibility": "every registered branch executed feasible or explicit NOT_APPLICABLE/HARD_INFEASIBLE reason",
            "causal_evidence": "CF_runtime_seal_sha256, actual_branch_path/SHA, context path/SHA, original candidate index/SHA; one-shot own KEEP branches"},
        "root_status": "UNPROVEN_INDEPENDENT_CAUSAL_BOUNDARIES; no causal-origin inference from runs, sample spacing, seed, click or branch count"})
    write_json("events/EVENT_CORPUS.json", {**common, "status": "COMPLETE_ALL24_ACTUAL_EVENT_SUPERVISION_CORPUS_WITH_EXPLICIT_ROOT_LIMITATION",
        "actual_counts": dict(total), "by_role": by_role,
        "all24_census": [{k: r[k] for k in ("sequence", "role", "sole_click_attempts", "failed_initializations_retained",
            "registered_CF_positions_not_independent_events", "all_registered_probe_rows_not_independent_events", "observed_interval_total",
            "C0_episode_frame_class_counts_not_independent_roots", "counts")} for r in rows],
        "overlap_union_components_not_independent_roots": union,
        "overlap_union_component_count_not_roots": len(union),
        "E0_E1_E2_E7_E8": "Mutually exclusive original baseline observation categories; E3/E4/E5/E9 additional observational tags, not independent counts",
        "E6_FALSE_OVERRIDE": "Actual arm current_N10; labels are recomputed, not merely a baseline category",
        "E9_LONG_PROPAGATION": "Original observed baseline interval tag plus raw branch N10 propagation intervals; neither interval list proves independent roots",
        "training_weighting": "Frozen equal video -> sole-click episode -> observed interval -> event probe -> distinct action config; duplicate configs excluded by frozen trainer. This controls correlation, NOT root proof.",
        "frozen_inputs_or_training_grid_changed": False})
    write_json("events/EVENT_LABEL_AUDIT.json", {**common, "status": "COMPLETE_ALL24_RAW_TRAJECTORY_ORIGINAL_GT_LABEL_RECOMPUTATION",
        "all_actual_label_fields_match_recomputation": True, "actual_counts": dict(total), "by_role": by_role,
        "raw_current_plus_future_arm_rows_recomputed": sum(r["raw_current_plus_future_arm_rows"] for r in rows),
        "original_label_worker_counters_rebuilt_exactly_all24": True, "current_future_component_sums_checked_all_horizons": list(HORIZONS),
        "all_runtime_vectors_and_H3_H8_accept_label_excluding_frozen_encoder": True,
        "UNKNOWN_verified_OTHER_NONE_separate": True, "incomplete_future_windows_not_negative_labels": True,
        "false_override_other_damage_memory_write_takeover_motion_prototype_and_fragmentation_recomputed": True,
        "own_KEEP_raw_outputs_semantic_states_AA_all_selected_windows": True,
        "actual_direct_tensor_inclusive_prestate_matching_arms": sum(r["actual_direct_tensor_inclusive_prestate_matching_arms"] for r in rows),
        "legacy_semantic_only_prestate_arms_NO_retrospective_tensor_proof": sum(r["legacy_semantic_only_prestate_arms_NO_retrospective_tensor_proof"] for r in rows),
        "preserved_original_failed_context_label_attempts": [r["sequence"] for r in rows if r["original_failed_logs_partials_reverified_NOT_overwritten"]]})
    write_json("events/DIRECT_VS_PROPAGATED.json", {**common, "status": "COMPLETE_ALL24_ONE_SHOT_DIRECT_VS_FUTURE_COMPONENT_AUDIT_NOT_ROOT_PROOF",
        "all24_census": [{"sequence": r["sequence"], "role": r["role"], **r["current_vs_future_overlapping_arm_components"]} for r in rows],
        "unit_warning": "Every count is over overlapping correlated counterfactual arms. Do NOT sum this into policy N01/N10, event precision/recall, independent benefits/harms, or full-video HOTA.",
        "single_branch_direct_action_limit": 1, "delayed_limit": "At most t+2, then own KEEP",
        "independent_event_precision": None, "independent_event_recall": None,
        "required_remaining_proof": "Actual own-policy action-onset branches plus independent causal-boundary adjudication; not chronological observed interval grouping"})
    validation_path = OUT / "events/TRAJECTORY_UTILITY_API_CLI_VALIDATION_V1.json"
    validation = read_json(validation_path)
    assert validation["status"] == "COMPLETE_ALL_PILOT_ARMS_ALL_NINE_METRICS_API_CLI_AA"
    assert validation["source_freeze"] == {name: sha256(ROOT / name) for name in validation["source_freeze"]}
    comparisons = [abs(delta) for row in validation["pilots"] for arm in row["arms"].values() for delta in arm["delta_API_vs_CLI"].values()]
    assert comparisons and max(comparisons) < 1e-10
    write_json("events/COUNTERFACTUAL_BRANCHES.json", {**common, "status": "COMPLETE_ALL24_REGISTERED_ACTUAL_ONE_SHOT_BRANCH_AND_WINDOW_EVIDENCE",
        "frozen_branch_protocol": reference(OUT / "events/COUNTERFACTUAL_PROTOCOL_V1.json"), "actual_counts": dict(total),
        "registered_positions_not_roots": sum(r["registered_CF_positions_not_independent_events"] for r in rows),
        "same_semantic_prestate_full_global_assignment_own_future_trajectory_verified": True,
        "no_future_baseline_state_reused_for_treatment": True,
        "representative_actual_CLI_window_results": [{"sequence": r["sequence"], "status": r["window_TrackEval_status"], "arms": r["window_TrackEval_arms"]} for r in rows],
        "actual_CLI_window_arms": sum(r["window_TrackEval_arms"] for r in rows),
        "API_CLI_validation": reference(validation_path), "actual_pilot_API_CLI_metric_comparisons": len(comparisons),
        "maximum_observed_API_CLI_absolute_difference": max(comparisons),
        "future_H100_actual_pinned_nine_metric_utilities_verified_by_artifact_and_own_KEEP_pair": True,
        "current_inclusive_101_frame_CLI_window_and_future_only100_frame_utility_are_DISTINCT": True,
        "overlapping_window_HOTA_AssA_never_summed_into_full_policy_metrics": True})
    feasible = defaultdict(Counter)
    for r in rows:
        for branch, values in r["action_feasibility"].items():
            feasible[branch].update(values)
    write_json("events/ACTION_FEASIBILITY.json", {**common, "status": "COMPLETE_ALL24_EVERY_REGISTERED_BRANCH_FEASIBILITY_ACCOUNTED",
        "registered_positions_not_roots": sum(r["registered_CF_positions_not_independent_events"] for r in rows),
        "by_branch": {k: dict(v) for k, v in feasible.items()},
        "no_applicable_or_hard_infeasible_case_dropped_or_replaced_with_KEEP_success": True,
        "feasibility_is_not_benefit_safety_or_policy_acceptance": True,
        "solver_and_candidate_matcher_identity_memory_association_NOT_changed": True})
    print({"event_reports_created": 6, "actual_counts": dict(total), "root_proof": "UNPROVEN", "science_complete": False}, flush=True)


def run():
    protocol = checked_protocol()
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Preserve prior audit attempt/marker; inspect exact owner before a versioned continuation")
    torch.set_num_threads(1)
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE_ACTUAL_RAW_ARM_LABEL_AUDIT",
             "protocol_sha256": sha256(OUT / PROTOCOL), "active": None, "completed": [], "failed_retained": []}
    write_json(marker, state)
    rows = []
    for sequence in protocol["sequences"]:
        storage(8 << 20)
        state["active"] = sequence
        write_json(marker, state, mutable=True)
        try:
            rows.append(audit_sequence(sequence))
        except Exception as error:
            state.update(status="FAILED_AUDIT_ATTEMPT_RETAINED", active=None)
            state["failed_retained"].append({"sequence": sequence, "error_type": type(error).__name__, "error": str(error)})
            write_json(marker, state, mutable=True)
            raise
        state["completed"].append(sequence)
        state["active"] = None
        write_json(marker, state, mutable=True)
    reports(rows)
    state.update(status="COMPLETE_ALL24_ACTUAL_RAW_ARM_LABEL_AUDIT_NOT_RESEARCH_CLOSURE", active=None)
    write_json(marker, state, mutable=True)
    append_log("ALL24_EVENT_DELIVERABLES_ACTUAL_RAW_RECOMPUTATION_COMPLETE", sequences=24, reports=6, independent_roots=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
