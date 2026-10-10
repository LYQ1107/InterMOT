"""Prospective controlled-treatment states and exact CF-current identity axes.

This is a state-source contrast, NOT model-generated on-policy closure.
Selection reads current feasibility only. Every nested action starts from a
real replayed treatment prestate and has its own original-frame KEEP future.
All video runtime is sealed before the separate offline labeling action.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, read_json, write_json, sha256, storage, development_sequence, preregistration, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_event_context_v2 import sealed_click
from scripts.n72r21r2_counterfactual import compact
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_open_set_data import current_axis_snapshot, label_uid
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.matched_event_observer import MatchedEventObserver, causal_bridge_fingerprint
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal, commit_action, step_keep
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint, fingerprint, serial
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.association.opportunity_solver import AssociationAction
from sam3_intermot.evaluation.event_causal_labels import label_actual_branch

PROTOCOL = OUT / "on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json"
SOURCE_CASES = ("KEEP", "REJECT_TARGET", "RAW_IDENTITY_TOP", "LEARNED_IDENTITY_TOP",
                "TOP_ALTERNATIVE", "RECOVERY", "DELAYED_CHALLENGER")
OFFSETS = (1, 5, 20, 50)
CODE = ("scripts/n72r21r2_joint_state_curriculum.py", "scripts/n72r21r2_event_context_v2.py",
        "scripts/n72r21r2_joint_state_driver.py",
        "scripts/n72r21r2_open_set_data.py", "scripts/n72r21r2_baseline.py",
        "sam3_intermot/one_click/matched_event_observer.py", "sam3_intermot/one_click/event_authority_runtime.py",
        "sam3_intermot/one_click/joint_intervention_primitives.py", "sam3_intermot/one_click/intervention_features.py",
        "sam3_intermot/one_click/causal_state_fingerprint.py", "sam3_intermot/one_click/open_set_verifier.py",
        "sam3_intermot/one_click/runtime_file_guard.py", "sam3_intermot/evaluation/event_causal_labels.py")


def freeze():
    p = preregistration()
    write_json("on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": p["goal"], "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "split": p["split"],
        "frozen_before_new_state_replay_or_labels": True, "source_sha256": {x: sha256(ROOT / x) for x in CODE},
        "source_cases": list(SOURCE_CASES), "state_probe_offsets": list(OFFSETS), "nested_future_H": 100,
        "selection": "For each registered valid click and source case: first chronological already registered CF frame with full future H150, applicable feasible current branch; non-KEEP/non-delayed must change current assignment. No future effect, truth, reward or successful sequence selection.",
        "no_eligible_case": "Explicit NOT_APPLICABLE; no extra frame/video replacement or oracle candidate",
        "source_policy": "Replay the sealed one-shot (or <=t+2 delayed) action on its own state; otherwise KEEP. Check original outputs/semantic and, when available, tensor fingerprints. No whole-video learned-policy claim.",
        "nested_policy": "Deployment-matched exact current branches; one actual current global action then its own KEEP for future k1..100. Never follow another arm's future scheduled actions.",
        "feature_history": "Exact existing LearnedEventBridge always-KEEP view on an isolated clone; accept ONLY past feature/confirmation metadata after actual own commit, never the view's core tracker/actor state.",
        "exact_CF_axis": "Every already frozen CF frame, all actual current UIDs plus explicit NONE before current C0 commit; no every32/nearest-frame imputation. Offline join to actual CF repairs is still required.",
        "memory_state": "Actual P0 no-write bank only. Safely-updated memory source requires separately demonstrated nonvacuous G4, not oracle memory or relabeling unsafe writes.",
        "actual_model_generated_on_policy": False, "candidate_or_association_code_changed": False,
        "all24_required_before_main_state_source_fits": True, "CONFIRM_VAL_TEST_SOT": False,
        "max_seconds_per_episode": 1800, "CPU_workers": 1, "CPU_threads": 1,
        "reserve_GiB": 60, "not_G1_G2_or_scientific_closure": True})


def select_cases(episode, seals, frame_count):
    cases = []
    for name in SOURCE_CASES:
        selected = None
        for seal in sorted(seals, key=lambda s: s["frame"]):
            if seal["frame"] + max(OFFSETS) + 100 >= frame_count:
                continue
            arm = next((a for a in seal["artifacts"] if a["branch"] == name), None)
            if arm is None or not arm["current_feasibility"]["feasible"]:
                continue
            if name not in ("KEEP", "DELAYED_CHALLENGER") and not arm["current_feasibility"]["assignment_changed"]:
                continue
            selected = {"source_case": name, "source_frame": int(seal["frame"]),
                        "source_trace": {"path": arm["path"], "sha256": arm["sha256"]},
                        "source_KEEP_trace": {k: next(a for a in seal["artifacts"] if a["branch"] == "KEEP")[k] for k in ("path", "sha256")},
                        "selected_by_current_feasibility_not_future_effect": True}
            break
        cases.append(selected or {"source_case": name, "status": "NOT_APPLICABLE_NO_REGISTERED_FULL_H150_CURRENT_CHANGE"})
    return {"episode_uid": episode, "cases": cases}


def prepare(sequence):
    development_sequence(sequence)
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {x: sha256(ROOT / x) for x in CODE}
    source_path = OUT / "events/counterfactual_sequences" / (sequence + ".json")
    source = read_json(source_path)
    assert all(sha256(ROOT / x) == h for x, h in source["source_freeze"].items())
    cf_plan_path = OUT / "events/counterfactual_plans" / (sequence + ".json")
    assert sha256(cf_plan_path) == source["plan_sha256"]
    cf_plan = read_json(cf_plan_path)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    episodes = []
    for event in init["inputs"]:
        registered = next(e for e in cf_plan["episodes"] if e["episode_uid"] == event["episode_uid"])
        seals = []
        for f in registered["frames"]:
            path = OUT / "events/counterfactual_seals" / event["episode_uid"] / ("frame" + str(f) + ".json")
            seal = read_json(path)
            assert seal["plan_sha256"] == source["plan_sha256"] and seal["source_freeze"] == source["source_freeze"]
            seals.append(seal)
        selected = select_cases(event["episode_uid"], seals, event["frames"])
        selected.update(exact_CF_axis_frames=registered["frames"], initialization_failure_not_replaced=event["initialization_failure"])
        episodes.append(selected)
    write_json("on_policy/joint_state_v1/plans/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "episodes": episodes, "protocol_sha256": sha256(PROTOCOL),
        "CF_sequence_sha256": sha256(source_path), "CF_frame_plan_sha256": sha256(cf_plan_path),
        "frozen_before_new_treatment_replay": True, "future_GT_or_rewards_used_for_selection": False})


@contextmanager
def writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
        try:
            def emit(row):
                proc.stdin.write((json.dumps(serial(row), sort_keys=True, allow_nan=False) + "\n").encode())
            yield emit
            proc.stdin.close()
            assert proc.wait() == 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()


def full_compact(result, bridge, before):
    state = bridge.tracker.states[bridge.tracker.target_public]
    return {**compact(result, bridge), "full_tracker_state_before_sha256": before,
            "full_tracker_state_after_sha256": full_tracker_fingerprint(bridge),
            "target_prototype_sha256": __import__("hashlib").sha256(state.prototype.tobytes()).hexdigest(),
            "target_motion_state": {"last_box": state.last_box.tolist(), "velocity": state.velocity.tolist()}}


def action(data):
    return AssociationAction(data["family"], data["public_id"], candidate_uid=data.get("candidate_uid"))


def nested_arms(parent, observer, frames, f, source_case, source_frame, keep_reference, emit):
    before, pending = causal_bridge_fingerprint(parent), observer.state_fingerprint()
    observation = observer.observe(parent, f, frames[f][1])
    source_keep_previous = keep_reference[f - source_frame - 1]
    tensor_changed = (None if "full_tracker_state_after_sha256" not in source_keep_previous else
                      full_tracker_fingerprint(parent) != source_keep_previous["full_tracker_state_after_sha256"])
    distribution = {"trusted_gap": f - parent.tracker.last_trusted_frame,
                    "observed_native_streak": parent.tracker.observed_native_streak,
                    "actor_bank_size": len(parent.identity.bank), "actor_bank_drift": 0.,
                    "bank_drift_semantics": "EMPTY_P0_BANK_NOT_A_LEARNED_MEMORY_MEASUREMENT"}
    signatures = {}
    count = 0
    for choice in observation["choices"]:
        arm = parent.clone()
        start = full_tracker_fingerprint(arm)
        prepared = prepare_proposal(arm, f, frames[f][1])
        result = commit_action(arm, f, frames[f][1], prepared, action(choice["action"]))
        trace = [full_compact(result, arm, start)]
        for payload, rows in frames[f + 1:f + 101]:
            current = int(payload["frame"])
            start = full_tracker_fingerprint(arm)
            trace.append(full_compact(step_keep(arm, current, rows), arm, start))
        assert len(trace) == 101
        signature = fingerprint([{k: r[k] for k in ("outputs", "full_tracker_state_after_sha256")} for r in trace])
        canonical = signatures.setdefault(signature, choice["branch"])
        emit({"record_type": "NESTED_ARM", "source_case": source_case, "source_frame": source_frame,
              "frame": f, "state_offset": f - source_frame, "branch": choice["branch"], "action": choice["action"],
              "runtime_features": choice["runtime"], "own_prestate_sha256": before,
              "treatment_tracker_state_changed_vs_source_own_KEEP": tensor_changed,
              "actor_snapshot_changed_vs_source_own_KEEP": parent.identity.snapshot() != source_keep_previous["actor_state"],
              "causal_state_distribution_observables": distribution,
              "public_axis": sorted(parent.tracker.states), "actual_full_global_trajectory": trace,
              "same_executed_trajectory_as": canonical if canonical != choice["branch"] else None,
              "future_from_own_arm_only": True, "actual_model_generated_on_policy": False})
        count += 1
    assert before == causal_bridge_fingerprint(parent) and pending == observer.state_fingerprint()
    return count


def collect_case(bridge, observer, frames, case, emit):
    source_frame, source_case = case["source_frame"], case["source_case"]
    assert sha256(case["source_trace"]["path"]) == case["source_trace"]["sha256"]
    reference = read_zstd_jsonl(Path(case["source_trace"]["path"]))
    assert sha256(case["source_KEEP_trace"]["path"]) == case["source_KEEP_trace"]["sha256"]
    keep_reference = read_zstd_jsonl(Path(case["source_KEEP_trace"]["path"]))
    parent, history = bridge.clone(), observer.clone()
    core_before, observer_before = causal_bridge_fingerprint(bridge), observer.state_fingerprint()
    trace, arms, actual_effects, tensor_checks = [], 0, [], 0
    for payload, rows in frames[source_frame:source_frame + max(OFFSETS) + 1]:
        f = int(payload["frame"])
        if f - source_frame in OFFSETS:
            arms += nested_arms(parent, history, frames, f, source_case, source_frame, keep_reference, emit)
        observation = history.observe(parent, f, rows)
        prepared = prepare_proposal(parent, f, rows)
        start = full_tracker_fingerprint(parent)
        original = reference[f - source_frame]
        result = commit_action(parent, f, rows, prepared, action(original["selected_action"]))
        for key in ("outputs", "target_uid", "state_before", "state_after", "selected_action"):
            assert result[key] == original[key], (source_case, f, key)
        if "full_tracker_state_before_sha256" in original:
            assert start == original["full_tracker_state_before_sha256"]
            assert full_tracker_fingerprint(parent) == original["full_tracker_state_after_sha256"]
            tensor_checks += 1
        if result["outputs"] != observation["own_KEEP_outputs"]:
            actual_effects.append(f)
        history.accept_commit(parent, f, observation, result)
        assert not result["joint_identity_memory_write"] and not parent.identity.bank
        trace.append(full_compact(result, parent, start))
    emit({"record_type": "SOURCE_PREFIX", "source_case": source_case, "source_frame": source_frame,
          "own_actual_source_prefix": trace, "effective_source_action_frames": actual_effects,
          "matched_original_source_tensor_frames": tensor_checks,
          "source_history_not_model_generated_on_policy": True, "actual_no_write_bank": True})
    assert core_before == causal_bridge_fingerprint(bridge) and observer_before == observer.state_fingerprint()
    return arms, len(actual_effects), tensor_checks


def runtime(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    p = read_json(PROTOCOL)
    source = {x: sha256(ROOT / x) for x in CODE}
    assert source == p["source_sha256"]
    plan_path = OUT / "on_policy/joint_state_v1/plans" / (sequence + ".json")
    plan = read_json(plan_path)
    assert plan["protocol_sha256"] == sha256(PROTOCOL)
    frames, index_sha = checked_frames(sequence)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    seals, totals = [], Counter()
    for event in init["inputs"]:
        episode = next(e for e in plan["episodes"] if e["episode_uid"] == event["episode_uid"])
        if event["initialization_failure"]:
            totals["initialization_failure_not_replaced"] += 1
            continue
        seal_path = OUT / "on_policy/joint_state_v1/runtime" / (event["episode_uid"] + ".json")
        if seal_path.exists():
            old = read_json(seal_path)
            assert old["source_sha256"] == source and old["plan_sha256"] == sha256(plan_path)
            assert sha256(old["artifact"]["path"]) == old["artifact"]["sha256"]
            totals.update(old["counts"])
            seals.append({"path": str(seal_path), "sha256": sha256(seal_path)})
            continue
        storage(128 << 20)
        anchor = np.array(anchors[event["anchor_index"]], np.float32)
        actor, identity = make_actor(event, anchor)
        bridge = SafeMOTIdentityBridge(sealed_click(event, anchor), actor, policy=GatePolicy(family="shadow"), frames=len(frames))
        bridge.configure_fps(event["fps"])
        observer = MatchedEventObserver()
        baseline_path = OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (event["episode_uid"] + ".json")
        baseline_seal = read_json(baseline_path)
        baseline_ref = artifact(baseline_seal, "trace")
        assert sha256(baseline_ref) == next(a["sha256"] for a in baseline_seal["artifacts"] if a["kind"] == "trace")
        baseline = read_zstd_jsonl(baseline_ref)
        destination = ASSETS / "joint_state_curriculum_v1/runtime" / (event["episode_uid"] + ".jsonl.zst")
        counts, began = Counter(), time.monotonic()
        with writer(destination) as emit:
            for payload, rows in frames:
                if time.monotonic() - began > p["max_seconds_per_episode"]:
                    raise RuntimeError("Bounded joint-state episode cap; preserve partial evidence")
                f = int(payload["frame"])
                if f in episode["exact_CF_axis_frames"]:
                    sample = current_axis_snapshot(bridge, f, rows)
                    emit({"record_type": "EXACT_CF_CURRENT_AXIS", **sample,
                          "episode_uid": event["episode_uid"], "source_history": "ACTUAL_C0_SHADOW_P0_EXACT_CF_FRAME"})
                    counts["exact_CF_current_axis_frames"] += 1
                for case in episode["cases"]:
                    if case.get("source_frame") == f:
                        arms, effects, tensor_frames = collect_case(bridge, observer, frames, case, emit)
                        counts["actual_source_cases"] += 1
                        counts["actual_nested_arms"] += arms
                        counts["effective_source_action_onsets_NOT_independent_events"] += effects
                        counts["source_tensor_AA_frames"] += tensor_frames
                        print({"joint_state_source": event["episode_uid"], "case": case["source_case"], "frame": f,
                               "nested_arms": arms, "source_effective_actions": effects}, flush=True)
                observation = observer.observe(bridge, f, rows) if f > event["frame"] else None
                result = bridge.step(f, rows)
                if observation is not None:
                    observer.accept_commit(bridge, f, observation, result)
                for key in ("outputs", "target_uid", "state_before", "state_after", "selected_action"):
                    assert result[key] == baseline[f][key], (event["episode_uid"], f, key)
                assert not result["joint_identity_memory_write"] and not bridge.identity.bank
        write_json("on_policy/joint_state_v1/runtime/" + event["episode_uid"] + ".json", {
            "stage": "N72R21R2", "sequence": sequence, "event": event, "source_sha256": source,
            "protocol_sha256": sha256(PROTOCOL), "plan_sha256": sha256(plan_path), "counts": dict(counts),
            "candidate_index_sha256": index_sha, "identity_source": identity, "seconds": time.monotonic() - began,
            "baseline_reference_sha256": sha256(baseline_path), "full_C0_outputs_semantic_states_AA": True,
            "source_and_nested_clone_tensor_actor_history_isolation": True, "actual_runtime_GT_file_guard": True,
            "actual_model_generated_on_policy": False, "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size}})
        totals.update(counts)
        seals.append({"path": str(seal_path), "sha256": sha256(seal_path)})
    write_json("on_policy/joint_state_v1/runtime_sequences/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "source_sha256": source, "protocol_sha256": sha256(PROTOCOL),
        "plan_sha256": sha256(plan_path), "episodes": seals, "counts": dict(totals),
        "all_registered_clicks_retained": True, "not_model_generated_on_policy_or_full_MOT_result": True})
    append_log("M7_CONTROLLED_TREATMENT_STATE_AND_M8_EXACT_CF_AXIS_RUNTIME", sequence=sequence, counts=dict(totals))


def label(sequence):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    development_sequence(sequence)
    runtime_path = OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json")
    runtime_seal = read_json(runtime_path)
    assert runtime_seal["source_sha256"] == {x: sha256(ROOT / x) for x in CODE}
    assert runtime_seal["protocol_sha256"] == sha256(PROTOCOL)
    episodes = []
    for ref in runtime_seal["episodes"]:
        assert sha256(ref["path"]) == ref["sha256"]
        seal = read_json(ref["path"])
        assert sha256(seal["artifact"]["path"]) == seal["artifact"]["sha256"]
        episodes.append(seal)
    # Verify all registered video runtime before first truth access.
    frames, index_sha = checked_frames(sequence)
    truth = read_json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {p["frame"]: strict_candidate_matching(rows, gt.get(p["frame"], [])) for p, rows in frames}
    counts = Counter()
    destination = ASSETS / "joint_state_curriculum_v1/supervision" / (sequence + ".jsonl.zst")
    with writer(destination) as emit:
        for seal in episodes:
            event = seal["event"]
            episode, target = event["episode_uid"], targets[event["episode_uid"]]
            records = read_zstd_jsonl(Path(seal["artifact"]["path"]))
            parents = {(r["source_case"], r["source_frame"]): r for r in records if r["record_type"] == "SOURCE_PREFIX"}
            keeps = {(r["source_case"], r["source_frame"], r["frame"]): r for r in records if r["record_type"] == "NESTED_ARM" and r["branch"] == "KEEP"}
            baseline = read_zstd_jsonl(artifact(read_json(OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (episode + ".json")), "trace"))
            for record in records:
                if record["record_type"] == "SOURCE_PREFIX":
                    continue
                f = record["frame"]
                visible = any(a["identity"] == target for a in gt.get(f, []))
                available = target in matched[f].values()
                meta = {"sequence": sequence, "role": event["role"], "episode_uid": episode,
                        "runtime_artifact_sha256": seal["artifact"]["sha256"], "candidate_index_sha256": index_sha,
                        "GT_labels_never_runtime_features": True}
                if record["record_type"] == "EXACT_CF_CURRENT_AXIS":
                    axis = [{**r, "current_outcome": label_uid(r["candidate_uid"], matched[f], target, available)} for r in record["axis"]]
                    emit({**record, **meta, "axis": axis, "target_candidate_available_label": available,
                          "target_visible_label": visible, "not_current_correctness_training_source_replacement": True})
                    counts["exact_CF_current_axis_groups"] += 1
                    continue
                source_frame = record["source_frame"]
                parent = parents[record["source_case"], source_frame]
                prefix = baseline[:source_frame] + parent["own_actual_source_prefix"][:f - source_frame]
                origins = {}
                for row in prefix:
                    for output in row["outputs"]:
                        identity = matched[row["frame"]][output["candidate_uid"]]
                        if identity is not None:
                            origins.setdefault(int(output["public_id"]), identity)
                active = {p: v for p, v in origins.items() if p in record["public_axis"]}
                keep = keeps[record["source_case"], source_frame, f]
                labels = label_actual_branch(record["actual_full_global_trajectory"], keep["actual_full_global_trajectory"],
                                             matched, target, active,
                                             {t: any(a["identity"] == target for a in gt.get(t, [])) for t in range(f, f + 101)})
                labels["same_executed_action_configuration_as"] = record["same_executed_trajectory_as"]
                current_harm = any(labels["current_t"][k] for k in ("N10", "non_target_damage", "verified_OTHER_writes", "UNKNOWN_writes"))
                labels["H100_any_harm_including_current_t_label"] = bool(current_harm or labels["future"]["H100"]["risk_label"])
                own_public = record["actual_full_global_trajectory"][0]["target_public_id"]
                previous = prefix[-1] if prefix else None
                parent_outcome = label_uid(None if previous is None else previous["target_uid"], matched[f - 1], target, target in matched[f - 1].values())
                event_uid = episode + "__" + record["source_case"] + "__source" + str(source_frame) + "__state" + str(f)
                emit({**meta, "record_type": "NESTED_ARM", "event_uid": event_uid, "frame": f,
                      "source_case": record["source_case"], "source_frame": source_frame, "state_offset": record["state_offset"],
                      "state_source": "MATCHED_BASELINE_STATE" if record["source_case"] == "KEEP" else "ACTUAL_CONTROLLED_TREATMENT_STATE",
                      "branch": record["branch"], "action": record["action"], "runtime_features": record["runtime_features"],
                      "treatment_tracker_state_changed_vs_source_own_KEEP": record["treatment_tracker_state_changed_vs_source_own_KEEP"],
                      "actor_snapshot_changed_vs_source_own_KEEP": record["actor_snapshot_changed_vs_source_own_KEEP"],
                      "causal_state_distribution_observables": record["causal_state_distribution_observables"],
                      "offline_supervision_labels": labels, "parent_current_identity_outcome_offline": parent_outcome,
                      "actual_source_effective_action_frames": parent["effective_source_action_frames"],
                      "correlation_group_not_proven_causal_origin": episode + "__source" + str(source_frame),
                      "prestate_origin_proxy_only_offline": active, "target_public_id": own_public,
                      "actual_model_generated_on_policy": False, "not_independent_correction_or_full_policy_result": True})
                counts["nested_arm_rows"] += 1
                counts["baseline_state_rows" if record["source_case"] == "KEEP" else "controlled_treatment_state_rows"] += 1
                counts["duplicate_executed_trajectory_rows"] += record["same_executed_trajectory_as"] is not None
                counts["H100_safe_positive_arms_NOT_independent_corrections"] += labels["future"]["H100"]["benefit_label"] is True and not current_harm
                counts["H100_harmful_arms"] += labels["H100_any_harm_including_current_t_label"]
                counts["verified_OTHER_parent_state_rows"] += parent_outcome == "VERIFIED_OTHER"
                counts["UNKNOWN_parent_state_rows_not_verified_wrong"] += parent_outcome == "UNKNOWN_UNMATCHED"
                counts["incorrect_NONE_parent_state_rows"] += parent_outcome == "INCORRECT_NONE"
                counts["actually_changed_tracker_state_rows_vs_source_KEEP"] += record["treatment_tracker_state_changed_vs_source_own_KEEP"] is True
    write_json("on_policy/joint_state_v1/supervision/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "runtime_sequence_sha256": sha256(runtime_path),
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": runtime_seal["source_sha256"], "counts": dict(counts),
        "GT_sha256": truth["GT_sha256"], "whole_video_runtime_verified_before_GT": True,
        "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size},
        "not_model_generated_on_policy_or_scientific_closure": True})
    append_log("M7_CONTROLLED_TREATMENT_STATE_OFFLINE_LABELS", sequence=sequence, counts=dict(counts))
    print({"joint_state_labels": sequence, "counts": dict(counts)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "prepare", "runtime", "label"])
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "label":
        label(args.sequence)
    else:
        with runtime_file_guard():
            {"prepare": prepare, "runtime": runtime}[args.action](args.sequence)
