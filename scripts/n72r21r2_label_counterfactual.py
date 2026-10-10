"""Separate offline actual-branch supervision after all sequence seals exist."""
import argparse
from collections import Counter
from pathlib import Path
import json
import subprocess
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, read_json, write_json, sha256, development_sequence, append_log, storage
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.one_click.intervention_features import feature_vector
from sam3_intermot.one_click.causal_state_fingerprint import fingerprint
from sam3_intermot.evaluation.safe_intervention_events import assignment_map
from sam3_intermot.evaluation.event_causal_labels import label_actual_branch


def verified(sequence):
    development_sequence(sequence)
    seal_path = OUT / "events/counterfactual_sequences" / (sequence + ".json")
    sequence_seal = read_json(seal_path)
    assert sequence_seal["source_C0_shadow_states_outputs_AA_after_all_future_branches"] and sequence_seal["actual_forbidden_GT_file_guard"]
    assert all(sha256(ROOT / path) == value for path, value in sequence_seal["source_freeze"].items())
    plan_path = OUT / "events/counterfactual_plans" / (sequence + ".json")
    assert sha256(plan_path) == sequence_seal["plan_sha256"]
    context_sequence = read_json(OUT / "events/current_context_sequences" / (sequence + ".json"))
    assert context_sequence["CF_sequence_seal_sha256"] == sha256(seal_path)
    plan = read_json(plan_path)
    events = []
    for episode in plan["episodes"]:
        for frame in episode["frames"]:
            path = OUT / "events/counterfactual_seals" / episode["episode_uid"] / ("frame" + str(frame) + ".json")
            seal = read_json(path)
            assert seal["source_freeze"] == sequence_seal["source_freeze"] and seal["plan_sha256"] == sha256(plan_path)
            assert seal["all_branches_same_starting_tracker_state"] and seal["future_state_from_own_branch_only"]
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            context_path = OUT / "events/current_context" / episode["episode_uid"] / ("frame" + str(frame) + ".json")
            context = read_json(context_path)
            assert context["CF_seal_sha256"] == sha256(path) and not context["runtime_GT_or_future_truth_used"]
            assert context["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_event_context.py")
            events.append((path, seal, context_path, context))
    assert len(events) == sequence_seal["actual_event_positions"] == context_sequence["actual_event_contexts"]
    return events, plan


def run(sequence):
    protocol = development_sequence(sequence)
    events, plan = verified(sequence)  # Entire registered sequence verified BEFORE GT labels.
    storage(512 << 20)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    output = ASSETS / "offline_supervision_cf_v1" / (sequence + ".jsonl.zst")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Preserve completed/partial supervision; use a new attempt namespace")
    # Freeze this video's window selection before parsing action effects.
    selected = None
    for event in sorted(init["inputs"], key=lambda e: (e["frame"], e["episode_uid"])):
        if event["initialization_failure"]:
            continue
        positions = next(e["frames"] for e in plan["episodes"] if e["episode_uid"] == event["episode_uid"])
        frame = next((f for f in positions if f + 100 < event["frames"]), None)
        if frame is not None:
            selected = {"episode_uid": event["episode_uid"], "frame": frame}
            break
    write_json("events/window_trackeval_selection/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "selection": selected, "status": "FROZEN_BEFORE_ACTION_LABELS" if selected else "NOT_RUN_NO_FULL_WINDOW",
        "protocol_sha256": sha256(OUT / "events/WINDOW_TRACKEVAL_PROTOCOL_V1.json"), "sampling_uses_action_effects": False})
    truth_file = OUT / "data/initialization_truth" / (sequence + ".json")
    truth = read_json(truth_file)
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    identities = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    frames, index_sha = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {int(p["frame"]): strict_candidate_matching(rows, gt.get(int(p["frame"]), [])) for p, rows in frames}
    corpus_path = OUT / "events/corpus_v2" / (sequence + ".json")
    corpus = read_json(corpus_path)
    probes = {e["episode_uid"]: {r["frame"]: r for r in e.get("event_probe_rows", [])} for e in corpus["episodes"]}
    prefixes = {}
    for episode in plan["episodes"]:
        if not episode["frames"]:
            continue
        seal = read_json(OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (episode["episode_uid"] + ".json"))
        prefixes[episode["episode_uid"]] = read_zstd_jsonl(artifact(seal, "trace"))
    counts = Counter()
    summaries, origin_cache, cursor_cache = [], {}, {}
    role = "FIT" if sequence in protocol["split"]["fit"] else "INNER"
    with output.open("xb") as handle:
        compressor = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
        try:
            for seal_path, seal, context_path, context in events:
                episode, frame = seal["episode_uid"], seal["frame"]
                target = identities[episode]
                origins = origin_cache.setdefault(episode, {})
                cursor = cursor_cache.get(episode, 0)
                while cursor < frame:
                    for o in prefixes[episode][cursor]["outputs"]:
                        identity = matched[cursor][o["candidate_uid"]]
                        if identity is not None:
                            origins.setdefault(int(o["public_id"]), identity)
                    cursor += 1
                cursor_cache[episode] = cursor
                active_origins = {p: identity for p, identity in origins.items() if p in context["public_axis"]}
                branches = {a["branch"]: read_zstd_jsonl(Path(a["path"])) for a in seal["artifacts"]}
                keep = branches["KEEP"]
                assert all(rows[0]["state_before"] == keep[0]["state_before"] for rows in branches.values())
                if "full_tracker_state_before_sha256" in keep[0]:
                    assert all(rows[0]["full_tracker_state_before_sha256"] == context["tensor_inclusive_prestate_sha256"] for rows in branches.values())
                visible = {f: any(a["identity"] == target for a in gt.get(f, [])) for f in range(frame, min(frame + 101, len(frames)))}
                probe = probes[episode][frame]
                event_summary = {"event_uid": episode + "__f" + str(frame), "frame": frame, "branches": {}, "infeasible_or_not_applicable": seal["not_applicable_or_hard_infeasible"]}
                signatures = {}
                for branch in seal["artifacts"]:
                    name = branch["branch"]
                    rows = branches[name]
                    assert branch["feature_vector"] == feature_vector(branch["features"]).tolist()
                    labels = label_actual_branch(rows, keep, matched, target, active_origins, visible)
                    actions = [r["frame"] for r, b in zip(rows, keep, strict=True) if r["selected_action"]["family"] != "KEEP" and assignment_map(r) != assignment_map(b)]
                    assert len(actions) <= 1, "One-shot/delayed diagnostic cannot become an adaptive policy"
                    signature = fingerprint([r["selected_action"] for r in rows])
                    canonical = signatures.setdefault(signature, name)
                    labels.update(effective_direct_action_frames=actions, direct_action_onsets_in_this_one_shot_branch=len(actions),
                                  distinct_correlated_window_not_independent_causal_origin=True,
                                  same_executed_action_configuration_as=canonical if canonical != name else None)
                    including_current_harm = bool(labels["current_t"]["N10"] or labels["current_t"]["non_target_damage"] or labels["current_t"]["verified_OTHER_writes"] or labels["current_t"]["UNKNOWN_writes"])
                    labels["H100_any_harm_including_current_t_label"] = bool(including_current_harm or labels["future"]["H100"]["risk_label"]) if labels["future"]["H100"]["complete"] else None
                    row = {"sequence": sequence, "role": role, "episode_uid": episode, "event_uid": event_summary["event_uid"], "frame": frame,
                           "state_source": "ACTUAL_FRESH_C0_SHADOW_P0", "branch": name, "action": branch["current_action"],
                           "runtime_features": {"features": branch["features"], "feature_vector": branch["feature_vector"],
                                                "causal_previous_feature_vectors": context["causal_previous_feature_vectors"],
                                                "current_context_path": str(context_path), "current_context_sha256": sha256(context_path)},
                           "offline_supervision_labels": labels, "offline_categories": probe["categories"],
                           "anonymous_identity_scope": probe["anonymous_identity_scope"],
                           "correlation_group_not_proven_causal_origin": episode + "__interval" + str(probe["observational_interval_start"]),
                           "CF_runtime_seal_sha256": sha256(seal_path), "actual_branch_path": branch["path"], "actual_branch_sha256": branch["sha256"],
                           "candidate_index_sha256": index_sha, "future_truth_not_runtime_feature": True,
                           "current_features_captured_before_branch_future": True, "prestate_origin_proxy_only_offline": active_origins}
                    compressor.stdin.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
                    counts["actual_arm_records"] += 1
                    counts["effective_one_shot_arms"] += bool(actions)
                    counts["duplicate_executed_action_configs"] += canonical != name
                    full = labels["future"]["H100"]
                    counts["complete_H100_arms"] += full["complete"]
                    counts["observed_safe_positive_H100_arms_NOT_independent_corrections"] += full["benefit_label"] is True and not including_current_harm
                    counts["observed_harmful_H100_arms"] += full["risk_label"] is True or including_current_harm
                    counts["severe_other_harm_H100_arms"] += bool(full["severe_non_target_harm_public_ids"])
                    event_summary["branches"][name] = {"direct_actions": actions, "duplicate_action_configuration_of": canonical if canonical != name else None,
                                                       "current_N01": labels["current_t"]["N01"], "current_N10": labels["current_t"]["N10"],
                                                       "H100": {k: full[k] for k in ("complete", "N01", "N10", "non_target_damage", "severe_non_target_harm_public_ids", "benefit_label", "risk_label")}}
                summaries.append(event_summary)
            compressor.stdin.close()
            assert compressor.wait() == 0
        finally:
            if compressor.poll() is None:
                compressor.terminate()
                compressor.wait()
    write_json("events/label_audit/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence, "role": role,
               "status": "COMPLETE_ACTUAL_OFFLINE_SAME_PRESTATE_CAUSAL_COMPONENT_LABELS", "events": len(events), "counts": dict(counts),
               "event_summaries": summaries, "source_sha256": sha256(Path(__file__)),
               "label_component_source_sha256": sha256(ROOT / "sam3_intermot/evaluation/event_causal_labels.py"),
               "artifact": {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size},
               "GT_sha256": truth["GT_sha256"], "corpus_sha256": sha256(corpus_path),
               "all_registered_sequence_runtime_verified_before_GT_labels": True, "UNKNOWN_separate_from_verified_OTHER": True,
               "incomplete_future_windows_not_negative_training_labels": True,
               "overlapping_CF_windows_not_independent_correction_onsets": True, "not_adaptive_full_policy_MOT_result": True,
               "scientific_G1_PASS_not_claimed": True})
    append_log("M3_FRESH_CF_OFFLINE_LABELS_COMPLETE", sequence=sequence, events=len(events), counts=dict(counts))
    print({"actual_CF_labels": sequence, "events": len(events), "counts": dict(counts)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    run(parser.parse_args().sequence)
