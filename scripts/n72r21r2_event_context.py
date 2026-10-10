"""Reconstruct GT-free current/past event context from its exact own prefix."""
import argparse
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, development_sequence, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal
from sam3_intermot.one_click.intervention_features import feature_vector, FEATURE_NAMES
from sam3_intermot.one_click.causal_state_fingerprint import tracker_state, full_tracker_fingerprint, fingerprint, serial
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard


def run(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    sequence_seal_path = OUT / "events/counterfactual_sequences" / (sequence + ".json")
    sequence_seal = read_json(sequence_seal_path)
    for source, expected in sequence_seal["source_freeze"].items():
        assert sha256(ROOT / source) == expected
    plan_path = OUT / "events/counterfactual_plans" / (sequence + ".json")
    assert sha256(plan_path) == sequence_seal["plan_sha256"]
    plan = read_json(plan_path)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    count = 0
    for selected in plan["episodes"]:
        event = next(e for e in init["inputs"] if e["episode_uid"] == selected["episode_uid"])
        if event["initialization_failure"]:
            assert not selected["frames"]
            continue
        actor, source = make_actor(event, np.array(anchors[event["anchor_index"]], np.float32))
        click = {"event_frame": event["frame"], "human_anchor": actor.anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
        bridge = SafeMOTIdentityBridge(click, actor, policy=GatePolicy(family="shadow"), frames=len(frames))
        bridge.configure_fps(event["fps"])
        base_seal = read_json(OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (event["episode_uid"] + ".json"))
        ref = next(a for a in base_seal["artifacts"] if a["kind"] == "trace")
        assert sha256(ref["path"]) == ref["sha256"]
        baseline = read_zstd_jsonl(Path(ref["path"]))
        history = {}
        for payload, rows in frames:
            f = int(payload["frame"])
            if f in selected["frames"]:
                prepared = prepare_proposal(bridge, f, rows)
                seal_path = OUT / "events/counterfactual_seals" / event["episode_uid"] / ("frame" + str(f) + ".json")
                cf_seal = read_json(seal_path)
                assert all(a["starting_tracker_state_sha256"] == prepared["preview"]["state_before"] for a in cf_seal["artifacts"])
                output = prepared["provisional"].model.last
                context = {"stage": "N72R21R2", "sequence": sequence, "episode_uid": event["episode_uid"], "frame": f,
                           "pre_intervention_tracker_state": tracker_state(bridge), "tensor_inclusive_prestate_sha256": full_tracker_fingerprint(bridge),
                           "legacy_semantic_digest_is_not_full_tensor_fingerprint": True,
                           "full_current_candidate_axis": [{k: serial(r[k]) for k in ("candidate_uid", "box_xyxy", "native_tid", "feature_sha256", "embedding_offset") if k in r} for r in rows],
                           "candidate_index_sha256": index_sha, "public_axis": prepared["preview"]["states_before_commit_axis"],
                           "base_global_score_matrix": prepared["preview"]["base_matrix"].tolist(),
                           "base_assignments": prepared["preview"]["base_assignments"], "baseline_action": {"family": "KEEP", "public_id": bridge.tracker.target_public},
                           "challenger_action": prepared["action"].to_dict(), "identity_joint_candidate_plus_NONE_scores": output["joint_probabilities"][0].detach().cpu().tolist(),
                           "identity_actor_snapshot": deepcopy(bridge.identity.snapshot()),
                           "identity_actor_pending_state": serial(deepcopy(bridge.authority_pending)),
                           "causal_previous_feature_vectors": {"H" + str(h): [history.get(t, [0.] * len(FEATURE_NAMES)) for t in range(f - h, f)] for h in (3, 8)},
                           "history_definition": "Sealed own previous post-click SHADOW features only, zero padded before click; excludes current and all future frames",
                           "CF_seal_sha256": sha256(seal_path), "reconstructed_prefix_sha256": ref["sha256"],
                           "source_sha256": sha256(Path(__file__)), "fingerprint_source_sha256": sha256(ROOT / "sam3_intermot/one_click/causal_state_fingerprint.py"),
                           "actual_forbidden_GT_file_guard": True, "runtime_GT_or_future_truth_used": False}
                write_json("events/current_context/" + event["episode_uid"] + "/frame" + str(f) + ".json", context)
                count += 1
            actual = bridge.step(f, rows)
            for key in ("outputs", "target_uid", "state_before", "state_after"):
                assert actual[key] == baseline[f][key]
            features = actual.get("authority", {}).get("features")
            if features is not None:
                history[f] = feature_vector(features).tolist()
    assert count == sequence_seal["actual_event_positions"]
    write_json("events/current_context_sequences/" + sequence + ".json", {"stage": "N72R21R2", "actual_event_contexts": count,
               "source_sha256": sha256(Path(__file__)), "CF_sequence_seal_sha256": sha256(sequence_seal_path),
               "all_original_full_prefix_states_outputs_AA": True, "runtime_actual_GT_file_guard": True})
    append_log("M3_CURRENT_PAST_EVENT_CONTEXT_COMPLETE", sequence=sequence, contexts=count)
    print({"current_causal_context_complete": sequence, "events": count}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    with runtime_file_guard():
        run(parser.parse_args().sequence)
