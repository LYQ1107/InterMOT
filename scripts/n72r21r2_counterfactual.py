"""Fresh actual same-prestate full-global action branches, no runtime GT.

prepare is explicitly offline sampling. runtime opens only the frame-only
plan, sealed click, frozen model and candidate tape under a file boundary.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, development_sequence, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.intervention_features import opportunity_features, feature_vector
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal, check_action, commit_action, step_keep
from sam3_intermot.one_click.acib_runtime import overlap
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.association.opportunity_solver import AssociationAction, action_for_candidate

PROTOCOL = OUT / "events/COUNTERFACTUAL_PROTOCOL_V1.json"
CODE = ["scripts/n72r21r2_counterfactual.py", "scripts/n72r21r2_baseline.py", "scripts/n72r21r2_common.py",
        "sam3_intermot/one_click/safe_mot_bridge.py", "sam3_intermot/one_click/joint_intervention_primitives.py",
        "sam3_intermot/one_click/intervention_features.py", "sam3_intermot/one_click/mot_bridge.py",
        "sam3_intermot/one_click/acib_trusted_runtime.py", "sam3_intermot/one_click/runtime_file_guard.py"]


def prepare(sequence):
    development_sequence(sequence)
    corpus_path = OUT / "events/corpus_v2" / (sequence + ".json")
    corpus = read_json(corpus_path)
    plan = {"stage": "N72R21R2", "sequence": sequence, "episodes": [], "protocol_sha256": sha256(PROTOCOL),
            "offline_sampling_source_sha256": sha256(corpus_path), "runtime_GT_or_future_label_fields": False,
            "sampling_frozen_before_branch_effects": True}
    for episode in corpus["episodes"]:
        plan["episodes"].append({"episode_uid": episode["episode_uid"], "frames": episode.get("preregistered_long_branch_frames", [])})
    write_json("events/counterfactual_plans/" + sequence + ".json", plan)
    print({"frame_only_CF_plan": sequence, "positions": sum(len(e["frames"]) for e in plan["episodes"])}, flush=True)


def choose_action(name, bridge, rows, prepared):
    public = bridge.tracker.target_public
    keep = AssociationAction("KEEP", public)
    preview = prepared["preview"]
    uids = [str(r["candidate_uid"]) for r in rows]
    if name in ("KEEP", "DELAYED_CHALLENGER"):
        return keep, None
    if name == "REJECT_TARGET":
        return AssociationAction("REJECT_TARGET", public), None
    if name == "RECOVERY" and preview["target_uid"] is not None and bridge.tracker.states[public].state != "LOST":
        return None, "NOT_APPLICABLE_NOT_RECOVERY"
    scores = [float(np.dot(bridge.tracker.event["human_anchor"], r["feature"])) for r in rows]
    eligible = list(range(len(rows)))
    if name == "LEARNED_IDENTITY_TOP":
        probabilities = prepared["provisional"].model.last["joint_probabilities"][0, :len(rows)].detach().cpu().numpy().copy()
        scores = [float(x) for x in probabilities]
    elif name == "TOP_ALTERNATIVE":
        eligible = [i for i in eligible if uids[i] != preview["target_uid"]]
    elif name == "FEASIBLE_GLOBAL_SWAP":
        owners = {o["candidate_uid"]: int(o["public_id"]) for o in preview["outputs"]}
        eligible = [i for i in eligible if owners.get(uids[i]) not in (None, public)]
    if not eligible:
        return None, "NO_APPLICABLE_CURRENT_CANDIDATE"
    slot = min(eligible, key=lambda i: (-scores[i], uids[i]))
    return action_for_candidate(public, uids[slot], preview["solver"]), None


def compact(result, bridge):
    return {key: result[key] for key in ("frame", "outputs", "target_uid", "target_public_id", "selected_action", "state_before", "state_after", "births", "deaths")} | {
        "identity_memory_write": bool(result["joint_identity_memory_write"]),
        "identity_memory_write_UID": result["joint_memory_write_candidate_uid"],
        "actor_state": bridge.identity.snapshot(),
        "native_after": bridge.tracker.states[bridge.tracker.target_public].last_native_tid,
        "prototype_anchor_cosine": float(np.dot(bridge.tracker.states[bridge.tracker.target_public].prototype, bridge.tracker.event["human_anchor"])),
        "runtime_GT_read": False, "runtime_future_GT_used": False}


def collect_event(source_bridge, frames, frame, episode_uid, source_freeze, index_sha, plan_sha):
    relative = "events/counterfactual_seals/" + episode_uid + "/frame" + str(frame) + ".json"
    if (OUT / relative).exists():
        previous = read_json(OUT / relative)
        assert previous["source_freeze"] == source_freeze and previous["plan_sha256"] == plan_sha
        assert all(sha256(a["path"]) == a["sha256"] for a in previous["artifacts"])
        return previous
    storage(512 << 20)
    protocol = read_json(PROTOCOL)
    artifacts, unavailable = [], []
    for name in protocol["branches"]:
        arm = source_bridge.clone()
        rows = frames[frame][1]
        prepared = prepare_proposal(arm, frame, rows)
        action, reason = choose_action(name, arm, rows, prepared)
        if action is None:
            unavailable.append({"branch": name, "status": reason})
            continue
        check = check_action(arm, frame, rows, prepared["preview"], action)
        if not check["feasible"]:
            unavailable.append({"branch": name, "status": "HARD_INFEASIBLE", "action": action.to_dict(), "reason": check["reason"]})
            continue
        uid = prepared["preview"]["target_uid"] if action.family == "KEEP" else action.candidate_uid
        # Feature-only proposal view; never mutate the committed actor proposal.
        feature_prepared = {**prepared, "proposal": {**prepared["proposal"], "selected_candidate_uid": uid}}
        features = deepcopy(opportunity_features(arm, frame, rows, feature_prepared, check))
        vector = feature_vector(features).tolist()
        current = commit_action(arm, frame, rows, prepared, action)
        trace = [compact(current, arm)]
        executed_frame = frame if name not in ("KEEP", "DELAYED_CHALLENGER") else None
        initial_delayed_candidate = next((r for r in rows if r["candidate_uid"] == prepared["proposal"]["selected_candidate_uid"]), None)
        delayed_previous = None if initial_delayed_candidate is None else {"feature": np.array(initial_delayed_candidate["feature"], copy=True), "box_xyxy": list(initial_delayed_candidate["box_xyxy"])}
        delayed_count = 1 if initial_delayed_candidate is not None else 0
        for payload, future_rows in frames[frame + 1:min(len(frames), frame + 101)]:
            f = int(payload["frame"])
            if name == "DELAYED_CHALLENGER" and executed_frame is None and f <= frame + 2:
                proposal = prepare_proposal(arm, f, future_rows)
                candidate = next((r for r in future_rows if r["candidate_uid"] == proposal["proposal"]["selected_candidate_uid"]), None)
                agreement = candidate is not None and delayed_previous is not None and float(np.dot(candidate["feature"], delayed_previous["feature"])) >= .9 and overlap(candidate["box_xyxy"], delayed_previous["box_xyxy"]) >= .3
                delayed_count = delayed_count + 1 if agreement else 1 if candidate is not None else 0
                delayed_previous = None if candidate is None else {"feature": np.array(candidate["feature"], copy=True), "box_xyxy": list(candidate["box_xyxy"])}
                tested = proposal["action"]
                tested_check = check_action(arm, f, future_rows, proposal["preview"], tested)
                approve = delayed_count >= 3 and tested_check["feasible"] and tested_check["assignment_changed"]
                actual = tested if approve else AssociationAction("KEEP", arm.tracker.target_public)
                result = commit_action(arm, f, future_rows, proposal, actual)
                if approve:
                    executed_frame = f
            else:
                result = step_keep(arm, f, future_rows)
            trace.append(compact(result, arm))
        path = ASSETS / "counterfactual_v1" / episode_uid / ("frame" + str(frame)) / (name + ".jsonl.zst")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
            try:
                for row in trace:
                    proc.stdin.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
                proc.stdin.close()
                assert proc.wait() == 0
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
        artifacts.append({"branch": name, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                          "frames": len(trace), "current_action": action.to_dict(), "executed_action_frame": executed_frame,
                          "starting_tracker_state_sha256": trace[0]["state_before"], "features": features, "feature_vector": vector,
                          "current_feasibility": {"feasible": True, "global_regret": check["global_cost"], "assignment_changed": check["assignment_changed"],
                                                  "displaced_public_ids": check["displaced_public_ids"]}})
    assert len({a["starting_tracker_state_sha256"] for a in artifacts}) == 1
    assert any(a["branch"] == "KEEP" for a in artifacts)
    seal = {"stage": "N72R21R2", "status": "COMPLETE_ACTUAL_SAME_PRESTATE_FULL_GLOBAL_BRANCHES", "episode_uid": episode_uid,
            "frame": frame, "artifacts": artifacts, "not_applicable_or_hard_infeasible": unavailable, "source_freeze": source_freeze,
            "candidate_index_sha256": index_sha, "plan_sha256": plan_sha, "protocol_sha256": sha256(PROTOCOL),
            "all_branches_same_starting_tracker_state": True, "future_state_from_own_branch_only": True,
            "actual_forbidden_GT_file_guard": True, "labels_not_yet_generated": True}
    write_json(relative, seal)
    return seal


def runtime(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    source_freeze = {p: sha256(ROOT / p) for p in CODE}
    plan_path = OUT / "events/counterfactual_plans" / (sequence + ".json")
    plan = read_json(plan_path)
    assert plan["protocol_sha256"] == sha256(PROTOCOL) and not plan["runtime_GT_or_future_label_fields"]
    assert all(set(e) == {"episode_uid", "frames"} for e in plan["episodes"])
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    total = 0
    for episode in plan["episodes"]:
        event = next(e for e in init["inputs"] if e["episode_uid"] == episode["episode_uid"])
        if event["initialization_failure"]:
            assert not episode["frames"]
            continue
        anchor = np.array(anchors[event["anchor_index"]], np.float32)
        actor, source = make_actor(event, anchor, policy="P0")
        click = {"event_frame": event["frame"], "human_anchor": anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
        bridge = SafeMOTIdentityBridge(click, actor, policy=GatePolicy(family="shadow"), frames=len(frames))
        bridge.configure_fps(event["fps"])
        original_seal = read_json(OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (event["episode_uid"] + ".json"))
        artifact = next(a for a in original_seal["artifacts"] if a["kind"] == "trace")
        assert sha256(artifact["path"]) == artifact["sha256"]
        original = read_zstd_jsonl(Path(artifact["path"]))
        for payload, rows in frames:
            f = int(payload["frame"])
            if f in episode["frames"]:
                assert f > event["frame"]
                seal = collect_event(bridge, frames, f, event["episode_uid"], source_freeze, index_sha, sha256(plan_path))
                total += 1
                print(json.dumps({"actual_CF_event_complete": event["episode_uid"], "frame": f, "arms": len(seal["artifacts"]), "sequence_events_complete": total}), flush=True)
            actual = bridge.step(f, rows)
            for key in ("outputs", "target_uid", "state_before", "state_after"):
                assert actual[key] == original[f][key], (event["episode_uid"], f, key)
    write_json("events/counterfactual_sequences/" + sequence + ".json", {"stage": "N72R21R2", "status": "COMPLETE_ACTUAL_FRESH_C0_COUNTERFACTUAL_RUNTIME",
               "sequence": sequence, "actual_event_positions": total, "source_freeze": source_freeze, "plan_sha256": sha256(plan_path),
               "protocol_sha256": sha256(PROTOCOL), "candidate_index_sha256": index_sha, "actual_forbidden_GT_file_guard": True,
               "source_C0_shadow_states_outputs_AA_after_all_future_branches": True, "scientific_success_not_claimed": True})
    append_log("M3_FRESH_C0_COUNTERFACTUAL_SEQUENCE_COMPLETE", sequence=sequence, actual_event_positions=total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "runtime"])
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.sequence)
    else:
        with runtime_file_guard():
            runtime(args.sequence)
