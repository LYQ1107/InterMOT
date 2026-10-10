"""Own-policy one-shot causal onset diagnostics, independent offline labeling."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, read_json, write_json, sha256, development_sequence, storage, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_counterfactual import compact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.simple_event_intervention import SimpleEventBridge
from sam3_intermot.one_click.joint_intervention_primitives import step_keep
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.event_causal_labels import label_actual_branch

PROTOCOL = OUT / "simple/ONSET_AUDIT_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_simple_onset_audit.py", "sam3_intermot/one_click/simple_event_intervention.py",
        "sam3_intermot/one_click/joint_intervention_primitives.py", "sam3_intermot/one_click/causal_state_fingerprint.py")


def trace_seals(sequence):
    development_sequence(sequence)
    runtime = read_json(OUT / "simple/runtime_sequences" / (sequence + ".json"))
    assert all(sha256(ROOT / p) == h for p, h in runtime["source_freeze"].items())
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    seals = []
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        for case in read_json(OUT / "simple/SIMPLE_PROTOCOL_V1.json")["cases"][2:]:
            path = OUT / "simple/runtime" / case / (event["episode_uid"] + ".json")
            seal = read_json(path)
            assert seal["source_freeze"] == runtime["source_freeze"]
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            seals.append((event, path, seal))
    return init, seals


def compact_full(result, bridge, before):
    state = bridge.tracker.states[bridge.tracker.target_public]
    row = compact(result, bridge)
    row.update(full_tracker_state_before_sha256=before, full_tracker_state_after_sha256=full_tracker_fingerprint(bridge),
               target_prototype_sha256=__import__("hashlib").sha256(np.asarray(state.prototype).tobytes()).hexdigest(),
               target_motion_state={"last_box": state.last_box.tolist(), "velocity": state.velocity.tolist()})
    return row


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
        try:
            for row in rows:
                proc.stdin.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            proc.stdin.close()
            assert proc.wait() == 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()


def runtime(sequence):
    torch.set_num_threads(1)
    init, seals = trace_seals(sequence)
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    source = {p: sha256(ROOT / p) for p in CODE}
    frames = None
    entries = []
    for event, seal_path, seal in seals:
        trace_path = artifact(seal, "trace")
        observed = read_zstd_jsonl(trace_path)
        positions = [r["frame"] for r in observed if r.get("authority", {}).get("effective_assignment_change")]
        assert len(positions) == seal["actual_effective_direct_decisions_NOT_independent_events"]
        selected, next_allowed = [], -1
        for frame in positions:
            if frame >= next_allowed:
                selected.append(frame)
                next_allowed = frame + 101
        summary = {"case": seal["case"], "episode_uid": event["episode_uid"], "event": event,
                   "source_runtime_seal_sha256": sha256(seal_path), "trace_path": str(trace_path), "trace_sha256": sha256(trace_path),
                   "all_effective_decision_frames": positions, "chronological_nonoverlapping_candidate_onsets": selected,
                   "status": "ACTUAL_ONSET_BRANCHES" if positions else "NOT_RUN_NO_EFFECTIVE_ACTION",
                   "nonvacuity": bool(positions), "branches": []}
        if not positions:
            entries.append(summary)
            continue
        storage(512 << 20)
        if frames is None:
            frames, _ = checked_frames(sequence)
        actor, _ = make_actor(event, np.array(anchors[event["anchor_index"]], np.float32))
        click = {"event_frame": event["frame"], "human_anchor": actor.anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
        bridge = SimpleEventBridge(click, actor, case=seal["case"], frames=len(frames))
        bridge.configure_fps(event["fps"])
        for payload, rows in frames:
            frame = int(payload["frame"])
            assert full_tracker_fingerprint(bridge) == observed[frame]["full_tracker_state_before_sha256"]
            if frame in positions:
                before = full_tracker_fingerprint(bridge)
                arms = {}
                for name in ("OWN_KEEP", "ACTUAL_CURRENT_THEN_KEEP"):
                    arm = bridge.clone()
                    assert full_tracker_fingerprint(arm) == before
                    branch = []
                    for future_payload, future_rows in frames[frame:min(frame + 101, len(frames))]:
                        f = int(future_payload["frame"])
                        start = full_tracker_fingerprint(arm)
                        result = arm.step(f, future_rows) if f == frame and name == "ACTUAL_CURRENT_THEN_KEEP" else step_keep(arm, f, future_rows)
                        if f == frame and name == "ACTUAL_CURRENT_THEN_KEEP":
                            assert result["outputs"] == observed[f]["outputs"]
                            assert full_tracker_fingerprint(arm) == observed[f]["full_tracker_state_after_sha256"]
                        branch.append(compact_full(result, arm, start))
                    path = ASSETS / "simple_onsets_v1" / seal["case"] / event["episode_uid"] / ("frame" + str(frame)) / (name + ".jsonl.zst")
                    save_rows(path, branch)
                    arms[name] = {"path": str(path), "sha256": sha256(path), "frames": len(branch)}
                assert full_tracker_fingerprint(bridge) == before
                summary["branches"].append({"frame": frame, "prestate_sha256": before, "public_axis": sorted(bridge.tracker.states),
                                           "chronological_nonoverlapping_candidate_onset": frame in selected, "artifacts": arms})
            result = bridge.step(frame, rows)
            assert result["outputs"] == observed[frame]["outputs"]
            assert full_tracker_fingerprint(bridge) == observed[frame]["full_tracker_state_after_sha256"]
        summary["tensor_inclusive_own_prefix_and_action_AA"] = True
        entries.append(summary)
    write_json("simple/onset_runtime/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_OWN_POLICY_ONSET_DIAGNOSTIC_RUNTIME",
        "source_freeze": source, "protocol_sha256": sha256(PROTOCOL), "entries": entries,
        "actual_GT_file_guard": True, "zero_actions_not_scientific_PASS": True})
    print({"simple_own_state_onset_runtime": sequence, "effective_decisions": sum(len(r["all_effective_decision_frames"]) for r in entries)}, flush=True)


def label(sequence):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    from sam3_intermot.evaluation.safe_intervention_events import assignment_map
    development_sequence(sequence)
    path = OUT / "simple/onset_runtime" / (sequence + ".json")
    runtime_seal = read_json(path)
    assert runtime_seal["protocol_sha256"] == sha256(PROTOCOL)
    assert runtime_seal["source_freeze"] == {p: sha256(ROOT / p) for p in CODE}
    # Verify the entire sequence BEFORE offline future truth.
    assert all(sha256(a["path"]) == a["sha256"] for e in runtime_seal["entries"] for b in e["branches"] for a in b["artifacts"].values())
    frames, _ = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {p["frame"]: strict_candidate_matching(rows, gt.get(p["frame"], [])) for p, rows in frames}
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in read_json(OUT / "data/initialization_truth" / (sequence + ".json"))["labels"]}
    entries = []
    for item in runtime_seal["entries"]:
        target = targets[item["episode_uid"]]
        trace = read_zstd_jsonl(Path(item["trace_path"]))
        assert sha256(item["trace_path"]) == item["trace_sha256"]
        origins, cursor, summaries = {}, 0, []
        for branch in item["branches"]:
            start = branch["frame"]
            while cursor < start:
                for p, uid in assignment_map(trace[cursor]).items():
                    identity = matched[cursor][uid]
                    if identity is not None:
                        origins.setdefault(p, identity)
                cursor += 1
            active = {p: identity for p, identity in origins.items() if p in branch["public_axis"]}
            rows = {name: read_zstd_jsonl(Path(a["path"])) for name, a in branch["artifacts"].items()}
            visible = {f: any(a["identity"] == target for a in gt.get(f, [])) for f in range(start, min(start + 101, len(frames)))}
            labels = label_actual_branch(rows["ACTUAL_CURRENT_THEN_KEEP"], rows["OWN_KEEP"], matched, target, active, visible)
            summaries.append({"frame": start, "chronological_nonoverlapping_candidate_onset": branch["chronological_nonoverlapping_candidate_onset"],
                              "labels": labels, "branch_seals": branch["artifacts"], "adaptive_policy_future_NOT_used_as_one_shot_reward": True})
        entries.append({"case": item["case"], "episode_uid": item["episode_uid"], "status": item["status"],
                        "effective_decisions": len(item["all_effective_decision_frames"]), "one_shot_diagnostics": summaries,
                        "nonvacuity": item["nonvacuity"], "G1_PASS_not_inferred_from_one_shot_diagnostics": True})
    write_json("simple/onset_audit/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_OWN_PREFIX_ONSET_DIAGNOSTICS",
        "runtime_seal_sha256": sha256(path), "source_freeze": runtime_seal["source_freeze"], "entries": entries,
        "zero_effective_decisions": all(e["effective_decisions"] == 0 for e in entries),
        "nonvacuity_PASS": any(e["effective_decisions"] > 0 for e in entries),
        "full_policy_G1_qualified": False, "scientific_success": None})
    append_log("M4_SIMPLE_OWN_PREFIX_ONSET_DIAGNOSTIC_COMPLETE", sequence=sequence, effective_decisions=sum(e["effective_decisions"] for e in entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["runtime", "label"])
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    if args.action == "runtime":
        with runtime_file_guard():
            runtime(args.sequence)
    else:
        label(args.sequence)
