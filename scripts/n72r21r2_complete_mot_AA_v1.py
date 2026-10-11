"""All24 raw-click C0/SHADOW complete-MOT-state supplemental replay audit.

No old source/result rewrite, fitting, scientific point, encoder run, GT parse
or future decision input. One CPU worker, same59 valid original human clicks;
failed initializations remain. Full tracker state excludes no instance fields.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, storage, preregistration, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_joint_state_curriculum import writer
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.complete_mot_state import canonical, assert_complete_mot_AA, complete_mot_snapshot
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint

PREFIX = "mot/complete_AA_v1"
PROTOCOL = OUT / "mot/COMPLETE_MOT_AA_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_complete_mot_AA_v1.py", "sam3_intermot/evaluation/complete_mot_state.py",
    "scripts/n72r21r2_baseline.py", "scripts/n72r21r2_joint_state_curriculum.py",
    "sam3_intermot/one_click/mot_bridge.py", "sam3_intermot/one_click/safe_mot_bridge.py",
    "sam3_intermot/one_click/joint_intervention_primitives.py", "sam3_intermot/one_click/runtime_file_guard.py",
    "sam3_intermot/one_click/causal_state_fingerprint.py", "sam3_intermot/association/opportunity_tracker.py",
    "sam3_intermot/association/causal_identity_tracker.py", "sam3_intermot/association/identity_state.py",
    "sam3_intermot/association/state_manager.py", "sam3_intermot/association/appearance_memory.py",
    "sam3_intermot/association/identity_authority.py", "sam3_intermot/association/opportunity_memory.py",
    "sam3_intermot/association/online_associator.py", "sam3_intermot/association/opportunity_solver.py",
    "sam3_intermot/one_click/acib_trusted_runtime.py", "sam3_intermot/one_click/acib_memory.py",
    "sam3_intermot/one_click/intervention_gate.py", "sam3_intermot/one_click/intervention_features.py",
    "scripts/n72r21_t2_replay.py")


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Preserve complete-AA protocol")
    p = preregistration(); storage(128 << 20)
    sequences = p["split"]["fit"] + p["split"]["inner"]
    refs = {}
    for sequence in sequences:
        for relative in ("data/initialization/", "data/candidate_integrity/", "mot/baseline_results/"):
            path = OUT / (relative + sequence + ".json"); refs[str(path)] = sha256(path)
    lineage_path = OUT / "audit/CHECKPOINT_LINEAGE.json"; lineage = read_json(lineage_path)
    identity = lineage["common_frozen_identity_model"]
    assert sha256(identity["path"]) == identity["sha256"]
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "sequences": sequences,
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "original_input_refs_sha256": refs,
        "identity_checkpoint": {"path": identity["path"], "sha256": identity["sha256"]},
        "state_scope": "Every tracker instance attribute, all IdentityState attributes, configs/event, dtype/shape/exact tensor bytes; reject unsupported opaque values. Identity actor and authority controller differ intentionally, and are NOT claimed equal to actor-free C0.",
        "legacy_subset_scope": "Existing full_tracker_fingerprint omits tracker pending/native streak/recent observation/trust fields. Old results and original fingerprints remain; do not retroactively call those complete-attribute AA.",
        "replay": "Original raw sealed click vector for both C0 and SHADOW, original actual candidates, frozen identity model/P0 shadow. Require complete-MOT-state equality before/after every original frame, semantic results equal sealed original traces, exact streamed trajectory SHA equal original C0/shadow.",
        "GT_content_parse_allowed": False, "new_encoder_inference_optimizer_policy_threshold_seed_choices": 0,
        "failed_initializations_replaced": False, "CPU_workers": 1, "max_seconds_per_episode": 1800,
        "bounded_supplemental_assets_GiB": .125, "storage_floor_GiB": 60,
        "CONFIRM_VAL_TEST_SOT_authorized": False, "scientific_success": None, "next_stage_authorized": False})
    print({"frozen_complete_MOT_AA": sha256(PROTOCOL), "sequences": len(sequences)}, flush=True)


def frozen():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL and p["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert sha256(p["identity_checkpoint"]["path"]) == p["identity_checkpoint"]["sha256"]
    return p


def pair(event, anchor, frames):
    actor, _ = make_actor(event, anchor)
    click = {"event_frame": event["frame"], "human_anchor": np.array(anchor, copy=True),
             "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
    left = MOTIdentityBridge(click, frames=frames)
    right = SafeMOTIdentityBridge(deepcopy(click), actor, policy=GatePolicy(family="shadow"), frames=frames)
    assert not np.shares_memory(left.tracker.event["human_anchor"], right.tracker.event["human_anchor"])
    for current in (left, right): current.configure_fps(event["fps"])
    assert_complete_mot_AA(left, right)
    return left, right


def sequence(sequence, *, smoke=False):
    p = frozen(); assert sequence in p["sequences"]
    torch.set_num_threads(1); storage(16 << 20)
    init_path = OUT / "data/initialization" / (sequence + ".json")
    assert sha256(init_path) == p["original_input_refs_sha256"][str(init_path)]
    init = read_json(init_path)
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    results = []
    for event in init["inputs"]:
        if event["initialization_failure"]: continue
        left, right = pair(event, np.array(anchors[event["anchor_index"]], np.float32), len(frames))
        old_seals, old = {}, {}
        for name in ("CLICK_C0", "ACIB_SHADOW_P0"):
            path = OUT / "mot/baseline_runtime" / name / (event["episode_uid"] + ".json")
            seal = read_json(path)
            assert seal["source_sha256"] == p["source_sha256"]["scripts/n72r21r2_baseline.py"]
            assert seal["event"] == event and seal["candidate_index_sha256"] == index_sha
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            old_seals[name] = {"path": str(path), "sha256": sha256(path)}
            old[name] = read_zstd_jsonl(artifact(seal, "trace"))
        mode = "smoke" if smoke else "full"
        path = ASSETS / "complete_mot_AA_v1" / mode / (event["episode_uid"] + ".jsonl.zst")
        began, actual_frames = time.monotonic(), 0
        trajectory = {name: hashlib.sha256() for name in old}
        attributes = complete_mot_snapshot(left)["all_tracker_attribute_names"]
        with writer(path) as emit:
            for payload, rows in (frames[:32] if smoke else frames):
                if time.monotonic() - began > p["max_seconds_per_episode"]:
                    raise RuntimeError("Preserve bounded supplemental AA partial")
                frame = int(payload["frame"])
                inputs_before = hashlib.sha256(json.dumps(canonical(rows), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                before = assert_complete_mot_AA(left, right)
                a, b = left.step(frame, rows), right.step(frame, rows)
                after = assert_complete_mot_AA(left, right)
                assert complete_mot_snapshot(left)["all_tracker_attribute_names"] == attributes
                inputs_after = hashlib.sha256(json.dumps(canonical(rows), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                assert inputs_before == inputs_after
                assert all(a[k] == b[k] for k in ("outputs", "target_uid", "state_before", "state_after"))
                for name, result in (("CLICK_C0", a), ("ACIB_SHADOW_P0", b)):
                    assert all(result[k] == old[name][frame][k] for k in ("outputs", "target_uid", "state_before", "state_after"))
                    assert result["extra_clicks"] == 0 and not result["runtime_gt_read"] and not result["runtime_future_gt_used"]
                    assert not result["joint_identity_memory_write"]
                    assert {o["candidate_uid"] for o in result["outputs"]} == {r["candidate_uid"] for r in rows}
                    assert len(result["outputs"]) == len(rows) == len({o["public_id"] for o in result["outputs"]})
                    trajectory[name].update(trajectory_text([result]).encode())
                assert not b.get("authority", {}).get("approved", False)
                emit({"frame": frame, "complete_MOT_before_AA_sha256": before, "complete_MOT_after_AA_sha256": after,
                      "legacy_subset_after_sha256": full_tracker_fingerprint(left), "all_tracker_attribute_names": attributes,
                      "candidate_count": len(rows), "target_uid": a["target_uid"], "extra_clicks": 0,
                      "unchanged_current_candidate_inputs_sha256": inputs_before,
                      "complete_output_ownership_unique": True, "actual_GT_file_guard": True})
                actual_frames += 1
        if not smoke:
            assert actual_frames == event["frames"]
            for name in old:
                seal = read_json(old_seals[name]["path"])
                assert trajectory[name].hexdigest() == next(a["sha256"] for a in seal["artifacts"] if a["kind"] == "trajectory")
        results.append({"episode_uid": event["episode_uid"], "frames": actual_frames,
            "all_tracker_attribute_names": attributes, "old_runtime_seals": old_seals,
            "streamed_trajectory_sha256": {name: digest.hexdigest() for name, digest in trajectory.items()},
            "artifact": {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size},
            "seconds": time.monotonic() - began})
        print({"actual_complete_MOT_AA": event["episode_uid"], "frames": actual_frames, "smoke_not_full_video": smoke}, flush=True)
    write_json(PREFIX + "/" + ("smoke" if smoke else "sequences") + "/" + sequence + ".json", {
        "stage": "N72R21R2", "goal": GOAL, "sequence": sequence, "protocol_sha256": sha256(PROTOCOL),
        "candidate_index_sha256": index_sha, "initialization_sha256": sha256(init_path), "episodes": results,
        "failed_initializations_retained": sum(bool(e["initialization_failure"]) for e in init["inputs"]),
        "all_complete_tracker_attributes_AA": True, "actual_GT_file_guard": True,
        "full_original_video": not smoke, "runtime_actor_controller_equality_NOT_claimed": True,
        "old_scientific_source_or_results_modified": False, "scientific_success": None, "next_stage_authorized": False})


def run():
    p = frozen(); marker = PREFIX + "/driver.json"
    if (OUT / marker).exists(): raise FileExistsError("Inspect scoped owner; no duplicate or implicit partial resume")
    state = {"stage": "N72R21R2", "goal": GOAL, "pid": os.getpid(), "status": "ACTIVE_ALL24_COMPLETE_MOT_AA",
        "protocol_sha256": sha256(PROTOCOL), "completed": [], "active": None, "failed_retained": [], "CPU_workers": 1}
    write_json(marker, state)
    for name in p["sequences"]:
        state["active"] = {"sequence": name, "pid": os.getpid()}; write_json(marker, state, mutable=True)
        try:
            with runtime_file_guard(): sequence(name)
        except Exception as error:
            state.update(status="ACTUAL_COMPLETE_ATTRIBUTE_AA_ATTEMPT_FAILED_PARTIAL_RETAINED", active=None)
            state["failed_retained"].append({"sequence": name, "error_type": type(error).__name__, "error": str(error)})
            write_json(marker, state, mutable=True); raise
        state["completed"].append(name); state["active"] = None; write_json(marker, state, mutable=True)
    state["status"] = "COMPLETE_ALL24_SUPPLEMENTAL_MOT_ATTRIBUTE_AA_NOT_SCIENTIFIC_PASS"
    write_json(marker, state, mutable=True); append_log("ACTUAL_ALL24_COMPLETE_MOT_AA_FINISHED", sequences=len(state["completed"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "smoke", "run")); parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "freeze": freeze()
    elif args.action == "run": run()
    else:
        with runtime_file_guard(): sequence(args.sequence, smoke=True)
