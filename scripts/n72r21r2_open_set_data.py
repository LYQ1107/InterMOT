"""Fresh current-UID/NONE/UNKNOWN data, with a separate post-seal GT action.

The C0 collector is not an intervention policy. Each axis is captured before
the current commit; candidate queries cannot advance any original state.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, read_json, write_json, sha256, storage, development_sequence, preregistration, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal
from sam3_intermot.one_click.open_set_verifier import candidate_axis
from sam3_intermot.one_click.causal_state_fingerprint import fingerprint, full_tracker_fingerprint
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.one_click.intervention_features import FEATURE_NAMES, feature_vector

PROTOCOL = OUT / "availability/CURRENT_AXIS_PROTOCOL_V1.json"
CODE = ["scripts/n72r21r2_open_set_data.py", "sam3_intermot/one_click/open_set_verifier.py",
        "sam3_intermot/one_click/intervention_features.py", "sam3_intermot/one_click/joint_intervention_primitives.py",
        "sam3_intermot/one_click/safe_mot_bridge.py", "sam3_intermot/one_click/runtime_file_guard.py",
        "scripts/n72r21r2_baseline.py", "sam3_intermot/one_click/causal_state_fingerprint.py"]


def freeze():
    protocol = preregistration()
    write_json("availability/CURRENT_AXIS_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": protocol["goal"], "formal_goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "frozen_before_new_current_axis_inputs_or_fits": True, "split": protocol["split"],
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "source_sha256": {p: sha256(ROOT / p) for p in CODE},
        "sampling": "Every32 original post-click frames from t+1; every actual current UID plus explicit NONE; no label/effect sampling or replacement",
        "sample_period": 32, "state_source": "ACTUAL_FRESH_C0_SHADOW_P0_WITH_OWN_CAUSAL_PENDING_HISTORY",
        "runtime_future_GT_input": False, "association_authority": False, "memory_writes": False,
        "labels": "Separate action, only after all registered clicks in the entire video have sealed runtime. Physical visibility and positive-candidate availability are distinct labels. UNKNOWN is an independent uncertainty class, never a verified-other identity negative.",
        "prospective_CF_missed_repair_measurement": "Every32 collector alone does not identify all CF events; exact CF-current-axis diagnostic requires a separate registered source, never nearest-frame imputation.",
        "feature_names": list(FEATURE_NAMES), "max_seconds_per_episode": 1800, "reserve_GiB": 60,
        "fit_scope": "All16 fresh FIT videos only; INNER8 for selection/calibration only; no CONFIRM/VAL/TEST",
        "not_future_intervention_safety_supervision": True})


def current_axis_snapshot(bridge, frame, rows):
    before = full_tracker_fingerprint(bridge)
    previous_identity_frame = bridge.identity.last_frame
    pending = fingerprint(bridge.authority_pending)
    bank_size = len(bridge.identity.bank)
    prepared = prepare_proposal(bridge, frame, rows)
    axis = candidate_axis(bridge, frame, rows, prepared)
    assert before == full_tracker_fingerprint(bridge)
    assert previous_identity_frame == bridge.identity.last_frame and bank_size == len(bridge.identity.bank)
    assert pending == fingerprint(bridge.authority_pending)
    assert [r["candidate_uid"] for r in axis] == [str(r["candidate_uid"]) for r in rows] + [None]
    for row in axis:
        assert row["feature_vector"] == feature_vector(row["features"]).tolist()
        assert np.isfinite(row["feature_vector"]).all()
    return {"frame": frame, "axis": axis, "own_current_KEEP_UID": prepared["preview"]["target_uid"],
            "target_public_id": bridge.tracker.target_public, "candidate_count": len(rows),
            "tensor_inclusive_C0_prestate_sha256": before, "current_queries_do_not_commit_or_mutate_original_state": True}


def runtime(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    protocol = read_json(PROTOCOL)
    source = {p: sha256(ROOT / p) for p in CODE}
    assert source == protocol["source_sha256"]
    frames, index_sha = checked_frames(sequence)
    init_path = OUT / "data/initialization" / (sequence + ".json")
    init = read_json(init_path)
    assert init["candidate_index_sha256"] == index_sha and sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    totals = Counter()
    with runtime_file_guard():
        for event in init["inputs"]:
            if event["initialization_failure"]:
                totals["initialization_failures_not_replaced"] += 1
                continue
            relative = "availability/current_axis_v1/runtime/" + event["episode_uid"] + ".json"
            if (OUT / relative).exists():
                old = read_json(OUT / relative)
                assert old["source_sha256"] == source and old["protocol_sha256"] == sha256(PROTOCOL)
                assert all(sha256(a["path"]) == a["sha256"] for a in old["artifacts"])
                totals.update(old["counts"])
                continue
            storage(64 << 20)
            anchor = np.array(anchors[event["anchor_index"]], np.float32)
            actor, identity = make_actor(event, anchor)
            click = {"event_frame": event["frame"], "human_anchor": anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
            bridge = SafeMOTIdentityBridge(click, actor, policy=GatePolicy(family="shadow"), frames=len(frames))
            bridge.configure_fps(event["fps"])
            reference_path = OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (event["episode_uid"] + ".json")
            reference = read_json(reference_path)
            assert all(sha256(a["path"]) == a["sha256"] for a in reference["artifacts"])
            old = read_zstd_jsonl(artifact(reference, "trace"))
            destination = ASSETS / "open_set_current_axis_v1/runtime" / (event["episode_uid"] + ".jsonl.zst")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("Preserve partial fresh current-axis runtime")
            began = time.monotonic()
            counts = Counter()
            with destination.open("xb") as stream:
                proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=stream)
                try:
                    for payload, rows in frames:
                        if time.monotonic() - began > protocol["max_seconds_per_episode"]:
                            raise RuntimeError("Current-axis episode cap: retain partial")
                        f = int(payload["frame"])
                        sample = current_axis_snapshot(bridge, f, rows) if f > event["frame"] and (f - event["frame"] - 1) % protocol["sample_period"] == 0 else None
                        actual = bridge.step(f, rows)
                        for field in ("outputs", "target_uid", "state_before", "state_after", "selected_action"):
                            assert actual[field] == old[f][field], (event["episode_uid"], f, field)
                        assert not actual["joint_identity_memory_write"] and not bridge.identity.bank
                        if sample is not None:
                            assert sample["own_current_KEEP_UID"] == actual["target_uid"]
                            sample.update(sequence=sequence, episode_uid=event["episode_uid"], role=event["role"],
                                          source_history="ACTUAL_FRESH_C0_SHADOW_P0", actual_committed_UID=actual["target_uid"],
                                          runtime_features_not_GT_or_future=True)
                            proc.stdin.write((json.dumps(sample, sort_keys=True, allow_nan=False) + "\n").encode())
                            counts["sampled_frame_groups"] += 1
                            counts["candidate_plus_NONE_rows"] += len(sample["axis"])
                    proc.stdin.close()
                    assert proc.wait() == 0
                finally:
                    if proc.poll() is None:
                        proc.terminate(); proc.wait()
            totals.update(counts)
            write_json(relative, {"stage": "N72R21R2", "status": "COMPLETE_ACTUAL_FRESH_ALL_CURRENT_CANDIDATES_AND_NONE",
                       "sequence": sequence, "event": event, "source_sha256": source, "protocol_sha256": sha256(PROTOCOL),
                       "candidate_index_sha256": index_sha, "initialization_sha256": sha256(init_path), "identity_source": identity,
                       "reference_C0_shadow_seal_sha256": sha256(reference_path), "all_original_frames": len(frames), "counts": dict(counts),
                       "actual_GT_file_guard": True, "full_C0_outputs_and_semantic_states_AA": True,
                       "sample_queries_preserve_full_tensor_tracker_state": True, "seconds": time.monotonic() - began,
                       "artifacts": [{"kind": "current_axis", "path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size}]})
            print({"fresh_current_axis": event["episode_uid"], "counts": dict(counts)}, flush=True)
    write_json("availability/current_axis_v1/runtime_sequences/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ALL_REGISTERED_CLICKS_CURRENT_AXIS_RUNTIME",
        "source_sha256": source, "protocol_sha256": sha256(PROTOCOL), "counts": dict(totals), "no_runtime_GT": True})


def label_uid(uid, matched, target, available):
    if uid is None:
        return "CORRECT_NONE" if not available else "INCORRECT_NONE"
    identity = matched.get(uid)
    return "UNKNOWN_UNMATCHED" if identity is None else "TARGET" if identity == target else "VERIFIED_OTHER"


def label(sequence):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    development_sequence(sequence)
    runtime_path = OUT / "availability/current_axis_v1/runtime_sequences" / (sequence + ".json")
    runtime_seal = read_json(runtime_path)
    assert runtime_seal["source_sha256"] == {p: sha256(ROOT / p) for p in CODE}
    assert runtime_seal["protocol_sha256"] == sha256(PROTOCOL)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    seals = []
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        path = OUT / "availability/current_axis_v1/runtime" / (event["episode_uid"] + ".json")
        seal = read_json(path)
        assert seal["source_sha256"] == runtime_seal["source_sha256"]
        assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
        seals.append((path, seal))
    # All video runtime artifacts verified above, before the first GT read.
    frames, index_sha = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    truth = read_json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    destination = ASSETS / "open_set_current_axis_v1/supervision" / (sequence + ".jsonl.zst")
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage(64 << 20)
    counts, taxonomy = Counter(), Counter()
    with destination.open("xb") as stream:
        proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=stream)
        try:
            for path, seal in seals:
                event = seal["event"]
                target = targets[event["episode_uid"]]
                data = read_zstd_jsonl(artifact(seal, "current_axis"))
                assert len(data) == seal["counts"]["sampled_frame_groups"]
                for record in data:
                    f = record["frame"]
                    rows = frames[f][1]
                    matched = strict_candidate_matching(rows, gt.get(f, []))
                    available = target in matched.values()
                    visible = any(r["identity"] == target for r in gt.get(f, []))
                    assert [a["candidate_uid"] for a in record["axis"]] == [str(r["candidate_uid"]) for r in rows] + [None]
                    axis = []
                    for proposal in record["axis"]:
                        outcome = label_uid(proposal["candidate_uid"], matched, target, available)
                        features = proposal["features"]
                        tags = []
                        if outcome == "VERIFIED_OTHER":
                            if features["anchor_cosine"] >= .7: tags.append("ANCHOR_LOOKALIKE_VERIFIED_OTHER")
                            if features["native_same"]: tags.append("NATIVE_CONTINUITY_VERIFIED_OTHER")
                        if features["competitor_owned"]: tags.append("OTHER_PUBLIC_ID_OWNED_CANDIDATE")
                        if outcome == "UNKNOWN_UNMATCHED": tags.append("UNKNOWN_NOT_VERIFIED_IDENTITY_NEGATIVE")
                        axis.append({**proposal, "current_outcome": outcome,
                                     "class": 0 if outcome in ("TARGET", "CORRECT_NONE") else 2 if outcome == "UNKNOWN_UNMATCHED" else 1,
                                     "verified_identity_negative": outcome == "VERIFIED_OTHER", "offline_taxonomy_not_features": tags})
                        counts[outcome] += 1
                        taxonomy.update(tags)
                    state = "TARGET_PRESENT_WITH_CANDIDATE" if available else "TARGET_PRESENT_WITHOUT_VALID_CANDIDATE" if visible else "TARGET_PHYSICALLY_UNAVAILABLE"
                    taxonomy[state] += 1
                    labeled = {**record, "axis": axis, "target_candidate_available_label": available, "target_visible_label": visible,
                               "availability_state_offline_only": state, "runtime_seal_sha256": sha256(path), "candidate_index_sha256": index_sha,
                               "UNKNOWN_not_verified_negative": True, "current_GT_labels_not_features": True}
                    proc.stdin.write((json.dumps(labeled, sort_keys=True, allow_nan=False) + "\n").encode())
                    counts["frame_groups"] += 1
            proc.stdin.close()
            assert proc.wait() == 0
        finally:
            if proc.poll() is None:
                proc.terminate(); proc.wait()
    write_json("availability/current_axis_v1/supervision/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_FRESH_CURRENT_AXIS_SUPERVISION",
        "source_sha256": runtime_seal["source_sha256"], "protocol_sha256": sha256(PROTOCOL), "runtime_sequence_sha256": sha256(runtime_path),
        "counts": dict(counts), "taxonomy": dict(taxonomy), "all_registered_video_runtime_sealed_before_GT": True,
        "GT_sha256": truth["GT_sha256"], "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size},
        "current_correctness_not_future_intervention_safety": True, "UNKNOWN_separate_class_not_verified_negative": True})
    append_log("M8_FRESH_CURRENT_AXIS_LABELS_COMPLETE", sequence=sequence, counts=dict(counts))
    print({"fresh_open_set_labels": sequence, "counts": dict(counts)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "runtime", "label"])
    parser.add_argument("--sequence")
    args = parser.parse_args()
    freeze() if args.action == "freeze" else {"runtime": runtime, "label": label}[args.action](args.sequence)
