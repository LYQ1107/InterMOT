"""Sealed actual six-policy bank states, then separate offline write labels.

Do not substitute oracle-positive memory, P0-only supervision or proposed
identity crops for actual current C0 commits. Bank vectors are reconstructed
only from SHA-verified strictly earlier committed candidate references.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, read_json, write_json, sha256, storage, preregistration, development_sequence, append_log
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_joint_state_curriculum import writer
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.zero_authority_memory import FIXED_MEMORY_CASES
from sam3_intermot.one_click.acib_runtime import Evidence, unit
from sam3_intermot.one_click.fresh_memory_risk import write_input, INPUT_NAMES
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "memory/RISK_WRITE_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_memory_risk_data.py", "scripts/n72r21r2_train_memory_risk.py",
        "scripts/n72r21r2_memory_risk_runtime.py", "scripts/n72r21r2_memory_risk_driver.py",
        "sam3_intermot/one_click/fresh_memory_risk.py", "sam3_intermot/one_click/committed_identity_memory.py",
        "sam3_intermot/one_click/committed_memory_bridge.py", "sam3_intermot/one_click/zero_authority_memory.py",
        "sam3_intermot/one_click/intervention_features.py", "sam3_intermot/one_click/runtime_file_guard.py")


def freeze():
    p = preregistration()
    write_json("memory/RISK_WRITE_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": p["goal"], "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "split": p["split"],
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "frozen_before_new_write_inputs_labels_fits_or_rollouts": True,
        "source_sha256": {x: sha256(ROOT / x) for x in CODE}, "existing_memory_source_protocol_sha256": sha256(OUT / "memory/ZERO_AUTHORITY_PROTOCOL_V1.json"),
        "source_cases": list(FIXED_MEMORY_CASES), "input_names": list(INPUT_NAMES), "sample_period": 8,
        "source": "Every8 original post-click frames from t+1, actual non-NONE C0 commit under all6 already sealed bank policies; strictly past accepted bank references and actual own pending. No label/outcome sampling or positive-memory oracle.",
        "source_tracker_click": "Preserve existing M-A V1 actor-normalized tracker input for exact same-state tensor AA vs FROZEN. Original raw-click baseline semantic/output/bytes/all9 equality also required; not a new claim of full tensor equivalence to unavailable historical C0 snapshots.",
        "labels": "TARGET / verified OTHER / UNKNOWN current committed crop, after all registered video runtime is sealed. UNKNOWN is not a verified identity negative. This is current write correctness, not future association safety.",
        "fit_scope": "All16 FIT only, all24 source receipts required before optimizer; no available subset or failed-click replacement",
        "loss": "FIT inverse-class-mass weighted3-class current correctness CE; common6 teacher-state INNER epoch selection. Temperature uses class-unweighted but hierarchy-weighted INNER NLL.",
        "families": ["LOGISTIC", "MLP"], "seeds": [730101, 730102, 730103],
        "training": {"epochs_max": 60, "patience": 8, "batch_size": 256, "learning_rate": .001, "weight_decay": .0001,
                     "gradient_clip": 10., "max_seconds_per_fit": 180, "CPU_threads": 1},
        "weights": "Equal video -> click -> teacher bank policy -> sampled actual commit. Teacher policies/frames/replicas are NOT independent units.",
        "temperature_grid": [.5, 1., 2.], "probability_grid": [.5, .8, .9, .95, .98, .995], "unknown_max": .02,
        "INNER_selection": "Common6 source-state weighted NLL temperature, then max correct retention among pooled sampled-source points with current wrong+UNKNOWN<=2%, >=50 accepted in>=3 videos, retention>=60%. This is not actual own-policy G4 or a population2% proof. If none, CALIBRATION_ABSTAIN.",
        "runtime_points": ["SELECTED_CURRENT_RISK_ONLY", "FIXED_P50_UNQUALIFIED_DIAGNOSTIC"],
        "fixed_diagnostic": {"status": "UNQUALIFIED_FIXED_CURRENT_WRITE_DIAGNOSTIC", "probability_min": .5, "unknown_max": .5},
        "hard_multicue_probability_anchor_motion_native_delay_gates_added": False,
        "runtime": "Actual full original-frame C0 + own bank, all6 fitted heads/all seeds/both points. Actual writes only committed UID. Never change matcher/association/core bank; compare tensor states to FROZEN and all9 pinned metrics to real C0.",
        "memory_capacity": 8, "aggregation": "attention", "association_authority": False,
        "M_B_or_safely_updated_source_automatic": False, "all24_required_for_final_memory_conclusion": True,
        "max_seconds_per_episode_case": 1800, "CPU_workers": 1, "reserve_GiB": 60, "estimated_additional_assets_GiB_max": 3.})


def reconstruct_bank_input(anchor, snapshot, previous_row, lookup, frame, candidate, features):
    if hashlib.sha256(anchor.tobytes()).hexdigest() != snapshot["anchor_sha256"]:
        raise ValueError("Actual teacher normalized anchor mismatch")
    bank = []
    for item in snapshot["bank"]:
        if item["frame"] >= frame:
            raise ValueError("Bank reference cannot read current/future crop")
        row = lookup[item["frame"]][item["candidate_uid"]]
        embedding = unit(row["feature"])
        if hashlib.sha256(embedding.tobytes()).hexdigest() != item["embedding_sha256"]:
            raise ValueError("Sealed past bank embedding SHA mismatch")
        bank.append(Evidence(embedding, item["recording_id"], item["frame"], item["camera"], item["timestamp_seconds"],
                             item["confidence"], item["source"], item["quality"], item["anchor_consistency"], item["candidate_uid"]))
    pending = snapshot["pending_only_not_trusted"]
    if pending is not None:
        if previous_row is None or previous_row["frame"] != pending["frame"] or previous_row["frame"] >= frame:
            raise ValueError("Pending must reference the real preceding committed crop")
        row = lookup[previous_row["frame"]][previous_row["target_uid"]]
        embedding = unit(row["feature"])
        if hashlib.sha256(embedding.tobytes()).hexdigest() != pending["embedding_SHA"]:
            raise ValueError("Pending crop SHA mismatch")
        pending = dict(frame=pending["frame"], count=pending["count"], feature=embedding,
                       box=row["box_xyxy"], native=(row.get("native_scope"), row["native_tid"]))
    return write_input(features, anchor, bank, pending, frame, candidate)


def collect(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {x: sha256(ROOT / x) for x in CODE}
    source_path = OUT / "memory/M_A/runtime_sequences" / (sequence + ".json")
    source = read_json(source_path)
    assert source["protocol_sha256"] == p["existing_memory_source_protocol_sha256"]
    assert all(sha256(ROOT / x) == h for x, h in source["source_freeze"].items())
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    lookup = {int(payload["frame"]): {r["candidate_uid"]: r for r in rows} for payload, rows in frames}
    seals = []
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        for case in p["source_cases"]:
            path = OUT / "memory/M_A/runtime" / case / (event["episode_uid"] + ".json")
            seal = read_json(path)
            assert seal["source_freeze"] == source["source_freeze"] and seal["protocol_sha256"] == source["protocol_sha256"]
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            seals.append((event, case, path, seal))
    storage(64 << 20)
    destination = ASSETS / "memory_risk_v1/runtime_inputs" / (sequence + ".jsonl.zst")
    counts = Counter()
    with runtime_file_guard(), writer(destination) as emit:
        for event, case, path, seal in seals:
            # Original actor -> committed-memory actor normalizes exactly twice.
            anchor = unit(unit(np.array(anchors[event["anchor_index"]], np.float32)))
            trace = read_zstd_jsonl(artifact(seal, "trace"))
            for i, row in enumerate(trace):
                f = int(row["frame"])
                if f <= event["frame"] or (f - event["frame"] - 1) % p["sample_period"]:
                    continue
                if row["target_uid"] is None:
                    counts["sampled_committed_NONE_no_crop_not_write_example"] += 1
                    continue
                candidate = lookup[f][row["target_uid"]]
                vector, values = reconstruct_bank_input(anchor, row["bank_before_current_score"], trace[i - 1] if i else None,
                                                        lookup, f, candidate, row["committed_observation_features"])
                assert values["own_pending_confirmations"] == row["committed_memory_diagnostic"]["confirmations"]
                emit({"sequence": sequence, "role": event["role"], "episode_uid": event["episode_uid"], "frame": f,
                      "source_case": case, "candidate_uid": row["target_uid"], "runtime_input_vector": vector.tolist(),
                      "own_bank_features": values, "source_runtime_seal_sha256": sha256(path),
                      "source_trace_sha256": next(a["sha256"] for a in seal["artifacts"] if a["kind"] == "trace"),
                      "current_and_past_only": True, "UNKNOWN_or_GT_label_not_runtime_input": True})
                counts["actual_sampled_commit_rows"] += 1
                counts["source_case_" + case] += 1
                counts["bank_changed_input_rows"] += bool(row["bank_before_current_score"]["bank"])
    write_json("memory/risk_v1/runtime_inputs/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"],
        "source_memory_sequence_sha256": sha256(source_path), "counts": dict(counts), "candidate_index_sha256": index_sha,
        "all_valid_clicks_all6_teacher_policies_verified": True,
        "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size},
        "initialization_failures_not_replaced": sum(e["initialization_failure"] for e in init["inputs"]),
        "actual_GT_file_guard": True, "new_own_risk_rollouts_not_claimed": True})
    print({"fresh_memory_risk_inputs": sequence, "counts": dict(counts)}, flush=True)


def label(sequence):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    development_sequence(sequence)
    path = OUT / "memory/risk_v1/runtime_inputs" / (sequence + ".json")
    receipt = read_json(path)
    p = read_json(PROTOCOL)
    assert receipt["protocol_sha256"] == sha256(PROTOCOL) and receipt["source_sha256"] == {x: sha256(ROOT / x) for x in CODE}
    assert sha256(receipt["artifact"]["path"]) == receipt["artifact"]["sha256"]
    # All valid clicks/all6 whole-video teacher runtimes were verified before
    # their new runtime-input seal, above. Current truth is read only now.
    frames, _ = checked_frames(sequence)
    truth = read_json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {payload["frame"]: strict_candidate_matching(rows, gt.get(payload["frame"], [])) for payload, rows in frames}
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    counts = Counter()
    destination = ASSETS / "memory_risk_v1/supervision" / (sequence + ".jsonl.zst")
    with writer(destination) as emit:
        for row in read_zstd_jsonl(Path(receipt["artifact"]["path"])):
            identity = matched[row["frame"]][row["candidate_uid"]]
            target = targets[row["episode_uid"]]
            outcome = "UNKNOWN" if identity is None else "TARGET" if identity == target else "VERIFIED_OTHER"
            label = 0 if outcome == "TARGET" else 1 if outcome == "VERIFIED_OTHER" else 2
            emit({**row, "current_committed_crop_outcome": outcome, "class": label,
                  "verified_identity_negative": outcome == "VERIFIED_OTHER", "GT_label_offline_only": True})
            counts[outcome] += 1
    write_json("memory/risk_v1/supervision/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"],
        "runtime_input_receipt_sha256": sha256(path), "counts": dict(counts), "GT_sha256": truth["GT_sha256"],
        "artifact": {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size},
        "all_teacher_video_runtime_verified_before_current_labels": True,
        "current_write_correctness_NOT_future_association_safety": True})
    append_log("M9_FRESH_OWN_BANK_WRITE_RISK_CURRENT_LABELS", sequence=sequence, counts=dict(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "collect", "label"])
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "collect":
        with runtime_file_guard():
            collect(args.sequence)
    else:
        label(args.sequence)
