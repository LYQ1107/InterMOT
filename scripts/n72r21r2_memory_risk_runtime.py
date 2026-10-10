"""Actual own-bank risk writer, zero MOT authority, then offline full metrics."""
import argparse
from collections import Counter
from pathlib import Path
import time
import subprocess
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, read_json, write_json, sha256, storage, development_sequence, append_log
from scripts.n72r21r2_memory_risk_data import PROTOCOL, CODE
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_joint_state_curriculum import writer
from scripts.n72r21r2_zero_authority_memory import bank_drift, METRICS
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.zero_authority_memory import ZeroAuthorityMemoryBridge
from sam3_intermot.one_click.fresh_memory_risk import FreshMemoryRiskPredictor, FreshRiskCommittedMemory, INPUT_NAMES
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard


def cases(p):
    for family in p["families"]:
        for seed in p["seeds"]:
            model_uid = family + "__seed" + str(seed)
            path = OUT / "memory/risk_v1/fits" / (model_uid + ".json")
            fit = read_json(path)
            if fit["status"] != "COMPLETE_ACTUAL_SAME_BANK_CURRENT_WRITE_RISK_FIT":
                raise ValueError("No runtime writer from failed/no-diversity fit")
            assert fit["protocol_sha256"] == sha256(PROTOCOL) and sha256(fit["checkpoint_path"]) == fit["checkpoint_sha256"]
            for point in p["runtime_points"]:
                yield model_uid + "__" + point, point, path, fit


def runtime(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {x: sha256(ROOT / x) for x in CODE}
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    registered = list(cases(p))
    storage(256 << 20)
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        base_path = OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json")
        base = read_json(base_path)
        assert all(sha256(a["path"]) == a["sha256"] for a in base["artifacts"])
        baseline = read_zstd_jsonl(artifact(base, "trace"))
        frozen_path = OUT / "memory/M_A/runtime/FROZEN" / (event["episode_uid"] + ".json")
        frozen_seal = read_json(frozen_path)
        assert all(sha256(a["path"]) == a["sha256"] for a in frozen_seal["artifacts"])
        frozen = read_zstd_jsonl(artifact(frozen_seal, "trace"))
        for case, point, fit_path, fit in registered:
            relative = "memory/risk_v1/runtime/" + case + "/" + event["episode_uid"] + ".json"
            if (OUT / relative).exists():
                old = read_json(OUT / relative)
                assert old["protocol_sha256"] == sha256(PROTOCOL) and old["checkpoint_sha256"] == fit["checkpoint_sha256"]
                assert all(sha256(a["path"]) == a["sha256"] for a in old["artifacts"])
                continue
            predictor = FreshMemoryRiskPredictor(fit["checkpoint_path"])
            selection = predictor.selection if point == "SELECTED_CURRENT_RISK_ONLY" else {**p["fixed_diagnostic"], "temperature": predictor.selection["temperature"]}
            original, lineage = make_actor(event, np.array(anchors[event["anchor_index"]], np.float32))
            actor = FreshRiskCommittedMemory(original.model, original.anchor, original.token, predictor=predictor, selection=selection)
            actor.start_recording(original.recording, fps=original.fps, width=original.width, height=original.height,
                                  camera=original.camera, initial_frame=original.initial_frame, initial_box=original.last_box)
            click = {"event_frame": event["frame"], "human_anchor": actor.anchor,
                     "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
            bridge = ZeroAuthorityMemoryBridge(click, actor, frames=len(frames))
            bridge.configure_fps(event["fps"])
            name = case + "__click" + str(event["slot"])
            trace_path = ASSETS / "memory_risk_v1/traces" / case / (event["episode_uid"] + ".jsonl.zst")
            trajectory_path = ASSETS / "memory_risk_v1/trackers" / name / "data" / (sequence + ".txt")
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            counts, inputs, began = Counter(), [], time.monotonic()
            with writer(trace_path) as emit, trajectory_path.open("x") as trajectory:
                for payload, rows in frames:
                    if time.monotonic() - began > p["max_seconds_per_episode_case"]:
                        raise RuntimeError("Preserve bounded own-risk memory partial")
                    f = int(payload["frame"])
                    before = full_tracker_fingerprint(bridge)
                    bank_before = bridge.identity.snapshot()
                    result = bridge.step(f, rows)
                    after = full_tracker_fingerprint(bridge)
                    assert result["outputs"] == baseline[f]["outputs"] and result["target_uid"] == baseline[f]["target_uid"]
                    assert result["state_before"] == baseline[f]["state_before"] and result["state_after"] == baseline[f]["state_after"]
                    assert before == frozen[f]["full_tracker_state_before_sha256"] and after == frozen[f]["full_tracker_state_after_sha256"]
                    prediction = result["identity_decision"]
                    result.update(full_tracker_state_before_sha256=before, full_tracker_state_after_sha256=after,
                                  bank_before_current_score=bank_before, bank_after_current_commit=bridge.identity.snapshot(),
                                  bank_drift_cosine_distance_to_anchor=bank_drift(bridge.identity),
                                  identity_claim_UID_before_global_feedback=None if prediction is None else prediction["proposed_candidate_uid"],
                                  identity_rank1_candidate_UID=None if prediction is None else prediction["rank1_candidate_uid"],
                                  association_authority_zero=True, identity_score_before_current_write=True,
                                  model_uid=case.split("__SELECTED", 1)[0].split("__FIXED", 1)[0], current_write_risk_point=point)
                    diagnostic = result.get("committed_memory_diagnostic", {})
                    if "writer_input_vector" in diagnostic:
                        inputs.append(diagnostic["writer_input_vector"])
                        counts["actual_nonNONE_current_writer_queries"] += 1
                    counts["actual_accepted_writes"] += bool(result["joint_identity_memory_write"])
                    counts["actual_original_frames"] += 1
                    emit(result)
                    trajectory.write(trajectory_text([result]))
            assert sha256(trajectory_path) == next(a["sha256"] for a in base["artifacts"] if a["kind"] == "trajectory")
            z = (np.asarray(inputs, np.float32) - predictor.mean) / predictor.scale if inputs else np.empty((0, len(INPUT_NAMES)))
            distribution = {"actual_writer_input_rows": len(inputs), "FIT_normalized_abs_z_above3_per_feature": (np.mean(np.abs(z) > 3., axis=0).tolist() if len(inputs) else None),
                            "own_input_quantiles_per_feature": (np.quantile(inputs, [.1, .5, .9], axis=0).tolist() if inputs else None),
                            "input_names": list(INPUT_NAMES), "P0_only_training_P1_deployment_NOT_hidden": True}
            write_json(relative, {"stage": "N72R21R2", "sequence": sequence, "event": event, "case": case, "point": point,
                "tracker_name": name, "counts": dict(counts), "selection": selection, "seconds": time.monotonic() - began,
                "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"], "candidate_index_sha256": index_sha,
                "fit_receipt_sha256": sha256(fit_path), "checkpoint_sha256": fit["checkpoint_sha256"], "identity_checkpoint_sha256": lineage["sha256"],
                "all_C0_outputs_semantic_states_trajectory_bytes_AA": True, "all_full_tensor_tracker_states_AA_vs_same_M_A_FROZEN": True,
                "original_C0_full_tensor_reference_not_invented": True, "own_bank_distribution_audit": distribution,
                "current_write_correctness_NOT_future_association_safety": True, "actual_GT_file_guard": True,
                "artifacts": [{"kind": k, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for k, path in (("trace", trace_path), ("trajectory", trajectory_path))]})
            print({"actual_own_memory_risk": sequence, "episode": event["episode_uid"], "case": case, "counts": dict(counts)}, flush=True)
    write_json("memory/risk_v1/runtime_sequences/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"],
        "all6_heads_all_seeds_both_points": [r[0] for r in registered], "association_authority_zero": True,
        "all_initialization_failures_retained_without_replacement": True, "scientific_success": None})


def evaluate(sequence):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome
    development_sequence(sequence)
    p = read_json(PROTOCOL)
    source_path = OUT / "memory/risk_v1/runtime_sequences" / (sequence + ".json")
    source = read_json(source_path)
    assert source["protocol_sha256"] == sha256(PROTOCOL) and source["source_sha256"] == {x: sha256(ROOT / x) for x in CODE}
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    seals = []
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        for case in source["all6_heads_all_seeds_both_points"]:
            path = OUT / "memory/risk_v1/runtime" / case / (event["episode_uid"] + ".json")
            seal = read_json(path)
            assert seal["protocol_sha256"] == sha256(PROTOCOL) and seal["source_sha256"] == p["source_sha256"]
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            seals.append(seal)
    # The entire registered video/click/head/point runtime is verified above.
    truth = read_json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    frames, _ = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {payload["frame"]: strict_candidate_matching(rows, gt.get(payload["frame"], [])) for payload, rows in frames}
    results = {}
    for seal in seals:
        event, counts, writes, max_drift = seal["event"], Counter(), [], 0.
        target = targets[event["episode_uid"]]
        for row in read_zstd_jsonl(artifact(seal, "trace")):
            f = row["frame"]
            if f <= event["frame"]:
                continue
            mapping = matched[f]
            positive = target in mapping.values()
            visible = any(r["identity"] == target for r in gt.get(f, []))
            competitive = positive and any(identity is not None and identity != target for identity in mapping.values())
            committed = candidate_identity_outcome(row["target_uid"], mapping, target)
            claim = candidate_identity_outcome(row["identity_claim_UID_before_global_feedback"], mapping, target)
            rank = candidate_identity_outcome(row["identity_rank1_candidate_UID"], mapping, target)
            counts["positive_available_frames"] += positive
            counts["physically_visible_frames"] += visible
            counts["competitive_available_frames"] += competitive
            counts["rank1_correct_when_positive_available"] += positive and rank == "TARGET"
            counts["competitive_rank1_correct"] += competitive and rank == "TARGET"
            counts["identity_correct_claim_available"] += positive and claim == "TARGET"
            counts["identity_correct_claim_visible"] += visible and claim == "TARGET"
            counts["identity_claim_" + claim] += 1
            counts["C0_committed_" + committed] += 1
            counts["correct_C0_committed_observations"] += committed == "TARGET"
            counts["false_presence_no_positive_candidate"] += not positive and claim != "NONE"
            accepted = bool(row["joint_identity_memory_write"])
            counts["accepted_writes"] += accepted
            if accepted:
                assert row["joint_memory_write_candidate_uid"] == row["target_uid"] and row["target_uid"] is not None
                counts["write_" + committed] += 1
                writes.append({"frame": f, "candidate_uid": row["target_uid"], "outcome": committed})
            max_drift = max(max_drift, row["bank_drift_cosine_distance_to_anchor"])
        denominator, correct = counts["accepted_writes"], counts["correct_C0_committed_observations"]
        results[seal["case"] + "/" + event["episode_uid"]] = {
            "raw_counts": dict(counts), "accepted_write_provenance": writes, "max_bank_drift": max_drift,
            "wrong_or_UNKNOWN_write_rate": (counts["write_VERIFIED_OTHER"] + counts["write_UNKNOWN"]) / denominator if denominator else None,
            "correct_write_retention": counts["write_TARGET"] / correct if correct else None,
            "target_rank1_available": counts["rank1_correct_when_positive_available"] / counts["positive_available_frames"] if counts["positive_available_frames"] else None,
            "target_rank1_competitive": counts["competitive_rank1_correct"] / counts["competitive_available_frames"] if counts["competitive_available_frames"] else None,
            "target_claim_recall_available": counts["identity_correct_claim_available"] / counts["positive_available_frames"] if counts["positive_available_frames"] else None,
            "zero_writes_NOT_safety_success": denominator == 0, "empirical_zero_errors_NOT_population_two_percent_proof": True,
            "unqualified_fixed_diagnostic_NOT_formal_selected_memory": seal["point"] == "FIXED_P50_UNQUALIFIED_DIAGNOSTIC"}
    evaluation = ASSETS / "memory_risk_v1/trackeval" / sequence
    evaluation.mkdir(parents=True, exist_ok=False)
    seqmap = evaluation / "seqmap.txt"
    with seqmap.open("x") as handle:
        handle.write("name\n" + sequence + "\n")
    metrics, receipt = {}, {"status": "NOT_RUN_NO_VALID_INITIALIZATION"}
    if seals:
        pinned = HISTORY / "third_party/MOTIP/TrackEval"
        assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
        names = sorted(s["tracker_name"] for s in seals)
        command = _trackeval_command(ASSETS / "memory_risk_v1/trackers", evaluation, names, seqmap, gt_split="train", gt_folder=TRAIN)
        command[2] = str(pinned / "scripts/run_mot_challenge.py")
        log = evaluation / "trackeval.log"
        with log.open("x") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        receipt = {"command": command, "returncode": process.returncode, "log_path": str(log), "log_sha256": sha256(log)}
        if process.returncode:
            raise RuntimeError("Retain failed actual risk-bank full TrackEval")
        metrics = {n: trackeval_summary(parse_trackeval(evaluation, n, [sequence])) for n in names}
        base = read_json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
        assert all(all(value[k] == base["CLICK_C0__click" + n.rsplit("__click", 1)[1]][k] for k in METRICS) for n, value in metrics.items())
    write_json("memory/risk_v1/results/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"],
        "whole_video_runtime_receipt_sha256": sha256(source_path), "identity_and_write_results": results,
        "all_nine_actual_metrics": metrics, "all_nine_equal_C0": bool(seals), "receipt": receipt,
        "source_to_own_bank_distribution_audit_recorded": True, "association_authority_zero": True,
        "constant_HOTA_NOT_memory_scientific_success": True, "G4_population_risk_NOT_inferred_from_few_zero_errors": True,
        "scientific_success": None, "next_stage_authorized": False})
    append_log("M9_ACTUAL_OWN_RISK_BANK_FULL_VIDEO_OFFLINE_COMPLETE", sequence=sequence, trackers=len(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["runtime", "evaluate"])
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    if args.action == "runtime":
        with runtime_file_guard():
            runtime(args.sequence)
    else:
        evaluate(args.sequence)
