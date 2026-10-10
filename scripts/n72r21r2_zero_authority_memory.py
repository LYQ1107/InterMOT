"""Actual fresh M-A rollouts plus separate identity/write/full-MOT labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, read_json, write_json, sha256, storage, development_sequence, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.zero_authority_memory import fixed_memory_policy, memory_actor, ZeroAuthorityMemoryBridge
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint, serial
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "memory/ZERO_AUTHORITY_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_zero_authority_memory.py", "sam3_intermot/one_click/zero_authority_memory.py",
        "sam3_intermot/one_click/committed_identity_memory.py", "sam3_intermot/one_click/committed_memory_bridge.py",
        "sam3_intermot/one_click/joint_intervention_primitives.py", "sam3_intermot/one_click/causal_state_fingerprint.py",
        "sam3_intermot/one_click/acib_runtime.py", "sam3_intermot/one_click/acib_trusted_runtime.py",
        "sam3_intermot/one_click/runtime_file_guard.py")
METRICS = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN")


def bank_drift(actor):
    if not actor.bank:
        return 0.
    average = np.mean([e.embedding for e in actor.bank], axis=0)
    return float(1. - np.dot(average, actor.anchor) / max(1e-8, np.linalg.norm(average)))


def runtime(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    protocol = read_json(PROTOCOL)
    assert protocol["frozen"] and protocol["immutable_anchor"]
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    source = {p: sha256(ROOT / p) for p in CODE}
    storage(512 << 20)
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        base_seal = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
        assert all(sha256(a["path"]) == a["sha256"] for a in base_seal["artifacts"])
        baseline = read_zstd_jsonl(artifact(base_seal, "trace"))
        frozen = None
        for case in protocol["cases"]:
            relative = "memory/M_A/runtime/" + case + "/" + event["episode_uid"] + ".json"
            if (OUT / relative).exists():
                seal = read_json(OUT / relative)
                assert seal["source_freeze"] == source and seal["protocol_sha256"] == sha256(PROTOCOL)
                assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
                if case == "FROZEN":
                    frozen = read_zstd_jsonl(artifact(seal, "trace"))
                continue
            original, lineage = make_actor(event, np.array(anchors[event["anchor_index"]], np.float32))
            actor = memory_actor(original, case)
            click = {"event_frame": event["frame"], "human_anchor": actor.anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
            bridge = ZeroAuthorityMemoryBridge(click, actor, frames=len(frames))
            bridge.configure_fps(event["fps"])
            name = case + "__click" + str(event["slot"])
            base = ASSETS / "memory_zero_authority_v1"
            trace_path = base / "traces" / case / (event["episode_uid"] + ".jsonl.zst")
            trajectory_path = base / "trackers" / name / "data" / (sequence + ".txt")
            for path in (trace_path, trajectory_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError("Preserve unsealed changing-memory replay; version recovery")
            writes, rollback, began = 0, 0, time.monotonic()
            with trace_path.open("xb") as stream, trajectory_path.open("x") as trajectory:
                proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=stream)
                try:
                    for payload, rows in frames:
                        f = int(payload["frame"])
                        before = full_tracker_fingerprint(bridge)
                        before_bank = bridge.identity.snapshot()
                        result = bridge.step(f, rows)
                        assert result["outputs"] == baseline[f]["outputs"] and result["target_uid"] == baseline[f]["target_uid"]
                        assert result["state_before"] == baseline[f]["state_before"] and result["state_after"] == baseline[f]["state_after"]
                        after = full_tracker_fingerprint(bridge)
                        if frozen is not None:
                            assert before == frozen[f]["full_tracker_state_before_sha256"] and after == frozen[f]["full_tracker_state_after_sha256"]
                        prediction = result["identity_decision"]
                        joint = None if prediction is None else actor.model.last["joint_probabilities"][0].detach().cpu().tolist()
                        proposed = None if prediction is None else prediction["proposed_candidate_uid"]
                        rank1 = None if prediction is None else prediction["rank1_candidate_uid"]
                        result.update(full_tracker_state_before_sha256=before, full_tracker_state_after_sha256=after,
                                      bank_before_current_score=before_bank, bank_after_current_commit=bridge.identity.snapshot(),
                                      bank_drift_cosine_distance_to_anchor=bank_drift(bridge.identity),
                                      current_candidate_axis=[r["candidate_uid"] for r in rows],
                                      current_identity_candidate_plus_NONE_probabilities=joint,
                                      identity_claim_UID_before_global_feedback=proposed, identity_rank1_candidate_UID=rank1,
                                      identity_current_score_captured_before_write=True,
                                      association_authority_zero=True, original_axis_frame=f)
                        writes += bool(result["joint_identity_memory_write"])
                        rollback += bool(result.get("committed_memory_diagnostic", {}).get("rollback"))
                        proc.stdin.write((json.dumps(serial(result), sort_keys=True, allow_nan=False) + "\n").encode())
                        trajectory.write(trajectory_text([result]))
                    proc.stdin.close()
                    assert proc.wait() == 0
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
            assert sha256(trajectory_path) == next(a["sha256"] for a in base_seal["artifacts"] if a["kind"] == "trajectory")
            write_json(relative, {"stage": "N72R21R2", "sequence": sequence, "case": case, "event": event, "tracker_name": name,
                       "status": "COMPLETE_ACTUAL_CHANGING_MEMORY_ZERO_AUTHORITY_FULL_MOT",
                       "source_freeze": source, "protocol_sha256": sha256(PROTOCOL), "candidate_index_sha256": index_sha,
                       "identity_checkpoint_sha256": lineage["sha256"], "memory_policy": serial(fixed_memory_policy(case).__dict__),
                       "all_original_frames": len(frames), "actual_writes": writes, "actual_rollbacks": rollback,
                       "complete_C0_outputs_semantic_states_trajectory_bytes_AA": True,
                       "changing_memory_tensor_inclusive_tracker_AA_vs_FROZEN": case != "FROZEN",
                       "actual_GT_file_guard": True, "association_authority_zero": True, "seconds": time.monotonic() - began,
                       "artifacts": [{"kind": k, "path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for k, p in (("trace", trace_path), ("trajectory", trajectory_path))]})
            if case == "FROZEN":
                frozen = read_zstd_jsonl(trace_path)
            print({"M_A_full_C0_memory": sequence, "episode": event["episode_uid"], "case": case, "writes": writes, "rollbacks": rollback}, flush=True)
    write_json("memory/M_A/runtime_sequences/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ALL_FIXED_ZERO_AUTHORITY_MEMORY_RUNTIME",
        "source_freeze": source, "protocol_sha256": sha256(PROTOCOL), "association_authority_zero": True,
        "all_initialization_failures_retained_without_replacement": True, "scientific_success": None})


def evaluate(sequence):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome
    development_sequence(sequence)
    runtime_path = OUT / "memory/M_A/runtime_sequences" / (sequence + ".json")
    runtime_seal = read_json(runtime_path)
    assert runtime_seal["source_freeze"] == {p: sha256(ROOT / p) for p in CODE}
    assert runtime_seal["protocol_sha256"] == sha256(PROTOCOL)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    seals = []
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        for case in read_json(PROTOCOL)["cases"]:
            seal = read_json(OUT / "memory/M_A/runtime" / case / (event["episode_uid"] + ".json"))
            assert seal["source_freeze"] == runtime_seal["source_freeze"]
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            seals.append(seal)
    frames, _ = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {p["frame"]: strict_candidate_matching(rows, gt.get(p["frame"], [])) for p, rows in frames}
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in read_json(OUT / "data/initialization_truth" / (sequence + ".json"))["labels"]}
    results = {}
    for seal in seals:
        event = seal["event"]
        target = targets[event["episode_uid"]]
        trace = read_zstd_jsonl(artifact(seal, "trace"))
        counts, max_drift, write_origins = Counter(), 0., []
        for row in trace:
            f = row["frame"]
            if f <= event["frame"]:
                continue
            truth = matched[f]
            positive = target in truth.values()
            visible = any(a["identity"] == target for a in gt.get(f, []))
            competitive = positive and any(identity is not None and identity != target for identity in truth.values())
            claim = candidate_identity_outcome(row["identity_claim_UID_before_global_feedback"], truth, target)
            rank = candidate_identity_outcome(row["identity_rank1_candidate_UID"], truth, target)
            committed = candidate_identity_outcome(row["target_uid"], truth, target)
            counts["physically_visible_frames"] += visible
            counts["positive_available_frames"] += positive
            counts["competitive_available_frames"] += competitive
            counts["rank1_correct_when_positive_available"] += positive and rank == "TARGET"
            counts["competitive_rank1_correct"] += competitive and rank == "TARGET"
            counts["identity_correct_claim_visible"] += visible and claim == "TARGET"
            counts["identity_correct_claim_available"] += positive and claim == "TARGET"
            counts["identity_claim_" + claim] += 1
            counts["false_presence_no_positive_candidate"] += not positive and claim != "NONE"
            counts["physically_unavailable_frames"] += not visible
            counts["false_presence_physically_unavailable"] += not visible and claim != "NONE"
            counts["correct_C0_committed_observations"] += committed == "TARGET"
            counts["C0_committed_" + committed] += 1
            counts["accepted_writes"] += row["joint_identity_memory_write"]
            if row["joint_identity_memory_write"]:
                assert row["joint_memory_write_candidate_uid"] == row["target_uid"]
                counts["write_" + committed] += 1
                write_origins.append({"frame": f, "candidate_uid": row["target_uid"], "outcome": committed})
            counts["rollback_events"] += bool(row.get("committed_memory_diagnostic", {}).get("rollback"))
            max_drift = max(max_drift, row["bank_drift_cosine_distance_to_anchor"])
        assert counts["accepted_writes"] == seal["actual_writes"]
        accepted = counts["accepted_writes"]
        correct = counts["correct_C0_committed_observations"]
        results[seal["case"] + "/" + event["episode_uid"]] = {
            "raw_counts": dict(counts), "accepted_write_provenance": write_origins, "max_bank_drift": max_drift,
            "wrong_verified_OTHER_write_rate": counts["write_VERIFIED_OTHER"] / accepted if accepted else None,
            "UNKNOWN_unverified_write_rate": counts["write_UNKNOWN"] / accepted if accepted else None,
            "wrong_or_UNKNOWN_write_rate": (counts["write_VERIFIED_OTHER"] + counts["write_UNKNOWN"]) / accepted if accepted else None,
            "correct_write_retention": counts["write_TARGET"] / correct if correct else None,
            "target_rank1_available": counts["rank1_correct_when_positive_available"] / counts["positive_available_frames"] if counts["positive_available_frames"] else None,
            "target_rank1_competitive": counts["competitive_rank1_correct"] / counts["competitive_available_frames"] if counts["competitive_available_frames"] else None,
            "target_claim_recall_visible": counts["identity_correct_claim_visible"] / counts["physically_visible_frames"] if counts["physically_visible_frames"] else None,
            "target_claim_recall_available": counts["identity_correct_claim_available"] / counts["positive_available_frames"] if counts["positive_available_frames"] else None,
            "zero_writes_not_safety_PASS": accepted == 0, "few_zero_error_writes_not_population_two_percent_guarantee": True}
    evaluation = ASSETS / "memory_zero_authority_v1/trackeval" / sequence
    evaluation.mkdir(parents=True, exist_ok=False)
    seqmap = evaluation / "seqmap.txt"
    with seqmap.open("x") as handle:
        handle.write("name\n" + sequence + "\n")
    metrics, receipt = {}, {"status": "NOT_RUN_NO_VALID_INITIALIZATION"}
    if seals:
        names = sorted({s["tracker_name"] for s in seals})
        pinned = HISTORY / "third_party/MOTIP/TrackEval"
        assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
        command = _trackeval_command(ASSETS / "memory_zero_authority_v1/trackers", evaluation, names, seqmap, gt_split="train", gt_folder=TRAIN)
        command[2] = str(pinned / "scripts/run_mot_challenge.py")
        log = evaluation / "trackeval.log"
        with log.open("x") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        receipt = {"command": command, "returncode": process.returncode, "log_path": str(log), "log_sha256": sha256(log)}
        if process.returncode:
            raise RuntimeError("Preserve failed actual independent-memory full TrackEval")
        metrics = {n: trackeval_summary(parse_trackeval(evaluation, n, [sequence])) for n in names}
        base = read_json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
        assert all(all(v[k] == base["CLICK_C0__click" + n.rsplit("__click", 1)[1]][k] for k in METRICS) for n, v in metrics.items())
    write_json("memory/M_A/results/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_CHANGING_BANK_INDEPENDENT_OF_AUTHORITY",
        "runtime_sequence_sha256": sha256(runtime_path), "protocol_sha256": sha256(PROTOCOL), "source_freeze": runtime_seal["source_freeze"],
        "all_nine_actual_metrics": metrics, "all_nine_equal_C0": bool(seals), "receipt": receipt,
        "identity_and_write_results": results, "association_authority_zero": True,
        "actual_changing_bank_success_NOT_inferred_from_constant_MOT": True,
        "scientific_success": None, "next_stage_authorized": False})
    append_log("M9_ZERO_AUTHORITY_MEMORY_ACTUAL_SEQUENCE_COMPLETE", sequence=sequence, actual_trackers=len(metrics))


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
