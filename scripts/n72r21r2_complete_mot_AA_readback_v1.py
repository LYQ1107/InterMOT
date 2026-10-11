"""Independent sealed A/A input/trace/trajectory readback, no tracker rollout.

Uses runtime_file_guard, never opens raw GT or future labels. Complete internal
states are asserted by the frozen paired replay, not reconstructed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, utcnow, storage
from scripts.n72r21r2_complete_mot_AA_v1 import frozen, PROTOCOL, PREFIX
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from sam3_intermot.evaluation.complete_mot_state import canonical
from sam3_intermot.evaluation.complete_mot_aa_evidence import sequence_scope, checked_frame, summarize_scopes

CODE = ("scripts/n72r21r2_complete_mot_AA_readback_v1.py",
        "sam3_intermot/evaluation/complete_mot_aa_evidence.py",
        "scripts/n72r20r1_build_base_score_tape.py", "scripts/n72r20r4_run_causal_tracker.py",
        "scripts/n72r21r2_baseline.py", "scripts/n72r21r2_events.py",
        "sam3_intermot/one_click/runtime_file_guard.py",
        "sam3_intermot/evaluation/complete_mot_state.py")


def run(suffix):
    if not suffix.replace("_", "").isalnum():
        raise ValueError("Immutable narrow readback suffix required")
    relative = "audit/COMPLETE_MOT_AA_RAW_READBACK_V1" + suffix + ".json"
    if (OUT / relative).exists():
        raise FileExistsError("Preserve earlier readbacks")
    torch.set_num_threads(1)
    storage(8 << 20)
    p = frozen()
    code = {name: sha256(ROOT / name) for name in CODE}
    refs = {str(PROTOCOL): sha256(PROTOCOL)}

    def hashed(path):
        path = Path(path)
        if not path.resolve().is_relative_to(OUT.resolve()) and not path.resolve().is_relative_to(ASSETS):
            raise ValueError("Evidence path escaped scoped stage output/assets")
        current = sha256(path)
        if str(path) in refs and refs[str(path)] != current:
            raise ValueError("Previously verified artifact changed during raw readback")
        refs[str(path)] = current
        return current

    def checked(path):
        hashed(path)
        return read_json(path)

    # Snapshot the sealed scope once. Concurrently completed sequences are for
    # another immutable readback, not a silently changing sample denominator.
    sealed = [sequence for sequence in p["sequences"]
              if (OUT / PREFIX / "sequences" / (sequence + ".json")).exists()]
    scopes = []
    for sequence in sealed:
        path = OUT / PREFIX / "sequences" / (sequence + ".json")
        receipt = checked(path)
        assert receipt["stage"] == "N72R21R2" and receipt["goal"] == GOAL
        init_path = OUT / "data/initialization" / (sequence + ".json")
        init = checked(init_path)
        assert refs[str(init_path)] == p["original_input_refs_sha256"][str(init_path)]
        assert hashed(init["anchor_path"]) == init["anchor_sha256"]
        integrity_path = OUT / "data/candidate_integrity" / (sequence + ".json")
        integrity = checked(integrity_path)
        assert refs[str(integrity_path)] == p["original_input_refs_sha256"][str(integrity_path)]
        index = checked(integrity["index_path"])
        assert refs[integrity["index_path"]] == integrity["index_sha256"]
        for kind in ("metadata", "embeddings"):
            assert hashed(index[kind]) == integrity[kind + "_sha256"]
        baseline_path = OUT / "mot/baseline_results" / (sequence + ".json")
        assert hashed(baseline_path) == p["original_input_refs_sha256"][str(baseline_path)]
        frames, index_sha = checked_frames(sequence)
        scope = sequence_scope(sequence, init, receipt, protocol_sha256=refs[str(PROTOCOL)],
                               initialization_sha256=refs[str(init_path)], candidate_index_sha256=index_sha)
        scope["sequence_receipt_sha256"] = refs[str(path)]
        scope["episodes_reverified"] = []
        valid = [event for event in init["inputs"] if not event["initialization_failure"]]
        for event, episode in zip(valid, receipt["episodes"], strict=True):
            assert len(frames) == event["frames"]
            evidence = episode["artifact"]
            assert hashed(evidence["path"]) == evidence["sha256"]
            assert Path(evidence["path"]).stat().st_size == evidence["bytes"]
            audit = read_zstd_jsonl(evidence["path"])
            old, trajectories = {}, {}
            assert set(episode["old_runtime_seals"]) == {"CLICK_C0", "ACIB_SHADOW_P0"}
            assert set(episode["streamed_trajectory_sha256"]) == set(episode["old_runtime_seals"])
            for name, ref in episode["old_runtime_seals"].items():
                assert Path(ref["path"]) == OUT / "mot/baseline_runtime" / name / (event["episode_uid"] + ".json")
                seal = checked(ref["path"])
                assert refs[ref["path"]] == ref["sha256"]
                assert seal["event"] == event and seal["initialization_sha256"] == refs[str(init_path)]
                assert seal["candidate_index_sha256"] == index_sha
                assert seal["source_sha256"] == p["source_sha256"]["scripts/n72r21r2_baseline.py"]
                for item in seal["artifacts"]:
                    assert hashed(item["path"]) == item["sha256"]
                old[name] = read_zstd_jsonl(artifact(seal, "trace"))
                trajectories[name] = next(item["sha256"] for item in seal["artifacts"] if item["kind"] == "trajectory")
                assert episode["streamed_trajectory_sha256"][name] == trajectories[name]
            assert len(audit) == len(old["CLICK_C0"]) == len(old["ACIB_SHADOW_P0"]) == len(frames)
            prior = None
            streams = {name: hashlib.sha256() for name in old}
            for frame, ((payload, candidates), row, c0, shadow) in enumerate(zip(
                    frames, audit, old["CLICK_C0"], old["ACIB_SHADOW_P0"], strict=True)):
                assert int(payload["frame"]) == frame
                assert c0["current_candidate_index_sha256"] == shadow["current_candidate_index_sha256"] == index_sha
                assert c0["episode_uid"] == shadow["episode_uid"] == event["episode_uid"]
                input_sha = hashlib.sha256(json.dumps(canonical(candidates), sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest()
                prior = checked_frame(row, frame=frame, candidates=candidates, input_sha256=input_sha,
                    attributes=episode["all_tracker_attribute_names"], previous_after=prior, c0=c0, shadow=shadow)
                for name, trace in (("CLICK_C0", c0), ("ACIB_SHADOW_P0", shadow)):
                    streams[name].update(trajectory_text([trace]).encode())
            assert {name: stream.hexdigest() for name, stream in streams.items()} == trajectories
            scope["episodes_reverified"].append({"episode_uid": event["episode_uid"], "frames": len(frames),
                "raw_current_inputs_and_global_output_axis_reverified": True,
                "recorded_complete_boundary_continuity_reverified": True,
                "original_trace_stream_recomputed_trajectory_sha256": trajectories})
        if valid:
            scope.update(status="VERIFIED_NONEMPTY_FULL_VIDEO_AA_RAW_EVIDENCE", complete_MOT_AA_proof=True)
        scopes.append(scope)
        print({"independent_raw_AA_readback": sequence, "status": scope["status"],
               "valid_clicks": len(valid), "frames_NOT_independent_units":
               scope["paired_frame_exposures_NOT_independent_sample_size"]}, flush=True)
    assert code == {name: sha256(ROOT / name) for name in CODE}
    frozen()
    assert all(sha256(path) == expected for path, expected in refs.items())
    report = {"stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "source_sha256": code,
        "frozen_replay_source_sha256": p["source_sha256"], "protocol_sha256": refs[str(PROTOCOL)],
        "refs_sha256": refs, "sequences_reverified": scopes, **summarize_scopes(p["sequences"], scopes),
        "GT_raw_content_or_future_labels_read": False, "actual_runtime_file_guard": True,
        "new_tracker_rollouts_training_thresholds_or_candidates": 0,
        "internal_state_equality_asserted_by_frozen_paired_runtime_NOT_independently_reconstructed_here": True,
        "actor_controller_equality_NOT_claimed": True, "application_goal_status": "ACTIVE"}
    write_json(relative, report)
    print({"receipt": relative, "sealed_sequences_reverified": len(scopes),
           "nonempty_videos": len(report["nonempty_full_video_AA_sequences"]),
           "all_failed_NOT_RUN": report["retained_all_failed_NOT_RUN_sequences"],
           "all24_dispositions_done": report["all_registered_sequence_dispositions_reverified"],
           "scientific_goal_complete": False}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    with runtime_file_guard():
        run(parser.parse_args().suffix)
