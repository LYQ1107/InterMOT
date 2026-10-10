"""Actual fixed simple full-global causal rollouts and separate offline metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, read_json, write_json, sha256, development_sequence, append_log, storage
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.simple_event_intervention import SimpleEventBridge
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint, serial
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "simple/SIMPLE_PROTOCOL_V1.json"
SOURCE = ["scripts/n72r21r2_simple.py", "scripts/n72r21r2_baseline.py", "scripts/n72r21r2_common.py",
          "sam3_intermot/one_click/simple_event_intervention.py", "sam3_intermot/one_click/joint_intervention_primitives.py",
          "sam3_intermot/one_click/intervention_features.py", "sam3_intermot/one_click/causal_state_fingerprint.py",
          "sam3_intermot/association/opportunity_tracker.py", "sam3_intermot/association/opportunity_scores.py",
          "sam3_intermot/association/opportunity_solver.py", "sam3_intermot/one_click/runtime_file_guard.py"]
METRICS = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN")


def runtime(sequence):
    development_sequence(sequence)
    torch.set_num_threads(1)
    storage(512 << 20)
    protocol = read_json(PROTOCOL)
    assert protocol["frozen"] and protocol["no_current_or_future_GT_in_runtime"]
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    source = {p: sha256(ROOT / p) for p in SOURCE}
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        for case in protocol["cases"][2:]:
            relative = "simple/runtime/" + case + "/" + event["episode_uid"] + ".json"
            if (OUT / relative).exists():
                saved = read_json(OUT / relative)
                assert saved["source_freeze"] == source and saved["protocol_sha256"] == sha256(PROTOCOL)
                assert all(sha256(a["path"]) == a["sha256"] for a in saved["artifacts"])
                continue
            actor, actor_source = make_actor(event, np.array(anchors[event["anchor_index"]], np.float32))
            click = {"event_frame": event["frame"], "human_anchor": actor.anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
            bridge = SimpleEventBridge(click, actor, case=case, frames=len(frames))
            bridge.configure_fps(event["fps"])
            tracker_name = case + "__click" + str(event["slot"])
            trace_path = ASSETS / "simple_v1/traces" / case / (event["episode_uid"] + ".jsonl.zst")
            trajectory_path = ASSETS / "simple_v1/trackers" / tracker_name / "data" / (sequence + ".txt")
            for path in (trace_path, trajectory_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError("Preserve unsealed policy attempt")
            direct, approved, native = 0, 0, 0
            started = time.monotonic()
            with trace_path.open("xb") as trace, trajectory_path.open("x") as trajectory:
                compressor = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=trace)
                try:
                    for payload, rows in frames:
                        f = int(payload["frame"])
                        before = full_tracker_fingerprint(bridge)
                        result = bridge.step(f, rows)
                        result.update(full_tracker_state_before_sha256=before, full_tracker_state_after_sha256=full_tracker_fingerprint(bridge),
                                      episode_uid=event["episode_uid"], current_candidate_index_sha256=index_sha)
                        authority = result["authority"]
                        direct += bool(authority.get("effective_assignment_change"))
                        approved += bool(authority.get("approved"))
                        native += bool(authority.get("selective_native_discount_active_this_frame_only"))
                        compressor.stdin.write((json.dumps(serial(result), sort_keys=True, allow_nan=False) + "\n").encode())
                        trajectory.write(trajectory_text([result]))
                    compressor.stdin.close()
                    assert compressor.wait() == 0
                finally:
                    if compressor.poll() is None:
                        compressor.terminate()
                        compressor.wait()
            write_json(relative, {"stage": "N72R21R2", "sequence": sequence, "case": case, "event": event, "tracker_name": tracker_name,
                       "status": "COMPLETE_ACTUAL_FIXED_FULL_GLOBAL_POLICY_RUNTIME", "source_freeze": source, "protocol_sha256": sha256(PROTOCOL),
                       "candidate_index_sha256": index_sha, "actor_source": actor_source, "all_original_frames": len(frames),
                       "actual_effective_direct_decisions_NOT_independent_events": direct, "approved_proposals": approved,
                       "selective_native_discount_frames": native, "seconds": time.monotonic() - started,
                       "actual_forbidden_GT_file_guard": True, "one_click_only": True, "own_policy_future_state_only": True,
                       "artifacts": [{"kind": kind, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for kind, path in (("trace", trace_path), ("trajectory", trajectory_path))]})
            print({"actual_simple_policy_complete": sequence, "episode": event["episode_uid"], "case": case, "changed_direct_decisions": direct}, flush=True)
    write_json("simple/runtime_sequences/" + sequence + ".json", {"stage": "N72R21R2", "status": "COMPLETE_ALL_REGISTERED_SIMPLE_RUNTIME",
               "sequence": sequence, "source_freeze": source, "protocol_sha256": sha256(PROTOCOL),
               "failed_clicks_retained_without_replacement": sum(e["initialization_failure"] for e in init["inputs"]),
               "GT_file_guard_actual": True, "scientific_success_not_claimed": True})
    append_log("M4_FIXED_SIMPLE_RUNTIME_COMPLETE", sequence=sequence)


def evaluate(sequence):
    from collections import Counter
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome
    development_sequence(sequence)
    status = read_json(OUT / "simple/runtime_sequences" / (sequence + ".json"))
    assert all(sha256(ROOT / p) == digest for p, digest in status["source_freeze"].items())
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    valid = [e for e in init["inputs"] if not e["initialization_failure"]]
    cases = read_json(PROTOCOL)["cases"][2:]
    seals = {}
    for event in valid:
        for case in cases:
            seal = read_json(OUT / "simple/runtime" / case / (event["episode_uid"] + ".json"))
            assert seal["source_freeze"] == status["source_freeze"] and seal["protocol_sha256"] == sha256(PROTOCOL)
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            seals[case, event["episode_uid"]] = seal
    frames, _ = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {p["frame"]: strict_candidate_matching(rows, gt.get(p["frame"], [])) for p, rows in frames}
    target_ids = {e["episode_uid"]: e["target_gt_identity"] for e in read_json(OUT / "data/initialization_truth" / (sequence + ".json"))["labels"]}
    components = {}
    for event in valid:
        target = target_ids[event["episode_uid"]]
        base_seal = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
        base = read_zstd_jsonl(artifact(base_seal, "trace"))
        for case in cases:
            trace = read_zstd_jsonl(artifact(seals[case, event["episode_uid"]], "trace"))
            counts = Counter()
            for a, b in zip(trace, base, strict=True):
                if a["frame"] <= event["frame"]:
                    continue
                truth = matched[a["frame"]]
                outcome = candidate_identity_outcome(a["target_uid"], truth, target)
                original = candidate_identity_outcome(b["target_uid"], truth, target)
                counts["target_correct_frames"] += outcome == "TARGET"
                counts["N01_frames_vs_full_C0"] += outcome == "TARGET" and original != "TARGET"
                counts["N10_frames_vs_full_C0"] += original == "TARGET" and outcome != "TARGET"
                counts["verified_OTHER_takeover_frames"] += outcome == "VERIFIED_OTHER"
                counts["UNKNOWN_frames"] += outcome == "UNKNOWN"
                counts["NONE_frames"] += outcome == "NONE"
                counts["effective_direct_decisions_NOT_independent_events"] += bool(a["authority"].get("effective_assignment_change"))
            components[case + "/" + event["episode_uid"]] = dict(counts)
    metrics, receipt = {}, {"status": "NOT_RUN_NO_VALID_INITIALIZATION"}
    if valid:
        evaluation = ASSETS / "simple_v1/trackeval" / sequence
        evaluation.mkdir(parents=True, exist_ok=False)
        seqmap = evaluation / "seqmap.txt"
        with seqmap.open("x") as handle:
            handle.write("name\n" + sequence + "\n")
        names = sorted({s["tracker_name"] for s in seals.values()})
        pinned = HISTORY / "third_party/MOTIP/TrackEval"
        assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
        command = _trackeval_command(ASSETS / "simple_v1/trackers", evaluation, names, seqmap, gt_split="train", gt_folder=TRAIN)
        command[2] = str(pinned / "scripts/run_mot_challenge.py")
        log = evaluation / "trackeval.log"
        with log.open("x") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        receipt = {"command": command, "returncode": process.returncode, "log_path": str(log), "log_sha256": sha256(log)}
        write_json("simple/invocations/" + sequence + ".json", receipt)
        if process.returncode:
            raise RuntimeError("Retain failed full-video simple TrackEval attempt")
        metrics = {n: trackeval_summary(parse_trackeval(evaluation, n, [sequence])) for n in names}
        assert all(all(m[k] is not None for k in METRICS) for m in metrics.values())
    baseline = read_json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
    deltas = {n: {k: m[k] - baseline["CLICK_C0__click" + n.rsplit("__click", 1)[1]][k] for k in METRICS} for n, m in metrics.items()}
    write_json("simple/results/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_SIMPLE_FULL_VIDEO_NINE_METRICS",
               "all_nine_metrics": metrics, "paired_deltas_vs_same_input_full_C0": deltas, "per_episode_raw_target_components": components,
               "receipt": receipt, "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(Path(__file__)),
               "causal_onset_non_target_harm_audit": "PENDING_OWN_KEEP_BRANCH_AUDIT_NOT_INFERRED_FROM_FRAME_COUNTS",
               "clicks_averaged_inside_video_for_statistics": True, "not_independent_confirmation": True, "scientific_success": None})
    append_log("M4_FIXED_SIMPLE_FULL_VIDEO_TRACK_EVAL_COMPLETE", sequence=sequence, actual_trackers=len(metrics))
    print({"simple_full_video_nine_metrics": sequence, "actual_trackers": len(metrics), "event_gate": "PENDING"}, flush=True)


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
