"""Actually rerun historical C0/FULL with exact inputs, then pinned nine metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from scripts.n72r21r2_common import ROOT, OUT, R1, HISTORY, ASSETS, TRAIN, sha256, read_json, write_json, storage, append_log
from scripts.n72r21r1_reproduce_v2 import checked_frames, make_actor, diagnostic_step
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge

CASES = [("CLICK_C0", None, {"family": "joint_baseline", "click": True})] + [("ACIB_FULL_SEED" + str(s), s, {"family": "T2", "policy": "FULL"}) for s in (72101, 72102, 72103)]
SEQUENCES = ["dancetrack0001", "dancetrack0002"]


def run_sequence(sequence):
    if sequence not in SEQUENCES:
        raise ValueError("Historical reproduction only")
    torch.set_num_threads(1)
    protocol = read_json(HISTORY / "outputs/N72R21/protocol/MOT_TRAIN_PILOT.json")
    init_path = HISTORY / "outputs/N72R21/mot_pilot/initialization" / (sequence + ".json")
    init = read_json(init_path)
    assert sha256(init["anchor_path"]) == init["anchor_sha256"] and not init["initialization_failure"]
    frames, index_sha = checked_frames(sequence)
    assert index_sha == init["candidate_index_sha256"]
    event = init["event"]
    anchor = np.array(np.load(init["anchor_path"], mmap_mode="r")[event["anchor_index"]], np.float32)
    click = {"event_frame": event["frame"], "human_anchor": anchor, "target_candidate_uid": init["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
    for case, seed, condition in CASES:
        seal_relative = "audit/reproduction_seals/" + case + "/" + sequence + ".json"
        if (OUT / seal_relative).exists():
            seal = read_json(OUT / seal_relative)
            assert seal["source_sha256"] == sha256(Path(__file__))
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            continue
        storage(256 << 20)
        old_seal_path = HISTORY / "outputs/N72R21/mot_pilot/runtime_seals" / case / (sequence + ".json")
        old_seal = read_json(old_seal_path)
        for path, digest in old_seal["code_sha256"].items():
            assert sha256(ROOT / path) == sha256(HISTORY / path) == digest
        old_trace_path = Path(next(a["path"] for a in old_seal["artifacts"] if a["kind"] == "trace"))
        old_trace = read_zstd_jsonl(old_trace_path)
        actor, model_source = make_actor(condition, seed, init, anchor, protocol)
        bridge = MOTIdentityBridge(click, actor, frames=len(frames))
        bridge.configure_fps(event["fps"])
        trace_path = ASSETS / "historical_reproduction/traces" / case / (sequence + ".jsonl.zst")
        trajectory_path = ASSETS / "historical_reproduction/trackers" / case / "data" / (sequence + ".txt")
        for path in (trace_path, trajectory_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError("Preserve partial attempt; no blind overwrite")
        started = time.monotonic()
        with trace_path.open("xb") as compressed, trajectory_path.open("x") as text:
            proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=compressed)
            try:
                for payload, rows in frames:
                    f = int(payload["frame"])
                    result = diagnostic_step(bridge, f, rows)
                    for key in ("outputs", "target_uid", "selected_action", "state_before", "state_after", "joint_identity_memory_write", "joint_memory_write_candidate_uid"):
                        assert result[key] == old_trace[f][key], (case, sequence, f, key)
                    output = result["outputs"]
                    assert len({o["candidate_uid"] for o in output}) == len({o["public_id"] for o in output}) == len(output) == len(rows)
                    text.write(trajectory_text([result]))
                    proc.stdin.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
                proc.stdin.close()
                assert proc.wait() == 0
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
        expected = next(a["sha256"] for a in old_seal["artifacts"] if a["kind"] == "trajectory")
        assert sha256(trajectory_path) == expected
        artifacts = [{"kind": kind, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for kind, path in [("trace", trace_path), ("trajectory", trajectory_path)]]
        write_json(seal_relative, {"stage": "N72R21R2", "status": "COMPLETE_ACTUAL_FULL_JOINT_REPRODUCTION", "case": case, "sequence": sequence,
                   "frames": len(frames), "seconds": time.monotonic() - started, "source_sha256": sha256(Path(__file__)),
                   "inherited_runtime_SHA": old_seal["code_sha256"], "candidate_index_sha256": index_sha, "old_seal_sha256": sha256(old_seal_path),
                   "initialization_sha256": sha256(init_path), "model_source": model_source, "artifacts": artifacts,
                   "all_states_outputs_and_trajectory_bytes_AA": True, "full_global_unique_ownership": True,
                   "GT_runtime": False, "one_click_only": True})
        append_log("M0_ACTUAL_OLD_REPLAY_SEALED", case=case, sequence=sequence, seconds=time.monotonic() - started)
        print(json.dumps({"reproduction_complete": case, "sequence": sequence, "frames": len(frames), "seconds": round(time.monotonic() - started, 2)}), flush=True)


def evaluate():
    for case, _, _ in CASES:
        for sequence in SEQUENCES:
            seal = read_json(OUT / "audit/reproduction_seals" / case / (sequence + ".json"))
            assert seal["source_sha256"] == sha256(Path(__file__))
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
    pinned = HISTORY / "third_party/MOTIP/TrackEval"
    commit = subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "12c8791b303e0a0b50f753af204249e622d0281a"
    names = [c[0] for c in CASES]
    root = ASSETS / "historical_reproduction"
    seqmap = root / "seqmap.txt"
    with seqmap.open("x") as handle:
        handle.write("name\n" + "\n".join(SEQUENCES) + "\n")
    eval_root = root / "trackeval_v1"
    eval_root.mkdir()
    command = _trackeval_command(root / "trackers", eval_root, names, seqmap, gt_split="train", gt_folder=TRAIN)
    command[2] = str(pinned / "scripts/run_mot_challenge.py")
    log_path = eval_root / "trackeval.log"
    with log_path.open("x") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    invocation = {"command": command, "returncode": result.returncode, "log_path": str(log_path), "log_sha256": sha256(log_path), "pinned_TrackEval_commit": commit}
    write_json("audit/BASELINE_TRACKEVAL_INVOCATION.json", invocation)
    if result.returncode:
        raise RuntimeError("Actual TrackEval failed; attempt retained")
    actual = {case: trackeval_summary(parse_trackeval(eval_root, case, SEQUENCES)) for case in names}
    old = read_json(HISTORY / "outputs/N72R21/mot_pilot/RESULT.json")["actual_TrackEval_metrics"]
    keys = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN")
    for case in names:
        for key in keys:
            assert actual[case][key] is not None and abs(actual[case][key] - old[case][key]) < 1e-12, (case, key)
            for sequence in SEQUENCES:
                assert abs(actual[case]["per_sequence"][sequence][key] - old[case]["per_sequence"][sequence][key]) < 1e-12
    write_json("audit/BASELINE_REPRODUCTION.json", {"stage": "N72R21R2", "status": "PASS_ACTUAL_REPRODUCTION_NOT_SCIENTIFIC_PASS", "actual_replays": 8,
               "conditions": names, "sequences": SEQUENCES, "actual_nine_metrics": actual, "invocation": invocation,
               "all_states_outputs_trajectory_bytes_and_metrics_equal": True, "new_fits": 0, "R1_scientific_failure_unchanged": True})
    append_log("M0_ACTUAL_PINNED_NINE_METRIC_REPRODUCTION_COMPLETE", actual_replays=8, metrics_AA=True)
    print(json.dumps({"status": "PASS_8_ACTUAL_REPLAYS_AND_9_METRICS_AA", "C0": actual["CLICK_C0"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["replay", "evaluate"])
    parser.add_argument("--sequence", choices=SEQUENCES)
    args = parser.parse_args()
    evaluate() if args.action == "evaluate" else run_sequence(args.sequence)
