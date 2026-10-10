"""Actual fixed-operating-point own-state FIT pilot; never main/confirm."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, read_json, write_json, sha256, storage, append_log, preregistration
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor, LearnedEventBridge
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint, serial
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "training/PILOT_POLICY_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_learned_pilot.py", "sam3_intermot/one_click/event_authority_runtime.py",
        "sam3_intermot/one_click/event_authority_learning.py", "sam3_intermot/one_click/event_authority_models.py",
        "sam3_intermot/one_click/joint_intervention_primitives.py", "sam3_intermot/one_click/runtime_file_guard.py")
METRICS = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN")


def identity(family, seed):
    return "PILOT__" + family + "__L3_HARM_AWARE__seed" + str(seed)


def inputs(sequence, family, seed):
    protocol = read_json(PROTOCOL)
    assert protocol["frozen"] and sequence in protocol["sequences"] and family in protocol["families"] and seed in protocol["seeds"]
    assert set(protocol["sequences"]).issubset(preregistration()["split"]["fit"])
    uid = identity(family, seed)
    training = read_json(OUT / "training/event_authority" / (uid + ".json"))
    assert training["pilot_not_main_fit_or_independent_confirmation"]
    assert sha256(training["checkpoint_path"]) == training["checkpoint_sha256"]
    assert all(sha256(ROOT / p) == s for p, s in training["source_freeze"].items())
    return protocol, uid, training


def runtime(sequence, family, seed):
    torch.set_num_threads(1)
    protocol, uid, training = inputs(sequence, family, seed)
    saved = torch.load(training["checkpoint_path"], map_location="cpu", weights_only=False)
    predictor = EventAuthorityPredictor(saved, development_diagnostic=True)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    source = {p: sha256(ROOT / p) for p in CODE}
    receipts = []
    storage(128 << 20)
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        actor, _ = make_actor(event, np.array(anchors[event["anchor_index"]], np.float32))
        click = {"event_frame": event["frame"], "human_anchor": actor.anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
        bridge = LearnedEventBridge(click, actor, predictor=predictor, point=protocol["point"], frames=len(frames))
        bridge.configure_fps(event["fps"])
        name = uid + "__click" + str(event["slot"])
        base = ASSETS / "learned_pilot_v1"
        trace_path = base / "traces" / uid / (event["episode_uid"] + ".jsonl.zst")
        tracker_path = base / "trackers" / name / "data" / (sequence + ".txt")
        for path in (trace_path, tracker_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError("Preserve partial/completed pilot policy replay")
        direct, writes, began = 0, 0, time.monotonic()
        with trace_path.open("xb") as handle, tracker_path.open("x") as tracker:
            proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
            try:
                for payload, rows in frames:
                    f = int(payload["frame"])
                    before = full_tracker_fingerprint(bridge)
                    result = bridge.step(f, rows)
                    result.update(full_tracker_state_before_sha256=before, full_tracker_state_after_sha256=full_tracker_fingerprint(bridge))
                    direct += bool(result["authority"].get("effective_assignment_change"))
                    writes += bool(result["joint_identity_memory_write"])
                    proc.stdin.write((json.dumps(serial(result), sort_keys=True, allow_nan=False) + "\n").encode())
                    tracker.write(trajectory_text([result]))
                proc.stdin.close()
                assert proc.wait() == 0
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
        assert writes == 0, "This pilot isolates authority under frozen P0 memory"
        base_seal = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
        if direct == 0:
            assert sha256(tracker_path) == next(a["sha256"] for a in base_seal["artifacts"] if a["kind"] == "trajectory")
        receipt = {"episode_uid": event["episode_uid"], "event": event, "tracker_name": name, "effective_direct_decisions_NOT_independent_onsets": direct,
                   "memory_writes": writes, "seconds": time.monotonic() - began,
                   "artifacts": [{"kind": k, "path": str(p), "sha256": sha256(p)} for k, p in (("trace", trace_path), ("trajectory", tracker_path))]}
        receipts.append(receipt)
        print({"actual_learned_pilot_episode": event["episode_uid"], "model": uid, "direct_decisions": direct}, flush=True)
    write_json("training/pilot_policy_runtime/" + uid + "/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "experiment_uid": uid, "status": "COMPLETE_ACTUAL_OWN_POLICY_FIT_PILOT",
        "checkpoint_sha256": training["checkpoint_sha256"], "source_freeze": source, "protocol_sha256": sha256(PROTOCOL),
        "candidate_index_sha256": index_sha, "episodes": receipts, "initialization_failures_not_replaced": sum(e["initialization_failure"] for e in init["inputs"]),
        "actual_GT_file_guard": True, "pilot_not_main_fit_or_generalization": True, "scientific_success": None})


def evaluate(sequence, family, seed):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome
    protocol, uid, training = inputs(sequence, family, seed)
    seal_path = OUT / "training/pilot_policy_runtime" / uid / (sequence + ".json")
    seal = read_json(seal_path)
    assert seal["source_freeze"] == {p: sha256(ROOT / p) for p in CODE}
    assert seal["protocol_sha256"] == sha256(PROTOCOL)
    assert all(sha256(a["path"]) == a["sha256"] for e in seal["episodes"] for a in e["artifacts"])
    frames, _ = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {p["frame"]: strict_candidate_matching(rows, gt.get(p["frame"], [])) for p, rows in frames}
    targets = {r["episode_uid"]: r["target_gt_identity"] for r in read_json(OUT / "data/initialization_truth" / (sequence + ".json"))["labels"]}
    counts = {}
    for episode in seal["episodes"]:
        truth_target = targets[episode["episode_uid"]]
        trace = read_zstd_jsonl(artifact(episode, "trace"))
        baseline = read_zstd_jsonl(artifact(read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (episode["episode_uid"] + ".json")), "trace"))
        counter = Counter()
        for a, b in zip(trace, baseline, strict=True):
            assert a["frame"] == b["frame"]
            if a["frame"] <= episode["event"]["frame"]:
                continue
            current = candidate_identity_outcome(a["target_uid"], matched[a["frame"]], truth_target)
            keep = candidate_identity_outcome(b["target_uid"], matched[a["frame"]], truth_target)
            counter["N01_frames"] += current == "TARGET" and keep != "TARGET"
            counter["N10_frames"] += keep == "TARGET" and current != "TARGET"
            counter[current + "_frames"] += 1
        counts[episode["episode_uid"]] = dict(counter)
    evaluation = ASSETS / "learned_pilot_v1/trackeval" / uid / sequence
    evaluation.mkdir(parents=True, exist_ok=False)
    seqmap = evaluation / "seqmap.txt"
    with seqmap.open("x") as handle:
        handle.write("name\n" + sequence + "\n")
    names = [e["tracker_name"] for e in seal["episodes"]]
    metrics, receipt = {}, {"status": "NOT_RUN_NO_VALID_INITIALIZATION"}
    if names:
        pinned = HISTORY / "third_party/MOTIP/TrackEval"
        assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
        command = _trackeval_command(ASSETS / "learned_pilot_v1/trackers", evaluation, names, seqmap, gt_split="train", gt_folder=TRAIN)
        command[2] = str(pinned / "scripts/run_mot_challenge.py")
        log = evaluation / "trackeval.log"
        with log.open("x") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        receipt = {"command": command, "returncode": process.returncode, "log_path": str(log), "log_sha256": sha256(log)}
        if process.returncode:
            raise RuntimeError("Retain failed actual learned-pilot TrackEval")
        metrics = {n: trackeval_summary(parse_trackeval(evaluation, n, [sequence])) for n in names}
        assert all(all(v[k] is not None for k in METRICS) for v in metrics.values())
    base = read_json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
    deltas = {n: {k: v[k] - base["CLICK_C0__click" + n.rsplit("__click", 1)[1]][k] for k in METRICS} for n, v in metrics.items()}
    write_json("training/pilot_policy_results/" + uid + "/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "experiment_uid": uid, "status": "COMPLETE_ACTUAL_FIT_PILOT_FULL_MOT_DIAGNOSTIC",
        "checkpoint_sha256": training["checkpoint_sha256"], "runtime_seal_sha256": sha256(seal_path), "protocol_sha256": sha256(PROTOCOL),
        "all_nine_metrics": metrics, "paired_deltas_vs_C0": deltas, "per_episode_target_components": counts, "receipt": receipt,
        "effective_direct_decisions_NOT_independent_onsets": sum(e["effective_direct_decisions_NOT_independent_onsets"] for e in seal["episodes"]),
        "causal_harm_onsets": "REQUIRED_IF_EFFECTIVE_ACTIONS_NONZERO_NOT_INFERRED_FROM_FRAME_COUNTS",
        "not_main_fit_or_independent_generalization": True, "scientific_success": None})
    append_log("M5_LEARNED_EVENT_FIT_PILOT_FULL_MOT_COMPLETE", sequence=sequence, experiment_uid=uid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["runtime", "evaluate"])
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.action == "runtime":
        with runtime_file_guard():
            runtime(args.sequence, args.family, args.seed)
    else:
        evaluate(args.sequence, args.family, args.seed)
