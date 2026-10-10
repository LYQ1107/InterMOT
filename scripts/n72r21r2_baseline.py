"""Fresh input sole-click preparation, GT-free joint replay and offline audit.

The runtime action reads only the sealed click, frozen model and current tape.
Offline truth is opened exclusively by the separate prepare/evaluate actions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, R1, sha256, read_json, write_json, storage, development_sequence, append_log
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.one_click.mot_bridge import MOTIdentityBridge
from sam3_intermot.one_click.safe_mot_bridge import SafeMOTIdentityBridge
from sam3_intermot.one_click.intervention_gate import GatePolicy
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork
from sam3_intermot.one_click.acib_trusted_runtime import TrustedACIBRecognizer
from scripts.n72r21_t2_replay import DecisionCapture


def checked_frames(sequence):
    development_sequence(sequence)
    seal_path = OUT / "data/candidate_integrity" / (sequence + ".json")
    seal = read_json(seal_path)
    assert seal["full_axis_unique_current_UIDs_features_offsets"] and not seal["runtime_GT_read"]
    index_path = Path(seal["index_path"])
    assert sha256(index_path) == seal["index_sha256"]
    index = read_json(index_path)
    for key in ("metadata", "embeddings"):
        assert sha256(index[key]) == seal[key + "_sha256"]
    frames = [(p, [r for r in rows if valid_geometry(r)]) for p, rows in load_candidate_frames(ASSETS, sequence)]
    assert [int(p["frame"]) for p, _ in frames] == list(range(seal["frames"]))
    return frames, seal["index_sha256"]


def prepare(sequence):
    # GT is legal ONLY as a simulated first human observation here.
    from sam3_intermot.one_click.datasets import dancetrack_annotations, dancetrack_info, strict_candidate_matching
    from sam3_intermot.identity_probe.dataset import GTBox
    from sam3_intermot.identity_probe.crop import load_person_crops
    from sam3_intermot.identity_probe.encoders import OSNetEncoder
    protocol = development_sequence(sequence)
    frames, index_sha = checked_frames(sequence)
    path = OUT / "data/initialization" / (sequence + ".json")
    if path.exists():
        saved = read_json(path)
        assert saved["source_sha256"] == sha256(Path(__file__)) and saved["candidate_index_sha256"] == index_sha
        assert sha256(saved["anchor_path"]) == saved["anchor_sha256"]
        return
    torch.set_num_threads(1)
    gt = dancetrack_annotations(TRAIN / sequence)
    info = dancetrack_info(TRAIN / sequence)
    first = {}
    for frame, rows in sorted(gt.items()):
        for row in sorted(rows, key=lambda r: r["identity"]):
            x, y, right, bottom = row["box"]
            if row["identity"] not in first and right - x >= 16 and bottom - y >= 48:
                first[row["identity"]] = (frame, row)
    selected = sorted(first.items(), key=lambda pair: (pair[1][0], pair[0]))[:3]
    checkpoint = read_json(OUT / "audit/CHECKPOINT_LINEAGE.json")["weights"]["osnet"]
    assert sha256(checkpoint["path"]) == checkpoint["sha256"]
    encoder = OSNetEncoder(Path(checkpoint["path"]), "cpu")
    inputs, labels, vectors = [], [], []
    for slot, (identity, (frame, row)) in enumerate(selected):
        x, y, right, bottom = row["box"]
        box = GTBox(sequence, frame + 1, identity, x, y, right - x, bottom - y, row["confidence"], 1, row["visibility"])
        crops = load_person_crops(TRAIN / sequence / "img1" / f"{frame+1:08d}.jpg", [box])
        vector = encoder.encode(torch.stack(crops)).cpu().numpy()[0].astype(np.float32)
        matching = strict_candidate_matching(frames[frame][1], gt.get(frame, []))
        positive = sorted(uid for uid, target in matching.items() if target == identity)
        assert len(positive) <= 1
        uid = positive[0] if positive else None
        episode = sequence + "__R2click" + str(slot)
        inputs.append({"sequence": sequence, "role": "FIT" if sequence in protocol["split"]["fit"] else "INNER",
                       "episode_uid": episode, "slot": slot, "frame": frame, "box_xyxy": list(row["box"]),
                       "clicked_candidate_uid": uid, "initialization_failure": uid is None,
                       "anchor_index": slot, "candidate_index_sha256": index_sha, **info,
                       "sole_click": True, "runtime_GT_identity_fields": False, "runtime_future_GT_used": False})
        labels.append({"episode_uid": episode, "target_gt_identity": identity, "offline_only": True})
        vectors.append(vector)
    if not vectors:
        raise ValueError("No eligible sole-human observations; report data limitation")
    anchor_path = ASSETS / "sole_anchors" / (sequence + ".npy")
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    with anchor_path.open("xb") as handle:
        np.save(handle, np.stack(vectors))
    write_json("data/initialization_truth/" + sequence + ".json", {"stage": "N72R21R2", "labels": labels,
               "GT_sha256": sha256(TRAIN / sequence / "gt/gt.txt"), "runtime_never_reads_this_file": True})
    write_json("data/initialization/" + sequence + ".json", {"stage": "N72R21R2", "inputs": inputs,
               "anchor_path": str(anchor_path), "anchor_sha256": sha256(anchor_path), "encoder_sha256": checkpoint["sha256"],
               "candidate_index_sha256": index_sha, "source_sha256": sha256(Path(__file__)),
               "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "strict_one_to_one_click_matching": True,
               "initialization_failures_retained_without_replacement": sum(e["initialization_failure"] for e in inputs)})
    append_log("M1_FRESH_SOLE_CLICK_PREPARATION", sequence=sequence, clicks=len(inputs), initialization_failures=sum(e["initialization_failure"] for e in inputs))
    print(json.dumps({"fresh_sole_clicks_prepared": sequence, "clicks": len(inputs), "failures": sum(e["initialization_failure"] for e in inputs)}), flush=True)


def make_actor(event, anchor, policy="P0"):
    source = read_json(OUT / "audit/CHECKPOINT_LINEAGE.json")["common_frozen_identity_model"]
    assert sha256(source["path"]) == source["sha256"] and sha256(source["fit_record_path"]) == source["fit_record_sha256"]
    assert event["sequence"] not in source["fit_sequences"] + [source["inner_sequence"], source["outer_sequence"]]
    saved = torch.load(source["path"], weights_only=True, map_location="cpu")
    assert saved["schema"] == read_json(source["fit_record_path"])["schema"]
    model = ACIBMemoryNetwork()
    model.load_state_dict(saved["model"], strict=True)
    wrapper = DecisionCapture(model).eval()
    actor = TrustedACIBRecognizer(wrapper, anchor, event["episode_uid"], capacity=8, policy=policy)
    actor.start_recording(event["sequence"], fps=event["fps"], width=event["width"], height=event["height"], initial_frame=event["frame"], initial_box=event["box_xyxy"])
    return actor, source


def replay(sequence):
    # No GT adapter import, truth-file reads or image inputs on this path.
    torch.set_num_threads(1)
    frames, index_sha = checked_frames(sequence)
    init_path = OUT / "data/initialization" / (sequence + ".json")
    init = read_json(init_path)
    assert init["candidate_index_sha256"] == index_sha and sha256(init["anchor_path"]) == init["anchor_sha256"]
    assert init["source_sha256"] == sha256(Path(__file__))
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    for event in init["inputs"]:
        if event["initialization_failure"]:
            write_json("mot/baseline_runtime/" + event["episode_uid"] + ".json", {"stage": "N72R21R2", "status": "INITIALIZATION_FAILURE_NOT_REPLACED", "event": event, "artifacts": []})
            continue
        anchor = np.array(anchors[event["anchor_index"]], np.float32)
        click = {"event_frame": event["frame"], "human_anchor": anchor, "target_candidate_uid": event["clicked_candidate_uid"], "target_box_xyxy": event["box_xyxy"]}
        for case in ("CLICK_C0", "ACIB_SHADOW_P0"):
            seal_relative = "mot/baseline_runtime/" + case + "/" + event["episode_uid"] + ".json"
            if (OUT / seal_relative).exists():
                seal = read_json(OUT / seal_relative)
                assert seal["source_sha256"] == sha256(Path(__file__))
                assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
                continue
            actor, source = (None, None) if case == "CLICK_C0" else make_actor(event, anchor)
            bridge = MOTIdentityBridge(click, frames=len(frames)) if actor is None else SafeMOTIdentityBridge(click, actor, policy=GatePolicy(family="shadow"), frames=len(frames))
            bridge.configure_fps(event["fps"])
            name = case + "__click" + str(event["slot"])
            trace_path = ASSETS / "fresh_baselines/traces" / case / (event["episode_uid"] + ".jsonl.zst")
            trajectory_path = ASSETS / "fresh_baselines/trackers" / name / "data" / (sequence + ".txt")
            for path in (trace_path, trajectory_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError("Preserve unsealed fresh replay attempt")
            started = time.monotonic()
            baseline = None
            if actor is not None:
                old = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
                baseline = read_zstd_jsonl(Path(next(a["path"] for a in old["artifacts"] if a["kind"] == "trace")))
            with trace_path.open("xb") as compressed, trajectory_path.open("x") as trajectory:
                proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=compressed)
                try:
                    for payload, rows in frames:
                        f = int(payload["frame"])
                        result = bridge.step(f, rows)
                        if baseline is not None:
                            for key in ("outputs", "target_uid", "state_before", "state_after"):
                                assert result[key] == baseline[f][key], (event["episode_uid"], f, key)
                        result.update(episode_uid=event["episode_uid"], case=case, current_candidate_index_sha256=index_sha)
                        proc.stdin.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
                        trajectory.write(trajectory_text([result]))
                    proc.stdin.close()
                    assert proc.wait() == 0
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
            if baseline is not None:
                assert sha256(trajectory_path) == next(a["sha256"] for a in old["artifacts"] if a["kind"] == "trajectory")
            write_json(seal_relative, {"stage": "N72R21R2", "status": "COMPLETE_ACTUAL_FRESH_FULL_JOINT_RUNTIME", "sequence": sequence,
                       "case": case, "tracker_name": name, "event": event, "source_sha256": sha256(Path(__file__)),
                       "candidate_index_sha256": index_sha, "initialization_sha256": sha256(init_path), "model_source": source,
                       "all_original_frames": len(frames), "seconds": time.monotonic() - started, "GT_runtime_read": False,
                       "one_click_only": True, "full_global_unique_ownership": True, "shadow_C0_complete_states_and_bytes_AA": baseline is not None,
                       "artifacts": [{"kind": kind, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for kind, path in (("trace", trace_path), ("trajectory", trajectory_path))]})
            print(json.dumps({"fresh_baseline_replay_complete": event["episode_uid"], "case": case, "seconds": time.monotonic() - started}), flush=True)
    append_log("M1_FRESH_BASELINE_REPLAYS_COMPLETE", sequence=sequence)


def evaluate(sequence):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    development_sequence(sequence)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    valid = [e for e in init["inputs"] if not e["initialization_failure"]]
    all_seals = {}
    for event in valid:
        for case in ("CLICK_C0", "ACIB_SHADOW_P0"):
            seal = read_json(OUT / "mot/baseline_runtime" / case / (event["episode_uid"] + ".json"))
            assert seal["source_sha256"] == sha256(Path(__file__))
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            all_seals[case, event["episode_uid"]] = seal
    frames, index_sha = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {int(p["frame"]): strict_candidate_matching(rows, gt.get(int(p["frame"]), [])) for p, rows in frames}
    labels = {r["episode_uid"]: r["target_gt_identity"] for r in read_json(OUT / "data/initialization_truth" / (sequence + ".json"))["labels"]}
    coverage = {}
    for event in init["inputs"]:
        target = labels[event["episode_uid"]]
        visible = available = correct = other = unknown = 0
        trace = None
        if not event["initialization_failure"]:
            seal = all_seals["CLICK_C0", event["episode_uid"]]
            trace = read_zstd_jsonl(Path(next(a["path"] for a in seal["artifacts"] if a["kind"] == "trace")))
        for f in range(event["frame"] + 1, len(frames)):
            is_visible = any(a["identity"] == target for a in gt.get(f, []))
            has_positive = target in matched[f].values()
            visible += is_visible
            available += has_positive
            if trace is not None:
                uid = trace[f]["target_uid"]
                identity = matched[f].get(uid)
                correct += bool(uid is not None and identity == target)
                other += bool(uid is not None and identity is not None and identity != target)
                unknown += bool(uid is not None and identity is None)
        coverage[event["episode_uid"]] = {"visible_frames": visible, "strict_positive_available_frames": available,
                                         "coverage_given_visible": available / visible if visible else None,
                                         "C0_correct_frames": correct, "C0_verified_OTHER_takeover_frames": other,
                                         "C0_UNKNOWN_frames": unknown, "initialization_failure": event["initialization_failure"]}
    metrics, invocation = {}, {"status": "NOT_RUN_NO_VALID_INITIALIZATION"}
    if valid:
        names = sorted({s["tracker_name"] for s in all_seals.values()})
        pinned = HISTORY / "third_party/MOTIP/TrackEval"
        assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
        eval_root = ASSETS / "fresh_baselines/trackeval" / sequence
        eval_root.mkdir(parents=True, exist_ok=False)
        seqmap = eval_root / "seqmap.txt"
        with seqmap.open("x") as handle:
            handle.write("name\n" + sequence + "\n")
        command = _trackeval_command(ASSETS / "fresh_baselines/trackers", eval_root, names, seqmap, gt_split="train", gt_folder=TRAIN)
        command[2] = str(pinned / "scripts/run_mot_challenge.py")
        log_path = eval_root / "trackeval.log"
        with log_path.open("x") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        invocation = {"command": command, "returncode": result.returncode, "log_path": str(log_path), "log_sha256": sha256(log_path)}
        write_json("mot/baseline_invocations/" + sequence + ".json", invocation)
        if result.returncode:
            raise RuntimeError("Fresh baseline TrackEval failed; preserve actual attempt")
        metrics = {name: trackeval_summary(parse_trackeval(eval_root, name, [sequence])) for name in names}
        for values in metrics.values():
            assert all(values[k] is not None for k in ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN"))
        for event in valid:
            a, b = (case + "__click" + str(event["slot"]) for case in ("CLICK_C0", "ACIB_SHADOW_P0"))
            for metric in ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN"):
                assert metrics[a][metric] == metrics[b][metric]
    write_json("mot/baseline_results/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_FRESH_INPUT_BASELINE",
               "actual_nine_metrics": metrics, "invocation": invocation, "coverage": coverage, "candidate_index_sha256": index_sha,
               "density_integrity_sha256": sha256(OUT / "data/candidate_integrity" / (sequence + ".json")),
               "GT_labels_offline_only": True, "clicks_not_independent_sequence_clusters": True, "scientific_success_not_established": True})
    append_log("M1_FRESH_BASELINE_NINE_METRIC_EVALUATION_COMPLETE", sequence=sequence, valid_clicks=len(valid), actual_trackers=len(metrics))
    print(json.dumps({"fresh_baseline_evaluation_complete": sequence, "valid_clicks": len(valid), "coverage": coverage, "actual_trackers": len(metrics)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "replay", "evaluate"])
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "replay": replay, "evaluate": evaluate}[args.action](args.sequence)
