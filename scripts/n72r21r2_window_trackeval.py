"""Actual preregistered paired window TrackEval, not global-policy metrics."""
import argparse
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, read_json, write_json, sha256, development_sequence, storage, append_log
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from scripts.n72r20r3r2r3_pipeline import _trackeval_command, parse_trackeval, trackeval_summary
from sam3_intermot.evaluation.mot_detection_audit import audit_unchanged_detections

METRICS = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN")


def run(sequence):
    development_sequence(sequence)
    storage(64 << 20)
    label_audit = read_json(OUT / "events/label_audit" / (sequence + ".json"))
    assert sha256(label_audit["artifact"]["path"]) == label_audit["artifact"]["sha256"]
    selection_path = OUT / "events/window_trackeval_selection" / (sequence + ".json")
    selection = read_json(selection_path)
    protocol_path = OUT / "events/WINDOW_TRACKEVAL_PROTOCOL_V1.json"
    assert selection["protocol_sha256"] == sha256(protocol_path) and not selection["sampling_uses_action_effects"]
    relative = "events/window_trackeval_results/" + sequence + ".json"
    if selection["selection"] is None:
        write_json(relative, {"stage": "N72R21R2", "sequence": sequence, "status": "NOT_RUN_NO_REGISTERED_COMPLETE_WINDOW", "reason": selection["status"]})
        return
    selected = selection["selection"]
    start = selected["frame"]
    seal_path = OUT / "events/counterfactual_seals" / selected["episode_uid"] / ("frame" + str(start) + ".json")
    seal = read_json(seal_path)
    assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
    event = next(e for e in read_json(OUT / "data/initialization" / (sequence + ".json"))["inputs"] if e["episode_uid"] == selected["episode_uid"])
    root = ASSETS / "counterfactual_window_trackeval_v1" / sequence
    root.mkdir(parents=True, exist_ok=False)
    gt_root = root / "GT" / sequence
    (gt_root / "gt").mkdir(parents=True)
    with (gt_root / "gt/gt.txt").open("x") as handle:
        for line in (TRAIN / sequence / "gt/gt.txt").read_text().splitlines():
            fields = line.split(",")
            if not fields[0].strip():
                continue
            original = int(fields[0])
            if start + 1 <= original <= start + 101:
                fields[0] = str(original - start)
                handle.write(",".join(fields) + "\n")
    with (gt_root / "seqinfo.ini").open("x") as handle:
        handle.write(f'[Sequence]\nname={sequence}\nimDir=img1\nframeRate={event["fps"]}\nseqLength=101\nimWidth={event["width"]}\nimHeight={event["height"]}\nimExt=.jpg\n')
    mapping, texts = {}, {}
    for i, arm in enumerate(seal["artifacts"]):
        rows = read_zstd_jsonl(Path(arm["path"]))
        assert [r["frame"] for r in rows] == list(range(start, start + 101))
        name = "ARM" + str(i)
        mapping[name] = {"branch": arm["branch"], "branch_sha256": arm["sha256"], "action": arm["current_action"]}
        texts[name] = trajectory_text([dict(r, frame=r["frame"] - start) for r in rows])
        path = root / "trackers" / name / "data" / (sequence + ".txt")
        path.parent.mkdir(parents=True)
        with path.open("x") as handle:
            handle.write(texts[name])
    keep = next(n for n, a in mapping.items() if a["branch"] == "KEEP")
    unchanged = {n: audit_unchanged_detections(texts[keep], text) for n, text in texts.items()}
    seqmap = root / "seqmap.txt"
    with seqmap.open("x") as handle:
        handle.write("name\n" + sequence + "\n")
    evaluation = root / "evaluation"
    evaluation.mkdir()
    pinned = HISTORY / "third_party/MOTIP/TrackEval"
    commit = subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip()
    assert commit == read_json(protocol_path)["TrackEval_commit"]
    command = _trackeval_command(root / "trackers", evaluation, list(mapping), seqmap, gt_split="train", gt_folder=root / "GT")
    command[2] = str(pinned / "scripts/run_mot_challenge.py")
    log = evaluation / "trackeval.log"
    started = time.monotonic()
    with log.open("x") as handle:
        process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    receipt = {"stage": "N72R21R2", "command": command, "returncode": process.returncode, "seconds": time.monotonic() - started,
               "log_path": str(log), "log_sha256": sha256(log), "TrackEval_commit": commit}
    write_json("events/window_trackeval_invocations/" + sequence + ".json", receipt)
    if process.returncode:
        raise RuntimeError("Preserve failed actual window evaluation")
    metrics = {n: trackeval_summary(parse_trackeval(evaluation, n, [sequence])) for n in mapping}
    assert all(all(v[k] is not None for k in METRICS) for v in metrics.values())
    write_json(relative, {"stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_PINNED_FULL_GLOBAL_WINDOW_METRICS",
               "selected_event": selected, "selection_sha256": sha256(selection_path), "protocol_sha256": sha256(protocol_path),
               "CF_seal_sha256": sha256(seal_path), "arms": mapping, "KEEP_reference": keep, "all_nine_metrics": metrics,
               "paired_delta_vs_own_KEEP": {n: {k: metrics[n][k] - metrics[keep][k] for k in METRICS} for n in metrics},
               "unchanged_detection_multiset_audit": unchanged, "receipt": receipt,
               "source_sha256": sha256(Path(__file__)), "original_GT_sha256": sha256(TRAIN / sequence / "gt/gt.txt"),
               "evaluation_only_original_frame_origin_remap": True, "public_IDs_not_remapped": True,
               "not_full_adaptive_policy_MOT_or_independent_confirmation": True, "overlapping_window_metrics_not_summed": True})
    append_log("M3_PREREGISTERED_ACTUAL_WINDOW_TRACKEVAL", sequence=sequence, arms=len(mapping), event=selected)
    print({"actual_window_TrackEval_complete": sequence, "arms": len(mapping), "original_window": [start, start + 100]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    run(parser.parse_args().sequence)
