"""Actual pinned offline future-window metric labels, separate from runtime."""
import argparse
from pathlib import Path
import subprocess
import json
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, HISTORY, read_json, write_json, sha256, development_sequence, preregistration, append_log, storage
from scripts.n72r21r2_events import artifact
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.evaluation.pinned_event_trajectory import PinnedEventEvaluator, METRIC_NAMES

PROTOCOL = OUT / "events/TRAJECTORY_UTILITY_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_trajectory_utility.py", "sam3_intermot/evaluation/pinned_event_trajectory.py")


def evaluator(sequence):
    development_sequence(sequence)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    event = next(e for e in init["inputs"] if not e["initialization_failure"])
    seal = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
    path = artifact(seal, "trajectory")
    ref = next(a for a in seal["artifacts"] if a["kind"] == "trajectory")
    assert sha256(path) == ref["sha256"]
    return PinnedEventEvaluator(HISTORY / "third_party/MOTIP/TrackEval", gt_root=TRAIN,
                               tracker_root=path.parents[2], reference_tracker=seal["tracker_name"],
                               sequence=sequence, frames=event["frames"])


def validate():
    differences = []
    for sequence in preregistration()["fresh_candidate_pilot"]:
        record = read_json(OUT / "events/window_trackeval_results" / (sequence + ".json"))
        selection = record["selected_event"]
        ev = evaluator(sequence)
        seal = read_json(OUT / "events/counterfactual_seals" / selection["episode_uid"] / ("frame" + str(selection["frame"]) + ".json"))
        actual = {}
        for name, info in record["arms"].items():
            arm = next(a for a in seal["artifacts"] if a["branch"] == info["branch"])
            assert sha256(arm["path"]) == arm["sha256"] == info["branch_sha256"]
            rows = read_zstd_jsonl(Path(arm["path"]))
            current = ev.evaluate(rows, first_frame=selection["frame"], length=101)
            delta = {k: current[k] - record["all_nine_metrics"][name][k] for k in METRIC_NAMES}
            assert max(abs(v) for v in delta.values()) < 1e-10, (sequence, name, delta)
            actual[name] = {"metrics": current, "delta_API_vs_CLI": delta}
        differences.append({"sequence": sequence, "source_window_receipt_sha256": sha256(OUT / "events/window_trackeval_results" / (sequence + ".json")), "arms": actual})
    write_json("events/TRAJECTORY_UTILITY_API_CLI_VALIDATION_V1.json", {
        "status": "COMPLETE_ALL_PILOT_ARMS_ALL_NINE_METRICS_API_CLI_AA", "protocol_sha256": sha256(PROTOCOL),
        "source_freeze": {p: sha256(ROOT / p) for p in CODE}, "pilots": differences,
        "scientific_success_not_claimed": True})
    print({"actual_API_vs_CLI": "AA", "pilot_sequences": len(differences), "metric_comparisons": sum(len(s["arms"]) * 9 for s in differences)}, flush=True)


def run(sequence):
    development_sequence(sequence)
    validation = read_json(OUT / "events/TRAJECTORY_UTILITY_API_CLI_VALIDATION_V1.json")
    assert validation["protocol_sha256"] == sha256(PROTOCOL)
    assert validation["source_freeze"] == {p: sha256(ROOT / p) for p in CODE}
    audit_path = OUT / "events/label_audit" / (sequence + ".json")
    audit = read_json(audit_path)
    assert sha256(audit["artifact"]["path"]) == audit["artifact"]["sha256"]
    labels = read_zstd_jsonl(Path(audit["artifact"]["path"]))
    # Verify all immutable runtime evidence before any metric truth use.
    assert all(sha256(r["actual_branch_path"]) == r["actual_branch_sha256"] for r in labels)
    storage(64 << 20)
    output = ASSETS / "offline_trajectory_utility_v1" / (sequence + ".jsonl.zst")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Retain prior/partial actual trajectory utility")
    ev = evaluator(sequence) if labels else None
    grouped = {}
    for row in labels:
        grouped.setdefault(row["event_uid"], []).append(row)
    count, complete = 0, 0
    started = time.monotonic()
    with output.open("xb") as handle:
        proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
        try:
            for uid, records in grouped.items():
                keep = next(r for r in records if r["branch"] == "KEEP")
                keep_rows = read_zstd_jsonl(Path(keep["actual_branch_path"]))
                full = keep["offline_supervision_labels"]["future"]["H100"]["complete"]
                base = ev.evaluate(keep_rows[1:], first_frame=keep["frame"] + 1, length=100) if full else None
                for row in records:
                    rows = read_zstd_jsonl(Path(row["actual_branch_path"]))
                    assert row["offline_supervision_labels"]["future"]["H100"]["complete"] == full
                    metrics = base if row["branch"] == "KEEP" else ev.evaluate(rows[1:], first_frame=row["frame"] + 1, length=100) if full else None
                    delta = {k: metrics[k] - base[k] for k in METRIC_NAMES} if full else None
                    result = {"event_uid": uid, "sequence": sequence, "role": row["role"], "branch": row["branch"],
                              "complete_H100": full, "original_future_axis": [row["frame"] + 1, row["frame"] + 100],
                              "actual_nine_metrics": metrics, "paired_delta_vs_own_KEEP": delta,
                              "L5_value_label": .5 * (delta["HOTA"] + delta["AssA"]) if full else None,
                              "actual_branch_sha256": row["actual_branch_sha256"], "KEEP_branch_sha256": keep["actual_branch_sha256"],
                              "not_full_adaptive_policy_MOT_result": True, "never_runtime_feature": True}
                    proc.stdin.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
                    count += 1
                    complete += full
                print({"actual_future_trajectory_utility_event": uid, "completed_arms": count}, flush=True)
            proc.stdin.close()
            assert proc.wait() == 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    write_json("events/trajectory_utility_audit/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_PINNED_EVENT_TRAJECTORY_UTILITY",
        "arms": count, "complete_H100_arms": complete, "seconds": time.monotonic() - started,
        "artifact": {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size},
        "source_freeze": {p: sha256(ROOT / p) for p in CODE}, "label_audit_sha256": sha256(audit_path),
        "GT_sha256": sha256(TRAIN / sequence / "gt/gt.txt"), "protocol_sha256": sha256(PROTOCOL),
        "API_CLI_validation_sha256": sha256(OUT / "events/TRAJECTORY_UTILITY_API_CLI_VALIDATION_V1.json"),
        "overlapping_windows_not_independent_correction_onsets": True})
    append_log("M3_ACTUAL_TRAJECTORY_UTILITY_COMPLETE", sequence=sequence, arms=count, complete_H100_arms=complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate", "label"])
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.action == "validate":
        validate()
    else:
        run(args.sequence)
