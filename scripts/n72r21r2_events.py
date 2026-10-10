"""Mine sealed fresh baseline event intervals; CF action labels stay pending.

This is an OFFLINE FIT/INNER supervision worker, not a deployable GT-aided
tracker. Runtime evidence and offline annotations occupy separate fields.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, TRAIN, read_json, write_json, sha256, development_sequence, append_log
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.evaluation.causal_identity_events import intervals, candidate_identity_outcome, stratified_onsets


def artifact(seal, kind):
    record = next(a for a in seal["artifacts"] if a["kind"] == kind)
    assert sha256(record["path"]) == record["sha256"]
    return Path(record["path"])


def run(sequence):
    development_sequence(sequence)
    result_path = OUT / "mot/baseline_results" / (sequence + ".json")
    read_json(result_path)  # Require actual pinned full baseline evaluation first.
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    truth = read_json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    identities = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    frames, index_sha = checked_frames(sequence)
    gt = dancetrack_annotations(TRAIN / sequence)
    matched = {int(p["frame"]): strict_candidate_matching(rows, gt.get(int(p["frame"]), [])) for p, rows in frames}
    episodes = []
    for event in init["inputs"]:
        if event["initialization_failure"]:
            episodes.append({"episode_uid": event["episode_uid"], "status": "INITIALIZATION_FAILURE_NO_REPLACEMENT", "events": []})
            continue
        base_seal_path = OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json")
        shadow_seal_path = OUT / "mot/baseline_runtime/ACIB_SHADOW_P0" / (event["episode_uid"] + ".json")
        base = read_zstd_jsonl(artifact(read_json(base_seal_path), "trace"))
        shadow = read_zstd_jsonl(artifact(read_json(shadow_seal_path), "trace"))
        target = identities[event["episode_uid"]]
        target_public = base[event["frame"]]["target_public_id"]
        table, runtime, previous_native, native_streak = [], {}, None, 0
        for f in range(event["frame"] + 1, len(frames)):
            now = base[f]
            candidates = frames[f][1]
            positive = sorted(uid for uid, label in matched[f].items() if label == target)
            visible = any(a["identity"] == target for a in gt.get(f, []))
            selected = now["target_uid"]
            outcome = candidate_identity_outcome(selected, matched[f], target)
            category = "E0_BASELINE_CORRECT" if outcome == "TARGET" else "E8_TARGET_UNAVAILABLE" if not visible else "E7_NO_VALID_CANDIDATE" if not positive else "E2_TARGET_LOST" if selected is None else "E1_WRONG_IDENTITY_MATCH"
            owner = {o["candidate_uid"]: int(o["public_id"]) for o in now["outputs"]}
            selected_row = next((r for r in candidates if r["candidate_uid"] == selected), None)
            native = None if selected_row is None else int(selected_row["native_tid"])
            native_streak = native_streak + 1 if native is not None and native == previous_native else 1 if native is not None else 0
            previous_native = native
            extra = []
            if positive and owner.get(positive[0]) != target_public:
                extra.append("E4_COMPETITOR_OWNED")
            if category == "E1_WRONG_IDENTITY_MATCH" and native_streak >= 3:
                extra.append("E5_NATIVE_LOCK_IN_OBSERVATIONAL")
            if positive and table and not table[-1]["positive_available"]:
                extra.append("E3_REAPPEARANCE")
            features = shadow[f].get("authority", {}).get("features")
            runtime[f] = {"prestate_sha256": now["state_before"], "baseline_uid": selected,
                          "challenger_uid": shadow[f].get("identity_decision", {}).get("proposed_candidate_uid"),
                          "current_UIDs": [r["candidate_uid"] for r in candidates], "candidate_owner_map": owner,
                          "proposal_features": features, "native_streak_current_past_only": native_streak,
                          "GT_or_future_label_fields": False}
            table.append({"frame": f, "category": category, "extra": extra, "positive_available": bool(positive),
                          "visible": visible, "baseline_outcome": outcome, "baseline_correct": outcome == "TARGET"})
        runs = intervals(table)
        starts = {r["start_frame"] for r in runs}
        pending = []
        for row in table:
            f = row["frame"]
            # Fixed temporal probes + offline category onsets, no action-effect selection.
            if f not in starts and (f - event["frame"] - 1) % 16 and "E3_REAPPEARANCE" not in row["extra"]:
                continue
            run = next(r for r in runs if r["start_frame"] <= f <= r["end_frame"])
            categories = [row["category"], *row["extra"]]
            if row["category"] in ("E1_WRONG_IDENTITY_MATCH", "E2_TARGET_LOST") and run["frames"] >= 20:
                categories.append("E9_LONG_PROPAGATION_ERROR_OBSERVATIONAL")
            future = {}
            for horizon in (1, 5, 20, 50, 100):
                window = [r for r in table if f <= r["frame"] < f + horizon]
                future["H" + str(horizon)] = {"complete": len(window) == horizon, "observed_frames": len(window),
                                              "C0_correct": sum(r["baseline_correct"] for r in window),
                                              "C0_verified_OTHER": sum(r["baseline_outcome"] == "VERIFIED_OTHER" for r in window)}
            pending.append({"event_uid": event["episode_uid"] + "__f" + str(f), "frame": f, "categories": categories,
                            "anonymous_identity_scope": hashlib.sha256((sequence + ":" + str(target)).encode()).hexdigest()[:16],
                            "observational_interval_start": run["start_frame"], "independent_causal_origin_proven": False,
                            "runtime_features": runtime[f], "offline_supervision_labels": {**row, "observed_interval_frames": run["frames"],
                                                                                           "actual_C0_future_observations_not_CF_action_rewards": future},
                            "counterfactual_action_labels": "PENDING_ACTUAL_SAME_PRESTATE_BRANCHES"})
        sampled = stratified_onsets(pending, maximum_additional=8)
        episodes.append({"episode_uid": event["episode_uid"], "status": "COMPLETE_ACTUAL_SEALED_BASELINE_EVENT_MINING",
                         "base_seal_sha256": sha256(base_seal_path), "shadow_seal_sha256": sha256(shadow_seal_path),
                         "frame_class_counts": dict(Counter(r["category"] for r in table)), "observational_intervals": runs,
                         "event_probe_rows": pending, "preregistered_long_branch_frames": sampled,
                         "class_counts_overlap_and_propagated_frames_not_independent_events": True})
    write_json("events/corpus/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence, "episodes": episodes,
               "candidate_index_sha256": index_sha, "baseline_result_sha256": sha256(result_path),
               "source_sha256": sha256(Path(__file__)), "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
               "GT_OFFLINE_ONLY": True, "same_prestate_counterfactual_labels_not_yet_generated": True,
               "scientific_event_success_not_claimed": True})
    append_log("M2_BASELINE_EVENT_INTERVAL_MINING_COMPLETE", sequence=sequence, valid_clicks=sum(e["status"].startswith("COMPLETE") for e in episodes),
               proposed_long_branch_positions=sum(len(e.get("preregistered_long_branch_frames", [])) for e in episodes))
    print({"sequence": sequence, "valid_episodes": sum(e["status"].startswith("COMPLETE") for e in episodes),
           "long_branch_positions": sum(len(e.get("preregistered_long_branch_frames", [])) for e in episodes), "actual_CF_labels": "PENDING"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    run(args.sequence)
